#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""nn_train —— 序列模型族的**公共训练/推理/评估全链**（iTransformer / TimeXer / 前缀条件化共用）。

它收敛掉什么
============
三种神经臂如果各写一遍训练循环，必然出现经典的四处漂移：划分不同、早停看了 val、
种子约定不同、单位门缺一个作用域。本模块把这些**一次写死**，模型只提供"网络 + 配置"：

    spec = NNSpec(name="iTransformer-plan", policy="plan70", use_personal=False, ...)
    train_neural(spec)          # registry → trainer(protocol) → infer → eval_trip + 落报告

关键设计（与源树 ``JJ_pathway`` / ``AA_prefixxer`` 同构）
========================================================
* **冻结基线当骨架**：``pred = base · exp(clamp(Δ, ±0.7))``。``base`` 是 SePT-70 的
  **冻结 plan-time 段级预测按趟（或后缀段）求和**；只训练 ``Δ``。
  ⇒ Δ=0 时预测=基线，于是"前缀信息值多少"是一个**直接测量**，不是两次独立实验的差。
* **个性化是一个开关**：``spec.use_personal``。关闭时个性化分支整个不建，
  网络退化为**纯 plan-time 外生模型**（= plan-time 基线 + 可有可无的神经修正）。
* **口径两维显式**：``base=full`` ⇒ ``(info_condition, y_full)``；
  ``base=rem`` ⇒ ``(info_condition, y_rem@phi)``。混排会被 registry / eval_trip 拦下。

三种基线来源（``spec.base``）
=============================
* ``"tree_full"`` ：冻结 SePT-70 → 按**全程段**求和（plan-time / y_full）
* ``"tree_rem"``  ：冻结 SePT-70 → 按**后缀段**求和（φ 由 ``spec.phi`` 给定）
* ``"frozen_npz"``：``spec.frozen_path`` 指向的**既有逐趟预测**（源树已冻结产物；
  重训代价过高时的合法入口，必须在 README 里声明是代理口径）

纪律（逐位复现的成败就在这里）
==============================
* 权重初始化发生在 ``apply_seed`` **之后**（``run_protocol_loop`` 负责 seed 与 init 顺序）。
* 早停**只看 train 内部 holdout**（框架固化，模型无从干预）。
* 一切归约走 mask；``Lmax`` 由数据包给定，**不按批内最大长度重排**。
* 同种子重跑逐位一致（框架的 ``repro_check``）；本模块默认开启。

本文件**不含 assert**。
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_RELEASE = Path(__file__).resolve().parents[1]
for _p in (_RELEASE / "framework", _RELEASE / "preprocess", _RELEASE):
    _s = str(_p)
    if _s in sys.path:
        sys.path.remove(_s)
    sys.path.insert(0, _s)

import dataio as DIO                                                 # noqa: E402
import eval_trip as ET                                               # noqa: E402
import features as FT                                                # noqa: E402
import infer as INF                                                  # noqa: E402
import nn_blocks as NB                                               # noqa: E402
import paths as pth                                                  # noqa: E402
import registry as reg                                               # noqa: E402
import trainer as TR                                                 # noqa: E402

__all__ = ["NNSpec", "NNRun", "NNError", "load_frozen_baseline", "tree_baseline",
           "build_inputs", "train_neural", "eval_from_handle", "list_runs"]


class NNError(RuntimeError):
    """神经臂全链的错误（基线缺失 / 形状不符 / 口径不明）。"""


# --------------------------------------------------------------------------- 配置
@dataclass
class NNSpec:
    """一条神经臂的完整定义（**口径两维 + 个性化开关 + 特征集**都在这里）。"""

    name: str
    family: str = "neural"
    arch: str = "itransformer"          # itransformer | timexer | prefix
    policy: str = "plan70"              # dataio.POLICIES 的名字
    base: str = "tree_full"             # tree_full | tree_rem | frozen_npz | none
    use_personal: bool = False          # **个性化总开关**
    use_cross: bool = False             # True=TimeXer 式跨注意力；False=拼接
    phi: float = 0.0
    d: int = 24
    heads: int = 4
    layers: int = 2
    d_ff: int = 48
    dropout: float = 0.1
    lr: float = 2e-3
    weight_decay: float = 1e-4
    max_steps: int = 1500
    batch: int = 64
    eval_every: int = 100
    patience_epochs: int = 6
    smoke: bool = False                 # True ⇒ 覆盖为"接口冒烟"配置（**不看数字**）
    seed: int = 42
    frozen_path: str = ""               # base="frozen_npz" 时用（.npz：file_, pred_wh[, phi]）
    frozen_key: str = "pred_wh"
    info_override: str = ""             # 覆盖 info_condition（个性化臂需声明 "realized"）
    notes: str = ""
    source: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- 口径
    @property
    def info_condition(self) -> str:
        """信息条件：``info_override`` 优先，否则取 ``policy`` 的。

        ★ 为什么要这个覆盖：本树的个性化臂用的是**同一套 70 项特征**（plan-time 外生），
          但**个性化分支吃的是实测前缀** ⇒ 整条臂的信息条件是 ``realized``。
          特征集（70 项）与信息条件（realized）是**两个正交的维度**，必须能分开声明；
          混成一个字段就会出现"φ=0.2 时特征悄悄从 70 变成 10"这类事故。
        """
        if self.info_override:
            return str(self.info_override)
        return DIO.get_policy(self.policy).info_condition

    @property
    def policy_obj(self) -> DIO.Policy:
        """数据口径策略：**特征集取 ``policy``（默认 plan70），口径两维取 §info_condition + φ**。"""
        base_pol = DIO.get_policy(self.policy)
        ic = self.info_condition
        if self.base in ("tree_rem", "frozen_npz") and self.phi > 0:
            return DIO.Policy(ic, "y_rem@%.6g" % bot_caliber(self.phi), self.phi,
                              base_pol.feats_list)
        return DIO.Policy(ic, "y_full", 0.0, base_pol.feats_list)

    @property
    def target_caliber(self) -> str:
        return self.policy_obj.target_caliber

    def to_row(self) -> dict[str, Any]:
        return {"name": self.name, "arch": self.arch, "family": self.family,
                "policy": self.policy, "base": self.base, "phi": self.phi,
                "use_personal": self.use_personal, "use_cross": self.use_cross,
                "info_condition": self.info_condition,
                "target_caliber": self.target_caliber,
                "d": self.d, "heads": self.heads, "layers": self.layers, "d_ff": self.d_ff,
                "dropout": self.dropout, "lr": self.lr, "wd": self.weight_decay,
                "max_steps": self.max_steps, "batch": self.batch, "seed": self.seed,
                "smoke": bool(self.smoke),
                "purpose": ("接口冒烟（证明注册可用 / 链路通 / 能产出标准 trip_preds.npz；"
                            "**不看收敛与精度**）" if self.smoke else "正式口径")}

    def smoked(self, n_trips: int = 24) -> "NNSpec":
        """返回"接口冒烟"配置：**几十步 / 1 个种子 / 小步数**，只证明链路通。

        ★ 为什么它必须是**另一个方法**而不是就地改字段：冒烟跑出来的数字**没有任何
          可比性**（没收敛），一旦和正式态混进同一张榜就是污染。``smoke=True`` 会随
          ``eval_report.json`` 落盘，于是"这个数字是冒烟来的"永远可查。
        """
        import dataclasses
        return dataclasses.replace(self, smoke=True, max_steps=30, eval_every=10,
                                   patience_epochs=99, batch=16,
                                   layers=min(self.layers, 1),
                                   extra=dict(self.extra, smoke_n_trips=int(n_trips)))


def bot_caliber(phi: float) -> float:
    """口径标签里的 φ（六位有效数字；只影响字符串，不影响数值）。"""
    return float("%.6g" % float(phi))


# --------------------------------------------------------------------------- 基线
_SEG_CACHE: dict[str, np.ndarray] = {}


def _tree_seg_pred(pack: DIO.TripPack, *, verbose: bool = True,
                   frozen: str | Path | None = None) -> np.ndarray:
    """冻结 SePT-70 的**段级**预测（Wh）—— 与 ``models/ours_tree`` 逐位同参同序。

    三级来源（**先便宜后昂贵**）：

    1. **进程内缓存**：同一个 ``pack`` 只拟合一次。
       ★ 这修的是一个真实性能缺陷：原先每次 ``build_inputs`` 都重训一棵 HGBR
         （134,212 段 × 70 特征，纯 CPU + OpenMP），一次全链里会被调用两次
         （注册用的 ``train_fn`` 一次 + 真正的训练一次）。
    2. **冻结段级预测**：``<产物根>/SePT-70_v1.0/seg_preds.npz``（先跑一次
       ``python -B models/ours_tree/model.py`` 即生成）。按 ``(file, seg_idx)`` 逐位核对，
       **不匹配就拒用**（绝不静默套用另一份键序）。
    3. **兜底**：现拟合（全量时约 8 秒；**冒烟态不该走到这里**）。
    """
    key = str(getattr(pack, "info", {}).get("smoke_label", "")) + "|" + str(pack.n_seg)
    if key in _SEG_CACHE and not getattr(pack, "info", {}).get("smoke_subset", False):
        return _SEG_CACHE[key]
    p = Path(frozen) if frozen else (pth.Paths().runs("SePT-70_v1.0") / "seg_preds.npz")
    if p.exists():
        try:
            z = np.load(p, allow_pickle=True)
            fk = "file_" if "file_" in z.files else "file"
            f = np.asarray(z[fk]).astype(str)
            si = np.asarray(z["seg_idx"], dtype=np.int64) if "seg_idx" in z.files \
                else None
            v = np.asarray(z["pred_wh"], dtype=np.float64).reshape(-1)
            src = pack.df[["file", "seg_idx"]].copy()
            src["file"] = src["file"].astype(str)
            if si is not None:
                zz = {k: i for i, k in enumerate(zip(f.tolist(), si.tolist()))}
                idx = [zz.get((str(a), int(b)), -1)
                       for a, b in zip(src["file"].tolist(), src["seg_idx"].tolist())]
                if all(i >= 0 for i in idx):
                    out = v[np.asarray(idx, dtype=np.int64)]
                    if verbose:
                        print("[nn_train] 冻结段级预测命中：%s（n=%d，按键逐位对齐）"
                              % (p, len(out)))
                    return out
        except Exception as e:                                # noqa: BLE001
            if verbose:
                print("[nn_train] 冻结段级预测不可用（%s: %s）⇒ 回退现拟合"
                      % (type(e).__name__, e))
    if getattr(pack, "info", {}).get("smoke_subset", False):
        raise NNError(
            "冒烟态**不允许**现拟合树基线（那是纯 CPU 的昂贵步骤，正是\"冒烟没冒到贵的部分\""
            "这个坑）。\n  正解：先跑一次 `python -B models/ours_tree/model.py` 生成 %s，"
            "再冒烟。" % p)
    from sklearn.ensemble import HistGradientBoostingRegressor
    tree = {"loss": "absolute_error", "learning_rate": 0.05, "max_iter": 500,
            "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 1.0,
            "early_stopping": False, "random_state": 42}
    is_tr = pack.df["split"].to_numpy(dtype=object).astype(str) == "train"
    net = HistGradientBoostingRegressor(**tree)
    net.fit(pack.raw[is_tr], pack.y_seg_wh[is_tr])
    out = np.asarray(net.predict(pack.raw), dtype=np.float64)
    _SEG_CACHE[key] = out
    return out


def tree_baseline(pack: DIO.TripPack, *, mode: str = "full", phi: float = 0.0,
                  seg_pred: np.ndarray | None = None) -> np.ndarray:
    """冻结树的**趟级**基线（Wh）。

    ``mode="full"``：全部段的预测求和 ⇒ 口径 ``y_full``（= 公布 SePT-70 的趟级预测）。
    ``mode="rem"`` ：φ 切点**之后**的段的预测求和 ⇒ 口径 ``y_rem@phi``（源树 ``base_rem``）。
    """
    seg = _tree_seg_pred(pack) if seg_pred is None else np.asarray(seg_pred, np.float64)
    if mode == "full":
        return np.bincount(pack.flat_trip, weights=seg, minlength=pack.n_trips)
    if mode != "rem":
        raise NNError("tree_baseline: mode 只支持 'full' | 'rem'，实得 %r" % mode)
    cuts = FT.cut_for_phi(pack.cuts, phi).set_index("file")
    n_pre = cuts.loc[pack.trips, "n_pre"].to_numpy(np.int64)
    out = np.zeros(pack.n_trips, np.float64)
    for i in range(pack.n_trips):
        s = int(pack.off[i]) + int(n_pre[i])
        e = int(pack.off[i]) + int(pack.nseg[i])
        out[i] = float(seg[s:e].sum())
    return out


def load_frozen_baseline(path: str | Path, trips: Sequence[Any], *,
                         key: str = "pred_wh", phi: float | None = None) -> np.ndarray:
    """装载**既有冻结逐趟预测**（``.npz``；按 ``file`` 对齐 ⇒ 键序随行，不按位置贴）。

    支持两种形态：
      * 摊平数组：``file`` + ``<key>``（长度 = 趟数）
      * 复合键：``file`` + ``phi`` + ``<key>``（取 ``phi`` 命中的那一层）
    找不到对齐的键 ⇒ 抛（**不静默回退到按位置**，那正是 18.24× 事故的成因）。
    """
    p = Path(path)
    if not p.exists():
        raise NNError("冻结预测文件不存在：%s" % p)
    z = np.load(p, allow_pickle=True)
    if "file" not in z.files:
        raise NNError("%s 里没有 'file' 键（可用 %s）—— 无法按键对齐" % (p, list(z.files)))
    if key not in z.files:
        raise NNError("%s 里没有 %r 键（可用 %s）" % (p, key, list(z.files)))
    f = np.asarray(z["file"]).astype(str)
    v = np.asarray(z[key], dtype=np.float64).reshape(-1)
    if "phi" in z.files and phi is not None:
        ph = np.asarray(z["phi"], dtype=np.float64).reshape(-1)
        m = np.isclose(ph, float(phi))
        if not bool(m.any()):
            raise NNError("%s 里没有 phi=%r 的层（可用 %s）"
                          % (p, phi, sorted(set(ph.tolist()))[:8]))
        f, v = f[m], v[m]
    idx = {str(k): i for i, k in enumerate(f)}
    miss = [t for t in trips if str(t) not in idx]
    if miss:
        raise NNError("%s 缺 %d 趟（例 %s）—— 键必须覆盖全部趟" % (p, len(miss), miss[:3]))
    return np.array([v[idx[str(t)]] for t in trips], dtype=np.float64)


# --------------------------------------------------------------------------- 输入构造
def build_inputs(spec: NNSpec, pack: DIO.TripPack) -> dict[str, Any]:
    """把数据包变成网络输入 + 目标 + mask（**唯一入口**）。

    返回：``exo``（分箱后的每变量 token）、``patch``/``patch_mask``（前缀 patch，可选）、
    ``base``（冻结基线 Wh）、``y``（目标 Wh）、``ok``（有效趟掩码，`y_rem` 短路为 False）、
    ``seg_pred``（冻结树段级预测，供复用）。
    """
    import torch
    seg_pred = _tree_seg_pred(pack, verbose=False)
    mode = "rem" if spec.base == "tree_rem" else "full"
    if spec.base == "frozen_npz":
        base = load_frozen_baseline(spec.frozen_path, pack.trips, key=spec.frozen_key,
                                    phi=(spec.phi if spec.phi > 0 else None))
    elif spec.base == "none":
        base = np.ones(pack.n_trips, np.float64)
    else:
        base = tree_baseline(pack, mode=mode, phi=spec.phi, seg_pred=seg_pred)

    y = pack.rem_target(spec.phi) if spec.target_caliber.startswith("y_rem@") else pack.y_trip
    ok = np.isfinite(y) & np.isfinite(base) & (base > 0)
    if int(ok.sum()) < 100:
        raise NNError("有效趟只有 %d 条（base>0 且 y 有限）—— 口径或切点有问题" % int(ok.sum()))

    X = torch.as_tensor(pack.X3, dtype=torch.float32)
    nseg = np.asarray(pack.nseg, dtype=np.int64)
    exo = NB.bin_sequence(X, nseg, pack.off, None, n_bins=spec.extra.get("n_bins", NB.N_BINS))
    out: dict[str, Any] = {"exo": exo, "base": base, "y": y, "ok": ok, "seg_pred": seg_pred,
                           "nseg": nseg}
    if spec.use_personal:
        cuts = FT.cut_for_phi(pack.cuts, spec.phi if spec.phi > 0 else 0.0).set_index("file")
        n_pre = cuts.loc[pack.trips, "n_pre"].to_numpy(np.int64)
        plen = NB.prefix_lens_of(n_pre, nseg)
        pmax = int(spec.extra.get("patch_max", NB.PATCH_MAX))
        pt, pm = NB.patchify(X, plen, patch_max=pmax)
        out.update({"patch": pt, "patch_mask": pm, "prefix_len": plen})
    return out


# --------------------------------------------------------------------------- 训练
@dataclass
class NNRun:
    """一次神经臂全链的结果（账目 + 指标 + 产物路径）。"""

    spec: NNSpec
    val_wmape: float = float("nan")
    dev_wmape: float = float("nan")
    pooled_wmape: float = float("nan")
    val_mape_ge1kwh: float = float("nan")
    val_mae_wh: float = float("nan")
    val_bias_wh: float = float("nan")
    val_rmse_wh: float = float("nan")
    unit_ratio: dict[str, float] = field(default_factory=dict)
    n_params: int = 0
    elapsed_s: float = 0.0
    verdict: str = "PASS"
    n_ok: int = 0
    npz: str = ""
    key_sha16: str = ""
    branch: dict[str, Any] = field(default_factory=dict)
    delta_stats: dict[str, Any] = field(default_factory=dict)
    baseline_wmape: dict[str, float] = field(default_factory=dict)
    spec_obj: Any = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        d = self.spec.to_row()
        d.update({"val_wmape": self.val_wmape, "dev_wmape": self.dev_wmape,
                  "pooled_wmape": self.pooled_wmape,
                  "val_mape_ge1kwh": self.val_mape_ge1kwh, "val_mae_wh": self.val_mae_wh,
                  "val_bias_wh": self.val_bias_wh, "val_rmse_wh": self.val_rmse_wh,
                  "n_params": self.n_params, "elapsed_s": self.elapsed_s,
                  "verdict": self.verdict, "n_ok": self.n_ok, "npz": self.npz,
                  "key_sha16": self.key_sha16,
                  "unit_ratio": self.unit_ratio, "branch": self.branch,
                  "delta_stats": self.delta_stats,
                  "baseline_wmape": self.baseline_wmape, "meta": self.meta})
        return d


_BOXES: dict[str, dict[str, Any]] = {}


def register_nn_spec(spec: NNSpec, *, source: str = "") -> reg.ModelSpec:
    """把一条神经臂登记进统一注册表（**唯一注册入口**）。"""
    if spec.name in _BOXES:
        box = _BOXES[spec.name]
    else:
        box = {}
    _BOXES[spec.name] = box

    def train_fn(ctx: TR.TrainContext) -> reg.ModelHandle:
        run = train_neural(spec, verbose=False, repro_check=False)
        h = box.get("handle")
        if h is None:
            raise NNError("train_fn 未能取到 ModelHandle（%s）" % spec.name)
        return h

    sp = reg.register(spec.name, spec.info_condition, spec.target_caliber, train_fn,
                      _predict_from_handle, version="1.0", family=spec.family,
                      notes=spec.notes or ("%s / base=%s / personal=%s"
                                           % (spec.arch, spec.base, spec.use_personal)),
                      features=tuple(DIO.get_policy(spec.policy).feats_list),
                      source=source or spec.source, override=True,
                      extra={"mode": "protocol", "spec": spec.to_row()})
    return sp


def probe_names_in(net: Any) -> tuple[str, ...]:
    """分支守卫要看的参数名（**按子串**与 ``snapshot_probes`` 的语义对齐）。"""
    return ("head_d.", "patch.")


def train_neural(spec: NNSpec, *, verbose: bool = True, repro_check: bool = True) -> NNRun:
    """一条神经臂的完整全链（**唯一入口**）。

    链路：``dataio → registry → trainer(mode=protocol) → infer → eval_trip``。
    ``mode="protocol"`` 之下：766/188/193 划分、train 内部 holdout 早停、
    warmup+cosine、patience、ckpt（``best_val.pt`` + ``last_1..5.pt``）、逐步日志、
    分支活性守卫、单位门**全部生效** —— 模型层无法绕过任何一条。
    """
    import torch
    t0 = time.time()
    NB.cap_threads()                      # ★ 线程数必须钉住（见 nn_blocks.cap_threads）
    P = pth.Paths()
    if spec.smoke:
        spec = spec.smoked()
    pack = DIO.load_pack(spec.policy_obj, P=P)
    if spec.smoke:
        # ★ 冒烟**必须冒到贵的那一段**：数据也缩到子集（否则仍要跑全量基线/全量推理）
        pack = DIO.limit_pack(pack, int(spec.extra.get("smoke_n_trips", 24)),
                              label="%s-smoke" % spec.name)
    data = build_inputs(spec, pack)

    exo_np = data["exo"].numpy()
    exo_mean = exo_np.reshape(-1, exo_np.shape[-1]).mean(0, keepdims=True)
    exo_std = exo_np.reshape(-1, exo_np.shape[-1]).std(0, keepdims=True)
    exo_std = np.where(exo_std >= 1e-9, exo_std, 1.0)
    exo_z = torch.as_tensor((exo_np - exo_mean) / exo_std, dtype=torch.float32)
    sc = NB.PerVarScaler(mu=exo_mean.reshape(-1), sd=exo_std.reshape(-1))

    cfg = NB.ExoConfig(n_var=pack.n_feat, n_bins=data["exo"].shape[-1], d=spec.d,
                       heads=spec.heads, layers=spec.layers, d_ff=spec.d_ff,
                       dropout=spec.dropout, use_cross=spec.use_cross,
                       use_personal=spec.use_personal)
    net = NB.PrefixNet(cfg)
    n_params = NB.count_params(net)
    y = np.asarray(data["y"], np.float64)
    base = np.asarray(data["base"], np.float64)
    ok = np.asarray(data["ok"], bool)
    patch = data.get("patch")
    pmask = data.get("patch_mask")

    def mk_batch(sel: np.ndarray, dev: str) -> tuple[Any, Any, Any, Any]:
        e = exo_z[sel].to(dev)
        p = patch[sel].to(dev) if patch is not None else None
        m = pmask[sel].to(dev) if pmask is not None else None
        b = torch.as_tensor(base[sel], dtype=torch.float32, device=dev)
        return e, p, m, b

    def loss_fn(sel: np.ndarray, dev: str) -> Any:
        e, p, m, b = mk_batch(sel, dev)
        o = net(e, p, m, base_wh=b)
        t = torch.as_tensor(y[sel], dtype=torch.float32, device=dev)
        # 目标：log 倍率的**相对误差**（HMAPE 式的对称损失，与 WMAPE 口径一致）
        r = torch.log(t.clamp_min(1.0)) - torch.log(o["pred"].clamp_min(1.0))
        return (r ** 2).mean()

    def infer_all(dev: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
        """全量推理（**分批**：不是按趟一次性过 —— 2012×70 的张量在 CPU 上没必要整个塞进去）。"""
        net.eval()
        preds = np.zeros(pack.n_trips, np.float64)
        deltas = np.zeros(pack.n_trips, np.float64)
        with torch.no_grad():
            for s in range(0, pack.n_trips, 128):
                sel = np.arange(s, min(s + 128, pack.n_trips))
                e, p, m, b = mk_batch(sel, dev)
                o = net(e, p, m, base_wh=b)
                preds[sel] = o["pred"].detach().cpu().numpy()
                deltas[sel] = o["delta"].detach().cpu().numpy()
        net.train()
        return preds, deltas

    # ---- 分支守卫要看的张量（零位移 = 死分支 ⇒ 抛）
    #   ★ 守卫必须**指名**要看哪些参数：把全部参数都塞进去，任何"设计上就该不动"的
    #     参数（如未参与该头的偏置）都会把守卫误报成死分支。所以只挑两支的权重。
    _named = dict(net.named_parameters())
    probes: dict[str, Any] = {k: v for k, v in _named.items()
                              if k.startswith("head_d.") and k.endswith("weight")}
    if not probes:
        raise NNError("网络里没有 head_d.*.weight 参数 —— PrefixNet 的结构被改动了")
    if spec.use_personal:
        _pk = {k: v for k, v in _named.items()
               if k.startswith("patch.") and k.endswith(("weight", "bias"))}
        if not _pk:
            raise NNError("use_personal=True 但网络里没有 patch.* 参数 —— 开关没接上")
        probes.update(_pk)

    def eval_fn() -> Mapping[str, Any]:
        pred, _ = infer_all("cpu")
        q = quick_eval(pred)
        return {"holdout_mse": q["holdout_mse"], "val_wmape": q["val_wmape"],
                "dev_wmape": q["dev_wmape"], "pred_true_ratio": q["pred_true_ratio"],
                "branch": {k: v.detach().clone() for k, v in probes.items()}}

    # 评价集掩码在**整个训练期间恒定** ⇒ 预先算好，别在每个 eval 点重算
    MASK_VAL = pack.mask_of("val")
    MASK_DEV = pack.mask_of("dev")
    HO = pack.idx_of("train")[::TR.HOLDOUT_STRIDE][:TR.HOLDOUT_N]

    def quick_eval(pred: np.ndarray) -> dict[str, float]:
        return {"holdout_mse": float(np.mean((pred[HO] - y[HO]) ** 2)),
                "val_wmape": _wmape(pred[MASK_VAL], y[MASK_VAL]),
                "dev_wmape": _wmape(pred[MASK_DEV], y[MASK_DEV]),
                "pred_true_ratio": float(pred[ok].sum() / max(y[ok].sum(), 1e-9))}

    all_idx = np.flatnonzero(ok & (pack.split.astype(str) == "train"))
    if all_idx.size < 50:
        raise NNError("train 有效趟只有 %d 条 —— 拒绝训练" % all_idx.size)

    def train_fn(ctx: TR.TrainContext) -> reg.ModelHandle:
        opt = torch.optim.AdamW(net.parameters(), lr=spec.lr, weight_decay=spec.weight_decay)

        def batch_fn(sel: Any, gen: Any) -> tuple[Any, int, Any]:
            sel_np = np.asarray(sel, dtype=np.int64)
            sel_np = all_idx[np.mod(sel_np, all_idx.size)]
            loss = loss_fn(sel_np, ctx.device)
            return loss, int(sel_np.size), None

        prot = TR.Protocol(lr=spec.lr, weight_decay=spec.weight_decay,
                           max_steps=spec.max_steps, eval_every=spec.eval_every,
                           batch=spec.batch, patience_epochs=spec.patience_epochs)
        r = TR.run_protocol_loop(ctx, net=net, opt=opt, batch_fn=batch_fn, eval_fn=eval_fn,
                                 probe_names=tuple(probes.keys()), required_probes=True)
        pred_all, deltas = infer_all(ctx.device)
        ctx.log("[%s] 训练完成：%d 有效趟 / params %d / Δ 均值 %.4f 标准差 %.4f"
                % (spec.name, int(ok.sum()), n_params, float(deltas[ok].mean()),
                   float(deltas[ok].std())))
        h = reg.ModelHandle(params=net, protocol="protocol", n_params=n_params,
                            ckpt_dir=ctx.ckpt.ck, log_csv=ctx.ckpt.log_csv, history=r)
        h.meta = {"pred_wh": pred_all, "deltas": deltas, "probes": r.get("branch", {}),
                  "allow_dead": () if spec.use_personal else ("patch",),
                  "note": "pred = 冻结基线 · exp(clamp(Δ, ±0.7))；个性化=%s"
                          % ("ON" if spec.use_personal else "OFF")}
        _BOXES.setdefault(spec.name, {})["handle"] = h
        return h

    # 注册与执行**共用同一份 train_fn**（见 register_nn_spec 的说明）
    def _registered_train_fn(ctx: TR.TrainContext) -> reg.ModelHandle:
        train_neural(spec, verbose=False, repro_check=False)
        h = _BOXES.get(spec.name, {}).get("handle")
        if h is None:
            raise NNError("注册用 train_fn 未能取到 ModelHandle（%s）" % spec.name)
        return h

    _BOXES.setdefault(spec.name, {})
    sp = reg.register(spec.name, spec.info_condition, spec.target_caliber,
                      _registered_train_fn,
                      _predict_from_handle, version="1.0", family=spec.family,
                      notes=spec.notes or ("%s / base=%s / personal=%s"
                                           % (spec.arch, spec.base, spec.use_personal)),
                      features=tuple(DIO.get_policy(spec.policy).feats_list),
                      source=spec.source, override=True, extra={"mode": "protocol",
                                                                "spec": spec.to_row()})
    out_dir = P.runs(spec.name)
    res = TR.train(sp, data=TR.prepare(), mode="protocol",
                   protocol=TR.Protocol(lr=spec.lr, weight_decay=spec.weight_decay,
                                        max_steps=spec.max_steps,
                                        eval_every=spec.eval_every, batch=spec.batch,
                                        patience_epochs=spec.patience_epochs),
                   out_dir=out_dir, seed=spec.seed, repro_check=repro_check)
    pred = np.asarray(res.handle.meta["pred_wh"], np.float64)
    deltas = np.asarray(res.handle.meta["deltas"], np.float64)

    inf = INF.predict_to_npz(_predict_from_handle, res.handle, None, out_dir=out_dir,
                             files=pack.trips, split=pack.split, y_wh=y,
                             info_condition=spec.info_condition,
                             target_caliber=spec.target_caliber, model=spec.name,
                             y_basis=("冻结基线·exp(clamp Δ)；基线=%s" % spec.base),
                             phi=(spec.phi if spec.target_caliber.startswith("y_rem@")
                                  else None))
    m_val = ET.eval_trip(inf.trip.select("validation"), info_condition=spec.info_condition,
                         target_caliber=spec.target_caliber, splits=("val",))
    m_dev = ET.eval_trip(inf.trip.select("future"), info_condition=spec.info_condition,
                         target_caliber=spec.target_caliber, splits=("dev",))
    m_all = ET.eval_trip(inf.trip, info_condition=spec.info_condition,
                         target_caliber=spec.target_caliber)
    # 基线自身指标（"Δ 值多少" 的分母）
    bl = {"val": _wmape(base[pack.mask_of("val")], y[pack.mask_of("val")]),
          "dev": _wmape(base[pack.mask_of("dev")], y[pack.mask_of("dev")])}

    run = NNRun(spec=spec,                val_wmape=float(m_val.metric("WMAPE", "val")),
                dev_wmape=float(m_dev.metric("WMAPE", "dev")),
                pooled_wmape=float(m_all.metric("WMAPE", "pooled")),
                val_mape_ge1kwh=float(m_val.metric("MAPE_ge1kWh", "val")),
                val_mae_wh=float(m_val.metric("MAE", "val")),
                val_bias_wh=float(m_val.metric("bias", "val")),
                val_rmse_wh=float(m_val.metric("RMSE", "val")),
                unit_ratio={k: float(v) for k, v in res.unit_ratio.items()},
                n_params=n_params, elapsed_s=time.time() - t0,
                n_ok=int(ok.sum()), npz=str(inf.npz), key_sha16=str(inf.meta.get("key_sha16")),
                branch={k: v for k, v in (res.branch or {}).items() if k != "rows"},
                delta_stats={"mean": float(deltas[ok].mean()),
                             "std": float(deltas[ok].std()),
                             "p01": float(np.percentile(deltas[ok], 1)),
                             "p99": float(np.percentile(deltas[ok], 99)),
                             "frac_clamped": float(np.mean(np.abs(deltas[ok])
                                                           >= NB.MULT_CLAMP - 1e-9))},
                baseline_wmape=bl, spec_obj=sp,
                meta={"scaling": sc.to_row(), "cfg": cfg.to_row(),
                                         "protocol": res.protocol.to_row() if res.protocol
                                         else None,
                                         "repro": res.repro, "out_dir": str(out_dir)})
    run.verdict = ("PASS" if (np.isfinite(run.val_wmape) and np.isfinite(run.dev_wmape)
                              and res.verdict == "PASS") else "FAIL")
    # ★ 逐位复现的**诚实声明**：框架的 repro_check 只会重跑 ``eval_fn``（不是重训），
    #   因此它**证明不了**"同种子重训逐位一致"。真正的证明要再训一遍（本类臂约 2–5 分钟，
    #   翻倍本臂代价）。这里给出**推理路径**的确定性证明（同权重两次前向逐位一致），
    #   并把"未做重训级复现"写进 meta —— 不许把它说成"已验证逐位可复现"。
    _p1, _d1 = infer_all("cpu")
    _p2, _d2 = infer_all("cpu")
    run.meta["determinism"] = {
        "inference_bitwise_equal": bool(np.array_equal(_p1, _p2) and np.array_equal(_d1, _d2)),
        "train_rerun_checked": bool(repro_check),
        "train_rerun_note": ("已开启框架 repro_check（重跑 eval_fn，**不是**重训）"
                             if repro_check else "未开启"),
        "honest_scope": "同权重两次前向逐位一致（已证）；同种子**重训**逐位一致（**未证**，"
                        "需人工重跑一次 --seed 相同并比对 pred_wh）。",
    }
    (out_dir / "eval_report.json").write_text(
        json.dumps(run.to_row(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if verbose:
        print("\n★ %s：%s" % (spec.name, run.verdict))
        print("   口径 %s / %s | base=%s | 个性化=%s | 参数 %d"
              % (spec.info_condition, spec.target_caliber, spec.base,
                 "ON" if spec.use_personal else "OFF", n_params))
        print("   WMAPE val=%.4f dev=%.4f pooled=%.4f | 基线 val=%.4f dev=%.4f | Δ(val)=%+.4f pp"
              % (run.val_wmape, run.dev_wmape, run.pooled_wmape, bl["val"], bl["dev"],
                 bl["val"] - run.val_wmape))
        print("   Δ 分布：均值 %+.4f / 标准差 %.4f / 触顶比例 %.3f | 有效趟 %d | %.1fs"
              % (run.delta_stats["mean"], run.delta_stats["std"],
                 run.delta_stats["frac_clamped"], run.n_ok, run.elapsed_s))
        print("   %s（key_sha16=%s）" % (run.npz, run.key_sha16))
    return run


def _wmape(p: Any, y: Any) -> float:
    p = np.asarray(p, np.float64)
    y = np.asarray(y, np.float64)
    m = np.isfinite(y) & np.isfinite(p)
    if not bool(m.any()) or float(np.abs(y[m]).sum()) <= 0:
        return float("nan")
    return float(100.0 * np.abs(p[m] - y[m]).sum() / np.abs(y[m]).sum())


def _predict_from_handle(handle: Any, ctx: Any = None, **kw: Any) -> dict[str, Any]:
    """``ModelHandle`` → ``{"trip_wh": ...}``（神经臂的 ``predict_fn``）。"""
    m = getattr(handle, "meta", None) or {}
    if "pred_wh" not in m:
        raise NNError("handle.meta 里没有 pred_wh —— 请经 train_neural 产出 handle"
                      "（或从 runs/<name>/ckpt/best_val.pt 重新推理，见 README）")
    return {"trip_wh": np.asarray(m["pred_wh"], dtype=np.float64)}


def eval_from_handle(handle: Any, spec: NNSpec) -> dict[str, Any]:
    """对已有 handle 复算指标（不重训）—— 供对比脚本用。"""
    pack = DIO.load_pack(spec.policy)
    pred = np.asarray(handle.meta["pred_wh"], np.float64)
    return {"val": _wmape(pred[pack.mask_of("val")], pack.y_trip[pack.mask_of("val")]),
            "dev": _wmape(pred[pack.mask_of("dev")], pack.y_trip[pack.mask_of("dev")])}


def list_runs(root: str | Path | None = None) -> list[dict[str, Any]]:
    """列出 ``runs/`` 下所有 ``eval_report.json``（一张可打印的对照表）。"""
    base = Path(root) if root else pth.Paths().runs()
    out = []
    for p in sorted(base.glob("*/eval_report.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception as e:                                    # noqa: BLE001
            out.append({"name": p.parent.name, "error": str(e)})
    return out
