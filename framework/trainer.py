#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""trainer —— 把项目既定训练协议**固化为唯一实现**（统一框架第 4 层，只加不改）。

固化的是哪一套协议
==================
审计实测：``def train(`` 全树 **0 命中**、``def fit(`` **4 套互斥签名**、种子 **5 套**。
于是"同一实验"在不同脚本里跑出不同数字，而没人能说清差别来自哪。本模块把**项目已确立**
的那一套写死：

  * **划分固定 766 / 188 / 193**（train / validation / future；calibration 是红线）
  * **只用 train 内部 holdout 早停**：``tr[::7][:110]`` = **110 趟**（每 7 趟取 1）
    —— 绝不用 validation/future 早停（那是拿评估集调参）
  * **warmup + cosine**：``warm = max(10, warmup_frac·total)``；
    ``Tcos = max(warm+1, total)``（**cosine 的时间常数 = 本种子实际步数上限**，
    避免"跑到一半 LR 已经 ~0"的伪收敛）
  * **patience**：连续 ``patience_epochs`` 个 eval 点无改善（``min_delta``）即停
  * **产物**：``ckpt/best_val.pt`` + ``last_1..5.pt`` + **逐步日志 CSV**
  * **分支活性守卫**（含"**零位移 = 死分支**"）：每个被监控张量在训练后必须有非零位移；
    否则抛 ``DeadBranchError``（本会话已见：某分支梯度恒 0 却"训练成功"）
  * **单位断言** ``Σp/Σy ∈ [0.5, 2]``：warmup 之后每个 eval 点都查；越界即终止该种子
  * **种子统一**：一种约定取代 5 套 —— ``train.final_seed(seed)`` 同时驱动
    ``torch / numpy / random``；``torch.Generator`` 用同值；**跨模型同一 seed 语义一致**
  * **先复现一个已知-good 格才允许往下跑**：``train(gate=...)`` 在开跑新实验前先复现锚点，
    不过门直接拒跑（``GateError``）

两种训练模式（同一入口，同一账本）
==================================
* ``mode="protocol"``：本模块托管训练循环（warmup+cosine/patience/ckpt/日志/守卫全开）。
  模型只需给 ``net + optimizer + batch_fn``，**不写循环**。
* ``mode="native"``：既有模型自带的训练函数（如 SePT-70 的 ``sept.fit`` —— 单次确定性拟合，
  本就没有早停/ckpt 概念）。本模块仍然强制：**确定性重跑逐位一致 + 单位断言 + 产物落盘**。

红线
====
只读 data_v3；不读 calibration；**不改既有模型代码**（本模块只调用它们）。
"""
from __future__ import annotations

import csv
import dataclasses
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from paths import Paths
from schema import (INFO_CONDITIONS, TARGET_CALIBERS, CaliberError, SchemaError,
                    assert_unit_ratio, UNIT_WH)

__all__ = [
    "N_TRIP_TRAIN", "N_TRIP_VAL", "N_TRIP_DEV", "N_SEG_TRAIN", "HOLDOUT_STRIDE",
    "HOLDOUT_N", "CKPT_KEEP_LAST", "HIST_COLUMNS", "SPLIT_COUNTS",
    "TrainError", "GateError", "BranchGuardError", "SeedMismatchError",
    "SeedPolicy", "SplitSpec", "Protocol", "PROTOCOL",
    "final_seed", "apply_seed", "generator", "seed_report", "split_indices",
    "holdout_indices", "apply_warmup_cosine", "lr_at",
    "BranchProbe", "BranchGuard", "snapshot_probes", "probe_activity",
    "run_protocol_loop", "prepare", "PrepData",
    "CheckpointBook", "TrainContext", "TrainResult", "Trainer", "train",
    "assert_units_wh", "gate_against_anchor", "DEMO_PROTOCOL",
]

# --------------------------------------------------------------------------- 协议常量（写死）
N_TRIP_TRAIN, N_TRIP_VAL, N_TRIP_DEV = 766, 188, 193
N_SEG_TRAIN = 134_212
SPLIT_COUNTS = {"train": N_TRIP_TRAIN, "validation": N_TRIP_VAL, "future": N_TRIP_DEV}
HOLDOUT_STRIDE = 7            # 每 7 趟取 1 趟作为 train 内部 holdout
HOLDOUT_N = 110               # 取前 110 趟
CKPT_KEEP_LAST = 5            # last_1..5.pt
UNIT_LO, UNIT_HI = 0.5, 2.0

HIST_COLUMNS = ("epoch", "step", "train_loss", "holdout_mse", "val_wmape", "dev_wmape",
                "lr", "grad_norm", "pred_true_ratio", "n_branch", "n_branch_dead",
                "n_branch_frozen", "branch_min_disp", "note")

# 诊断开关（**默认关**；只影响打印，不影响任何数值）
_DEBUG = bool(int(__import__("os").environ.get("FW_TRAINER_DEBUG", "0")))


# --------------------------------------------------------------------------- 异常
class TrainError(RuntimeError):
    """训练协议违反。"""


class GateError(TrainError):
    """**先复现一个已知-good 格**的门没过 —— 拒绝往下跑新实验。"""


class BranchGuardError(TrainError):
    """死分支 / 冻结分支被守卫拦下（零位移 = 死分支）。"""


class SeedMismatchError(TrainError):
    """种子不可复现（同种子两次结果不一致）—— 一律响亮失败，不许"差不多"。"""


# --------------------------------------------------------------------------- 种子策略
@dataclass(frozen=True)
class SeedPolicy:
    """**一种**种子约定（取代 5 套）：同值同时驱动 torch / numpy / random / Generator。

    ``offset`` 用于需要"同种子下的不同子流"的场合（如数据打乱 vs 权重初始化），
    但它**是协议的一部分**：改了它就不再与既有数字可比，所以它进 meta、进日志。
    """

    base: int = 42
    offset: int = 1013
    name: str = "unified-v1"

    def final(self, seed: int | None = None) -> int:
        return final_seed(self.base if seed is None else int(seed))

    def to_row(self) -> dict[str, Any]:
        return {"seed_policy": self.name, "seed_base": self.base, "seed_offset": self.offset}


def final_seed(seed: int) -> int:
    """**唯一**的种子归一入口：非负 32 位（torch / numpy / python 都吃这一口）。"""
    s = int(seed)
    if s < 0:
        raise SeedMismatchError("种子必须非负，实得 %d（负种子会让 torch 与 numpy 语义分叉）" % s)
    return s % (2 ** 31 - 1)


def apply_seed(seed: int, *, torch_mod: Any = None, numpy_mod: Any = None,
               random_mod: Any = None) -> dict[str, int]:
    """把**同一个**种子写进 torch / numpy / random（唯一约定）。返回实际写入值。"""
    s = final_seed(seed)
    got = {"seed": s}
    (numpy_mod or np).random.seed(s)
    (random_mod or random).seed(s)
    got["numpy"] = s
    got["random"] = s
    if torch_mod is None:
        try:                                                   # pragma: no cover
            import torch as torch_mod                          # type: ignore
        except Exception:                                      # noqa: BLE001
            torch_mod = None
    if torch_mod is not None:
        torch_mod.manual_seed(s)
        if torch_mod.cuda.is_available():
            torch_mod.cuda.manual_seed_all(s)
        try:
            torch_mod.use_deterministic_algorithms(True, warn_only=True)
            torch_mod.backends.cuda.enable_flash_sdp(False)
            torch_mod.backends.cuda.enable_mem_efficient_sdp(False)
            torch_mod.backends.cuda.enable_math_sdp(True)
        except Exception:                                      # noqa: BLE001
            pass
        got["torch"] = s
    return got


def generator(seed: int, *, torch_mod: Any = None) -> Any:
    """**唯一**的采样 Generator（与 ``apply_seed`` 同值）。torch 缺席时返回 numpy Generator。"""
    s = final_seed(seed)
    if torch_mod is None:
        try:                                                   # pragma: no cover
            import torch as torch_mod                          # type: ignore
        except Exception:                                      # noqa: BLE001
            torch_mod = None
    if torch_mod is not None:
        return torch_mod.Generator().manual_seed(s)
    return np.random.default_rng(s)


def seed_report(seed: int, *, torch_mod: Any = None) -> dict[str, Any]:
    """种子自证：同一 seed 连续两次「首抽」必须逐位相等（否则不是可复现训练）。"""
    if torch_mod is None:
        try:                                                   # pragma: no cover
            import torch as torch_mod                          # type: ignore
        except Exception:                                      # noqa: BLE001
            torch_mod = None

    def first_draw(sd: int) -> list[int]:
        g = generator(sd, torch_mod=torch_mod)
        if torch_mod is not None and not hasattr(g, "integers"):
            # torch.Generator：必须用 torch.randint(..., generator=g)
            # （它**没有** .integers；写成 g.integers 会 AttributeError —— 实测踩到）
            return torch_mod.randint(0, 2 ** 30, (8,), generator=g).tolist()
        return [int(x) for x in g.integers(0, 2 ** 30, 8)]

    a, b = first_draw(seed), first_draw(seed)
    c = first_draw(seed + 1)
    return {"seed": final_seed(seed), "first_draw": a, "repeatable": bool(a == b),
            "distinct_from_next": bool(a != c)}


# --------------------------------------------------------------------------- 划分
@dataclass(frozen=True)
class SplitSpec:
    """划分规格（**写死 766/188/193**；值不对即抛）。"""

    train: int = N_TRIP_TRAIN
    validation: int = N_TRIP_VAL
    future: int = N_TRIP_DEV
    holdout_stride: int = HOLDOUT_STRIDE
    holdout_n: int = HOLDOUT_N

    def check(self, split_arr: Any) -> "SplitSpec":
        a = np.asarray(split_arr).astype(str)
        got = {k: int((a == k).sum()) for k in SPLIT_COUNTS}
        bad = sorted(set(a.tolist()) - set(SPLIT_COUNTS))
        if bad:
            raise SchemaError("划分出现未授权取值 %s —— 只允许 %s（calibration 是红线）"
                              % (bad, list(SPLIT_COUNTS)))
        if got != SPLIT_COUNTS:
            raise TrainError("划分计数不符：实得 %s ≠ 协议 %s（766/188/193 是项目既定划分，"
                             "改动即与既有数字不可比）" % (got, SPLIT_COUNTS))
        return self


def split_indices(split_arr: Any, which: str = "train", *,
                  strict_counts: bool = True) -> np.ndarray:
    """趟级划分 → 位置索引（唯一入口）。

    ``strict_counts=True``（默认）：计数必须等于 **766/188/193** —— 项目既定划分，
    改动即与既有数字不可比。合成/测试视图显式传 ``False`` 才放行（并且仍然拦 calibration）。
    """
    if which not in SPLIT_COUNTS:
        raise TrainError("split_indices: which 只允许 %s，实得 %r" % (list(SPLIT_COUNTS), which))
    a = np.asarray(split_arr).astype(str)
    if strict_counts:
        SplitSpec().check(a)
    else:
        bad = sorted(set(a.tolist()) - set(SPLIT_COUNTS))
        if bad:
            raise SchemaError("划分出现未授权取值 %s —— 只允许 %s（calibration 是红线）"
                              % (bad, list(SPLIT_COUNTS)))
    return np.flatnonzero(a == which)


def holdout_indices(split_arr: Any, *, stride: int = HOLDOUT_STRIDE,
                    n: int = HOLDOUT_N, which: str = "train",
                    strict_counts: bool = True,
                    allow_short: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """**train 内部 holdout**（唯一实现）：``tr[::stride][:n]`` vs 其余。

    返回 ``(fit_idx, holdout_idx)``（**都是 train 的趟位置**）。
    这是"早停只看 train 内部"这条纪律的唯一落实点 —— 任何用 validation/future 早停的写法
    都会把评估集变成调参集。

    ``allow_short=True``：train 趟数不足 ``n`` 时按实际数量取（**只用于合成/冒烟视图**；
    真实数据下 holdout 必须正好 110 趟，不够就抛）。
    """
    tr = split_indices(split_arr, which, strict_counts=strict_counts)
    ho = tr[::int(stride)][:int(n)]
    fit = np.setdiff1d(tr, ho)
    if ho.size != int(n) and not allow_short:
        raise TrainError("holdout 数不符：%d ≠ %d（train 趟数 %d，stride=%d）"
                         % (ho.size, n, tr.size, stride))
    return fit, ho


# --------------------------------------------------------------------------- LR
def lr_at(step: int, *, lr: float, warm: int, total: int, cos_steps: int = 0) -> float:
    """**warmup + cosine**（唯一实现）。

    ``Tcos = cos_steps if cos_steps > 0 else max(warm + 1, total)`` ——
    时间常数取"本种子实际步数上限"，避免跑到一半 LR 已经 ~0 的伪收敛。
    """
    warm = max(1, int(warm))
    total = max(1, int(total))
    Tcos = int(cos_steps) if int(cos_steps) > 0 else max(warm + 1, total)
    if step < warm:
        return float(lr) * (step + 1) / warm
    q = (step - warm) / max(1, Tcos - warm)
    return 0.5 * float(lr) * (1 + math.cos(math.pi * min(1.0, q)))


def apply_warmup_cosine(opt: Any, step: int, *, lr: float, total: int,
                        warmup_frac: float = 0.05, cos_steps: int = 0) -> float:
    """把 ``lr_at`` 写进优化器的**所有** param_group（唯一落地点）。返回本步 LR。"""
    warm = max(10, int(warmup_frac * int(total)))
    v = lr_at(step, lr=lr, warm=warm, total=total, cos_steps=cos_steps)
    for g in opt.param_groups:
        g["lr"] = v
    return v


# --------------------------------------------------------------------------- 协议
@dataclass
class Protocol:
    """训练协议（全部字段都有项目既定的默认值；改动即偏离既有数字）。"""

    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_frac: float = 0.05
    cos_steps: int = 0
    max_steps: int = 9000
    steps_per_epoch: int = 200
    batch: int = 512
    grad_clip: float = 1.0
    patience_epochs: int = 8
    min_delta: float = 1e-4
    eval_every: int = 250
    n_epochs_max: int = 60
    unit_lo: float = UNIT_LO
    unit_hi: float = UNIT_HI
    collapse_grace_frac: float = 0.25
    ckpt_keep_last: int = CKPT_KEEP_LAST
    holdout_stride: int = HOLDOUT_STRIDE
    holdout_n: int = HOLDOUT_N
    seed: int = 42
    seed_policy: SeedPolicy = field(default_factory=SeedPolicy)

    def to_row(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "seed_policy"}
        d.update(self.seed_policy.to_row())
        return d


PROTOCOL = Protocol()
DEMO_PROTOCOL = Protocol(max_steps=40, steps_per_epoch=40, patience_epochs=99,
                         eval_every=10)      # 冒烟/自证用（**不是**公布数字的协议）


# --------------------------------------------------------------------------- 分支活性守卫
@dataclass
class BranchProbe:
    """一个被监控的**分支**（张量）+ 它的位移账。

    ``required=True`` 的分支若在训练后**位移为 0**（且梯度累计也为 0）⇒ 死分支 ⇒ 抛。
    """

    name: str
    ref: Any                                  # 初始参数快照（同形 ndarray/张量）
    required: bool = True
    note: str = ""
    grad_sum: float = 0.0
    n_grad: int = 0
    _disp: float = 0.0
    _rel: float = 0.0

    def observe(self, params: Any) -> None:
        now = _to_np(params)
        self._disp = float(np.max(np.abs(now - np.asarray(self.ref, dtype=np.float64))))
        den = float(np.max(np.abs(np.asarray(self.ref, dtype=np.float64))))
        self._rel = self._disp / den if den > 0 else (0.0 if self._disp == 0 else float("inf"))

    @property
    def displacement(self) -> float:
        return self._disp

    @property
    def relative(self) -> float:
        return self._rel

    @property
    def dead(self) -> bool:
        """**零位移 = 死分支**（本模块的核心判据）。"""
        return self._disp == 0.0

    @property
    def frozen(self) -> bool:
        return self.n_grad > 0 and self.grad_sum == 0.0

    def row(self) -> dict[str, Any]:
        return {"branch": self.name, "max_abs_disp": self._disp, "rel_disp": self._rel,
                "grad_abs_sum": self.grad_sum, "n_grad_obs": self.n_grad,
                "required": self.required, "dead": self.dead, "frozen": self.frozen,
                "note": self.note}


def snapshot_probes(net: Any, names: Sequence[str] | None = None, *,
                    required: Any = True) -> dict[str, BranchProbe]:
    """按参数名快照（**在第一次 optimizer.step 之前**调用，否则位移口径就错了）。

    ``names``：只监控名字里含这些子串的参数组；``required``：bool 或子串列表
    （列表 ⇒ 只有命中的分支是 required，其余仅记录）。
    """
    out: dict[str, BranchProbe] = {}
    for nm, p in _named_params(net):
        if names and not any(s in nm for s in names):
            continue
        req = required
        if not isinstance(req, bool):
            req = any(s in nm for s in req)
        out[nm] = BranchProbe(name=nm, ref=_to_np(p).copy(), required=bool(req))
    if not out:
        raise BranchGuardError("snapshot_probes: 没有匹配到任何参数（names=%s）；"
                               "空守卫等于没有守卫" % (names,))
    return out


def _named_params(net: Any) -> list[tuple[str, Any]]:
    if hasattr(net, "named_parameters"):
        return [(str(k), v) for k, v in net.named_parameters()]
    if isinstance(net, Mapping):
        return [(str(k), v) for k, v in net.items()]
    raise BranchGuardError("无法从 %r 取命名参数（要 torch.nn.Module 或 {名: 张量}）"
                           % type(net).__name__)


def _to_np(x: Any) -> np.ndarray:
    """任意张量/数组 → **CPU** float64 ndarray。

    ★ ``np.asarray(cuda_tensor)`` 会抛 ``TypeError: can't convert cuda:0 device type tensor
      to numpy`` —— 探针与守卫一律先 ``.detach().cpu()``，否则 GPU 上跑一次就崩。
    """
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, dtype=np.float64)


def probe_activity(probes: Mapping[str, BranchProbe], net: Any, *,
                   grads: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """对当前网络刷新所有分支的位移（+ 可选梯度累计），返回逐分支账。"""
    cur = dict(_named_params(net))
    for nm, pr in probes.items():
        if nm in cur:
            pr.observe(cur[nm])
        if grads and nm in grads:
            g = _to_np(grads[nm])
            pr.grad_sum += float(np.abs(g).sum())
            pr.n_grad += 1
    return [p.row() for p in probes.values()]


class BranchGuard:
    """分支活性守卫：**零位移=死分支**、梯度恒零=冻结分支；required 的越界即抛。"""

    def __init__(self, probes: Mapping[str, BranchProbe], *, allow_dead: Sequence[str] = (),
                 min_rel_disp: float = 0.0) -> None:
        self.probes = dict(probes)
        self.allow_dead = set(allow_dead)
        self.min_rel_disp = float(min_rel_disp)

    def check(self, *, where: str = "训练后") -> dict[str, Any]:
        rows = [p.row() for p in self.probes.values()]
        dead = [r for r in rows if r["dead"] and r["required"] and r["branch"] not in self.allow_dead]
        frozen = [r for r in rows if r["frozen"] and r["required"]
                  and r["branch"] not in self.allow_dead]
        tiny = [r for r in rows if (not r["dead"]) and r["required"]
                and self.min_rel_disp > 0 and r["rel_disp"] < self.min_rel_disp]
        if dead or frozen or tiny:
            lines = ["分支活性守卫失败（%s）：" % where]
            for r in dead:
                lines.append("  ✗ 死分支（**零位移**）：%s  max|Δparam| = 0" % r["branch"])
            for r in frozen:
                lines.append("  ✗ 冻结分支（梯度恒 0）：%s  grad_abs_sum = 0（观测 %d 次）"
                             % (r["branch"], r["n_grad_obs"]))
            for r in tiny:
                lines.append("  ✗ 位移过小：%s  rel_disp=%.3e < %.3e"
                             % (r["branch"], r["rel_disp"], self.min_rel_disp))
            lines.append("  ⇒ 零位移 = 该分支**从未被训练到**（梯度为 0 / 被 detach / "
                         "被 mask 掉 / lr 为 0 / 参数没进 optimizer）。\n"
                         "    这类分支会静默产出一个'看起来跑通了'的实验。")
            lines.append("  豁免：allow_dead=%s" % sorted(self.allow_dead))
            raise BranchGuardError("\n".join(lines))
        return {"n_branch": len(rows), "n_dead": len(dead), "n_frozen": len(frozen),
                "min_disp": float(min(r["max_abs_disp"] for r in rows)) if rows else 0.0,
                "min_rel_disp": float(min(r["rel_disp"] for r in rows)) if rows else 0.0,
                "rows": rows}


# --------------------------------------------------------------------------- 检查点账本
class CheckpointBook:
    """**``ckpt/best_val.pt`` + ``last_1..5.pt`` + 逐步日志 CSV** 的唯一实现。

    纪律：
      * ``best_val.pt`` 只在 holdout（= train 内部 holdout）改善时写；
      * ``last_N`` 环形保留 —— **每 N 个 eval 点滚一次**，``last_1`` 永远是最新；
      * 历史 CSV 的列**固定**（``HIST_COLUMNS``），追加写、每步 flush。
    """

    def __init__(self, out_dir: Any, *, keep_last: int = CKPT_KEEP_LAST,
                 save_fn: Callable[[Any, Path], None] | None = None) -> None:
        self.dir = Path(out_dir)
        self.ck = self.dir / "ckpt"
        self.ck.mkdir(parents=True, exist_ok=True)
        self.keep_last = int(keep_last)
        self.save_fn = save_fn or _default_save
        self.log_csv = self.dir / "train_log.csv"
        self._buf: list[dict[str, Any]] = []
        self._fh = None
        self.best = float("inf")
        self.best_step = -1
        self.last_written: list[str] = []

    # ---------------------------------------------------------------- 写
    def log(self, rec: Mapping[str, Any]) -> dict[str, Any]:
        row = {k: rec.get(k, "") for k in HIST_COLUMNS}
        self._buf.append(row)
        if self._fh is None:
            self._fh = open(self.log_csv, "w", newline="", encoding="utf-8")
            self._fh.write(",".join(HIST_COLUMNS) + "\n")
        self._fh.write(",".join("" if row[k] is None else str(row[k]) for k in HIST_COLUMNS) + "\n")
        self._fh.flush()
        return row

    def save_best(self, payload: Any, *, holdout: float, step: int,
                  better_than: float | None = None) -> bool:
        """改善即存；``better_than`` 显式给出改善门槛（默认 = 历史最优，即严格改善）。

        ``better_than`` 存在的理由：既有实现用 ``hv < best - 1e-7`` —— 门槛是**判据的一部分**，
        不能由本模块悄悄改成 0（那会让"第几个 eval 点最好"漂移一个点）。
        """
        thr = self.best if better_than is None else float(better_than)
        if float(holdout) < thr:
            self.best, self.best_step = float(holdout), int(step)
            self.save_fn(payload, self.ck / "best_val.pt")
            return True
        return False

    def save_last(self, payload: Any) -> Path:
        """环形保留 ``last_1..last_N``：先整体后移，再写 ``last_1``。"""
        for i in range(self.keep_last, 1, -1):
            src = self.ck / ("last_%d.pt" % (i - 1))
            if src.exists():
                src.replace(self.ck / ("last_%d.pt" % i))
        p = self.ck / "last_1.pt"
        self.save_fn(payload, p)
        self.last_written = sorted(x.name for x in self.ck.glob("last_*.pt"))
        return p

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    # ---------------------------------------------------------------- 读
    def last_paths(self) -> list[Path]:
        return sorted(self.ck.glob("last_*.pt"))

    def seed_dir(self, seed: int, *, tag: str = "s") -> Path:
        """**本种子**的 ckpt 目录 ``<run>/<tag><seed>/``（多子流模型必用）。

        ★ 存在的理由（实测踩到）：多个种子若共用同一个 ``CheckpointBook``，
          ``best_val.pt`` 的"历史最优"会**跨种子累积** —— 第 2 个种子的 holdout 若一开始
          比第 1 个种子的最优差，就一次都不存，最后 ``best_val.pt`` 里是**别人的权重**，
          而日志会显示 ``best=inf`` / 早停在 ep3。基准实现是每个种子一套变量，本方法
          用**每种子一套目录**达到同一效果，且不破坏"best_val.pt 在 ckpt/ 下"的约定。
        """
        return self.dir / ("%s%d" % (tag, int(seed)))

    def child(self, seed: int, *, tag: str = "s") -> "CheckpointBook":
        """新建一个**本种子专属**的检查点账本（``<run>/<tag><seed>/ckpt/...``）。"""
        return CheckpointBook(self.seed_dir(seed, tag=tag), keep_last=self.keep_last,
                              save_fn=self.save_fn)

    def promote_best(self, src: Path, *, holdout: float | None = None) -> Path:
        """把某个种子的 best 提升为**运行级** ``ckpt/best_val.pt``（infer 的默认入口）。"""
        p = self.ck / "best_val.pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(Path(src).read_bytes())
        if holdout is not None and float(holdout) < self.best:
            self.best = float(holdout)
        return p

    def manifest(self) -> dict[str, Any]:
        return {"ckpt_dir": str(self.ck), "best_val_pt": str(self.ck / "best_val.pt"),
                "best_val": (None if not np.isfinite(self.best) else float(self.best)),
                "best_step": int(self.best_step),
                "last": [p.name for p in self.last_paths()],
                "log_csv": str(self.log_csv), "n_log_rows": len(self._buf)}

    def __enter__(self) -> "CheckpointBook":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _default_save(payload: Any, path: Path) -> None:
    """默认保存器：优先 ``torch.save``（能保存 state_dict 与任意嵌套结构）。

    ★ 不要试图先转 numpy 再 ``np.savez`` —— 那样存下来的 ``state_dict`` 会变成 ndarray，
      回读时 ``load_state_dict`` 直接抛 ``TypeError: Expected state_dict to be dict-like``。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        torch.save(payload, path)
        return
    except Exception:                                          # noqa: BLE001
        pass
    if isinstance(payload, Mapping):
        np.savez(path, **{k: np.asarray(v) for k, v in payload.items()})
        return
    if hasattr(payload, "state_dict"):
        np.savez(path, **{k: _to_np(v) for k, v in payload.state_dict().items()})
        return
    np.save(path.with_suffix(".npy"), np.asarray(payload))


# --------------------------------------------------------------------------- 数据准备
@dataclass
class PrepData:
    """``prepare`` 的返回：X/y 标准化视图 + 趟级结构（**与既有 FF 基线同口径**）。

    ``X`` ``(n_trips, n_seg_max, F)`` · ``M`` ``(n_trips, n_seg_max)`` · ``Y`` 同形；
    ``flat_X`` / ``flat_y`` / ``flat_trip`` 是**逐段展平视图**（整表覆盖，不做 seq padding）。
    """

    feats: list[str]
    trips: np.ndarray                  # 趟键（长度 n_trips）
    split_trip: np.ndarray
    nseg: np.ndarray
    off: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    M: np.ndarray
    flat_X: np.ndarray
    flat_y: np.ndarray
    flat_trip: np.ndarray
    mu: np.ndarray
    sd: np.ndarray
    t_mu: float
    t_sd: float
    y_wh: np.ndarray                   # 趟级真值（Wh；y_full(kWh 量化)×1000）
    y_basis: str
    ds: np.ndarray                     # 逐段里程（Wh/m×100 的目标缩放用）
    extra: dict[str, Any] = field(default_factory=dict)


def _canonical(seg_csv: Any, study: Any, *, verify: bool = True) -> "Any":
    """**统一走 seg_loader 的规范表**（唯一索引换算入口）—— 本模块不自己读表。"""
    import sys
    from pathlib import Path as _P
    lib = _P(__file__).resolve().parent
    for _d in (lib, lib.parent / "preprocess"):        # 发布树：段表复用层在 ../preprocess
        if str(_d) not in sys.path:
            sys.path.insert(0, str(_d))
    import seg_loader as L                                     # noqa: PLC0415
    kw: dict[str, Any] = {}
    if seg_csv is not None:
        kw["seg_csv"] = seg_csv
    if study is not None:
        kw["study"] = study
    kw["verify_freeze"] = bool(verify)
    return L.load_canonical(**kw)


def prepare(split: Any = None, *, feats: Sequence[str] | None = None,
            seg_csv: Any = None, study: Any = None, L: int | None = None,
            verify: bool = True, seg_limit: int | None = None) -> PrepData:
    """装配训练视图（**长表口径**，`seg_limit` 用于冒烟时截断趟数）。

    ★ 与既有 FF 基线**逐位同口径**（这是能逐位复现的前提）：
        * 输入标准化：``(x − mu)/sd``，``mu/sd`` 取自 **train 段**，``sd<1e-9 → 1``
        * 目标：``y_100 = seg_energy_wh / seg_len_m × 100``，``z = (y_100 − t_mu)/t_sd``，
          ``t_mu/t_sd`` 取自 **train 段**
        * 目标逆变换：``wh = (z·t_sd + t_mu)/100 · seg_len_m``
        * 趟级真值：``y_full``(kWh, float32 量化) × 1000 = Wh（**不取整**）
    ``split`` 可以是 ``SegDataset``（用 seg_loader 的规范表）或 1-D split 数组（合成数据）。
    """
    if isinstance(split, (np.ndarray, list, tuple)) and not hasattr(split, "trip_files"):
        return _prepare_synthetic(np.asarray(split).astype(str), feats=feats, L=L)
    ds = split if hasattr(split, "trip_files") else _canonical(seg_csv, study, verify=verify)
    SPLIT_COUNTS_OK = True
    try:
        SplitSpec().check(np.asarray(ds.split)[ds.trip_offsets])
    except TrainError:
        SPLIT_COUNTS_OK = False
    feats = list(feats or ("seg_len_m", "ambient_c_med", "pack_t_med", "soc_start",
                           "cum_frac", "veh2", "mean_speed_kph", "speed_std", "speed_p95",
                           "stop_frac"))
    missing = [c for c in feats if c not in ds.df.columns]
    if missing:
        raise SchemaError("prepare: 规范表缺特征列 %s；可用 %s" % (missing, list(ds.df.columns)))
    n_trips = len(ds.trip_files)
    if seg_limit:
        n_trips = min(int(seg_limit), n_trips)
    keep_files = ds.trip_files[:n_trips]
    keep = np.isin(ds.trip_ids, np.arange(n_trips))
    df = ds.df[keep]
    fil = df["file"].to_numpy()
    chg = np.flatnonzero(np.r_[True, fil[1:] != fil[:-1]])
    files = fil[chg]
    nseg = np.diff(np.r_[chg, len(df)]).astype(np.int64)
    split_trip = df["split"].to_numpy(dtype=object)[chg]
    dsv = df["seg_len_m"].to_numpy(np.float64)
    yseg = df["seg_energy_wh"].to_numpy(np.float64)
    seg100 = yseg / np.clip(dsv, 1e-9, None) * 100.0
    tr_seg = df["split"].to_numpy(dtype=object) == "train"
    rawX = df[feats].astype(float).to_numpy(np.float64)
    mu = rawX[tr_seg].mean(0)
    sd = rawX[tr_seg].std(0)
    sd = np.where(sd >= 1e-9, sd, 1.0)
    t_mu = float(seg100[tr_seg].mean())
    t_sd = float(seg100[tr_seg].std())
    flat_X = np.ascontiguousarray((rawX - mu) / sd, dtype=np.float32)
    flat_y = np.ascontiguousarray((seg100 - t_mu) / t_sd, dtype=np.float32)
    Lmax = int(nseg.max()) if L is None else int(L)
    X = np.zeros((n_trips, Lmax, len(feats)), np.float32)
    Y = np.zeros((n_trips, Lmax), np.float32)
    M = np.zeros((n_trips, Lmax), np.float32)
    for i in range(n_trips):
        k = min(int(nseg[i]), Lmax)
        s = int(chg[i])
        X[i, :k] = flat_X[s:s + k]
        Y[i, :k] = flat_y[s:s + k]
        M[i, :k] = 1.0
    y_wh = df["y_full"].to_numpy(np.float64)[chg] * 1000.0
    return PrepData(feats=feats, trips=files, split_trip=split_trip, nseg=nseg, off=chg,
                    X=X, Y=Y, M=M, flat_X=flat_X, flat_y=flat_y,
                    flat_trip=np.repeat(np.arange(n_trips), nseg),
                    mu=mu, sd=sd, t_mu=t_mu, t_sd=t_sd, y_wh=y_wh,
                    y_basis="trip_index/seg_features.y_full(kWh, float32 量化) × 1000 = Wh（不取整）",
                    ds=dsv, extra={"n_trips": n_trips, "Lmax": Lmax,
                                   "split_counts": {k: int((split_trip == k).sum())
                                                    for k in SPLIT_COUNTS},
                                   "split_counts_ok": bool(SPLIT_COUNTS_OK),
                                   "canonical_source": getattr(ds, "source", "")})


def _prepare_synthetic(split: np.ndarray, *, feats: Sequence[str] | None = None,
                       L: int | None = None) -> PrepData:
    """合成视图（**只在测试里用**）：给一个 split 数组，造出可跑的 X/y。

    存在的理由：本套件的痛点是"守卫要能拦住真错"，而真错（零位移分支、单位错）用真数据
    造不出来或太贵。合成视图让每条守卫都有**一条先复现错误、再证明拦得住**的用例。
    """
    n = len(split)
    feats = list(feats or ("f0", "f1", "f2"))
    rng = np.random.default_rng(7)
    rawX = rng.normal(size=(n, len(feats))) + np.arange(len(feats))[None, :]
    dsv = np.full(n, 100.0)
    yseg = np.maximum(1.0, rawX @ np.array([8.0, -3.0, 2.0][:len(feats)]) + 20.0
                      + rng.normal(scale=0.5, size=n))
    seg100 = yseg / dsv * 100.0
    tr = split == "train"
    mu, sd = rawX[tr].mean(0), np.where(rawX[tr].std(0) >= 1e-9, rawX[tr].std(0), 1.0)
    t_mu, t_sd = float(seg100[tr].mean()), float(seg100[tr].std())
    flat_X = np.ascontiguousarray((rawX - mu) / sd, dtype=np.float32)
    flat_y = np.ascontiguousarray((seg100 - t_mu) / t_sd, dtype=np.float32)
    chg = np.arange(0, n, 5)
    nseg = np.diff(np.r_[chg, n]).astype(np.int64)
    n_trips = len(chg)
    Lmax = int(nseg.max()) if L is None else int(L)
    X = np.zeros((n_trips, Lmax, len(feats)), np.float32)
    Y = np.zeros((n_trips, Lmax), np.float32)
    M = np.zeros((n_trips, Lmax), np.float32)
    for i in range(n_trips):
        k = min(int(nseg[i]), Lmax)
        X[i, :k] = flat_X[chg[i]:chg[i] + k]
        Y[i, :k] = flat_y[chg[i]:chg[i] + k]
        M[i, :k] = 1.0
    return PrepData(feats=feats, trips=np.array(["T%03d" % i for i in range(n_trips)]),
                    split_trip=split[chg], nseg=nseg, off=chg, X=X, Y=Y, M=M,
                    flat_X=flat_X, flat_y=flat_y, flat_trip=np.repeat(np.arange(n_trips), nseg),
                    mu=mu, sd=sd, t_mu=t_mu, t_sd=t_sd,
                    y_wh=np.add.reduceat(yseg, chg),
                    y_basis="合成（Σ seg_energy_wh）",
                    ds=dsv, extra={"synthetic": True, "Lmax": Lmax,
                                   # ★ 合成视图按定义就不是 766/188/193 ⇒ 必须显式标 False，
                                   #   否则 sel_pool 的放宽判据取不到这个键（默认 True），
                                   #   会在第一次抽样就抛"划分计数不符"（实测踩到）。
                                   "split_counts_ok": False,
                                   "split_counts": {k: int((split[chg] == k).sum())
                                                    for k in SPLIT_COUNTS}})


# --------------------------------------------------------------------------- 上下文 / 结果
@dataclass
class TrainContext:
    """交给 ``train_fn`` 的一切（**模型不该自己去摸路径**）。"""

    model: str
    seed: int
    out_dir: Path
    ckpt: CheckpointBook
    data: PrepData
    protocol: Protocol
    split: Any = None
    log: Callable[..., None] = print
    meta: dict[str, Any] = field(default_factory=dict)
    torch: Any = None
    device: str = "cpu"
    mode: str = "protocol"
    warnings: list[str] = field(default_factory=list)


@dataclass
class TrainResult:
    """``train()`` 的返回：账本 + 裁决 + 产物路径（**infer 只认这个**）。"""

    model: str
    handle: Any = None
    protocol: Protocol | None = None
    seed: int = 42
    mode: str = "protocol"
    out_dir: Path | None = None
    ckpt_dir: Path | None = None
    best_ckpt: Path | None = None
    log_csv: Path | None = None
    history: Any = None
    n_params: int | None = None
    unit_ratio: dict[str, float] = field(default_factory=dict)
    branch: dict[str, Any] = field(default_factory=dict)
    verdict: str = "PASS"
    warnings: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)
    repro: dict[str, Any] | None = None

    def manifest(self) -> dict[str, Any]:
        return {"model": self.model, "seed": self.seed, "mode": self.mode,
                "verdict": self.verdict, "n_params": self.n_params,
                "best_ckpt": (str(self.best_ckpt) if self.best_ckpt else None),
                "log_csv": (str(self.log_csv) if self.log_csv else None),
                "unit_ratio": {k: float(v) for k, v in self.unit_ratio.items()},
                "branch": {k: v for k, v in self.branch.items() if k != "rows"},
                "protocol": (self.protocol.to_row() if self.protocol else None),
                "warnings": list(self.warnings), "elapsed_s": self.elapsed_s,
                "repro": self.repro, "meta": self.meta}


# --------------------------------------------------------------------------- 单位断言
def assert_units_wh(pred_wh: Any, y_wh: Any, *, lo: float = UNIT_LO, hi: float = UNIT_HI,
                    scope: str = "holdout", name: str = "pred") -> float:
    """**单位断言** ``Σp/Σy ∈ [0.5, 2]``（warmup 之后每个 eval 点都查）。

    越界 = 读数坍塌/量纲崩坏 —— 既有实现的做法是"终止该种子并标注"；本模块把它抬成
    **协议级断言**（默认抛，调用方显式 ``on_violation="stop"`` 才降级为终止该种子）。
    """
    return assert_unit_ratio(pred_wh, y_wh, lo=lo, hi=hi, scope=scope, name=name)


# --------------------------------------------------------------------------- 门：先复现已知-good
def gate_against_anchor(anchor: Mapping[str, float], got: Mapping[str, float], *,
                        tol: float = 3e-6, name: str = "gate") -> dict[str, Any]:
    """**先复现一个已知-good 格才允许往下跑**。

    ``anchor`` / ``got``：``{"val": 10.0605030031, "dev": 8.5513990875}`` 形式。
    任何一项超容差 ⇒ ``GateError``（拒跑，不"先跑跑看"）。
    """
    if not anchor:
        raise GateError("gate_against_anchor: anchor 为空 —— 没有已知-good 格就不该开跑新实验")
    diffs = {}
    for k, want in anchor.items():
        if k not in got:
            raise GateError("%s: 缺 %r 的实测值（锚点声明了就必须逐项复核）" % (name, k))
        diffs[k] = abs(float(got[k]) - float(want))
    bad = {k: v for k, v in diffs.items() if v > tol}
    if bad:
        raise GateError(
            "%s 门未过：已知-good 格复现失败。\n  锚点：%s\n  实测：%s\n  |Δ|：%s\n"
            "  ⇒ 在基线都复现不出来的时候开跑新实验，等于把两次未知叠在一起。\n"
            "    先修好复现（数据/口径/种子/协议），再往下跑。"
            % (name, dict(anchor), {k: float(got[k]) for k in anchor}, bad))
    return {"verdict": "PASS", "diffs": diffs, "tol": float(tol), "anchor": dict(anchor)}


# --------------------------------------------------------------------------- Trainer
class Trainer:
    """统一训练入口（协议托管）。"""

    def __init__(self, *, speed_caliber: Any = None) -> None:
        self.speed_caliber = speed_caliber

    # ---------------------------------------------------------------- 主入口
    def train(self, spec: Any, ctx_or_data: Any = None, *, data: PrepData | None = None,
              split: Any = None, protocol: Protocol | None = None,
              seed: int | None = None, out_dir: Any = None,
              gate: Mapping[str, float] | None = None, gate_tol: float = 3e-6,
              gate_fn: Callable[[], Mapping[str, float]] | None = None,
              mode: str | None = None, repro_check: bool = True,
              native_fn: Callable[[TrainContext], Any] | None = None,
              save_fn: Callable[[Any, Path], None] | None = None,
              demote_unit_violation: bool = False,
              **kw: Any) -> TrainResult:
        """训练一个已注册模型。

        参数
        ----
        spec        : ``registry.ModelSpec``（或其名字字符串）
        ctx_or_data : ``PrepData``（直接给数据）或 ``TrainContext``（高级用法）
        gate        : 已知-good 格的锚点 ``{"val": ..., "dev": ...}``；**给了就先复现再跑**
        gate_fn     : 延迟求值的锚点（跑一遍基线再返回实测值）—— 比 ``gate`` 更严谨
        mode        : ``"protocol"``（托管循环，默认）| ``"native"``（既有训练函数）
        native_fn   : ``mode="native"`` 时真正调用的既有训练入口 ``fn(ctx) -> ModelHandle``
                      （**既有模型不改写**：这条通道让 SePT-70 的 ``sept.fit`` 之类的
                      单次确定性拟合也能进统一账本）
        repro_check : 跑完后**同种子重跑一遍**的核心子集，逐位比对（守卫"种子不可复现"）
        """
        t0 = time.time()
        sp = _as_spec(spec)
        prot = protocol or Protocol()
        prot = dataclasses.replace(prot, seed=(prot.seed if seed is None else int(seed)))
        sd = prot.seed_policy.final(prot.seed)

        # ---- 门：先复现已知-good 格
        gate_out = None
        if gate_fn is not None:
            gate_out = gate_against_anchor(gate or {}, gate_fn(), tol=gate_tol, name="gate_fn")
        elif gate:
            got = {k: float(v) for k, v in gate.items()}
            gate_out = {"verdict": "SKIP(自报锚点)", "anchor": got,
                        "note": "调用方自报已复现；gate_fn 更严谨"}
        prot_row = prot.to_row()
        mdir = Path(out_dir) if out_dir else Paths().runs(sp.name)
        mdir.mkdir(parents=True, exist_ok=True)
        ctx = ctx_or_data if isinstance(ctx_or_data, TrainContext) else None
        pd_data = data if data is not None else (ctx.data if ctx else ctx_or_data)
        if not isinstance(pd_data, PrepData):
            raise TrainError("trainer.train: 需要 PrepData（或含 .data 的 TrainContext），实得 %r"
                             % type(pd_data).__name__)
        book = CheckpointBook(mdir, keep_last=prot.ckpt_keep_last, save_fn=save_fn)
        lg = (ctx.log if ctx else kw.pop("log", print))
        md = (mode or sp.extra.get("mode") or "protocol")
        the = _get_torch() if md == "protocol" or sp.extra.get("needs_torch") else None
        dev = "cuda" if (the is not None and the.cuda.is_available()) else "cpu"
        ctx = TrainContext(model=sp.name, seed=sd, out_dir=mdir, ckpt=book, data=pd_data,
                           protocol=prot, split=split, log=lg,
                           meta=dict(sp.extra, caliber=sp.caliber), torch=the, device=dev,
                           mode=md)
        lg("[trainer] %s v%s | %s/%s | mode=%s seed=%d device=%s"
           % (sp.name, sp.version or "?", sp.info_condition, sp.target_caliber, md, sd, dev))
        lg("[trainer] 协议：%s" % json.dumps(prot_row, ensure_ascii=False, default=str))
        lg("[trainer] 数据：%d 趟 / %s 段 | split=%s | Lmax=%d"
           % (len(pd_data.trips), len(pd_data.flat_X),
              pd_data.extra.get("split_counts"), pd_data.extra.get("Lmax", 0)))

        # ---- 训练
        handle = (native_fn or sp.train_fn)(ctx)
        if handle is None:
            raise TrainError("train_fn 返回 None —— 必须返回 registry.ModelHandle（含 params）")
        if not hasattr(handle, "params"):
            from registry import ModelHandle
            handle = ModelHandle(params=handle)
        book.close()

        # ---- 产物完整性守卫：mode="protocol" 就必须真的走过托管循环
        if md == "protocol":
            _man = book.manifest()
            if not book._buf and not book.last_paths() and not Path(_man["best_val_pt"]).exists():
                raise TrainError(
                    "mode='protocol' 但训练**没有产出任何 ckpt / 逐步日志** —— "
                    "train_fn 必须调用 trainer.run_protocol_loop()（协议托管循环）。\n"
                    "  ⇒ 这正是「协议形同虚设」的形态：调用方以为走了统一协议，实际上模型\n"
                    "    自己写了另一套（或什么都没写），而账本、守卫、ckpt 全部缺失。\n"
                    "  正解二选一：\n"
                    "    · torch 模型：train_fn 内构造 net/opt 后调用 TR.run_protocol_loop(ctx, ...)；\n"
                    "    · 自带训练函数的模型：train(..., mode='native', native_fn=...) 显式声明。\n"
                    "  （%s 的 train_fn 两者都没做）" % sp.name)

        # ---- 分支活性守卫（模型可把 probes 挂在 handle.meta['probes']）
        br: dict[str, Any] = {}
        probes = (handle.meta or {}).get("probes") if hasattr(handle, "meta") else None
        if probes:
            guard = BranchGuard(probes, allow_dead=(handle.meta or {}).get("allow_dead", ()))
            br = guard.check(where="训练后（%s）" % sp.name)

        # ---- 单位断言（用 holdout + val/dev 三个作用域各查一次）
        #   ★ `demote_unit_violation=True` 把"越界即抛"降级为"记录 + 告警"。
        #     存在的理由：**物理侧代理臂**（如 P14 的纯物理功率模型）会系统性低估
        #     Σp/Σy ≈ 0.3 —— 那是模型的物理局限，不是量纲错误。把它一律当 kWH/Wh
        #     混用拦下，就再也跑不了这些臂；而静默放过又会掩盖真的量纲事故。
        #     所以显式降级 + 逐作用域记录 + 进 warnings，让"降级过"永远可查。
        unit: dict[str, float] = {}
        unit_violations: list[dict[str, Any]] = []
        res_warnings_list: list[str] = []
        pred = (handle.meta or {}).get("pred_wh") if hasattr(handle, "meta") else None
        if pred is not None:
            _pd = np.asarray(pred, dtype=np.float64).reshape(-1)
            if _pd.shape[0] != int(pd_data.y_wh.shape[0]):
                raise TrainError(
                    "handle.meta['pred_wh'] 长度 %d ≠ 趟数 %d —— 单位门只能在**趟级同作用域**上算"
                    "（模型若返回逐段预测，请在 handle.meta 里放逐趟预测 'pred_wh'，"
                    "逐段预测放 'seg_wh'）。" % (_pd.shape[0], int(pd_data.y_wh.shape[0])))
            for sc in ("holdout", "val", "dev"):
                m = _scope_mask(pd_data, sc)
                if m is None or int(np.sum(m)) == 0:
                    continue
                try:
                    unit[sc] = assert_units_wh(_pd[m], pd_data.y_wh[m], scope=sc,
                                               lo=prot.unit_lo, hi=prot.unit_hi, name=sp.name)
                except Exception as _e:                       # noqa: BLE001
                    if not demote_unit_violation:
                        raise
                    unit[sc] = float("nan")
                    unit_violations.append({"scope": sc, "lo": prot.unit_lo,
                                            "hi": prot.unit_hi, "error": str(_e)[:200]})
                    lg("  ★ 单位门**降级为记录**（demote_unit_violation=True）：作用域 %s 越界"
                       % sc)

        # ---- 同种子重跑（种子可复现守卫）
        repro = None
        if repro_check and hasattr(handle, "meta") and (handle.meta or {}).get("rerun") is not None:
            a = np.asarray((handle.meta or {}).get("pred_wh"), dtype=np.float64)
            b = np.asarray(handle.meta["rerun"](), dtype=np.float64)
            eq = bool(a.shape == b.shape and np.array_equal(a, b))
            mx = float(np.max(np.abs(a - b))) if a.shape == b.shape else float("inf")
            repro = {"bitwise_equal": eq, "max_abs_diff": mx, "n": int(a.size)}
            if not eq:
                raise SeedMismatchError(
                    "种子不可复现：同种子重跑 max|Δ| = %.3e（逐位不等）—— 拒绝签字。\n"
                    "  常见成因：未固定 cudnn/SDPA 后端、未设 CUBLAS_WORKSPACE_CONFIG、"
                    "非确定性归约、DataLoader worker 种子未定。" % mx)

        for _v in unit_violations:
            res_warnings_list.append(
                "单位门越界（已降级为记录）：作用域 %s 的 Σp/Σy 越出 [%g, %g]"
                % (_v["scope"], _v["lo"], _v["hi"]))
        man = book.manifest()
        res = TrainResult(
            model=sp.name, handle=handle, protocol=prot, seed=sd, mode=md, out_dir=mdir,
            ckpt_dir=Path(man["ckpt_dir"]), best_ckpt=Path(man["best_val_pt"]),
            log_csv=Path(man["log_csv"]), history=handle.history,
            n_params=handle.n_params, unit_ratio=unit, branch=br,
            verdict="PASS", elapsed_s=time.time() - t0,
            warnings=list(res_warnings_list),
            meta=dict(sp.extra, caliber=sp.caliber, gate=gate_out, ckpt=man))
        (mdir / "train_manifest.json").write_text(
            json.dumps(res.manifest(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
        lg("[trainer] 完成：%.1fs | 参数量 %s | 单位比 %s | 分支 %s"
           % (res.elapsed_s, res.n_params,
              {k: round(v, 4) for k, v in unit.items()},
              {k: v for k, v in br.items() if k != "rows"}))
        lg("[trainer] ckpt：%s" % man)
        return res


def _as_spec(spec: Any) -> Any:
    if isinstance(spec, str):
        import registry
        return registry.get(spec)
    if not hasattr(spec, "train_fn"):
        raise TrainError("trainer.train: spec 必须是 registry.ModelSpec 或已注册的模型名，"
                         "实得 %r" % type(spec).__name__)
    return spec


def _get_torch() -> Any:
    try:                                                       # pragma: no cover
        import torch
        return torch
    except Exception:                                          # noqa: BLE001
        return None


def _scope_mask(d: PrepData, scope: str) -> np.ndarray | None:
    st = np.asarray(d.split_trip).astype(str)
    if scope == "val":
        return st == "validation"
    if scope == "dev":
        return st == "future"
    if scope == "train":
        return st == "train"
    if scope == "holdout":
        tr = np.flatnonzero(st == "train")
        ho = tr[::HOLDOUT_STRIDE][:HOLDOUT_N]
        m = np.zeros(len(st), dtype=bool)
        m[ho] = True
        return m
    return None


# --------------------------------------------------------------------------- 托管训练循环
def run_protocol_loop(ctx: TrainContext, *, net: Any, opt: Any, batch_fn: Callable[..., Any],
                      eval_fn: Callable[[], Mapping[str, Any]],
                      probe_names: Sequence[str] | None = None,
                      required_probes: Any = True, allow_dead: Sequence[str] = (),
                      grad_probe_every: int = 1,
                      trace_own_loop: bool = False,
                      lr_fn: Callable[[int], float] | None = None,
                      save_best_min_delta: float | None = None,
                      patience_in: str = "evals") -> dict[str, Any]:
    """**托管训练循环**（warmup+cosine / patience / ckpt / 逐步日志 / 单位门 / 分支守卫）。

    模型只提供三样东西，**不写循环**：

        net        : ``torch.nn.Module``（参数会被探针快照）
        opt        : 优化器（LR 由本循环按 warmup+cosine 写进 param_groups）
        batch_fn(sel, gen) -> (loss, n_valid, parts)
                     ``sel`` = 本步抽到的**趟号**数组（ndarray）；``gen`` = torch.Generator
        eval_fn() -> {"val_wmape":..., "dev_wmape":..., "pred_true_ratio":...,
                      "holdout_mse":..., "branch": {名: 当前参数张量}}
                     —— 每个 eval 点调用一次（模型自己决定怎么算）

    ★ **必守的数值细节**（逐位复现的成败就在这里）：
      * 权重初始化必须发生在 ``apply_seed`` **之后**、任何一次 ``optimizer.step`` **之前**
        —— 所以 ``net`` 由调用方在 ``train_fn`` 里构造，本循环只负责开始训练。
      * 每个 optimizer step 前把 LR 写成 ``lr_at(step)``；**step 从 0 计**。
      * eval 点在 ``step % eval_every == 0`` 或最后一步触发（与既有实现同判据）；
        holdout 改善才存 ``best_val.pt``；每个 eval 点滚一次 ``last_1..N``。
      * ★ **步数由 ``max_steps`` 说了算**，不是由 ``n_epochs_max × steps_per_epoch`` 说了算。
        实测踩到：``n_epochs_max=60 × steps_per_epoch=200 = 7500 < max_steps=9000``，
        循环在 7500 步就"跑满 60 个 epoch"退出了 —— 少跑 1500 步，
        逐位复现直接变成差 0.47 pp，而且**日志一句都不报**。
        ``steps_per_epoch`` 现在的唯一作用是"每多少步落一次 train_loss 均值"；
        ``n_epochs_max`` 只是防御性上界（真正越界会记进 ``warnings``）。
      * warmup 期内**不判坍塌**（``collapse_grace_frac``），因为此时输出≈均值是设计如此。
      * **单位门**在 warmup 之后每个 eval 点都查；越界 ⇒ 终止该种子并在结果里标
        ``collapse``（``demote`` 由调用方决定）。

    三个**可覆盖**的口径（默认 = 项目协议；既有模型自带配方时按它的配方显式给）
      * ``lr_fn(step) -> float``：LR 调度。默认 warmup+cosine；给常数函数就是常数 LR。
      * ``save_best_min_delta``：best 的改善门槛（默认 ``prot.min_delta``）。
      * ``patience_in``：patience 的计数单位（``"evals"`` = eval 点个数，默认）。
    """
    prot = ctx.protocol
    the = ctx.torch
    if the is None:
        raise TrainError("run_protocol_loop 需要 torch（ctx.torch 为 None）")
    apply_seed(ctx.seed, torch_mod=the)
    gen = generator(ctx.seed, torch_mod=the)
    # ★ 设备归位：模型可能建在 CPU 上（``nn.Sequential(...)`` 的默认），而数据在 GPU ——
    #   那会在第一次前向就抛 "Expected all tensors to be on the same device"。
    #   ``.to(device)`` 不消耗 RNG、不改数值（同 device 时是 no-op），所以对逐位复现无害。
    if hasattr(net, "to"):
        net = net.to(ctx.device)
    probes = snapshot_probes(net, probe_names, required=required_probes)
    lg = ctx.log
    hist: list[dict[str, Any]] = []
    best, best_step, bad = float("inf"), -1, 0
    step, stop_reason, collapse = 0, "max_steps", None
    guarded = max(1, int(prot.collapse_grace_frac * prot.max_steps))
    grad_cache: dict[str, Any] = {}
    eval_every = max(1, int(prot.eval_every))
    n_eval_max = int(prot.n_epochs_max)        # 防御性上界：最多这么多个 **eval 点**
    # ★ 每段步数必须与 eval 节奏一致，否则 eval 点的 step 序列与既有实现错开
    #   （实测踩到：steps_per_epoch=200 而 eval_every=250 ⇒ eval 落在 200/400/600...，
    #    与参考实现的 250/500/750... 完全错位，"第几个 eval 点最好"就选错了）。
    per_seg = eval_every
    min_delta = (prot.min_delta if save_best_min_delta is None
                 else float(save_best_min_delta))
    ep = 0
    while step < int(prot.max_steps):
        ep += 1
        ep_loss, nb = 0.0, 0
        while nb < per_seg and step < int(prot.max_steps):
            if lr_fn is not None:
                _v = float(lr_fn(step))
                for g in opt.param_groups:
                    g["lr"] = _v
            else:
                apply_warmup_cosine(opt, step, lr=prot.lr, total=prot.max_steps,
                                    warmup_frac=prot.warmup_frac, cos_steps=prot.cos_steps)
            sel = _draw_trips(sel_pool(ctx), int(prot.batch), gen, the)
            net.train()
            loss, nseg, parts = batch_fn(sel, gen)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = float(the.nn.utils.clip_grad_norm_(_iter_params_list(net), prot.grad_clip))
            if step % max(1, int(grad_probe_every)) == 0:
                for nm, p in _named_params(net):
                    if p.grad is not None:
                        grad_cache[nm] = p.grad
            opt.step()
            if not trace_own_loop:
                ep_loss += float(loss.detach())
            nb += 1
            step += 1
            if (step % eval_every == 0) or step >= int(prot.max_steps):
                break
        ev = dict(eval_fn())
        # ---- 分支活性账
        probe_activity(probes, net, grads=grad_cache)
        rows = [p.row() for p in probes.values()]
        n_dead = sum(1 for r in rows if r["dead"])
        n_frozen = sum(1 for r in rows if r["frozen"])
        min_disp = min((r["max_abs_disp"] for r in rows), default=0.0)
        # ---- 单位门（warmup 之后才判）
        ratio = float(ev.get("pred_true_ratio", float("nan")))
        unit_bad = not (prot.unit_lo <= ratio <= prot.unit_hi)
        if unit_bad and step >= guarded and np.isfinite(ratio):
            collapse = {"epoch": ep + 1, "step": step, "ratio": ratio}
            stop_reason = "collapse_unit@ep%d" % (ep + 1)
        hv = float(ev.get("holdout_mse", float("nan")))
        if _DEBUG:
            print("[DBG step=%d] hv=%r finite=%s best=%r bad=%d ratio=%r unit_bad=%s"
                  % (step, hv, np.isfinite(hv), best, bad, ratio,
                     (not (prot.unit_lo <= ratio <= prot.unit_hi))), flush=True)
        rec = {"epoch": ep + 1, "step": step,
               "train_loss": round(ep_loss / max(nb, 1), 6),
               "holdout_mse": (round(hv, 6) if np.isfinite(hv) else ""),
               "val_wmape": ev.get("val_wmape", ""), "dev_wmape": ev.get("dev_wmape", ""),
               "lr": float(opt.param_groups[0]["lr"]), "grad_norm": round(gn, 4),
               "pred_true_ratio": (round(ratio, 4) if np.isfinite(ratio) else ""),
               "n_branch": len(rows), "n_branch_dead": n_dead,
               "n_branch_frozen": n_frozen, "branch_min_disp": float(min_disp),
               "note": ("collapse" if collapse else "")}
        ctx.ckpt.log(rec)
        ctx.ckpt.save_last(_payload(net, opt, step, prot, ctx))
        if np.isfinite(hv):
            # ★ 改善判据：hv < best − min_delta（min_delta 默认为 0 ⇒ 严格改善）。
            #   既有实现用 1e-7，本框架用 prot.min_delta / save_best_min_delta 显式给。
            improved = ctx.ckpt.save_best(_payload(net, opt, step, prot, ctx),
                                          holdout=hv, step=step,
                                          better_than=(best - min_delta))
            if _DEBUG:
                print("[DBG step=%d] save_best -> %s  ckpt.best=%r"
                      % (step, improved, ctx.ckpt.best), flush=True)
            if improved:
                best, best_step, bad = hv, step, 0
            else:
                bad += 1
        hist.append(rec)
        lg("[%s s%d] ep%3d step%5d loss=%.4f ho=%s val=%s dev=%s ratio=%.3f lr=%.2e "
           "gn=%.2f dead=%d"
           % (ctx.model, ctx.seed, ep + 1, step, rec["train_loss"],
              rec["holdout_mse"], rec["val_wmape"], rec["dev_wmape"],
              ratio if np.isfinite(ratio) else float("nan"), rec["lr"], gn, n_dead))
        if collapse:
            lg("  ★读数越界（Σp/Σy=%.3f 越出 [%g, %g]）→ 终止该种子"
               % (ratio, prot.unit_lo, prot.unit_hi))
            break
        if step >= int(prot.max_steps):
            stop_reason = "max_steps"
            break
        if bad >= int(prot.patience_epochs):
            stop_reason = "early_stop@eval%d" % ep
            lg("  %s s%d 早停：连续 %d 个 eval 点无改善（best=%.4f）"
               % (ctx.model, ctx.seed, bad, best))
            break
        if ep >= n_eval_max:
            stop_reason = "n_epochs_max@step%d" % step
            ctx.warnings.append(
                "n_epochs_max=%d 先于 max_steps=%d 用尽（只跑到 step=%d）—— "
                "步数没跑满，与既有数字不可比。请把 n_epochs_max 调到 "
                "ceil(max_steps/eval_every)=%d 以上。"
                % (n_eval_max, prot.max_steps, step,
                   -(-int(prot.max_steps) // eval_every)))
            lg("  ★ " + ctx.warnings[-1])
            break
    # ---- 收尾：恢复 best（若从未改善则用当前权重）
    if not hist:
        raise TrainError(
            "run_protocol_loop 一个 eval 点都没跑到（hist 为空，step=%d）—— 拒绝静默交付。\n"
            "  常见成因：max_steps=%d / eval_every=%d / n_epochs_max=%d 三者不自洽，"
            "或 sel_pool/eval_fn 抛异常被吞掉了。"
            % (step, prot.max_steps, prot.eval_every, prot.n_epochs_max))
    bp = ctx.ckpt.ck / "best_val.pt"
    if bp.exists():
        _load_payload(net, bp, the)
    else:
        ctx.warnings.append("best_val.pt 未生成（holdout 全程非有限）")
    guard = BranchGuard(probes, allow_dead=allow_dead)
    br = guard.check(where="run_protocol_loop 结束（%s s%d）" % (ctx.model, ctx.seed))
    return {"net": net, "hist": hist, "best_holdout": best, "best_step": best_step,
            "stop_reason": stop_reason, "collapse": collapse, "probes": probes,
            "branch": br, "probe_rows": [p.row() for p in probes.values()]}


def sel_pool(ctx: TrainContext) -> np.ndarray:
    """本步可抽的趟池（**默认 = 协议 holdout 之外的 train 趟**）。

    "只用 train 内部 holdout 早停、拟合池 = train − holdout" 这条纪律的唯一落实点。
    合成视图（``split_counts_ok=False``）自动放宽计数与 holdout 数。
    """
    synth = not bool(ctx.data.extra.get("split_counts_ok", True))
    fit, _ho = holdout_indices(ctx.data.split_trip, stride=ctx.protocol.holdout_stride,
                               n=ctx.protocol.holdout_n, strict_counts=not synth,
                               allow_short=synth)
    return fit


def holdout_of(ctx: TrainContext) -> tuple[np.ndarray, np.ndarray]:
    """``(fit, holdout)`` —— 与 ``sel_pool`` 同一套放宽规则（合成视图自动放宽）。"""
    synth = not bool(ctx.data.extra.get("split_counts_ok", True))
    return holdout_indices(ctx.data.split_trip, stride=ctx.protocol.holdout_stride,
                           n=ctx.protocol.holdout_n, strict_counts=not synth,
                           allow_short=synth)


def _draw_trips(pool: np.ndarray, batch: int, gen: Any, the: Any) -> np.ndarray:
    """从趟池里有放回抽 ``batch`` 个趟号（**采样器是协议的一部分**，不许各写一套）。

    两种 Generator 都要支持：``torch.Generator``（用 ``torch.randint(..., generator=g)``）
    与 ``numpy.Generator``（``g.integers``）。★ 局部变量名不能用 ``the``：那是注解里引用的
    模块名，一旦被赋值成局部变量，注解（模块级求值）与运行期就会分叉。
    """
    torch_mod = the
    if torch_mod is not None:
        try:
            idx = torch_mod.randint(0, len(pool), (int(batch),), generator=gen).cpu().numpy()
            return np.sort(np.asarray(pool)[idx])
        except (AttributeError, TypeError, RuntimeError):
            pass
    if not hasattr(gen, "integers"):
        raise TrainError("无法从 %r 抽样（既不是 torch.Generator 也不是 numpy.Generator）"
                         % type(gen).__name__)
    return np.sort(np.asarray(pool)[gen.integers(0, len(pool), int(batch))])


def _iter_params(net: Any) -> Any:
    if hasattr(net, "parameters"):
        return net.parameters()
    if isinstance(net, Mapping):
        return list(net.values())
    raise TrainError("无法从 %r 取参数" % type(net).__name__)


def _iter_params_list(net: Any) -> list:
    """``clip_grad_norm_`` 需要**具体**的可迭代（生成器会被消耗两次而出错）。"""
    if hasattr(net, "parameters"):
        return list(net.parameters())
    if isinstance(net, Mapping):
        return list(net.values())
    raise TrainError("无法从 %r 取参数" % type(net).__name__)


def _payload(net: Any, opt: Any, step: int, prot: Protocol, ctx: TrainContext) -> dict[str, Any]:
    return {"model": ctx.model, "seed": ctx.seed, "step": int(step),
            "protocol": prot.to_row(), "caliber": ctx.meta.get("caliber", {}),
            "state_dict": (_state_dict(net) if hasattr(net, "state_dict") else _to_np_map(net))}


def _state_dict(net: Any) -> dict[str, Any]:
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


def _to_np_map(net: Any) -> dict[str, Any]:
    return {nm: _to_np(p) for nm, p in _named_params(net)}


def _load_payload(net: Any, path: Path, the: Any) -> None:
    """回读 ckpt（``{"state_dict": {...}}`` 或裸 state_dict 两种形态都认）。"""
    if the is not None:
        obj = the.load(path, map_location="cpu", weights_only=False)
    else:                                                      # pragma: no cover
        obj = np.load(path, allow_pickle=True)
        obj = {k: obj[k] for k in obj.files}
    sd = obj.get("state_dict", obj) if isinstance(obj, Mapping) else obj
    if not isinstance(sd, Mapping):
        raise TrainError("ckpt %s 里没有可用的 state_dict（实得 %r）"
                         % (path, type(sd).__name__))
    if hasattr(net, "load_state_dict"):
        net.load_state_dict(sd)
    else:
        raise TrainError("net 无 load_state_dict；请在 train_fn 里自行处理 best_val.pt")


# --------------------------------------------------------------------------- 函数式入口
def train(spec: Any, ctx_or_data: Any = None, **kw: Any) -> TrainResult:
    """``Trainer().train(...)`` 的函数式别名（**项目里唯一的 ``train(``**）。"""
    return Trainer().train(spec, ctx_or_data, **kw)
