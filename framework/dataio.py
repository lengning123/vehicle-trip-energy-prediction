#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""dataio —— **五种模型族共用的数据契约**（发布树里唯一一份"怎么把段表变成张量"）。

为什么需要它
============
统一框架把"训练/推理/评估"托管了，但没有规定"模型看到的张量长什么样"。若每个适配器
各写一遍取数，就会出现经典的**口径漂移**：一个按趟求和、另一个按段平均，单位从 Wh 变
kWh，划分从 766/188/193 变成别的。本模块把这件事钉死：

* **行序即身份**：段表按 ``(file, seg_idx)`` 升序；趟按首次出现顺序；键序指纹可比对。
* **标准化只用 train 段**：``(x − mu)/sd``，``mu/sd`` 取自 train（与既有 FF 基线同口径）。
* **两种目标口径**（``target_caliber``）：
    - ``y_full``     ：趟级全程能量（Wh）= ``y_full``(kWh, float32 量化) × 1000
    - ``y_rem@phi``  ：切点 φ 之后的剩余能量（Wh）= Σ 后缀段 ``seg_energy_wh``
* **两种信息条件**（``info_condition``）：
    - ``plan-time``  ：70 项 plan-time 特征（**realized 项 0 个**）
    - ``realized``   ：10 项实测运动特征（与既有 FF-MLP 基线同列，**不可与 plan-time 比**）

红线：只读冻结 ``data_v3`` 与既有只读产物；不读 calibration；不写任何输入目录。
本文件**不含 assert**。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

_RELEASE = Path(__file__).resolve().parents[1]
for _p in (_RELEASE / "framework", _RELEASE / "preprocess", _RELEASE):
    _s = str(_p)
    if _s in sys.path:
        sys.path.remove(_s)
    sys.path.insert(0, _s)

import features as FT                                               # noqa: E402
import trainer as TR                                                 # noqa: E402
from paths import Paths                                              # noqa: E402

__all__ = [
    "FEATS_PLAN70", "FEATS_REALIZED10", "FEATS_BY_CONDITION", "MISSING_FILL",
    "DataError", "TripPack", "Policy", "POLICIES", "get_policy",
    "load_pack", "seq_batches", "pack_summary", "clear_cache",
]

# --------------------------------------------------------------------------- 两种特征集
# 70 项 plan-time（顺序即冻结顺序；与 SePT-70 / AA_prefixxer 的清单逐字一致）
FEATS_PLAN70: list[str] = list(FT.FEATURES70)
# 10 项 realized（与既有 FF-MLP 基线同列；段表原生列，非派生）
FEATS_REALIZED10: list[str] = ["seg_len_m", "ambient_c_med", "pack_t_med", "soc_start",
                               "cum_frac", "veh2", "mean_speed_kph", "speed_std",
                               "speed_p95", "stop_frac"]
FEATS_BY_CONDITION: dict[str, list[str]] = {"plan-time": FEATS_PLAN70,
                                            "realized": FEATS_REALIZED10}
MISSING_FILL = FT.MISSING_FILL
# 冻结段表的 canonical 行数（100 m 档）—— 行数不对说明读的不是那一份
PP_ROW_EXPECTED = 201_271

# 划分名（框架内部一律用段表原生的 validation / future；显示层再叫 val / dev）
SPLIT_TO_SHORT = {"train": "train", "validation": "val", "future": "dev"}


class DataError(RuntimeError):
    """数据契约违反（缺列 / 形状不符 / 划分不对 / 键序不稳）—— 消息里给正解。"""


# --------------------------------------------------------------------------- 口径策略
@dataclass(frozen=True)
class Policy:
    """一个模型族的数据口径（**显式两维 + 特征集**）。

    ``info_condition`` / ``target_caliber`` 就是注册表要求的两维；``phi`` 只在
    ``target_caliber="y_rem@phi"`` 时有意义。
    """

    info_condition: str = "plan-time"
    target_caliber: str = "y_full"
    phi: float = 0.0
    feats: Sequence[str] = ()

    def __post_init__(self) -> None:
        if self.info_condition not in FEATS_BY_CONDITION:
            raise DataError("info_condition=%r 非法；合法 %s"
                            % (self.info_condition, sorted(FEATS_BY_CONDITION)))
        if not str(self.target_caliber).startswith(("y_full", "y_rem@")):
            raise DataError("target_caliber=%r 非法；合法 y_full | y_rem@<phi>"
                            % (self.target_caliber,))
        if self.info_condition == "plan-time" and self.target_caliber.startswith("y_rem@"):
            raise DataError(
                "口径组合 (plan-time, y_rem@phi) 在本发布树里没有已冻结的对照锚点 —— "
                "拒绝悄悄跑一个无法归位的格子。\n"
                "  正解：plan-time 用 target_caliber='y_full'；"
                "y_rem@phi 用 info_condition='realized'（前缀条件化）。")

    @property
    def feats_list(self) -> list[str]:
        return list(self.feats) if self.feats else list(FEATS_BY_CONDITION[self.info_condition])

    @property
    def short(self) -> str:
        return "%s|%s" % (self.info_condition, self.target_caliber)


POLICIES: dict[str, Policy] = {
    "plan70": Policy("plan-time", "y_full", 0.0, FEATS_PLAN70),
    "realized10": Policy("realized", "y_full", 0.0, FEATS_REALIZED10),
    "rem_p02": Policy("realized", "y_rem@0.2", 0.2, FEATS_REALIZED10),
    "rem_p04": Policy("realized", "y_rem@0.4", 0.4, FEATS_REALIZED10),
    "rem_p06": Policy("realized", "y_rem@0.6", 0.6, FEATS_REALIZED10),
}


def get_policy(name: str) -> Policy:
    """按名取口径策略（未知名 ⇒ 抛，并列出可用名）。"""
    if name not in POLICIES:
        raise DataError("未知口径策略 %r；可用：%s" % (name, sorted(POLICIES)))
    return POLICIES[name]


# --------------------------------------------------------------------------- 趟级数据包
@dataclass
class TripPack:
    """所有模型族共用的**趟级数据包**（只读视图 + 少量派生张量）。

    字段
    ----
    feats        : 特征列名（顺序即张量末维顺序）
    trips        : 趟键 ``file``（长度 n_trips；**行序即身份**）
    split        : 趟级划分（``train`` / ``validation`` / ``future``）
    nseg         : 每趟段数
    off          : 每趟在展平数组里的起始行（``np.diff(np.r_[off, n_seg_all])`` = nseg）
    X3           : ``(n_trips, Lmax, F)`` float32，已按 train 段标准化（右侧零填充）
    M3           : ``(n_trips, Lmax)`` float32，1=有效段 / 0=填充
    flat_X       : ``(n_seg, F)`` float32，标准化后的展平视图
    flat_trip    : ``(n_seg,)`` int64，段 → 趟下标
    raw          : ``(n_seg, F)`` float64，**未标准化**的原始特征（可解释性 / 树模型用）
    y_seg_wh     : ``(n_seg,)`` float64，段级真值能量（Wh）
    y_full_wh    : ``(n_trips,)`` float64，趟级全程真值（Wh；y_full 基）
    y_trip       : ``(n_trips,)`` float64，**本口径**的趟级目标（Wh）
    phi          : 目标口径的 φ（``y_full`` 时为 0.0）
    mu, sd       : train 段的特征均值 / 标准差（标准化参数，随行留痕）
    t_mu, t_sd   : train 段的目标（每 100 m 能量）均值 / 标准差
    seg_len_m    : ``(n_seg,)`` float64，段长（目标逆变换用）
    df           : 段级宽表（含 ``cum_end_m`` / ``split`` / ``y_full`` 等，只读引用）
    cuts         : 切点表（φ 网格；``build_cuts`` 的产物）
    info         : 数据来源与冻结自检的声明
    """

    feats: list[str]
    trips: np.ndarray
    split: np.ndarray
    nseg: np.ndarray
    off: np.ndarray
    X3: np.ndarray
    M3: np.ndarray
    flat_X: np.ndarray
    flat_trip: np.ndarray
    raw: np.ndarray
    y_seg_wh: np.ndarray
    y_full_wh: np.ndarray
    y_trip: np.ndarray
    phi: float
    mu: np.ndarray
    sd: np.ndarray
    t_mu: float
    t_sd: float
    seg_len_m: np.ndarray
    df: pd.DataFrame
    cuts: pd.DataFrame
    info: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- 基本量
    @property
    def n_trips(self) -> int:
        return int(len(self.trips))

    @property
    def n_seg(self) -> int:
        return int(len(self.flat_X))

    @property
    def n_feat(self) -> int:
        return len(self.feats)

    def rows_of(self, i: int) -> np.ndarray:
        """第 ``i`` 趟在展平数组里的行位置。"""
        s = int(self.off[i])
        return np.arange(s, s + int(self.nseg[i]), dtype=np.int64)

    def mask_of(self, which: str) -> np.ndarray:
        """趟级布尔掩码：``train`` / ``validation`` / ``future``（也接受 ``val`` / ``dev``）。"""
        w = {"val": "validation", "dev": "future"}.get(which, which)
        return self.split.astype(str) == w

    def idx_of(self, which: str) -> np.ndarray:
        """趟级位置索引（``which`` 同上）。"""
        return np.flatnonzero(self.mask_of(which))

    def seg_mask_of(self, which: str) -> np.ndarray:
        """段级布尔掩码（同划分）。"""
        m = self.mask_of(which)
        return m[self.flat_trip]

    def rem_target(self, phi: float) -> np.ndarray:
        """切点 φ 的趟级剩余能量（Wh），按 ``trips`` 顺序对齐；短路（``cut_ok=False``）为 NaN。"""
        c = FT.cut_for_phi(self.cuts, phi).set_index("file")
        miss = [f for f in self.trips if f not in c.index]
        if miss:
            raise DataError("切点表缺 %d 趟（例 %s）—— 键必须覆盖全部趟" % (len(miss), miss[:3]))
        y = c.loc[self.trips, "y_rem_wh"].to_numpy(np.float64)
        ok = c.loc[self.trips, "cut_ok"].to_numpy(bool)
        return np.where(ok, y, np.nan)

    def full_target_of_phi(self, phi: float) -> np.ndarray:
        """切点 φ 的趟级全程能量（Wh），与 :meth:`rem_target` 同序（闭合用）。"""
        c = FT.cut_for_phi(self.cuts, phi).set_index("file")
        return c.loc[self.trips, "y_full_wh"].to_numpy(np.float64)

    def write_feature_table(self, path: str | Path) -> Path:
        """把 70 列特征宽表落盘（**不含任何标签**；供"自备特征文件"的用户对齐列名）。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        cols = ["file", "seg_idx", "split"] + self.feats
        out = self.df[cols].copy()
        out.to_csv(p, index=False)
        return p


# --------------------------------------------------------------------------- 装载（进程内缓存）
_CACHE: dict[tuple, TripPack] = {}


def clear_cache() -> None:
    """清空进程内缓存（改了输入根 / 冻结依赖后必须调用）。"""
    _CACHE.clear()


def load_pack(policy: Policy | str = "plan70", *, P: Paths | None = None,
              verify: bool = True, with_cuts: bool = True,
              verbose: bool = False) -> TripPack:
    """装载趟级数据包（**唯一入口**）。

    ``policy`` 可以是 :class:`Policy` 或 :data:`POLICIES` 里的名字（``plan70`` /
    ``realized10`` / ``rem_p02`` / ``rem_p04`` / ``rem_p06``）。
    同一进程内按 ``(输入根, 口径, 特征集)`` 记忆化 —— 框架里的 ``trainer.prepare``
    已经做过一次同样的读表，重复调用不会重复付这份代价。
    """
    pol = get_policy(policy) if isinstance(policy, str) else policy
    if not isinstance(pol, Policy):
        raise DataError("load_pack: policy 必须是 Policy 或 POLICIES 里的名字，实得 %r"
                        % type(policy).__name__)
    pp = P or Paths()
    key = (str(pp.seg), pol.short, tuple(pol.feats_list))
    if key in _CACHE:
        return _CACHE[key]

    # 宽表只组一次：70 列特征 + 段级键/标签列（依赖哈希门在 load_wide 里）
    df = FT.load_wide(pp, verbose=verbose, verify=verify)
    feats = pol.feats_list
    FT.assert_plan_only(feats, pol.info_condition)
    miss = [c for c in feats if c not in df.columns]
    if miss:
        raise DataError("特征列缺失 %s；可用列 %s" % (miss, list(df.columns)))
    n_seg = int(len(df))
    if n_seg != PP_ROW_EXPECTED:
        raise DataError("段表行数 %d ≠ %d —— 段表不是冻结的那一份（拒绝继续）"
                        % (n_seg, PP_ROW_EXPECTED))

    fil = df["file"].to_numpy(dtype=object)
    chg = np.flatnonzero(np.r_[True, fil[1:] != fil[:-1]])
    if chg.size == 0 or not np.array_equal(fil[chg], np.unique(fil)):
        # 趟必须是"按 file 字典序成块出现"的连续区段 —— 否则"段→趟"换算不成立
        raise DataError("段表不是按 (file, seg_idx) 成块升序 —— 段→趟换算不成立；"
                        "请检查段表来源（冻结段表满足此性质）")
    files = fil[chg]
    trip_of_row = np.repeat(np.arange(len(chg)), np.diff(np.r_[chg, n_seg])).astype(np.int64)
    trip_split = np.asarray(df["split"])[chg]
    n_trips = int(len(files))
    nseg = np.diff(np.r_[chg, n_seg]).astype(np.int64)
    off = chg.astype(np.int64)
    Lmax = int(nseg.max())
    F = len(feats)

    # 标准化：**只用 train 段**（与既有 FF 基线同口径）；sd<1e-9 ⇒ 1
    is_tr = df["split"].to_numpy(dtype=object).astype(str) == "train"
    raw = df[feats].astype(float).to_numpy(np.float64)
    n_missing = int(np.isnan(raw).sum())
    if n_missing:                     # 与 preprocess.build_features 同策略（填充 -999.0）
        raw = np.where(np.isnan(raw), MISSING_FILL, raw)
    if not np.isfinite(raw).all():
        raise DataError("特征矩阵含非有限值（NaN 已填 %g，inf 不填）—— 拒绝继续" % MISSING_FILL)
    mu = raw[is_tr].mean(0)
    sd = raw[is_tr].std(0)
    sd = np.where(sd >= 1e-9, sd, 1.0)
    flat_X = np.ascontiguousarray((raw - mu) / sd, dtype=np.float32)

    # 目标：每 100 m 能量（与 trainer.prepare 逐位同口径）
    seg_len = df["seg_len_m"].to_numpy(np.float64)
    y_seg = df["seg_energy_wh"].to_numpy(np.float64)
    seg100 = y_seg / np.clip(seg_len, 1e-9, None) * 100.0
    t_mu = float(seg100[is_tr].mean())
    t_sd = float(seg100[is_tr].std())

    X3 = np.zeros((n_trips, Lmax, F), np.float32)
    M3 = np.zeros((n_trips, Lmax), np.float32)
    for i in range(n_trips):
        k = int(nseg[i])
        X3[i, :k] = flat_X[off[i]:off[i] + k]
        M3[i, :k] = 1.0

    # y_full 是**趟级常量列**（同一趟所有段同值）⇒ 取每趟首个段的值作为趟级真值
    y_full_col = df["y_full"].to_numpy(np.float64)
    y_full_wh = y_full_col[off] * 1000.0
    if not np.allclose(y_full_col, np.repeat(y_full_wh, nseg) / 1000.0,
                       rtol=1e-12, atol=1e-9):
        raise DataError("同趟内 y_full 不恒定 —— 趟边界或列语义被改动，拒绝继续")
    phi = float(pol.phi) if pol.target_caliber.startswith("y_rem@") else 0.0
    cuts = FT.build_cuts(df, P=pp, verbose=False) if with_cuts else pd.DataFrame()
    if pol.target_caliber.startswith("y_rem@"):
        c = FT.cut_for_phi(cuts, phi).set_index("file")
        y_trip = np.where(c.loc[files, "cut_ok"].to_numpy(bool),
                          c.loc[files, "y_rem_wh"].to_numpy(np.float64), np.nan)
    else:
        y_trip = y_full_wh.copy()

    info = {"input_root": str(pp.seg), "release_root": str(pp.release),
            "n_seg": int(n_seg), "n_trips": int(n_trips), "Lmax": Lmax,
            "n_feat": int(F), "info_condition": pol.info_condition,
            "target_caliber": pol.target_caliber, "phi": phi,
            "missing_filled": int(n_missing),
            "trip_by_split": {k: int((trip_split.astype(str) == k).sum())
                              for k in ("train", "validation", "future")},
            "y_basis": ("seg_features.y_full(kWh, float32 量化)×1000 = Wh" if phi == 0.0
                        else "Σ 后缀 seg_energy_wh（精确 Wh）"),
            "verify": {"plan_only_ok": True} if verify else {"plan_only_ok": False}}
    pack = TripPack(feats=list(feats), trips=files, split=trip_split, nseg=nseg,
                    off=off, X3=X3, M3=M3, flat_X=flat_X, flat_trip=trip_of_row,
                    raw=raw, y_seg_wh=y_seg, y_full_wh=y_full_wh, y_trip=y_trip, phi=phi,
                    mu=mu, sd=sd, t_mu=t_mu, t_sd=t_sd, seg_len_m=seg_len,
                    df=df, cuts=cuts, info=info)
    _CACHE[key] = pack
    if verbose:
        print("[dataio] %s" % pack_summary(pack))
    return pack


def limit_pack(pack: TripPack, n_trips: int, *, label: str = "smoke") -> TripPack:
    """取前 ``n_trips`` 趟的**子集视图**（只给冒烟用）。

    ★ 冒烟纪律：冒烟必须把**最贵的步骤**换成子集或冻结产物，而不是只把迭代数改小。
      本树的神经臂最贵的一步是"冻结树基线"（一次 HGBR 拟合，134,212 段 × 70 特征，
      纯 CPU + OpenMP）；只把 ``max_steps`` 从 1200 改成 30 而仍跑全量基线，
      结果是"省了便宜的 30 步、留下昂贵的多线程拟合" ⇒ 实测挂满 15 分钟、
      GPU 0%、CPU 1300%。**冒烟要冒到贵的那一段。**
    """
    import dataclasses
    n = int(min(max(int(n_trips), 3), pack.n_trips))
    idx = np.arange(n, dtype=np.int64)
    nseg = np.asarray(pack.nseg[idx], dtype=np.int64)
    off = np.r_[0, np.cumsum(nseg)[:-1]].astype(np.int64)
    rows = np.concatenate([np.arange(int(pack.off[i]), int(pack.off[i]) + int(nseg[k]))
                           for k, i in enumerate(idx)]) if nseg.sum() else np.zeros(0, np.int64)
    Lmax = int(nseg.max())
    X3 = np.zeros((n, Lmax, pack.n_feat), np.float32)
    M3 = np.zeros((n, Lmax), np.float32)
    for k in range(n):
        s, e = int(off[k]), int(off[k] + nseg[k])
        X3[k, :int(nseg[k])] = pack.flat_X[s:e]
        M3[k, :int(nseg[k])] = 1.0
    trips = np.asarray(pack.trips)[idx]
    split = np.asarray(pack.split)[idx]
    cuts = pack.cuts[pack.cuts["file"].isin(set(trips.tolist()))].reset_index(drop=True) \
        if len(pack.cuts) else pack.cuts
    info = dict(pack.info)
    info.update({"smoke_subset": True, "smoke_label": label, "n_trips_full": pack.n_trips,
                 "Lmax": Lmax,
                 "trip_by_split": {k: int((split.astype(str) == k).sum())
                                   for k in ("train", "validation", "future")}})
    return dataclasses.replace(
        pack, trips=trips, split=split, nseg=nseg, off=off, X3=X3, M3=M3,
        flat_X=np.ascontiguousarray(pack.flat_X[rows]), flat_trip=np.repeat(
            np.arange(n, dtype=np.int64), nseg),
        raw=np.ascontiguousarray(pack.raw[rows]), y_seg_wh=np.asarray(pack.y_seg_wh)[rows],
        y_full_wh=np.asarray(pack.y_full_wh)[idx], y_trip=np.asarray(pack.y_trip)[idx],
        seg_len_m=np.asarray(pack.seg_len_m)[rows],
        df=pack.df.iloc[rows].reset_index(drop=True), cuts=cuts, info=info)


def pack_summary(pack: TripPack) -> str:
    """一行声明（日志/报告里贴这一行）。"""
    t = pack.info.get("trip_by_split", {})
    return ("%s | %s | 段 %d / 趟 %d (train/val/dev = %s/%s/%s) | 特征 %d | Lmax %d | y 基：%s"
            % (pack.info.get("info_condition"), pack.info.get("target_caliber"),
               pack.n_seg, pack.n_trips, t.get("train"), t.get("validation"),
               t.get("future"), pack.n_feat, pack.info.get("Lmax"), pack.info.get("y_basis")))


# --------------------------------------------------------------------------- 序列批
def seq_batches(pack: TripPack, idx: np.ndarray, *, batch: int, shuffle: bool = False,
                seed: int = 42, device: Any = None, torch_mod: Any = None):
    """按趟产出 ``(i, X, M)` 小批（``X`` ``(b, Lmax, F)``、``M`` ``(b, Lmax)``）。

    ``Lmax`` 由 ``pack`` 决定（右侧零填充 + mask），所以**同一批里的趟可以是任意长度**。
    这是"序列模型必须自己处理变长"的唯一约定：**mask 即契约**，不要按 batch 内最大长度
    重排 —— 那会让同一趟在不同 batch 里得到不同的张量。
    """
    sel = np.array(idx, dtype=np.int64)
    if shuffle:
        g = np.random.default_rng(int(seed))
        g.shuffle(sel)
    for s in range(0, len(sel), int(batch)):
        chunk = sel[s:s + int(batch)]
        Xs = pack.X3[chunk]
        Ms = pack.M3[chunk]
        if torch_mod is not None:
            Xs = torch_mod.as_tensor(Xs, device=device)
            Ms = torch_mod.as_tensor(Ms, device=device)
        yield chunk, Xs, Ms


# --------------------------------------------------------------------------- CLI
def _main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="dataio —— 档位/口径自检与概览")
    ap.add_argument("--policy", default="plan70", help="POLICIES 里的名字")
    ap.add_argument("--all", action="store_true", help="逐个策略打印概览")
    a = ap.parse_args(argv)
    names = sorted(POLICIES) if a.all else [a.policy]
    for nm in names:
        pack = load_pack(nm, verbose=False)
        print("%-11s %s" % (nm, pack_summary(pack)))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
