#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""preprocess —— plan-time 特征装配 + 前缀/切点构造（统一框架第 6 层，只加不改）。

它是什么（以及**不是什么**）
============================
本模块是既有件的**薄封装 + 校验**，**不重新实现任何特征**：

* **70 特征**：逐列名冻结的 SePT-70 清单（``PLAN6+TT17+OSM35+TIME5+TRIPKM+STOP6``）。
  真实取值由既有冻结产物提供 —— 本模块**调用** ``models/G_stoptiming/g2_features_100m_trmap``
  （TT 列）、``models/E_stops/seg_features_100m``（OSM/时段/STOP6/trip_km）、
  ``models/E_stops/seg_stop_pred_100m``（冻结的起停预测），并在每个依赖上断言 SHA256 前 16。
  ★ 与 ``models/SePT/sept.py:build_features()`` **逐列同源**（同一批依赖、同一 fillna=-999.0）；
    本模块**不改写、不搬动**它，只是把"依赖哈希 + 列顺序 + 缺失列"这套契约暴露成可断言的门。
* **前缀/切点**：复用 ``models/L_prefixstyle/`` 的 ``y_rem`` 协议 —— 切点落在累计里程恰好
  跨过 ``φ·L`` 的那个段的**末端**；前缀 = 段 ``0..c``（含 c），后缀 = ``c+1..end``；
  ``y_rem = Σ 后缀 seg_energy_wh``（Wh），``y_pre + y_rem == 全程``（闭合）。
  既有的逐 (趟 × φ) 表在 ``L_prefixstyle/cuts_meta.csv.gz``；本模块提供**增量重算**入口
  （新数据/新 φ）以保证闭合断言，二者逐位对账见 ``verify_prefix``。

纪律
====
* **plan-time 只能吃 plan-time 特征**：``info_condition="plan-time"`` 时任何 realized 列
  （``mean_speed_kph`` / ``speed_std`` / ``speed_p95`` / ``stop_frac``）出现即抛。
* 列序即口径：一律按**列名**取数（``select(names)``），禁止按位置切片。
* 缺失填 ``-999.0``（既有约定）并**逐列计数留痕**（不静默）。
* 只读；不写 ``data_v3``；不读 calibration。

红线：不改 ``seg_loader`` 行为（本模块只 import 它的规范表）。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from paths import Paths

__all__ = [
    "PLAN6", "TT17", "OSM35", "TIME5", "TRIPKM", "STOP6", "OSM_DROP", "FEATURES70",
    "REALIZED_COLS", "MISSING_FILL", "PHI_GRID", "DEP_HASHES",
    "PreprocessError", "sha16", "check_deps", "load_wide", "build_features",
    "feature_matrix", "assert_plan_only", "build_cuts", "verify_prefix", "load_cuts",
    "prefix_features", "cut_for_phi",
]

# --------------------------------------------------------------------------- 特征清单（逐列名冻结）
PLAN6 = ["seg_len_m", "ambient_c_med", "pack_t_med", "soc_start", "cum_frac", "veh2"]
TT17 = ["tt_sl_wmean", "tt_sl_min", "tt_sl_max", "tt_urban_frac", "tt_seg_speed_kph",
        "tt_seg_dur_s", "tt_cong_ratio", "tt_speed_deficit", "tt_dur_s", "tt_len_m",
        "tt_trip_speed_kph", "tt_delay_s", "tt_traffic_len_m",
        "tt_n_turn", "tt_n_turn60", "tt_turn_deg", "tt_turn_deg_per_km"]
TIME5 = ["dep_hour", "dep_dow", "dep_is_weekend", "dep_hour_sin", "dep_hour_cos"]
STOP6 = ["p_stop", "pred_stop_frac", "pred_n_ep", "p_ep", "pred_dwell_s", "pred_stop_frac_plan"]
TRIPKM = ["trip_km"]
OSM_DROP = ("osm_stoplike_500", "osm_crossing_500", "osm_junction_500")
OSM35 = ["osm_covered", "osm_cov_p1",
         "osm_traffic_signals_100", "osm_traffic_signals_250", "osm_traffic_signals_500",
         "osm_traffic_signals_ndist",
         "osm_stop_100", "osm_stop_250", "osm_stop_500", "osm_stop_ndist",
         "osm_give_way_100", "osm_give_way_250", "osm_give_way_500", "osm_give_way_ndist",
         "osm_traffic_calming_100", "osm_traffic_calming_250", "osm_traffic_calming_500",
         "osm_traffic_calming_ndist",
         "osm_level_crossing_100", "osm_level_crossing_250", "osm_level_crossing_500",
         "osm_level_crossing_ndist",
         "osm_mini_roundabout_100", "osm_mini_roundabout_250", "osm_mini_roundabout_500",
         "osm_mini_roundabout_ndist",
         "osm_stoplike_100", "osm_stoplike_250", "osm_stoplike_ndist",
         "osm_crossing_100", "osm_crossing_250", "osm_crossing_ndist",
         "osm_stoplike_fwd200", "osm_sig_fwd200", "osm_sig_bwd100"]

FEATURES70: list[str] = PLAN6 + TT17 + OSM35 + TIME5 + TRIPKM + STOP6
BLOCKS = {"PLAN6": PLAN6, "TT": TT17, "OSM": OSM35, "TIME": TIME5, "trip_km": TRIPKM,
          "STOP6": STOP6}
REALIZED_COLS = ("mean_speed_kph", "speed_std", "speed_p95", "stop_frac")
MISSING_FILL = -999.0
PHI_GRID = (0.0, 0.2, 0.4, 0.6, 0.8)

DEP_HASHES = {
    "data_v3/segments_100m.csv.gz": "298f65ca0dc5c2f5",
    "models/G_stoptiming/g2_features_100m_trmap.csv.gz": "fd6eb1264ff2de43",
    "models/E_stops/seg_features_100m.csv.gz": "c2e5e7c6c03319cc",
    "models/E_stops/seg_stop_pred_100m.csv.gz": "6148ef0f306e687c",
}

if len(FEATURES70) != 70 or len(set(FEATURES70)) != 70:      # 契约自检（import 时即失败）
    raise RuntimeError("70 特征清单不成立：len=%d / unique=%d"
                       % (len(FEATURES70), len(set(FEATURES70))))


class PreprocessError(RuntimeError):
    """预处理层错误（依赖哈希不符 / 缺列 / plan-time 混入 realized / 闭合失败）。"""


# --------------------------------------------------------------------------- 依赖门
def sha16(path: Any) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def check_deps(P: Paths | None = None) -> dict[str, str]:
    """逐个断言冻结依赖的 SHA256 前 16（**读取前**，不符即拒跑）。"""
    p = P or Paths()
    got: dict[str, str] = {}
    for rel, want in DEP_HASHES.items():
        f = p.seg / rel
        if not f.exists():
            raise PreprocessError("缺冻结依赖：%s" % f)
        h = sha16(f)
        if h != want:
            raise PreprocessError("%s 哈希不符：%s ≠ %s（冻结依赖被动过？）" % (rel, h, want))
        got[rel] = h
    return got


# --------------------------------------------------------------------------- 宽表
def load_wide(P: Paths | None = None, *, verbose: bool = True,
              verify: bool = True) -> pd.DataFrame:
    """组出 SePT-70 的 70 列宽表（**只读既有冻结产物**；与 ``sept.build_features`` 同源）。

    返回的 DataFrame 含：``file, seg_idx, split, seg_energy_wh, y_full`` + 70 特征
    + ``vehicle/trip_km`` 等辅助列。行序 = ``(file, seg_idx)`` 升序（= canonical 序）。
    """
    p = P or Paths()
    t0 = __import__("time").time()
    hashes = check_deps(p) if verify else {}
    seg = (pd.read_csv(p.seg_csv).sort_values(["file", "seg_idx"]).reset_index(drop=True))
    if "calibration" in set(seg["split"].astype(str)):
        raise PreprocessError("红线：基表出现 calibration")
    tmap = pd.read_csv(p.G_stoptiming("g2_features_100m_trmap.csv.gz"))
    df = seg.merge(tmap, on=["file", "seg_idx"], how="left", validate="one_to_one")
    if len(df) != len(seg):
        raise PreprocessError("trmap 合并后行数改变：%d → %d" % (len(seg), len(df)))
    feat = (pd.read_csv(p.E_stops("seg_features_100m.csv.gz"))
            .sort_values(["file", "seg_idx"]).reset_index(drop=True))
    if "calibration" in set(feat["split"].astype(str)):
        raise PreprocessError("红线：特征表出现 calibration")
    extra = [c for c in feat.columns if c not in df.columns]
    df = df.merge(feat[["file", "seg_idx"] + extra], on=["file", "seg_idx"], how="left",
                  validate="one_to_one")
    if len(df) != len(seg):
        raise PreprocessError("特征表合并后行数改变：%d → %d" % (len(seg), len(df)))
    # ★ merge 后的 split 列形态取决于"右边有没有重名列"与 pandas 版本：
    #   · 右边无 split 列        ⇒ 只有 `split`；
    #   · 右边有 split 且带后缀  ⇒ `split_x`/`split_y`（旧）或 `split_left`/`split_right`（新）。
    #   三种都接受，但**必须恰好一组**，且两列（若有两列）必须逐位相等 —— 不硬编码后缀。
    cols = [c for c in df.columns if c == "split" or c.startswith("split_")]
    if len(cols) == 1:
        pass                                     # 只有一列 split ⇒ 直接可用
    elif len(cols) == 2:
        a, b = cols
        if not (df[a].astype(str).to_numpy() == df[b].astype(str).to_numpy()).all():
            raise PreprocessError("基表 split 被覆盖（%s ≠ %s）" % (a, b))
        df = df.rename(columns={a: "split"}).drop(columns=[b])
    else:
        raise PreprocessError(
            "merge 后 split 列形态异常（候选 %s）—— 拒绝瞎猜；请检查两表的列名。" % cols)
    sp = pd.read_csv(p.E_stops("seg_stop_pred_100m.csv.gz"), float_precision="round_trip")
    if list(sp.columns) != ["file", "seg_idx", "split"] + STOP6:
        raise PreprocessError("起停预测产物列不符：%s" % list(sp.columns))
    if len(sp) != len(seg):
        raise PreprocessError("起停预测产物行数 %d ≠ 段数 %d" % (len(sp), len(seg)))
    st = sp.set_index(["file", "seg_idx"])
    mi = pd.MultiIndex.from_arrays([df["file"], df["seg_idx"]])
    df = pd.concat([df, pd.DataFrame({c: st[c].reindex(mi).to_numpy(float) for c in STOP6},
                                     index=df.index)], axis=1).copy()
    dup = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
    if dup:
        raise PreprocessError("合并产生重名列：%s" % dup)
    if not df[STOP6].notna().all().all():
        raise PreprocessError("STOP6 存在 NaN（键未全覆盖）")
    miss = [c for c in FEATURES70 if c not in df.columns]
    if miss:
        raise PreprocessError("特征列缺失：%s" % miss)
    if verbose:
        print("[preprocess] 宽表 %s 行 × %d 列（含 70 特征）；依赖哈希 %s；%.1fs"
              % (format(len(df), ","), df.shape[1],
                 "已核对" if hashes else "跳过", __import__("time").time() - t0))
    return df


def assert_plan_only(names: Iterable[str], info_condition: str = "plan-time") -> None:
    """``info_condition="plan-time"`` 时禁止任何 realized 列（**响亮失败**）。"""
    if str(info_condition) != "plan-time":
        return
    bad = [c for c in names if c in REALIZED_COLS]
    if bad:
        raise PreprocessError(
            "info_condition='plan-time' 但特征里含 realized 列 %s —— 这是**口径混排**：\n"
            "  realized 列（%s）在出发前拿不到，混进 plan-time 臂会让它看起来更强。\n"
            "  正解：要么去掉这些列，要么把 info_condition 改成 'realized'。" % (bad,
                                                                     list(REALIZED_COLS)))


def build_features(df: pd.DataFrame | None = None, *, names: Sequence[str] | None = None,
                   info_condition: str = "plan-time", P: Paths | None = None,
                   verbose: bool = False) -> Any:
    """给定段表 → ``FeatureMatrix``（**列名取数**；plan-time 门默认开）。

    返回 ``preprocess.FeatureMatrix``（含 ``.X`` / ``.feature_names`` / ``.units`` /
    ``.column(name)`` / ``.select(names)`` / ``.seg_key()``）。
    """
    if df is None:
        df = load_wide(P, verbose=verbose)
    names = list(names or FEATURES70)
    if not names:
        raise PreprocessError("build_features: names 为空")
    assert_plan_only(names, info_condition)
    miss = [c for c in names if c not in df.columns]
    if miss:
        raise PreprocessError("build_features 缺列 %s；可用列 %s" % (miss, list(df.columns)))
    X = df[names].astype(float)
    n_missing = X.isna().sum()
    if int(n_missing.sum()):
        if verbose:
            print("[preprocess] 缺失填充 -999.0：%s"
                  % n_missing[n_missing > 0].to_dict())
        X = X.fillna(MISSING_FILL)
    return FeatureMatrix(file=df["file"].to_numpy(dtype=object),
                         seg_idx=df["seg_idx"].to_numpy(dtype=np.int64),
                         split=df["split"].to_numpy(dtype=object),
                         X=X.to_numpy(dtype=np.float64),
                         feature_names=names,
                         units={c: None for c in names},
                         info_condition=info_condition,
                         source="preprocess.build_features（依赖 = E_stops/G_stoptiming 冻结产物）",
                         extra={"n_missing_filled": int(n_missing.sum()),
                                "missing_by_col": {k: int(v) for k, v in n_missing.items()
                                                   if v > 0},
                                "y_seg_wh": df["seg_energy_wh"].to_numpy(dtype=np.float64),
                                "y_full_wh": df["y_full"].to_numpy(dtype=np.float64) * 1000.0,
                                "gamma": None})


@dataclass
class FeatureMatrix:
    """段级特征矩阵（**列名即口径**）+ 段表键 + 标签（只读视图）。"""

    file: np.ndarray
    seg_idx: np.ndarray
    split: np.ndarray
    X: np.ndarray
    feature_names: list[str]
    units: dict[str, Any] = field(default_factory=dict)
    info_condition: str = "plan-time"
    target_caliber: str = "y_full"
    source: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.feature_names = list(self.feature_names)
        self.X = np.asarray(self.X, dtype=np.float64)
        if self.X.ndim != 2 or self.X.shape[1] != len(self.feature_names):
            raise PreprocessError("X 形状 %s 与列名数 %d 不符" % (self.X.shape,
                                                           len(self.feature_names)))
        if len(set(self.feature_names)) != len(self.feature_names):
            raise PreprocessError("feature_names 有重复")
        if not np.isfinite(self.X).all():
            raise PreprocessError("X 含非有限值（缺失应已填 %g）" % MISSING_FILL)

    # ---------------------------------------------------------------- 取数
    def column(self, name: str) -> np.ndarray:
        try:
            return self.X[:, self.feature_names.index(name)]
        except ValueError:
            raise PreprocessError("无列 %r；可用：%s" % (name, self.feature_names[:12])) from None

    def select(self, names: Sequence[str]) -> np.ndarray:
        idx = []
        for nm in names:
            if nm not in self.feature_names:
                raise PreprocessError("select 缺列 %r（**按列名取数**，不允许按位置）" % nm)
            idx.append(self.feature_names.index(nm))
        return self.X[:, idx]

    def seg_key(self) -> pd.DataFrame:
        return pd.DataFrame({"file": self.file, "seg_idx": self.seg_idx})

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def trips(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(files, trip_of_row, split_trip)`` —— 段 → 趟的唯一换算（不手写算术）。"""
        fil = self.file
        chg = np.flatnonzero(np.r_[True, fil[1:] != fil[:-1]])
        return fil[chg], np.repeat(np.arange(len(chg)), np.diff(np.r_[chg, len(fil)])), \
            self.split[chg]


# --------------------------------------------------------------------------- 切点 / y_rem
def build_cuts(df: pd.DataFrame | None = None, *, phi: Sequence[float] = PHI_GRID,
               P: Paths | None = None, verbose: bool = False) -> pd.DataFrame:
    """按 ``L_prefixstyle`` 的协议**增量重算**逐 (趟 × φ) 切点与 ``y_rem``。

    切点：沿累计里程 ``cum_end_m`` 恰好跨过 ``φ·L`` 的那个段的**末端**（``cut_idx``）；
    前缀 = 段 ``0..cut_idx``（含），后缀 = ``cut_idx+1..end``。
    返回列：``file, split, phi, n_seg, cut_idx, n_pre, n_suf, L_m, cut_m, pre_km, suf_km,
    y_pre_wh, y_rem_wh, y_full_wh``，并断言 ``y_pre + y_rem == 全程``。
    """
    if df is None:
        df = load_wide(P, verbose=verbose)
    rows = []
    for f, g in df.groupby("file", sort=False):
        g = g.sort_values("seg_idx")
        cum = g["cum_end_m"].to_numpy(dtype=np.float64)
        e = g["seg_energy_wh"].to_numpy(dtype=np.float64)
        L = float(cum[-1])
        spl = str(g["split"].iloc[0])
        n = len(g)
        for ph in phi:
            target = float(ph) * L
            c = int(np.searchsorted(cum, target, side="left")) if target > 0 else -1
            c = min(c, n - 1)
            n_pre = c + 1
            y_pre = float(e[:n_pre].sum())
            y_rem = float(e[n_pre:].sum())
            rows.append({"file": f, "split": spl, "phi": float(ph), "n_seg": n,
                         "cut_idx": int(c), "n_pre": int(n_pre), "n_suf": int(n - n_pre),
                         "L_m": L, "cut_m": float(cum[c]) if c >= 0 else 0.0,
                         "pre_km": (float(cum[c]) / 1000.0) if c >= 0 else 0.0,
                         "suf_km": (L - float(cum[c])) / 1000.0 if c >= 0 else L / 1000.0,
                         "y_pre_wh": y_pre, "y_rem_wh": y_rem, "y_full_wh": y_pre + y_rem})
    cuts = pd.DataFrame(rows)
    err = float(np.abs(cuts["y_pre_wh"] + cuts["y_rem_wh"] - cuts["y_full_wh"]).max())
    if err > 1e-6:
        raise PreprocessError("能量闭合失败：max|y_pre+y_rem−y_full| = %.3e Wh" % err)
    # ★ 短趟 + 大 φ ⇒ 切点落在最后一段的末端，后缀为空。
    #   既有协议（``L_prefixstyle``）的做法是**把这趟的该 φ 排除**（P0_build.json：
    #   n_cut_rows=5590 / n_trips_with_cuts=1118 < 1147），而不是把整张表判死。
    #   本函数照做：标 ``cut_ok=False`` 并**数清楚**，由调用方决定怎么用。
    cuts["cut_ok"] = cuts["n_suf"] >= 1
    n_bad = int((~cuts["cut_ok"]).sum())
    if n_bad and verbose:
        print("[preprocess] %d 个 (趟 × φ) 组合后缀为空（短趟 + 大 φ）⇒ cut_ok=False，"
              "例：%s" % (n_bad, cuts.loc[~cuts["cut_ok"], "file"].head(3).tolist()))
    if verbose:
        print("[preprocess] 切点表 %d 行（%d 趟 × %d phi；可用 %d 行）；闭合 max|Δ| = %.2e Wh"
              % (len(cuts), cuts["file"].nunique(), len(phi), int(cuts["cut_ok"].sum()), err))
    return cuts


def load_cuts(P: Paths | None = None) -> pd.DataFrame:
    """读既有 ``L_prefixstyle/cuts_meta.csv.gz``（**只读**；用于与增量重算逐位对账）。"""
    p = (P or Paths()).L_prefixstyle("cuts_meta.csv.gz")
    if not p.exists():
        raise PreprocessError("缺既有切点表：%s" % p)
    return pd.read_csv(p)


def verify_prefix(P: Paths | None = None, *, phi: Sequence[float] = PHI_GRID,
                  tol: float = 1e-6) -> dict[str, Any]:
    """**增量重算 vs 既有表**逐位对账（只读既有表；不改它）。

    比对列：``n_pre / n_suf / cut_idx / y_pre_wh / y_rem_wh``；键 = ``(file, phi)``。
    """
    p = P or Paths()
    mine = build_cuts(P=p, phi=phi)
    mine = mine[mine["cut_ok"]]                       # 只比"两侧都有切点"的行
    ref = load_cuts(p)
    key = ["file", "phi"]
    ref = ref[ref["phi"].isin([float(x) for x in phi])]
    a = ref.set_index(key).sort_index()
    b = mine.set_index(key).sort_index()
    common = a.index.intersection(b.index)
    out: dict[str, Any] = {"n_ref": int(len(a)), "n_mine": int(len(b)),
                           "n_common": int(len(common))}
    worst = {}
    for c in ("n_pre", "n_suf", "cut_idx", "y_pre_wh", "y_rem_wh"):
        if c not in a.columns:
            continue
        d = np.abs(a.loc[common, c].to_numpy(float) - b.loc[common, c].to_numpy(float))
        worst[c] = float(np.nanmax(d)) if d.size else float("nan")
    out["max_abs_diff"] = worst
    out["verdict"] = "PASS" if all(v <= tol for v in worst.values() if np.isfinite(v)) else "FAIL"
    out["tol"] = float(tol)
    return out


def cut_for_phi(cuts: pd.DataFrame, phi: float) -> pd.DataFrame:
    """取某个 φ 的切点切片（**按值取，不按顺序**）。"""
    m = np.isclose(cuts["phi"].to_numpy(float), float(phi))
    if not bool(m.any()):
        raise PreprocessError("切点表里没有 φ=%r（可用 %s）"
                              % (phi, sorted(set(cuts['phi'].tolist()))[:8]))
    return cuts[m].reset_index(drop=True)


def prefix_features(fm: FeatureMatrix, cuts_phi: pd.DataFrame) -> dict[str, Any]:
    """按某个 φ 的切点切出**前缀特征**（严格因果：只用 ``0..c`` 的段）。

    返回 ``{"idx_pre": {file: 前缀行位置}, "pre_len": {file: n_pre},
    "y_pre_wh": (n_trips,), "y_rem_wh": (n_trips,), "trips": files, "split": trip_split}``。
    ★ 因果纪律：任何用于预测 ``y_rem`` 的特征都**只能**来自前缀段；本函数只给出**行位置**，
    聚合由调用方做（避免在这里偷偷用了后缀信息）。
    """
    files, trip_of_row, trip_split = fm.trips()
    pos = {f: i for i, f in enumerate(files)}
    c = cuts_phi.set_index("file")
    miss = [f for f in files if f not in c.index]
    if miss:
        raise PreprocessError("切点表缺 %d 趟（例：%s）—— 键必须覆盖全部趟"
                              % (len(miss), miss[:3]))
    n_pre = c.loc[files, "n_pre"].to_numpy(np.int64)
    idx_pre: dict[str, np.ndarray] = {}
    off = 0
    sizes = np.bincount(trip_of_row, minlength=len(files))
    for i, f in enumerate(files):
        idx_pre[str(f)] = np.arange(off, off + int(n_pre[i]), dtype=np.int64)
        off += int(sizes[i])
    return {"idx_pre": idx_pre, "pre_len": n_pre, "trips": files, "split": trip_split,
            "trip_of_row": trip_of_row,
            "y_pre_wh": c.loc[files, "y_pre_wh"].to_numpy(float),
            "y_rem_wh": c.loc[files, "y_rem_wh"].to_numpy(float),
            "y_full_wh": c.loc[files, "y_full_wh"].to_numpy(float),
            "phi": float(cuts_phi["phi"].iloc[0]),
            "info_condition": fm.info_condition,
            "target_caliber": "y_rem@%.6g" % float(cuts_phi["phi"].iloc[0])}
