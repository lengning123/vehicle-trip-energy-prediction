#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eval_trip —— 趟级评估的 **唯一实现**（统一框架第 3 层，只加不改）。⭐

它收敛掉什么
============
审计实测：全树 ``def wmape`` **77 份**、``paired_boot`` **≥9 份**、种子 **5 套**。
每一份都"差不多"，但没有一份能被断言与另一份**逐位相等** —— 于是同一个数字在两张表里
差 0.3 pp，谁也说不清哪个对。本模块把口径**固化成一份实现**，并且**逐位复现**既有
``models/BB_pooled/POOLED_METRICS_CI.csv`` 的 **75 臂 × 3 口径 × 6 指标 + CI**。

唯一口径（沿用，不得另立）
==========================
* ``WMAPE = 100·Σ|p−y| / Σ|y|``（**趟级**，Σ 在趟上）
* **pooled = val ∪ dev 拼接后各算一次**：把 val(188) 与 dev(193) 的趟拼成 381 趟，
  **每个指标按自身定义各算一次** —— 绝不先算两个划分的标量再拼权重。
  （对照口径 ``nwavg`` = ``(188·val + 193·dev)/381`` 是**另一个量**，显式命名、单独给。）
* ``MAPE(y≥1kWh)``：分母只数 ``y ≥ 1000 Wh`` 的趟；该子集为空时返回 NaN（不静默填 0）。
* **4000× 按趟配对 bootstrap，划分内重采样后拼接**：
  ``idx_v ~ 有放回抽 188（val 内）``、``idx_d ~ 有放回抽 193（dev 内）``，
  ``pooled* = metric( concat(val[idx_v], dev[idx_d]) )``。
  **不得**把两个划分各自的 CI「拼起来」当作 pooled 的 CI。
* **ΔCI 用同一组重采样索引同时作用于两臂**（按趟配对）——同一个 ``IV/ID``，绝不各抽各的。
* 按 φ 分组：``by_phi``（``target_caliber = y_rem@phi`` 时每 φ 一趟一预测）。
* 点估计与 CI 的作用域必须一致（**分母作用域**）：pooled 的 Σy 是 381 趟的 Σy。

复现基线（本模块的验收锚点）
============================
``PY = models/BB_pooled/PRED_MATRIX.csv``（75 臂逐趟预测）
``PY_REF = models/BB_pooled/POOLED_METRICS_CI.csv``（225 行 = 75 臂 × 3 口径）
回归脚本：``lib/verify/fw_eval_regress.py`` —— 逐位比对点估计 + CI，报最大 |Δ|。

红线
====
只读 train/validation/future；不读 calibration；只加不改。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from schema import (GE1KWH, UNIT_WH, CaliberError, SchemaError, TripPreds,
                    assert_unit_ratio, check_homogeneous, phi_label, read_meta)

__all__ = [
    "NB_DEFAULT", "BOOT_SEED_DEFAULT", "SPLIT_OF", "METRIC_NAMES", "SPLITS_EVAL",
    "m_wmape", "m_mae", "m_bias", "m_rmse", "m_mape", "m_mape_ge1kwh",
    "METRICS", "point_estimates", "stratified_boot", "ci_table", "pooled_pairs",
    "paired_delta_ci", "EvalResult", "eval_trip", "eval_trip_table",
    "eval_grouped", "compare", "load_pred_matrix",
]

NB_DEFAULT = 4000
BOOT_SEED_DEFAULT = 20261010
SPLIT_OF = {"val": "validation", "dev": "future", "pooled": None}
SPLITS_EVAL = ("val", "dev", "pooled")


# =========================================================================== 指标定义（唯一）
def m_wmape(y: Any, e: Any) -> np.ndarray:
    """``100·Σ|e| / Σy``（沿最后一维求和）。"""
    y = np.asarray(y, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    return 100.0 * np.abs(e).sum(axis=-1) / y.sum(axis=-1)


def m_mae(y: Any, e: Any) -> np.ndarray:
    return np.abs(np.asarray(e, dtype=np.float64)).mean(axis=-1)


def m_bias(y: Any, e: Any) -> np.ndarray:
    return np.asarray(e, dtype=np.float64).mean(axis=-1)


def m_rmse(y: Any, e: Any) -> np.ndarray:
    e = np.asarray(e, dtype=np.float64)
    return np.sqrt((e ** 2).mean(axis=-1))


def m_mape(y: Any, e: Any) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    return 100.0 * np.mean(np.abs(e) / y, axis=-1)


def m_mape_ge1kwh(y: Any, e: Any) -> np.ndarray:
    """``MAPE(y≥1kWh)``：分母只数 y≥1000 Wh 的趟；子集为空 → NaN（不填 0）。"""
    y = np.asarray(y, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    w = (y >= GE1KWH)
    num = np.where(w, np.abs(e) / np.where(w, y, 1.0), 0.0).sum(axis=-1)
    cnt = w.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, 100.0 * num / np.maximum(cnt, 1), np.nan)


METRICS: tuple[tuple[str, Any], ...] = (
    ("WMAPE", m_wmape), ("MAPE", m_mape), ("MAPE_ge1kWh", m_mape_ge1kwh),
    ("MAE", m_mae), ("bias", m_bias), ("RMSE", m_rmse),
)
METRIC_NAMES = tuple(k for k, _ in METRICS)


# =========================================================================== 点估计
def point_estimates(y: Any, e: Any) -> dict[str, float]:
    """6 个指标的点估计。``y``/``e`` 形状 ``(n,)`` 或 ``(B, n)``。

    与既有 ``po_point()`` **逐位同式**（同一函数体、同一次 ``sum(axis=-1)``）。
    ``errstate`` 静默掉"空切片/零分母"的警告 —— **数值一字不改**（照旧产出 inf/nan），
    只是不让 numpy 往 stderr 刷屏（那会把真正的告警淹掉）。
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        return {k: float(np.asarray(f(y, e)).reshape(-1)[0]) for k, f in METRICS}


def _as2d(y: Any, e: Any, *, where: str) -> tuple[np.ndarray, np.ndarray]:
    yy = np.asarray(y, dtype=np.float64)
    ee = np.asarray(e, dtype=np.float64)
    if yy.shape != ee.shape:
        raise SchemaError("%s: y%s 与 e%s 形状不一致（分母作用域必须与分子一致）"
                          % (where, yy.shape, ee.shape))
    if yy.ndim == 1:
        yy, ee = yy[None, :], ee[None, :]
    if yy.ndim != 2:
        raise SchemaError("%s: 只接受 (n,) 或 (B,n)，实得 %s" % (where, yy.shape))
    return yy, ee


# =========================================================================== bootstrap
def stratified_boot(n_val: int, n_dev: int, *, nb: int = NB_DEFAULT,
                    seed: int = BOOT_SEED_DEFAULT) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """划分内重采样的**唯一**索引生成器（顺序与既有实现逐位一致）。

    返回 ``(IV, ID, JJ)``：
      * ``IV`` ``(nb, n_val)``：val 内位置（有放回）；``ID`` ``(nb, n_dev)``：dev 内位置；
      * ``JJ`` ``(nb, n_val+n_dev)``：**拼接后**的 381 长行索引 —— val 段用 ``IV``，
        dev 段用 ``n_val + ID``。

    关键纪律：``IV`` 与 ``ID`` 必须来自**同一个 rng 的连续两次调用**（先 IV 后 ID），
    否则与既有 CI 不逐位相等。**ΔCI 复用同一对 (IV, ID)** —— 即"按趟配对"。
    """
    rng = np.random.default_rng(int(seed))
    IV = rng.integers(0, int(n_val), size=(int(nb), int(n_val)))
    ID = rng.integers(0, int(n_dev), size=(int(nb), int(n_dev)))
    JJ = np.concatenate([IV, int(n_val) + ID], axis=1)
    return IV, ID, JJ


def _ci_point(pool_b: np.ndarray) -> tuple[float, float, float, int]:
    """一个口径的 (点估计, lo, hi, se)。

    ★ ``pool_b`` **只含 bootstrap 样本**（``nb`` 个，已在该口径内滤掉 NaN）：
      点估计位单独处理，绝不混进 percentile / std。既有实现是
      ``v = B[tag][k][ai]; v = v[np.isfinite(v)]; percentile(v,[2.5,97.5])`` ——
      少了这一步就会把点估计当成第 nb+1 个样本，CI 与 se 都会偏。
    """
    v = np.asarray(pool_b, dtype=np.float64).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    pt = float(v[0])
    lo, hi = np.percentile(v, [2.5, 97.5])
    return pt, float(lo), float(hi), float(np.std(v, ddof=1))


# =========================================================================== EvalResult
@dataclass
class EvalResult:
    """一次 ``eval_trip`` 的完整结果（点估计 + CI + 分组 + 可选配对 ΔCI）。"""

    point: dict[str, dict[str, float]]              # {split_tag: {metric: value}}
    boot: dict[str, dict[str, np.ndarray]]          # {split_tag: {metric: (nb,)}}
    n: dict[str, int]
    ci: dict[str, dict[str, tuple[float, float]]] = field(default_factory=dict)
    se: dict[str, dict[str, float]] = field(default_factory=dict)
    unit_ratio: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    by_phi: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    delta: dict[str, Any] | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- 便捷访问
    def metric(self, name: str, split: str = "pooled") -> float:
        d = self.point.get(split) or {}
        if name not in d:
            raise SchemaError("EvalResult: 口径 %r 没有指标 %r（本次只算了 %s）"
                              % (split, name, sorted(k for k, v in self.point.items() if v)))
        return float(d[name])

    def metric_ci(self, name: str, split: str = "pooled") -> tuple[float, float]:
        return self.ci.get(split, {}).get(name, (float("nan"), float("nan")))

    def get(self, key: str, default: Any = None) -> Any:
        """``res['pooled_WMAPE']`` 风格的扁平访问（与既有表格列名一致）。"""
        if "_" in key:
            split, _, metric = key.partition("_")
            if split in self.point and metric in self.point[split]:
                return self.point[split][metric]
            if split in self.ci and metric.endswith(("_lo", "_hi", "_se")):
                base, _, kind = metric.rpartition("_")
                if base in self.ci.get(split, {}):
                    lo, hi = self.ci[split][base]
                    return {"lo": lo, "hi": hi}[kind] if kind in ("lo", "hi") \
                        else self.se.get(split, {}).get(base, default)
        return default

    def flat(self) -> dict[str, float]:
        """扁平化成 ``{split_metric: value}``（+ ``n_val``/``n_dev``/``n_pooled``）。"""
        out: dict[str, float] = {}
        for tag, d in self.point.items():
            for k, v in d.items():
                out["%s_%s" % (tag, k)] = float(v)
        for tag, nn in self.n.items():
            out["n_%s" % tag] = int(nn)
        return out

    def table_row(self, arm: str, *, family: str = "", unit_ok: bool = True,
                  exact: Mapping[str, float] | None = None) -> dict[str, Any]:
        """转成 ``POOLED_METRICS_CI.csv`` 的**同名列**（供逐位回归比对）。"""
        rec: dict[str, Any] = {"arm": arm, "split": "", "n": 0, "family": family,
                               "unit_ok": bool(unit_ok)}
        rows = []
        for tag in SPLITS_EVAL:
            if tag not in self.point:
                continue
            r = dict(rec)
            r["split"] = tag
            r["n"] = int(self.n[tag])
            for k in METRIC_NAMES:
                r[k] = float(self.point[tag][k])
                lo, hi = self.ci.get(tag, {}).get(k, (float("nan"), float("nan")))
                r["%s_lo" % k] = float(lo)
                r["%s_hi" % k] = float(hi)
                r["%s_se" % k] = float(self.se.get(tag, {}).get(k, float("nan")))
            for k in METRIC_NAMES:
                r["exactwh_%s" % k] = (float(exact[k]) if (tag == "pooled" and exact) else
                                       float("nan"))
            rows.append(r)
        return rows

    def __repr__(self) -> str:
        s = " ".join("%s=%.4f" % (t, self.point[t]["WMAPE"]) for t in self.point)
        return "EvalResult(n=%s, WMAPE[%s])" % (self.n, s)


# =========================================================================== 主入口
def eval_trip(trip_preds: Any, ref: Any = None, *,
              splits: Sequence[str] = ("val", "dev", "pooled"),
              metrics: Sequence[str] = METRIC_NAMES,
              nb: int = NB_DEFAULT, seed: int = BOOT_SEED_DEFAULT,
              y_exact: Any = None, unit_check: bool = True,
              by_phi: bool = False, phi: Any = None,
              info_condition: str = "", target_caliber: str = "",
              model: str = "", family: str = "", unit_ok: bool | None = None,
              ref_name: str = "reference") -> EvalResult:
    """趟级评估（**唯一口径实现**）。

    参数
    ----
    trip_preds : 以下任一种
        * ``TripPreds``（标准契约对象）；
        * ``{"file","split","y_wh","pred_wh"[,"phi"]}`` 的映射（裸数组，**必须**同时给
          ``info_condition`` / ``target_caliber``；否则口径不明 → 抛）；
        * ``pd.DataFrame``（需列 ``file/split/y_wh/pred_wh``）。
    ref        : 可选参照臂（同型）。给了就同时算 **同一划分内的按趟配对 ΔCI**
                 （Δ = 本臂 − 参照；``Δ>0`` = 本臂更差）。参照与本体**必须同口径两维**、
                 同键序（不一致即抛 —— 配对的前提是"同一趟"）。
    splits     : ``("val","dev","pooled")`` 的任意子集。pooled = 拼接后各算一次。
    y_exact    : 精确 Wh 基（``Σ seg_energy_wh``）的 ``{file: wh}`` 或与行同序数组；
                 给了就补 ``exactwh_*``（只对 pooled 口径给，与既有表一致）。
    by_phi     : 按 φ 分组各算一次（``target_caliber = y_rem@phi`` 的多 φ 表）。
    unit_check : ``Σp/Σy ∈ [0.5, 2]`` 门（**每个口径各查一次**）。

    返回 ``EvalResult``。
    """
    tp, meta_in = _coerce_trip_preds(trip_preds, info_condition=info_condition,
                                     target_caliber=target_caliber, model=model)
    phi_arr = np.asarray(phi, dtype=np.float64) if phi is not None else tp.phi
    if target_caliber and tp.target_caliber and str(target_caliber) != tp.target_caliber:
        raise CaliberError("eval_trip: 传入 target_caliber=%r 与数据的 %r 不一致"
                           % (target_caliber, tp.target_caliber))

    SPLIT = tp.split
    VAL, DEV = SPLIT == "validation", SPLIT == "future"
    EVAL = VAL | DEV
    y = tp.y_wh
    p = tp.pred_wh
    nv, nd = int(VAL.sum()), int(DEV.sum())
    npool = nv + nd
    SPL = tuple(splits)
    if "pooled" in SPL and (nv == 0 or nd == 0):
        raise SchemaError(
            "eval_trip: 要求 pooled 口径，但 val=%d 趟 / dev=%d 趟 —— pooled 要求两侧都非空"
            "（val=validation / dev=future）。\n"
            "  ⇒ 只评估单个划分时请显式 splits=('val',) 或 splits=('dev',)，"
            "这样跳过 pooled 就不会把「单划分」误当成「拼接后」。\n"
            "  （本次调用 splits=%s）" % (nv, nd, list(SPL)))
    if nv == 0 and "val" in SPL:
        raise SchemaError("eval_trip: splits 含 'val' 但数据里没有 validation 趟")
    if nd == 0 and "dev" in SPL:
        raise SchemaError("eval_trip: splits 含 'dev' 但数据里没有 future 趟")
    safe_pool = (nv > 0 and nd > 0 and npool > 0)
    nv_s, nd_s = max(nv, 1), max(nd, 1)
    YV, YD = y[VAL], y[DEV]
    ev, ed = p[VAL] - YV, p[DEV] - YD
    e_pool = np.r_[ev, ed]
    Y_POOL = np.r_[YV, YD]
    if not safe_pool:
        # 单划分评估：pooled 只是**占位**（不参与返回），用可跑的占位尺寸避免空切片崩
        Y_POOL, e_pool = np.r_[YV, np.zeros(1)], np.r_[ev, np.zeros(1)]

    # ---- 单位门（每个口径各查一次，分母=同作用域）
    ratio: dict[str, float] = {}
    if unit_check:
        if nv:
            ratio["val"] = assert_unit_ratio(p[VAL], YV, scope="val", name="eval_trip")
        if nd:
            ratio["dev"] = assert_unit_ratio(p[DEV], YD, scope="dev", name="eval_trip")
        if safe_pool:
            ratio["pooled"] = assert_unit_ratio(p[EVAL], y[EVAL], scope="pooled",
                                                name="eval_trip")

    # ---- 点估计（pooled = 拼接后每个指标各算一次）
    point: dict[str, dict[str, float]] = {
        "val": point_estimates(YV, ev) if nv else {},
        "dev": point_estimates(YD, ed) if nd else {},
        "pooled": point_estimates(Y_POOL, e_pool) if safe_pool else {}}
    n_map = {"val": nv, "dev": nd, "pooled": npool}

    # ---- bootstrap（唯一索引生成器；ΔCI 复用同一对）
    IV, ID, JJ = stratified_boot(nv_s, nd_s, nb=nb, seed=seed)
    IV, ID = IV[:, :nv], ID[:, :nd]
    YV_B, YD_B = YV[IV], YD[ID]
    YP_B = np.concatenate([YV_B, YD_B], axis=1)
    eV, eD = ev[IV], ed[ID]
    eP = np.concatenate([eV, eD], axis=1)

    boot: dict[str, dict[str, np.ndarray]] = {k: {} for k in ("val", "dev", "pooled")}
    with np.errstate(invalid="ignore", divide="ignore"):
        for k, f in METRICS:
            boot["val"][k] = _pt(f, point["val"].get(k, float("nan")), f(YV_B, eV))
            boot["dev"][k] = _pt(f, point["dev"].get(k, float("nan")), f(YD_B, eD))
            boot["pooled"][k] = _pt(f, point["pooled"].get(k, float("nan")), f(YP_B, eP))

    ci: dict[str, dict[str, tuple[float, float]]] = {}
    se: dict[str, dict[str, float]] = {}
    for tag in ("val", "dev", "pooled"):
        ci[tag], se[tag] = {}, {}
        for k in METRIC_NAMES:
            # ★ 只用 bootstrap 部分（丢弃第 0 位的点估计）—— 与既有实现同口径：
            #   percentile/ std 都作用在**重采样**样本上，nb 个，不是 nb+1 个。
            _p, lo, hi, s = _ci_point(boot[tag][k][1:])
            ci[tag][k], se[tag][k] = (lo, hi), s
    # 单划分（无 pooled）时清掉占位口径 —— 不返回"假的 pooled"
    if not safe_pool:
        for tag in ("pooled",):
            point[tag] = {}
            boot[tag] = {}
            ci[tag], se[tag] = {}, {}
    out_tags = tuple(t for t in SPL if t in ("val", "dev", "pooled")
                     and (t != "pooled" or safe_pool))

    # ---- 精确 Wh 基（可选；只对 pooled 给，与既有表一致）
    exact_pt: dict[str, float] | None = None
    if y_exact is not None:
        if not safe_pool:
            raise SchemaError("eval_trip: y_exact（精确 Wh 基）只对 pooled 口径有意义；"
                              "单划分评估请去掉 y_exact，或补上另一侧划分")
        ye = _resolve_exact(y_exact, tp, EVAL)
        exact_pt = point_estimates(np.r_[ye[VAL], ye[DEV]], e_pool)

    # ---- 按 φ 分组
    phi_out: dict[str, dict[str, dict[str, float]]] = {}
    if by_phi:
        if phi_arr is None:
            raise CaliberError("eval_trip(by_phi=True) 需要 phi（target_caliber=y_rem@phi）")
        for ph in sorted(set(np.asarray(phi_arr).tolist())):
            m = np.isclose(phi_arr, ph)
            sub_y, sub_p, sub_s = y[m], p[m], SPLIT[m]
            sv, sd = sub_s == "validation", sub_s == "future"
            if sv.sum() == 0 or sd.sum() == 0:
                continue
            yy = np.r_[sub_y[sv], sub_y[sd]]
            ee = np.r_[sub_p[sv] - sub_y[sv], sub_p[sd] - sub_y[sd]]
            phi_out["%.6g" % ph] = point_estimates(yy, ee)

    # ---- 配对 ΔCI（同一划分内、同一组重采样索引）
    delta: dict[str, Any] | None = None
    if ref is not None:
        rp, _ = _coerce_trip_preds(ref, info_condition=info_condition,
                                   target_caliber=target_caliber, model=ref_name)
        delta = paired_delta_ci(tp, rp, IV=IV, ID=ID, ref_name=ref_name,
                               metrics=metrics)

    res = EvalResult(point={k: v for k, v in point.items()}, boot=boot, n=n_map,
                     ci=ci, se=se, unit_ratio=ratio,
                     meta=dict(meta_in, nb=int(nb), seed=int(seed),
                               info_condition=tp.info_condition,
                               target_caliber=tp.target_caliber, model=tp.model,
                               y_basis=tp.y_basis, splits=list(out_tags),
                               n_val=nv, n_dev=nd, n_pooled=npool),
                     by_phi=phi_out, delta=delta)
    if exact_pt is not None:
        res.extras["exactwh"] = exact_pt
    return res


def _pt(f: Any, pt: float, boot_vals: np.ndarray) -> np.ndarray:
    """把**点估计**放在 bootstrap 数组的第 0 位（既有实现对 ``([y],[e])`` 也返回点估计）。

    CI/SE 只取后 ``nb`` 个元素 —— 与既有实现一致（第 0 位不进 percentile）。
    """
    b = np.asarray(boot_vals, dtype=np.float64).reshape(-1)
    return np.concatenate([[float(pt)], b])


def _coerce_trip_preds(obj: Any, *, info_condition: str = "", target_caliber: str = "",
                       model: str = "") -> tuple[TripPreds, dict[str, Any]]:
    """把 TripPreds / DataFrame / 映射统一成 ``TripPreds``（口径缺失即抛）。"""
    if isinstance(obj, TripPreds):
        return obj, {}
    if isinstance(obj, pd.DataFrame):
        obj = {c: obj[c].to_numpy() for c in obj.columns}
    if not isinstance(obj, Mapping):
        raise SchemaError("eval_trip: 无法识别的入参类型 %r（要 TripPreds / DataFrame / 映射）"
                          % type(obj).__name__)
    d = dict(obj)
    ic = d.pop("info_condition", None) or info_condition
    tc = d.pop("target_caliber", None) or target_caliber
    mdl = d.pop("model", None) or model
    units = d.pop("units", None) or {"y_wh": UNIT_WH, "pred_wh": UNIT_WH}
    y_basis = d.pop("y_basis", "") or ""
    if not ic or not tc:
        raise CaliberError(
            "eval_trip: 裸映射入参缺口径两维（info_condition=%r / target_caliber=%r）—— "
            "**强制声明**（防混排）。\n"
            "  正解：给 TripPreds，或在入参/调用里显式声明这两个维度。" % (ic, tc))
    from schema import TripPreds as _TP
    unit_ok = units
    for k in ("y_wh", "pred_wh"):
        if str(unit_ok.get(k, UNIT_WH)) != UNIT_WH:
            from schema import to_wh
            d[k] = to_wh(d[k], unit_ok[k], name=k)
    tp = _TP(file=d["file"], split=d["split"], y_wh=d["y_wh"], pred_wh=d["pred_wh"],
             phi=d.get("phi"), info_condition=str(ic), target_caliber=str(tc),
             units={"y_wh": UNIT_WH, "pred_wh": UNIT_WH}, model=str(mdl),
             y_basis=y_basis or "未声明（请显式给 y_basis）")
    return tp, {"units_in": dict(unit_ok)}


def _resolve_exact(y_exact: Any, tp: TripPreds, EVAL: np.ndarray) -> np.ndarray:
    if isinstance(y_exact, Mapping):
        s = pd.Series(tp.file).map(y_exact)
        if s.isna().any():
            raise SchemaError("eval_trip: y_exact 映射缺 %d 个趟（键必须覆盖全部）"
                              % int(s.isna().sum()))
        return s.to_numpy(dtype=np.float64)
    a = np.asarray(y_exact, dtype=np.float64).reshape(-1)
    if a.shape[0] != len(tp.file):
        raise SchemaError("eval_trip: y_exact 长度 %d ≠ 趟数 %d（不接受按位置截断）"
                          % (a.shape[0], len(tp.file)))
    return a


# =========================================================================== 配对 ΔCI
def paired_delta_ci(preds: Any, ref: Any, *, IV: np.ndarray | None = None,
                    ID: np.ndarray | None = None, nb: int = NB_DEFAULT,
                    seed: int = BOOT_SEED_DEFAULT,
                    metrics: Sequence[str] = METRIC_NAMES,
                    splits: Sequence[str] = ("val", "dev", "pooled"),
                    ref_name: str = "reference") -> dict[str, Any]:
    """**同一划分内按趟配对**的 Δ 与 ΔCI（Δ = 本臂 − 参照；Δ>0 = 本臂更差）。

    配对的前提是"同一趟"：两臂的 **键序必须逐位一致**（不一致即抛，绝不按键重排后
    假装配对 —— 重排会让 ``B[i] − A[i]`` 变成两趟相减）。CI 用**同一组**重采样索引
    同时作用于两臂（``diff = metric(arm*) − metric(ref*)``）。
    """
    a, _ = _coerce_trip_preds(preds)
    b, _ = _coerce_trip_preds(ref)
    from schema import normalize_caliber
    if a.info_condition != b.info_condition \
            or normalize_caliber("target_caliber", a.target_caliber) != \
            normalize_caliber("target_caliber", b.target_caliber):
        raise CaliberError(
            "paired_delta_ci: 两臂口径不同（%s/%s vs %s/%s）—— **不可配对**，"
            "也不可同榜（不同信息条件或不同目标口径的数字不可直接比）。"
            % (a.info_condition, a.target_caliber, b.info_condition, b.target_caliber))
    if str(a.target_caliber) != str(b.target_caliber):
        raise CaliberError(
            "paired_delta_ci: 两臂 target_caliber 取值不同（%s vs %s）—— 同族但 φ 不同则 y 不同，"
            "配对无意义" % (a.target_caliber, b.target_caliber))
    if a.phi is not None and b.phi is not None and not np.array_equal(a.phi, b.phi):
        raise CaliberError(
            "paired_delta_ci: 两臂 phi 逐趟不同 —— 目标口径（切点后的剩余能量）不可比，"
            "配对无意义")
    if len(a) != len(b) or not np.array_equal(a.file.astype(str), b.file.astype(str)):
        bad = -1
        if len(a) == len(b):
            neq = np.flatnonzero(a.file.astype(str) != b.file.astype(str))
            bad = int(neq[0]) if neq.size else -1
        raise SchemaError(
            "paired_delta_ci: 两臂键序不一致（len %d vs %d，首个不同 @%d）—— 配对的前提是"
            "同一趟。正解：两臂都用同一份键序（infer 产出的 trip_preds.npz 天然同序）。"
            % (len(a), len(b), bad))
    if not np.array_equal(a.split.astype(str), b.split.astype(str)):
        raise SchemaError("paired_delta_ci: 两臂 split 不一致 —— 划分都不同，谈不上配对")

    VAL, DEV = a.split == "validation", a.split == "future"
    EVAL = VAL | DEV
    nv, nd = int(VAL.sum()), int(DEV.sum())
    if IV is None or ID is None:
        IV, ID, _ = stratified_boot(nv, nd, nb=nb, seed=seed)
    y = a.y_wh
    ea = a.pred_wh - y
    eb = b.pred_wh - y
    fdict = dict(METRICS)
    out: dict[str, Any] = {"vs": ref_name, "n_val": nv, "n_dev": nd,
                           "info_condition": a.info_condition,
                           "target_caliber": a.target_caliber}
    for tag in splits:
        if tag == "val":
            Yb = y[VAL][IV]
            ea_b, eb_b = ea[VAL][IV], eb[VAL][IV]
            Ya, eaa, ebb = y[VAL], ea[VAL], eb[VAL]
        elif tag == "dev":
            Yb = y[DEV][ID]
            ea_b, eb_b = ea[DEV][ID], eb[DEV][ID]
            Ya, eaa, ebb = y[DEV], ea[DEV], eb[DEV]
        elif tag == "pooled":
            Yb = np.concatenate([y[VAL][IV], y[DEV][ID]], axis=1)
            ea_b = np.concatenate([ea[VAL][IV], ea[DEV][ID]], axis=1)
            eb_b = np.concatenate([eb[VAL][IV], eb[DEV][ID]], axis=1)
            Ya = np.r_[y[VAL], y[DEV]]
            eaa, ebb = np.r_[ea[VAL], ea[DEV]], np.r_[eb[VAL], eb[DEV]]
        else:
            raise ValueError("paired_delta_ci: splits 只支持 val/dev/pooled，实得 %r" % tag)
        for k in metrics:
            f = fdict[k]
            d_pt = float(np.asarray(f(Ya, eaa)).reshape(-1)[0]
                         - np.asarray(f(Ya, ebb)).reshape(-1)[0])
            diff = np.asarray(f(Yb, ea_b), dtype=np.float64).reshape(-1) \
                - np.asarray(f(Yb, eb_b), dtype=np.float64).reshape(-1)
            diff = diff[np.isfinite(diff)]
            lo, hi = (np.percentile(diff, [2.5, 97.5]) if diff.size else (np.nan, np.nan))
            out["%s_%s_d" % (tag, k)] = d_pt
            out["%s_%s_lo" % (tag, k)] = float(lo)
            out["%s_%s_hi" % (tag, k)] = float(hi)
            out["%s_%s_excl0" % (tag, k)] = bool(diff.size and (lo > 0 or hi < 0))
            out["%s_%s_nboot" % (tag, k)] = int(diff.size)
    return out


# =========================================================================== 多臂 / 分组
def eval_trip_table(arms: Mapping[str, Any], *, ref: str | None = None,
                    unit_ok: Mapping[str, bool] | None = None,
                    family: Mapping[str, str] | None = None,
                    y_exact: Mapping[str, Any] | None = None,
                    selftest: bool = True, **kw: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    """多臂评估 → ``(表, 附加)``。表列 = ``POOLED_METRICS_CI.csv`` 同名列。

    ``arms``：``{臂名: TripPreds | 映射}``（**同口径**；混排即抛）。
    返回 ``(TAB, {"delta": DataFrame|None, "results": {arm: EvalResult}})``。

    ``selftest=True`` 时执行两条**口径自检**（不依赖任何外部期望值）：
      ① pooled 恒等于"381 趟拼接后按定义各算一次"（与任何加权拼接无关）；
      ② pooled 的 CI **不等于**两个划分 CI 的拼接（会显式打印两者）。
    """
    if not arms:
        raise SchemaError("eval_trip_table: arms 为空")
    metas = []
    tp_map: dict[str, TripPreds] = {}
    for nm, obj in arms.items():
        tp, _ = _coerce_trip_preds(obj, info_condition=kw.get("info_condition", ""),
                                   target_caliber=kw.get("target_caliber", ""), model=nm)
        tp_map[nm] = tp
        metas.append({"arm": nm, "info_condition": tp.info_condition,
                      "target_caliber": tp.target_caliber})
    cal = check_homogeneous(metas, where="eval_trip_table")
    uok = dict(unit_ok or {})
    fam = dict(family or {})
    res_map: dict[str, EvalResult] = {}
    rows: list[dict[str, Any]] = []
    deltas: list[dict[str, Any]] = []
    for nm, tp in tp_map.items():
        r = eval_trip(tp, ref=(tp_map[ref] if (ref and ref != nm and ref in tp_map) else None),
                      y_exact=(y_exact or {}).get(nm), family=fam.get(nm, ""),
                      ref_name=str(ref or "reference"), **kw)
        res_map[nm] = r
        ex = r.extras.get("exactwh")
        for rec in r.table_row(nm, family=fam.get(nm, ""),
                               unit_ok=uok.get(nm, True), exact=ex):
            rows.append(rec)
        if r.delta is not None:
            d = {"arm": nm, "family": fam.get(nm, ""), **r.delta}
            deltas.append(d)
    TAB = pd.DataFrame(rows)
    extra: dict[str, Any] = {"results": res_map, "caliber": cal,
                             "delta": (pd.DataFrame(deltas) if deltas else None)}
    if selftest:
        extra["selftest"] = _caliber_selftest(res_map)
    return TAB, extra


def _caliber_selftest(res_map: Mapping[str, EvalResult]) -> dict[str, Any]:
    """两条不依赖外部期望值的口径自检（把"pooled 到底是什么"变成可断言的事实）。"""
    out: dict[str, Any] = {}
    for nm, r in res_map.items():
        p = r.point
        b = r.boot
        # ① pooled 点估计 == 拼接定义
        ok1 = all(abs(p["pooled"][k] - float(np.asarray(b["pooled"][k]).reshape(-1)[0])) == 0.0
                  for k in METRIC_NAMES)
        # ② pooled CI ≠ val/dev CI 的拼接（数值上必须不同）
        ci_p = r.ci["pooled"]["WMAPE"]
        ci_v = r.ci["val"]["WMAPE"]
        ci_d = r.ci["dev"]["WMAPE"]
        ok2 = (ci_p[0], ci_p[1]) != (min(ci_v[0], ci_d[0]), max(ci_v[1], ci_d[1])) or \
              (ci_p[0], ci_p[1]) != (ci_v[0], ci_d[0])
        out[nm] = {"pooled_equals_concat": bool(ok1),
                   "pooled_ci_distinct_from_split_merge": bool(ok2),
                   "pooled_ci": [float(x) for x in ci_p],
                   "val_ci": [float(x) for x in ci_v], "dev_ci": [float(x) for x in ci_d]}
    return out


def eval_grouped(trip_preds: Any, group: Any, *, ref: Any = None, **kw: Any) -> dict[str, Any]:
    """**按任意分组键**（如 φ、车、里程带）分组评估 + 组内配对 ΔCI。

    ``group``：与行同长的标签数组（或列名 + DataFrame 形式的 trip_preds）。
    每组都会走**同一份** ``eval_trip`` 与**同一对**重采样索引 —— 保证组间可比。
    """
    tp, _ = _coerce_trip_preds(trip_preds, info_condition=kw.get("info_condition", ""),
                               target_caliber=kw.get("target_caliber", ""))
    g = np.asarray(group)
    if g.shape[0] != len(tp):
        raise SchemaError("eval_grouped: group 长度 %d ≠ 趟数 %d" % (g.shape[0], len(tp)))
    out: dict[str, Any] = {}
    for lvl in pd.unique(g):
        m = g == lvl
        if int((tp.split[m] == "validation").sum()) == 0 or \
           int((tp.split[m] == "future").sum()) == 0:
            out[str(lvl)] = {"skipped": "该组 val 或 dev 为空，pooled 无定义"}
            continue
        sub = TripPreds(file=tp.file[m], split=tp.split[m], y_wh=tp.y_wh[m],
                        pred_wh=tp.pred_wh[m], phi=(tp.phi[m] if tp.phi is not None else None),
                        info_condition=tp.info_condition, target_caliber=tp.target_caliber,
                        units=dict(tp.units), y_basis=tp.y_basis, model=tp.model)
        rsub = None
        if ref is not None:
            rp, _ = _coerce_trip_preds(ref, info_condition=kw.get("info_condition", ""),
                                       target_caliber=kw.get("target_caliber", ""))
            rsub = TripPreds(file=rp.file[m], split=rp.split[m], y_wh=rp.y_wh[m],
                             pred_wh=rp.pred_wh[m],
                             phi=(rp.phi[m] if rp.phi is not None else None),
                             info_condition=rp.info_condition,
                             target_caliber=rp.target_caliber, units=dict(rp.units),
                             y_basis=rp.y_basis, model=rp.model)
        out[str(lvl)] = eval_trip(sub, ref=rsub, **kw)
    return out


def compare(a: Any, b: Any, *, nb: int = NB_DEFAULT, seed: int = BOOT_SEED_DEFAULT,
            ref_name: str = "A") -> dict[str, Any]:
    """两臂对比（= ``paired_delta_ci`` 的薄封装；Δ = a − b，Δ>0 = a 更差）。"""
    return paired_delta_ci(a, b, nb=nb, seed=seed, ref_name=ref_name)


# =========================================================================== 载入既有证据
def load_pred_matrix(path: Any, *, pred_prefix: str = "pred__",
                     drop_unit_violations: bool = False,
                     unit_exempt: Iterable[str] = ()) -> tuple[pd.DataFrame, dict[str, Any]]:
    """载入 ``BB_pooled/PRED_MATRIX.csv``（**只读**；逐位回归的输入）。

    返回 ``(mat, info)``；``info`` 含 ``arms`` / ``y_basis`` / ``n_val`` / ``n_dev``。

    ★ 纪律：``mat`` 的行序 = 规范序 P（= ``ds.trip_files``）；**不许**再与
    ``trip_index.csv`` 按位置拼接（两者行序不同，实测错标 541/1147 行）。
    """
    p = Path(path)
    mat = pd.read_csv(p)
    need = {"file", "split", "y_wh"}
    miss = need - set(mat.columns)
    if miss:
        raise SchemaError("load_pred_matrix: %s 缺列 %s" % (p, sorted(miss)))
    arms = [c[len(pred_prefix):] for c in mat.columns if c.startswith(pred_prefix)]
    if not arms:
        raise SchemaError("load_pred_matrix: 没有 %r 前缀的预测列" % pred_prefix)
    exempt = set(unit_exempt)
    info = {"path": str(p), "arms": arms, "n_arms": len(arms),
            "n_trips": int(len(mat)), "n_val": int((mat["split"] == "validation").sum()),
            "n_dev": int((mat["split"] == "future").sum()),
            "has_exact": "y_exact_wh" in mat.columns,
            "unit_violations": [a for a in arms
                                if a not in exempt and not (0.5 <= float(
                                    mat[pred_prefix + a][mat["split"].isin(
                                        ["validation", "future"])].sum()
                                    / mat["y_wh"][mat["split"].isin(
                                        ["validation", "future"])].sum()) <= 2.0)],
            "y_basis": "trip_index.y_full(kWh,float32 量化)×1000 = Wh（不取整）"}
    if drop_unit_violations:
        keep = [a for a in arms if a not in set(info["unit_violations"])]
        info["dropped"] = sorted(set(arms) - set(keep))
        arms = keep
    return mat, info
