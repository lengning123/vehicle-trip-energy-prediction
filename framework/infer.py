#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""infer —— ``ckpt → 标准 trip_preds.npz``（统一框架第 5 层，只加不改）。

它收敛掉什么
============
审计实测：``preds_io`` 守住了"读得进来"，但**没有任何一处规定"该写出去什么"**。
于是每个模型各写一份自己的预测文件（CSV / npy / npz，键序与单位各异），下游每读一次
就得重新猜一次键序与量纲 —— 这正是"按位置对齐 18.24×"事故的土壤。

本模块把"推理产物"收敛成**一种格式**：

    trip_preds.npz
      file      (object)  趟键（**行序 = 身份**）
      split     (str)     train/validation/future
      y_wh      (float64) 趟级真值 Wh
      pred_wh   (float64) 趟级预测 Wh
      phi       (float64, 可选) 前缀比例（target_caliber=y_rem@phi 时）
      meta      (json str) **口径元数据**：info_condition / target_caliber / phi /
                           units / y_basis / model / n

三条硬纪律
==========
1. **键序随行**：``trip_preds.npz`` 的行序由**推理时的键序**决定，且 meta 里带
   ``key_sha16``（键序的指纹）。下游一律按 ``file`` 对齐；要逐位比对就必须用同一份。
2. **单位显式**：``pred_wh`` / ``y_wh`` 恒为 **Wh**；kWh 输入必须在门口换算（留痕）。
3. **口径显式**：``info_condition`` / ``target_caliber`` **写入时强制声明**（缺省即抛）。
   读回时若 meta 缺这两维 ⇒ 拒读（口径不明 = 不可用）。

红线
====
只读 data_v3 / ckpt；不读 calibration；不覆盖既有产物（只写 ``runs/``）。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from paths import Paths
from schema import (CaliberError, SegPreds, SchemaError, TripPreds, UNIT_WH,
                    assert_unit_ratio, make_seg_preds, make_trip_preds, read_meta)

__all__ = [
    "InferError", "key_sha16", "InferResult", "predict_trips", "save_trip_preds",
    "load_trip_preds", "predict_to_npz", "attach_seg", "ckpt_files", "infer",
]


class InferError(RuntimeError):
    """推理层错误（ckpt 缺 / 口径缺 / 键序不符）。"""


# --------------------------------------------------------------------------- 键序指纹
def key_sha16(files: Sequence[Any]) -> str:
    """键序指纹（**顺序敏感**）：换行序就换指纹 —— 让"是不是同一份键序"变成可比对的字符串。"""
    h = hashlib.sha256()
    for f in files:
        h.update(str(f).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def key_sha16_from_npz(path: Any) -> str:
    """从 ``trip_preds.npz`` 取键序指纹（缺 ``key_sha16`` 就现场重算并在 meta 里注明）。"""
    z = np.load(path, allow_pickle=True)
    m = read_meta(z)
    k = m.get("key_sha16")
    fk = "file_" if "file_" in z.files else "file"
    return str(k) if k else key_sha16(z[fk])


# --------------------------------------------------------------------------- 结果
@dataclass
class InferResult:
    """一次推理的交付：标准对象 + 落盘路径 + 口径元数据。"""

    trip: TripPreds
    npz: Path | None = None
    seg: SegPreds | None = None
    seg_npz: Path | None = None
    unit_ratio: float = float("nan")
    meta: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return ("InferResult(n=%d, %s/%s, Σp/Σy=%.4f, npz=%s)"
                % (len(self.trip), self.trip.info_condition, self.trip.target_caliber,
                   self.unit_ratio, self.npz))


# --------------------------------------------------------------------------- ckpt
def ckpt_files(d: Any, *, prefer: str = "best_val.pt") -> dict[str, Any]:
    """列出 ``ckpt/`` 下的产物（**best 优先，last_1..N 兜底**）。

    纪律：推理**默认只用 best_val.pt**（early stop 选出来的那一个）；用 last 必须显式
    ``prefer="last_1.pt"`` 并留痕 —— 否则"我评估的是哪个 ckpt"就说不清。
    """
    p = Path(d)
    ck = p if p.name == "ckpt" else (p / "ckpt")
    if not ck.is_dir():
        raise InferError("ckpt 目录不存在：%s" % ck)
    best = ck / "best_val.pt"
    lasts = sorted(ck.glob("last_*.pt"))
    chosen = ck / prefer
    if not chosen.exists():
        if best.exists():
            chosen = best
        elif lasts:
            chosen = lasts[0]
        else:
            raise InferError("ckpt 目录里既没有 best_val.pt 也没有 last_*.pt：%s" % ck)
    return {"dir": ck, "best": (best if best.exists() else None), "last": lasts,
            "chosen": chosen, "chosen_name": chosen.name,
            "log_csv": (p / "train_log.csv") if (p / "train_log.csv").exists() else None,
            "manifest": (p / "train_manifest.json") if (p / "train_manifest.json").exists()
            else None}


# --------------------------------------------------------------------------- 推理
def predict_trips(predict_fn: Callable[..., Any], handle: Any, ctx: Any = None, **kw: Any) -> Any:
    """调用 ``predict_fn(handle, ctx)`` 并把结果规整成**逐段 Wh** 或**逐趟 Wh**。

    模型侧只需返回以下任一：
      * ``(n_seg,)`` 逐段 Wh（框架按趟求和 → 趟级 Wh）
      * ``(n_trips,)`` 逐趟 Wh
      * ``{"trip_wh": ..., "seg_wh": ...}``（最明确，推荐）
      * ``(n_trips, L)`` 定长 padding 的逐段 Wh（按 mask 求和 —— 需要 ctx 给 mask）
    """
    out = predict_fn(handle, ctx, **kw) if ctx is not None else predict_fn(handle, **kw)
    return out


def _as_trip_wh(out: Any, *, files: Sequence[Any], y_wh: Any, split: Any,
                trip_of_row: np.ndarray | None, mask: Any = None) -> tuple[np.ndarray, Any]:
    """把模型的任意输出规整成 ``(trip_wh, seg_wh|None)``。"""
    n_tr = len(files)
    if isinstance(out, Mapping):
        trip = out.get("trip_wh")
        seg = out.get("seg_wh")
        if trip is None and seg is None:
            raise InferError("predict_fn 返回的字典里既没有 trip_wh 也没有 seg_wh；实得键 %s"
                             % sorted(out))
        if trip is not None:
            t = np.asarray(trip, dtype=np.float64).reshape(-1)
            if t.shape[0] != n_tr:
                raise InferError("trip_wh 长度 %d ≠ 趟数 %d" % (t.shape[0], n_tr))
            return t, (None if seg is None else np.asarray(seg, dtype=np.float64).reshape(-1))
        return None, np.asarray(seg, dtype=np.float64).reshape(-1)   # type: ignore[return-value]
    a = np.asarray(out, dtype=np.float64)
    if a.ndim == 1 and a.shape[0] == n_tr:
        return a, None
    if a.ndim == 1 and trip_of_row is not None and a.shape[0] == len(trip_of_row):
        return np.bincount(trip_of_row, weights=a, minlength=n_tr), a
    if a.ndim == 2 and trip_of_row is not None and mask is not None:
        m = np.asarray(mask, dtype=np.float64)
        if m.shape != a.shape:
            raise InferError("逐段 (n_trips,L) 输出需要同形 mask：pred%s vs mask%s"
                             % (a.shape, m.shape))
        return (a * m).sum(axis=1), None
    raise InferError(
        "无法把 predict_fn 的输出规整成趟级 Wh：形状 %s（趟数 %d，逐段行数 %s）。\n"
        "  正解：返回 {\"trip_wh\": (n_trips,)} 或 {\"seg_wh\": (n_seg,)} 或 (n_trips,L)+mask。"
        % (a.shape, n_tr, (len(trip_of_row) if trip_of_row is not None else "?")))


def infer(predict_fn: Callable[..., Any], handle: Any, ctx: Any = None, *,
          files: Sequence[Any], split: Sequence[Any], y_wh: Any,
          info_condition: str, target_caliber: str, phi: Any = None,
          model: str = "", trip_of_row: np.ndarray | None = None, mask: Any = None,
          y_basis: str = "", unit_check: bool = True, **kw: Any) -> InferResult:
    """``predict_fn → TripPreds``（含单位门与口径校验）。"""
    out = predict_trips(predict_fn, handle, ctx, **kw)
    trip, seg = _as_trip_wh(out, files=files, y_wh=y_wh, split=split,
                            trip_of_row=trip_of_row, mask=mask)
    if trip is None:
        if trip_of_row is None:
            raise InferError("predict_fn 只给了 seg_wh，但没给 trip_of_row（无法按趟求和）")
        if seg is None or len(seg) != len(trip_of_row):
            raise InferError("seg_wh 长度 %s ≠ 逐段行数 %d"
                             % ((None if seg is None else len(seg)), len(trip_of_row)))
        trip = np.bincount(trip_of_row, weights=seg, minlength=len(files))
    tp = make_trip_preds(files, split, y_wh, trip, info_condition=info_condition,
                         target_caliber=target_caliber, phi=phi, model=model,
                         y_basis=(y_basis or "未声明"), validate=bool(unit_check))
    r = assert_unit_ratio(tp.pred_wh, tp.y_wh, scope="all", name=model or "infer") \
        if unit_check else float("nan")
    sp = None
    if seg is not None and trip_of_row is not None:
        fi = np.asarray(files)[np.asarray(trip_of_row)]
        sp = make_seg_preds(fi, _seg_idx_of(ctx, len(fi)), _split_of(split, trip_of_row),
                            _seg_y_of(ctx, len(fi)), seg, info_condition=info_condition,
                            target_caliber=target_caliber, model=model, validate=False)
    return InferResult(trip=tp, seg=sp, unit_ratio=r,
                       meta={"trip_of_row": ("yes" if trip_of_row is not None else "no"),
                             "e2e": "trainer → infer → eval_trip"})


def _seg_idx_of(ctx: Any, n: int) -> np.ndarray:
    d = getattr(ctx, "data", None)
    if d is not None and hasattr(d, "extra") and "seg_idx" in d.extra:
        return np.asarray(d.extra["seg_idx"], dtype=np.int64)
    return np.arange(n, dtype=np.int64)


def _split_of(split: Any, trip_of_row: np.ndarray) -> np.ndarray:
    return np.asarray(split)[np.asarray(trip_of_row)]


def _seg_y_of(ctx: Any, n: int) -> np.ndarray:
    d = getattr(ctx, "data", None)
    if d is not None and hasattr(d, "extra") and "seg_energy_wh" in d.extra:
        return np.asarray(d.extra["seg_energy_wh"], dtype=np.float64)
    return np.full(n, np.nan)


# --------------------------------------------------------------------------- 落盘 / 读回
def save_trip_preds(tp: TripPreds, path: Any, *, seg: SegPreds | None = None,
                    seg_path: Any = None) -> dict[str, Any]:
    """落盘标准 ``trip_preds.npz``（+ 可选 ``seg_preds.npz``），返回清单。

    meta 里额外写 ``key_sha16``（键序指纹）与 ``units`` —— 让"是不是同一份键序/量纲"
    成为可断言的字符串，而不是靠人记。
    """
    p = Path(path)
    tp.extra = dict(tp.extra or {})
    tp.extra["key_sha16"] = key_sha16(tp.file)
    tp.extra["units_declared"] = {"y_wh": UNIT_WH, "pred_wh": UNIT_WH}
    tp.to_npz(p)
    meta = dict(tp.meta_block())                 # ← extra 里已含 key_sha16
    meta.setdefault("key_sha16", tp.extra["key_sha16"])
    out = {"trip_preds": str(p), "key_sha16": tp.extra["key_sha16"],
           "n": len(tp), "meta": meta}
    if seg is not None:
        sp = Path(seg_path) if seg_path else p.with_name("seg_preds.npz")
        seg.to_npz(sp)
        out["seg_preds"] = str(sp)
    # ★ 随行 JSON 与 npz 的 meta **必须是同一块**（否则"读 npz"与"读 json"得到不同口径）
    (p.parent / "trip_preds.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_trip_preds(path: Any, *, verify: bool = True) -> TripPreds:
    """读回标准 ``trip_preds.npz``（口径/单位不明 ⇒ 拒读）。"""
    tp = TripPreds.from_npz(path)
    if verify:
        tp.validate()
    return tp


def predict_to_npz(predict_fn: Callable[..., Any], handle: Any, ctx: Any = None, *,
                   out_dir: Any = None, files: Sequence[Any], split: Sequence[Any],
                   y_wh: Any, info_condition: str, target_caliber: str, phi: Any = None,
                   model: str = "", **kw: Any) -> InferResult:
    """一步到位：``predict_fn → 规整 → 单位门 → trip_preds.npz``（``infer`` 的薄封装）。"""
    r = infer(predict_fn, handle, ctx, files=files, split=split, y_wh=y_wh,
              info_condition=info_condition, target_caliber=target_caliber, phi=phi,
              model=model, **kw)
    od = Path(out_dir) if out_dir else Paths().fw("infer", model or "model")
    od.mkdir(parents=True, exist_ok=True)
    man = save_trip_preds(r.trip, od / "trip_preds.npz", seg=r.seg)
    r.npz = Path(man["trip_preds"])
    r.seg_npz = (Path(man["seg_preds"]) if "seg_preds" in man else None)
    r.meta.update(man)
    return r


def attach_seg(tp: TripPreds, *, file: Sequence[Any], seg_idx: Sequence[Any],
               y_wh: Any, pred_wh: Any, **kw: Any) -> SegPreds:
    """给趟级预测挂上段级明细（同一口径）。"""
    return make_seg_preds(file, seg_idx, tp.split if len(tp) == len(file) else
                          np.asarray(kw.pop("split")), y_wh, pred_wh,
                          info_condition=tp.info_condition,
                          target_caliber=tp.target_caliber, model=tp.model, **kw)
