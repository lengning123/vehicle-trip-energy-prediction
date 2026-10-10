#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""preprocess.make_frozen_preds —— 把源树的**既有逐趟预测**打包成发布树的冻结件。

它做什么
========
论文复现臂（``models/paper/``）**不在本发布树里重训**（源树的 A1 要 45–60 个 epoch 的
1D-CNN + 20 万行 BDT，A2 要 1200–3000 步的 7 通道编码器），因此它们的"推理"入口是
**冻结逐趟预测 + 同口径重算**。本脚本把源树的逐趟预测 CSV 转成
``models/paper/frozen/<臂>.npz``（键 ``file_`` + ``pred_wh``），并落一份索引
``FROZEN_INDEX.json``（含每件的字节数与 SHA256 前 16）。

★ 键名为什么是 ``file_`` 而不是 ``file``
=======================================
``np.savez*`` 的第一个**形参**就叫 ``file``，所以 ``savez(f, **{"file": ...})`` 会抛
``TypeError: got multiple values for argument 'file'``。键名写 ``file_`` 即可 ——
``framework/infer.py`` 的 ``key_sha16_from_npz`` 明确认
``"file_" if "file_" in z.files else "file"``，下游无需特判。

对齐纪律
========
**按键（``file``）对齐，不按位置贴**。键序取 ``data_v3/segments_100m.csv.gz`` 里
趟首次出现的顺序（= canonical 顺序）。缺键即抛 —— 按位置对齐正是
"18.24× 事故"（val WMAPE 7.28 → 132.83）的成因。

跑法::

    python -B preprocess/make_frozen_preds.py \
        --src /root/autodl-tmp/DSH_experiments/07_segments \
        --out models/paper/frozen

.. note::
   上面 ``--src`` 里的 ``/root/autodl-tmp/...`` 是**示例路径**；实际默认值由
   ``framework/paths.py`` 派生（证据文件判据；取不到才退化到
   ``paths.SEGMENTS_CANDIDATE``），显式传 ``--src`` 或设 ``DSH_SEG_DIR`` 仍然优先。

只读源树；不读 calibration；本文件**不含 assert**。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_RELEASE = Path(__file__).resolve().parents[1]
for _p in (_RELEASE / "framework", _RELEASE / "preprocess", _RELEASE):
    _s = str(_p)
    if _s in sys.path:
        sys.path.remove(_s)
    sys.path.insert(0, _s)

import paths as _paths                                              # noqa: E402
import seg_loader as _seg_loader                                     # noqa: E402

# 臂 → (源文件, 列名, 语义)
PLAN: tuple[tuple[str, str, str, str], ...] = (
    ("A1_cnn_bdt", "models/X_papercomp/PRED_windows.csv",
     "pred__A1_cnn_bdt", "A1 主结果（CNN + BDT 二级）"),
    ("A1_cnn", "models/X_papercomp/PRED_windows.csv",
     "pred__A1_cnn", "A1 第一级（CNN）对照"),
    ("A2_cnn7_cov", "models/X_papercomp/PRED_windows.csv",
     "pred__A2_cnn7_cov", "A2 主结果（7 通道协方差编码）"),
    ("A12_calib_phys_plan", "models/X_papercomp/PRED_routes.csv",
     "pred__A12_calib_phys_plan", "A12 plan-time 物理标定变体（合成标签那一支已排除）"),
    ("G10_two_stage_plan", "models/X_papercomp/PRED_routes.csv",
     "pred__G10_two_stage_plan", "G10 论文主路径（两阶段）"),
    ("G04_phys_plan", "models/X2_papercomp/PRED_routes_x2.csv",
     "pred__G04x_phys_plan_speed_proxy", "G04 公平口径（X2 代理版，逐 fold 重标定）"),
    ("G04_phys_plan_upstream", "models/X_papercomp/PRED_routes.csv",
     "pred__G04_phys_plan_speed_plan", "G04 X1 对应列（对照；X1 无 pred__G04_phys_plan）"),
    ("P14_M100_base", "models/X_papercomp/PRED_p14.csv",
     "pred__P14_M100_base", "P14 唯一精确复现臂"),
    ("A1x_cnn_bdt", "models/X2_papercomp/PRED_windows_x2.csv",
     "pred__A1x_cnn_bdt", "A1 X2 代理版（对照）"),
    ("G10x_two_stage_plan", "models/X2_papercomp/PRED_routes_x2.csv",
     "pred__G10x_two_stage_plan_proxy", "G10 X2 代理版（对照）"),
)


def default_src() -> Path:
    """``--src`` 的默认值：**由 ``framework/paths.py`` 派生**，不写死绝对路径。

    判定顺序（与 ``paths.repo_root`` 同一条优先级，但**先认证据**，避免在发布树里把
    "存在但不是段表仓"的目录（如 ``/root/autodl-tmp`` 本身）当成输入根）：

      1. 显式环境变量 ``DSH_SEG_DIR`` / ``DSH_REPO_ROOT`` / ``SEG_DIR``（尊重之，即便还没就位）；
      2. 证据文件判据：候选里有 ``data_v3/segments_100m.csv.gz`` 的那个（``paths._EVIDENCE``）；
      3. ``paths.SEGMENTS_CANDIDATE``（云端默认位置，常量只在 paths.py 里定义一次）。

    真正的失败仍留给 ``build()`` 的 ``FileNotFoundError``（消息里已给正解）。
    """
    try:
        return Path(_seg_loader.S)                 # 与 seg_loader 同一个根（已先认证据）
    except Exception:                                          # noqa: BLE001
        return Path(_paths.SEGMENTS_CANDIDATE)


def canonical_trips(seg_csv: Path) -> tuple[list, list]:
    """从冻结段表取 canonical 趟序与趟级划分（**唯一键序来源**）。"""
    seg = pd.read_csv(seg_csv, usecols=["file", "split"])
    fil = seg["file"].to_numpy(object)
    chg = np.flatnonzero(np.r_[True, fil[1:] != fil[:-1]])
    return list(fil[chg]), list(seg["split"].to_numpy(object)[chg])


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def build(src: Path, out: Path, *, verbose: bool = True) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    seg_csv = src / "data_v3" / "segments_100m.csv.gz"
    if not seg_csv.exists():
        raise FileNotFoundError("缺冻结段表 %s（--src 应指向 07_segments 根）" % seg_csv)
    trips, split = canonical_trips(seg_csv)
    if verbose:
        print("canonical 趟数 = %d | 划分 = %s"
              % (len(trips), {k: split.count(k) for k in ("train", "validation", "future")}))
    meta, skipped = [], []
    for arm, rel, col, note in PLAN:
        p = src / rel
        if not p.exists():
            skipped.append((arm, "缺源文件 %s" % p))
            continue
        df = pd.read_csv(p)
        if col not in df.columns:
            skipped.append((arm, "源文件 %s 无列 %r" % (rel, col)))
            continue
        d = df.set_index("file")
        miss = [t for t in trips if t not in d.index]
        if miss:
            skipped.append((arm, "缺 %d 趟（例 %s）" % (len(miss), miss[:3])))
            continue
        pred = d.loc[trips, col].to_numpy(np.float64)
        dst = out / ("%s.npz" % arm)
        np.savez_compressed(str(dst),
                            file_=np.array(trips, dtype=object),
                            split=np.array(split, dtype=object),
                            pred_wh=pred,
                            src_file=np.array([rel]),
                            src_col=np.array([col]),
                            note=np.array([note]))
        meta.append({"arm": arm, "file": dst.name, "src_file": rel, "src_col": col,
                     "note": note, "n": int(len(pred)), "bytes": dst.stat().st_size,
                     "sha16": sha16(dst)})
        if verbose:
            print("  -> %-32s %-44s n=%d sha16=%s" % (dst.name, col, len(pred), sha16(dst)))
    idx = {"n_trips": len(trips),
           "key_order": "canonical：data_v3/segments_100m.csv.gz 里 file 首次出现的顺序",
           "split_counts": {k: split.count(k) for k in ("train", "validation", "future")},
           "key_note": "npz 键名用 file_（savez 的形参叫 file，不能重名）",
           "artifacts": meta, "skipped": skipped}
    (out / "FROZEN_INDEX.json").write_text(
        json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print("\n写出 %d 件冻结预测" % len(meta))
        for s in skipped:
            print("  !! 跳过 %s：%s" % s)
    return idx


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="打包源树既有逐趟预测为发布树冻结件")
    ap.add_argument("--src", default=str(default_src()),
                    help="源树根（含 data_v3/ 与 models/）；默认由 framework/paths.py 派生")
    ap.add_argument("--out", default=str(_RELEASE / "models" / "paper" / "frozen"))
    a = ap.parse_args(argv)
    idx = build(Path(a.src), Path(a.out))
    return 0 if idx["artifacts"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
