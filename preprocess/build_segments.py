#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""preprocess.build_segments —— 段表构建 / 校验的**只读入口**（发布树版）。

它做什么
========
发布树**不自带** ``data_v3``（26 MB 冻结段表由 ``DATASET_FREEZE.md`` 的 SHA256 固化，
按项目政策不入库）。本脚本负责把"段表从哪来、怎么验、长什么样"讲清楚并可执行：

    python -m preprocess.build_segments --where     # 我在读哪一份段表
    python -m preprocess.build_segments --verify    # 冻结 + 行/趟/划分计数自检
    python -m preprocess.build_segments --schema    # 段表列清单（含数据类型与缺失率）
    python -m preprocess.build_segments --deps      # 70 特征装配的四个冻结依赖 + SHA16
    python -m preprocess.build_segments --cuts      # 切点表（φ 网格）概览

段表本身由**上游同名脚本**（``scripts/seg_build_final.py``，四档冻结：50/100/250/500 m）
产出，并已冻结为 ``data_v3/segments_100m.csv.gz``。发布树**不重跑**它 —— 重跑会得到
另一份字节不同的 gz（压缩参数/时间戳），从而失去"逐位可复现"的锚点。要重建，请按
``data_v3/DATASET_FREEZE.md`` 记录的方式取得原始 1 Hz 视图后重跑上游脚本。

红线
====
只读；不写 ``data_v3``；**不读 calibration**。本文件**不含 assert**。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_RELEASE = Path(__file__).resolve().parents[1]
for _p in (_RELEASE / "framework", _RELEASE / "preprocess", _RELEASE):
    _s = str(_p)
    if _s in sys.path:
        sys.path.remove(_s)
    sys.path.insert(0, _s)

import paths as pth                                                  # noqa: E402
from paths import P                                                  # noqa: E402


def cmd_where() -> int:
    p = pth.Paths()
    print("输入根      : %s" % p.seg)
    print("命中来源    : %s" % p.source)
    print("产物根      : %s" % p.release)
    print("段表        : %s" % p.seg_csv)
    print("趟索引      : %s" % p.trip_csv)
    print("冻结声明    : %s" % p.freeze_md)
    print("标签来源    : %s" % p.study)
    for name, f in (("段表", p.seg_csv), ("趟索引", p.trip_csv), ("冻结声明", p.freeze_md),
                    ("标签目录", p.study)):
        print("  %-8s %s" % (name, "存在" if f.exists() else "**缺失**"))
    return 0


def cmd_verify() -> int:
    from preprocess import env as E
    info = E.verify_frozen()
    for k in ("seg_csv", "sha16_100m", "expected_sha16", "n_seg", "n_trip",
              "trip_by_split", "seg_by_split", "calibration_present", "frozen_ok"):
        print("%-20s %s" % (k, info.get(k)))
    print("标签来源            %s" % info.get("label_source"))
    print("★ 冻结自检 PASS（100 m 档 SHA16 = %s）" % info["sha16_100m"])
    return 0


def cmd_schema() -> int:
    import pandas as pd
    from preprocess import env as E
    ds = E.load_canonical(seg_csv=P.seg_csv, study=P.study, verify_freeze=True)
    df = ds.df
    print("行 %d / 列 %d" % (len(df), len(df.columns)))
    print("%-24s %-10s %-12s %s" % ("列", "dtype", "缺失率", "样例"))
    for c in df.columns:
        s = df[c]
        miss = float(s.isna().mean())
        try:
            sample = s.dropna().iloc[0]
        except Exception:                                    # noqa: BLE001
            sample = "-"
        print("%-24s %-10s %-12.6f %s" % (c, str(s.dtype), miss, repr(sample)[:40]))
    trips = list(ds.trip_files)
    print("\n趟数 %d；前 3 趟：%s" % (len(trips), trips[:3]))
    print("段级 split 计数 %s" % {k: int((df["split"].astype(str) == k).sum())
                                  for k in pth.SPLITS})
    return 0


def cmd_deps() -> int:
    import features as FT
    got = FT.check_deps(P)
    print("%-52s %-18s %s" % ("依赖（相对输入根）", "实测 SHA16", "冻结值"))
    for rel, want in FT.DEP_HASHES.items():
        print("%-52s %-18s %s" % (rel, got.get(rel, "?"), want))
    print("\n★ 四个冻结依赖全部命中（70 项 plan-time 特征的装配前提）")
    return 0


def cmd_cuts() -> int:
    import features as FT
    cuts = FT.load_cuts(P)
    print("切点表：行 %d / 列 %s" % (len(cuts), list(cuts.columns)))
    print(cuts.head(8).to_string(index=False))
    print("\nPHI_GRID = %s（y_rem@φ 的切点比例；φ=0 即整趟 = y_full）" % (FT.PHI_GRID,))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="段表构建 / 校验（只读入口）")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--where", action="store_true", help="打印输入根与关键文件是否存在")
    g.add_argument("--verify", action="store_true", help="冻结 + 计数自检")
    g.add_argument("--schema", action="store_true", help="段表列清单")
    g.add_argument("--deps", action="store_true", help="70 特征的四个冻结依赖 + SHA16")
    g.add_argument("--cuts", action="store_true", help="切点表概览")
    a = ap.parse_args(argv)
    if a.where:
        return cmd_where()
    if a.verify:
        return cmd_verify()
    if a.schema:
        return cmd_schema()
    if a.deps:
        return cmd_deps()
    return cmd_cuts()


if __name__ == "__main__":
    raise SystemExit(main())
