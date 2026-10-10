#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""preprocess.env —— 预处理层的**唯一前置**：把路径与冻结校验收在一处。

它解决什么
==========
"我读的是哪份段表、哪个依赖、哪一档"这件事，原先散落在每个脚本的第一屏里，各写各的。
本模块把它收敛成一次调用：

    import _bootstrap                      # 1) 挂 framework/ 与 preprocess/ 到 sys.path
    from preprocess import env as E        # 2) 拿路径 + 冻结自检
    info = E.verify_frozen()               # 3) 一行完成「100 m 档 SHA16 + 行/趟/划分计数」

四个入口
========
* ``E.paths()``        —— 唯一的 ``Paths`` 实例（输入根 + 产物根）
* ``E.verify_frozen()``—— 按【档位】校验冻结段表哈希，并核对 canonical 行/趟/划分计数
* ``E.deps()``         —— 70 特征装配的四个冻结依赖（逐项 SHA16 门）
* ``E.summary()``      —— 一份可打印的"我在什么数据上跑"的声明行

红线
====
只读；不写 ``data_v3``；**不读 calibration**（本模块连它的路径都不构造）。
本文件**不含 assert**：一切自检走显式 ``raise``（消息里给正解）。
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- 路径引导
# 本文件在 <release>/preprocess/env.py ⇒ release = parents[1]，framework = release/framework
_RELEASE = Path(__file__).resolve().parents[1]
for _p in (_RELEASE / "framework", _RELEASE / "preprocess", _RELEASE):
    _s = str(_p)
    if _s in sys.path:
        sys.path.remove(_s)
    sys.path.insert(0, _s)

import paths as _paths                                              # noqa: E402
from paths import (N_SEG_BY_SPLIT, N_SEG_EXPECTED, N_TRIP_BY_SPLIT,   # noqa: E402,F401
                   N_TRIP_EXPECTED, P, SEG_SHA16_100M, SPLITS, Paths)

__all__ = [
    "RELEASE", "FRAMEWORK", "PREPROCESS", "SEG_SHA16_100M", "SPLITS",
    "N_SEG_EXPECTED", "N_TRIP_EXPECTED", "N_SEG_BY_SPLIT", "N_TRIP_BY_SPLIT",
    "P", "Paths", "paths", "sha16", "check_freeze", "verify_frozen", "deps", "summary",
    "load_canonical", "seg_loader", "trip_blocks",
]

RELEASE: Path = _RELEASE
FRAMEWORK: Path = _RELEASE / "framework"
PREPROCESS: Path = _RELEASE / "preprocess"


# --------------------------------------------------------------------------- seg_loader 复用
def seg_loader() -> Any:
    """既有的三索引空间换算模块（**只读复用**，一个字节都没改）。

    ``preprocess/seg_loader.py`` 是本项目"canonical 201,271 / split 134,212 / sequence"
    三套索引之间的**唯一**换算入口。发布树直接放它，不做任何包装。
    """
    import importlib
    return importlib.import_module("seg_loader")


def load_canonical(**kw: Any) -> Any:
    """``seg_loader.load_canonical`` 的转发（默认走本机解析出的输入根）。"""
    return seg_loader().load_canonical(**kw)


# --------------------------------------------------------------------------- 基本工具
def paths(root: str | Path | None = None, *, probe: bool = True,
          release: str | Path | None = None) -> Paths:
    """唯一的 ``Paths`` 实例（输入根 + 产物根）。"""
    return Paths(root, probe=probe, release=release)


def sha16(path: str | Path) -> str:
    """文件 SHA256 的前 16 位（与全项目锚点口径一致）。"""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def check_freeze(p: Paths | None = None) -> str:
    """按【档位】校验冻结段表 ⇒ 返回 100 m 档 SHA16（不通过即抛）。"""
    pp = p or Paths()
    return seg_loader().check_freeze(seg_csv=pp.seg_csv, freeze_md=pp.freeze_md)


# --------------------------------------------------------------------------- 冻结自检
def trip_blocks(df: Any) -> tuple[Any, Any]:
    """段级宽表 → ``(files, trip_split)``（**每趟一条**，按首次出现顺序）。

    段表按 ``(file, seg_idx)`` 升序 ⇒ 同一趟的段是**连续区段**；本函数就取这些区段的
    第一个行作为该趟的代表。段级 split 与趟级 split 同值（划分是趟级的），但**计数不同**
    （段 134,212/30,484/36,575 vs 趟 766/188/193）—— 这两个数经常被混用，所以分开给。
    """
    import numpy as np
    fil = df["file"].to_numpy(dtype=object)
    chg = np.flatnonzero(np.r_[True, fil[1:] != fil[:-1]])
    return fil[chg], np.asarray(df["split"], dtype=object)[chg]


def verify_frozen(p: Paths | None = None, *, canonical: bool = True) -> dict[str, Any]:
    """一行完成「冻结 + 计数」自检。

    ``canonical=True``（默认）会真正读一遍段表（约 5 MB gz / 201,271 行）并核对：
    段数 201,271、趟数 1,147、**趟级**划分 766 / 188 / 193、**段级**划分
    134,212 / 30,484 / 36,575，且**不出现 calibration**。
    任何一项不符 ⇒ ``raise``（消息给出实测 vs 期望，以及正解）。

    ``canonical=False`` 只做哈希校验（快，不做行数核对）。
    """
    pp = p or Paths()
    got = check_freeze(pp)
    info: dict[str, Any] = {"root": str(pp.seg), "sha16_100m": got,
                            "expected_sha16": SEG_SHA16_100M,
                            "seg_csv": str(pp.seg_csv)}
    if not canonical:
        return info
    ds = load_canonical(seg_csv=pp.seg_csv, study=pp.study, verify_freeze=True)
    n_seg = int(len(ds.df))
    bad = sorted(set(ds.df["split"].astype(str).unique()) - set(SPLITS))
    if bad:
        raise RuntimeError("段表出现未授权划分取值 %s —— 只允许 %s（calibration 是红线）"
                           % (bad, list(SPLITS)))
    files, trip_split = trip_blocks(ds.df)
    n_trip = int(len(files))
    trip_counts = {k: int((trip_split.astype(str) == k).sum()) for k in SPLITS}
    seg_counts = {k: int((ds.df["split"].astype(str) == k).sum()) for k in SPLITS}
    problems: list[str] = []
    if n_seg != N_SEG_EXPECTED:
        problems.append("段数 %d ≠ %d" % (n_seg, N_SEG_EXPECTED))
    if n_trip != N_TRIP_EXPECTED:
        problems.append("趟数 %d ≠ %d" % (n_trip, N_TRIP_EXPECTED))
    if trip_counts != dict(N_TRIP_BY_SPLIT):
        problems.append("**趟级**划分 %s ≠ %s" % (trip_counts, dict(N_TRIP_BY_SPLIT)))
    if seg_counts != dict(N_SEG_BY_SPLIT):
        problems.append("**段级**划分 %s ≠ %s" % (seg_counts, dict(N_SEG_BY_SPLIT)))
    if problems:
        raise RuntimeError(
            "冻结自检未通过（拒绝在此基础上跑任何实验）：\n  - %s\n"
            "  正解：确认 DSH_SEG_DIR 指向 100 m 档冻结段表仓；"
            "段表哈希=%s（应为 %s）。\n"
            "  提示：段级计数（134,212/30,484/36,575）与趟级计数（766/188/193）是两套数，"
            "别互相当成对方。" % ("\n  - ".join(problems), got, SEG_SHA16_100M))
    info.update({"n_seg": n_seg, "n_trip": n_trip,
                 "trip_by_split": trip_counts, "seg_by_split": seg_counts,
                 "calibration_present": False, "frozen_ok": True,
                 "label_source": str(pp.study)})
    return info


# --------------------------------------------------------------------------- 70 特征依赖
def deps(p: Paths | None = None, *, verify: bool = True) -> dict[str, str]:
    """70 项 plan-time 特征装配的冻结依赖（见 ``framework/features.py`` 的 ``DEP_HASHES``）。

    返回 ``{相对路径: sha16}``；``verify=True`` 时逐项比对冻结哈希（不符即抛）。
    """
    import features as FT
    pp = p or Paths()
    if verify:
        return FT.check_deps(pp)
    return {k: (sha16(pp.seg / k) if (pp.seg / k).exists() else "") for k in FT.DEP_HASHES}


def summary(p: Paths | None = None) -> str:
    """一行声明"我在什么数据/什么根上跑"（日志与报告里贴这一行就够）。"""
    pp = p or Paths()
    return ("input_root=%s (source=%s) | release=%s | 段表=%s"
            % (pp.seg, pp.source, pp.release, pp.seg_csv))
