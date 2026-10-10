#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""_bootstrap —— **发布树唯一的路径引导**（release tree 的 import 前置）。

它是干什么的
============
`framework/` 里的模块彼此按**裸模块名**互相 import（``import registry``、``from paths
import Paths`` …），这是原工程 ``lib/`` 的既定风格，也是"复制即用、逐字可比对"的前提。
本文件只做一件事：**把 ``framework/`` 与 ``preprocess/`` 挂到 ``sys.path`` 上**，并登记
发布树的根目录常量。

三条纪律
========
1. **不猜根目录**：发布树根 = 本文件所在目录（``parents[0]``）。不扫、不回溯、不看环境。
2. **只加不改**：本文件不 import 框架任何模块，因此可以被任何脚本**在最前面**安全 import。
3. **不写盘、不读数据**：纯路径声明，import 一次即幂等。

用法（每个入口脚本的第一行）::

    import _bootstrap  # noqa: F401  （必须在 import framework 之前）
    import registry, trainer, infer, eval_trip

只想要路径常量::

    import _bootstrap as B
    B.RELEASE, B.FRAMEWORK, B.PREPROCESS, B.MODELS, B.RUNS
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = [
    "RELEASE", "FRAMEWORK", "PREPROCESS", "MODELS", "RUNS", "REPORTS",
    "bootstrap", "release_root", "framework_path", "preprocess_path",
    "entry", "ensure_dir",
]

# --------------------------------------------------------------------------- 路径常量
RELEASE: Path = Path(__file__).resolve().parent
FRAMEWORK: Path = RELEASE / "framework"
PREPROCESS: Path = RELEASE / "preprocess"
MODELS: Path = RELEASE / "models"
RUNS: Path = RELEASE / "runs"
REPORTS: Path = RELEASE / "reports"


def release_root() -> Path:
    """发布树根（唯一来源：本文件位置）。"""
    return RELEASE


def framework_path() -> Path:
    """``framework/`` 目录（框架层的五个模块 + 预处理装配都在这里可 import）。"""
    return FRAMEWORK


def preprocess_path() -> Path:
    """``preprocess/`` 目录（段表复用与特征装配入口）。"""
    return PREPROCESS


def _prepend(p: Path) -> bool:
    """把 ``p`` 放到 ``sys.path`` 最前面（幂等）。返回是否本次新加入。"""
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
        return True
    if sys.path[0] != s:                      # 已在表里但不在首位 ⇒ 提到最前
        sys.path.remove(s)
        sys.path.insert(0, s)
        return True
    return False


def bootstrap(*, verbose: bool = False) -> dict:
    """挂好 ``framework/`` 与 ``preprocess/``；返回本次实际做了什么（幂等、可反复调用）。"""
    added = {"framework": _prepend(FRAMEWORK), "preprocess": _prepend(PREPROCESS),
             "release": _prepend(RELEASE)}
    # 产物根：让 `runs/` 成为约定的落盘位置（**不**改环境变量之外的任何全局状态）
    os.environ.setdefault("DSH_RELEASE", str(RELEASE))
    if verbose:
        print("[_bootstrap] release    = %s" % RELEASE)
        print("[_bootstrap] framework  = %s" % FRAMEWORK)
        print("[_bootstrap] preprocess = %s" % PREPROCESS)
        print("[_bootstrap] added      = %s" % {k: v for k, v in added.items() if v})
    return {"release": str(RELEASE), "added": added}


def entry(name: str, *parts: str) -> Path:
    """发布树内的一条路径（相对 RELEASE）。"""
    return RELEASE.joinpath(name, *parts) if parts else (RELEASE / name)


def ensure_dir(p: str | os.PathLike) -> Path:
    """建目录并返回（``runs/`` 之类的产物根用）。"""
    q = Path(p)
    q.mkdir(parents=True, exist_ok=True)
    return q


# import 本文件即完成引导 —— 这是"第一行就 import _bootstrap"能成立的原因
bootstrap()
