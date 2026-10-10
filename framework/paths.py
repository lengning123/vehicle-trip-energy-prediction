#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""paths —— 仓库 / 模型 / 产物根路径的 **唯一来源**（统一框架第 0 层，只加不改）。

本模块存在的理由
================
审计实测：全树散落 **425 处** 形如 ``/root/autodl-tmp/DSH_experiments/07_segments/...``
的绝对路径字面量。它们带来三类事故：

  ① 换机器 / 换挂载点即全树失效（本地 D:\\ 与云端 /root/ 是两套）；
  ② 同一份"规范趟表"被写成三份不同的相对路径字面量 → 各自拼接、各自出错；
  ③ 产物写到哪、读的是哪一份，无法在运行前一次问清（"我读的是哪份 ckpt"）。

本模块**不搬运任何目录**，只提供一套**间接层**：

    from paths import P
    P.seg                      # 源数据仓根（= 既有 seg_loader.S，同一个值）
    P.model("SePT")            # <源仓>/models/SePT（**只读输入**）
    P.release                  # 发布树根（DSH_release）
    P.runs("SePT-70_v1.0")     # <发布树>/runs/...（**本框架自己的产物根**）
    P.check("frozen")          # 冻结哈希自检（100 m 档）

两种根，别混
============
* **输入根** ``self.root``（= ``P.seg``）：冻结数据与既有只读产物的所在。
  判据 = ``data_v3/DATASET_FREEZE.md`` 存在（**不是**"目录存在"这种弱证据）。
  默认指向源数据仓；发布树**不自带** ``data_v3``，也不复制任何大文件。
* **产物根** ``P.release``：发布树自己的 ``runs/``。训练 / 推理 / 评估的一切落盘都写这里，
  于是"跑一遍发布树"永远不会污染源数据仓。

旗舰入口
========
``repo_root()`` 按下列顺序解析输入根（**先命中先用，可打印来源**）：

    1. 显式参数 / ``DSH_SEG_DIR`` / ``DSH_REPO_ROOT`` / ``SEG_DIR`` 环境变量；
    2. 本文件自身位置向上回溯（``<lib>/../..`` 的候选）；
    3. 已知候选清单（云端 ``/root/autodl-tmp/DSH_experiments/07_segments``
       —— **示例路径**，实际字面量只在下面 ``SEGMENTS_CANDIDATE`` / ``_CANDIDATES``
       里出现一次；另有本地 ``D:\\python\\homework\\POC\\...`` 等）。

红线
====
* **只加不改**：本模块不移动、不重命名、不覆盖任何既有文件。
* 只读；不写 ``data_v3``；不读 calibration。
* 本文件**不含 assert**：所有自检都是显式 ``raise``（消息里给正解）。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = [
    "Paths", "P", "repo_root", "repo_root_source", "candidate_roots", "release_root",
    "SEG_SHA16_100M", "SPLITS", "N_SEG_BY_SPLIT", "N_TRIP_BY_SPLIT",
    "N_SEG_EXPECTED", "N_TRIP_EXPECTED", "ArtifactRoot", "artifact",
]

# --------------------------------------------------------------------------- 冻结常量
# 与 preprocess/seg_loader.py 的常量**同源同值**（本模块不重新定义数据，只转发口径）。
SEG_SHA16_100M = "298f65ca0dc5c2f5"      # 100 m 档；按档位取，不许取第一行
SPLITS = ("train", "validation", "future")   # calibration 是红线，不在其中
N_SEG_BY_SPLIT = {"train": 134_212, "validation": 30_484, "future": 36_575}
N_TRIP_BY_SPLIT = {"train": 766, "validation": 188, "future": 193}
N_SEG_EXPECTED = 201_271
N_TRIP_EXPECTED = 1_147

# 判定"这是输入根"的证据文件（相对输入根）
_EVIDENCE = ("data_v3/DATASET_FREEZE.md", "data_v3/segments_100m.csv.gz")

# --------------------------------------------------------------------------- 根候选清单
# 顺序即优先级；每一项都可以不存在（只是候选，不是断言）。
# ★ 全树的绝对路径**字面量只在这里定义一次**；其他模块一律 import 本模块取用
#   （如 ``SEGMENTS_CANDIDATE``），不各自再写一遍 —— 换挂载点只改这一处。
SEGMENTS_CANDIDATE = "/root/autodl-tmp/DSH_experiments/07_segments"   # 段表仓（输入根）
REPORTS_CANDIDATE = "/root/autodl-tmp/reports"                        # 云端报告 / 标签根
_CANDIDATES: tuple[str, ...] = (
    "D:/python/homework/POC/_staging/cloud_mirror/07_segments",
    SEGMENTS_CANDIDATE,
    "D:/python/homework/POC/07_segments",
)

_ENV_KEYS = ("DSH_SEG_DIR", "DSH_REPO_ROOT", "SEG_DIR")

_RELEASE_ENV = "DSH_RELEASE"
_RELEASE_DEFAULT = "/root/autodl-tmp/DSH_release"

_SOURCE = ""          # 上一次 repo_root() 命中的来源（供 --where / 日志用）


def release_root(explicit: str | os.PathLike | None = None) -> Path:
    """发布树根（**产物**根）：显式参数 > ``DSH_RELEASE`` > 已知默认位置。

    与输入根不同，这里**不做证据检查** —— 产物目录本来就允许还不存在（会被创建）。
    """
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get(_RELEASE_ENV)
    if env:
        return Path(env)
    return Path(_RELEASE_DEFAULT)


def candidate_roots() -> list[Path]:
    """全部候选输入根（按优先级；含环境变量与"本文件向上回溯"的推导值）。

    ``DSH_SEG_DIR`` 是**段表仓根**；``DSH_REPO_ROOT`` 是它的上一层 —— 两者都给，
    避免调用方各自猜。
    """
    out: list[Path] = []
    for k in _ENV_KEYS:
        v = os.environ.get(k)
        if not v:
            continue
        p = Path(v)
        out.append(p if p.name == "07_segments" else p / "07_segments")
    here = Path(__file__).resolve()
    # 本文件在 <seg>/lib/paths.py 或 <release>/framework/paths.py ⇒ 同时容忍两种布局
    for up in (1, 2, 3):
        try:
            out.append(here.parents[up])
        except IndexError:      # pragma: no cover - 只在极短路径下发生
            pass
    out.extend(Path(c) for c in _CANDIDATES)
    seen: list[Path] = []
    for p in out:
        if p not in seen:
            seen.append(p)
    return seen


def _looks_like_seg_dir(p: Path) -> bool:
    """唯一判据：证据文件存在（弱判据"目录在"一律不算）。"""
    try:
        return p.is_dir() and all((p / e).exists() for e in _EVIDENCE)
    except OSError:                                          # pragma: no cover
        return False


def repo_root(explicit: str | os.PathLike | None = None, *, probe: bool = True) -> Path:
    """解析**输入根**（唯一入口）。

    ``probe=False`` 时不做证据检查（只用于"我就是要拼一个还不存在的产物目录"）。
    """
    global _SOURCE
    if explicit is not None:
        p = Path(explicit)
        p = p if p.name == "07_segments" else p / "07_segments"
        _SOURCE = "explicit"
        return p
    env_hit = None
    for k in _ENV_KEYS:
        if os.environ.get(k):
            q = Path(os.environ[k])
            env_hit = q if q.name == "07_segments" else q / "07_segments"
            break
    cands = candidate_roots()
    if probe:
        for p in cands:
            if _looks_like_seg_dir(p):
                if env_hit is not None and p == env_hit:
                    _SOURCE = "env"
                elif p == Path(__file__).resolve().parents[1]:
                    _SOURCE = "derived"
                else:
                    _SOURCE = "candidate"
                return p
    else:
        for p in cands:
            if p.is_dir():
                _SOURCE = "candidate(no-probe)"
                return p
    if env_hit is not None:                      # 环境变量显式指定 ⇒ 尊重它（即便还没就位）
        _SOURCE = "env"
        return env_hit
    raise RuntimeError(
        "未能定位输入根（段表仓）—— 拒绝瞎猜。\n"
        "  候选（都缺证据文件 %s）：\n%s\n"
        "  正解：设 DSH_SEG_DIR=<...>/07_segments，或显式传 repo_root(<路径>)。"
        % (list(_EVIDENCE), "\n".join("    %s" % c for c in cands)))


def repo_root_source() -> str:
    """上一次 ``repo_root()`` 的命中来源（``env`` / ``derived`` / ``candidate`` / ``explicit``）。"""
    return _SOURCE


# --------------------------------------------------------------------------- 路径对象
@dataclass(frozen=True)
class ArtifactRoot:
    """一个产物根（名字 + 路径 + 用途），让"写在哪儿"成为可打印的对象而非字面量。"""

    name: str
    path: Path
    note: str = ""

    def __call__(self, *parts: str | os.PathLike) -> Path:
        return self.path.joinpath(*[str(p) for p in parts])

    def mkdir(self) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        return self.path

    def __str__(self) -> str:
        return "%s=%s%s" % (self.name, self.path, ("  # " + self.note) if self.note else "")


class Paths:
    """两种根的唯一来源（**只读输入 + 可写产物**；不改任何既有文件）。

    属性一览：

        seg        输入根：段表仓（冻结 data_v3 与既有只读产物）      Path
        release    产物根：发布树根                                  Path
        data_v3    <seg>/data_v3（冻结数据，只读）                    Path
        lib        <seg>/lib                                          Path
        models     <seg>/models（**只读输入**：既有实现与 ckpt）       Path
        study      EVB01 标签目录（y_full 来源）                      Path

    用法::

        from paths import P
        ck = P.runs("SePT-70_v1.0") / "ckpt" / "best_val.pt"
        tab = P.artifact("eval", "trip_preds.npz")
    """

    def __init__(self, root: str | os.PathLike | None = None, *, probe: bool = True,
                 release: str | os.PathLike | None = None):
        self.root = repo_root(root, probe=probe)
        self.source = repo_root_source()
        self.rel = release_root(release)

    # ---------------------------------------------------------------- 输入目录
    @property
    def seg(self) -> Path:
        return self.root

    @property
    def data_v3(self) -> Path:
        return self.root / "data_v3"

    @property
    def lib(self) -> Path:
        return self.root / "lib"

    @property
    def tests(self) -> Path:
        return self.lib / "tests"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def scripts(self) -> Path:
        return self.root / "scripts"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def study(self) -> Path:
        """EVB01 标签目录（y_full 来源）—— 由 ``REPORTS_CANDIDATE`` 派生，不另写字面量。"""
        return Path(REPORTS_CANDIDATE) / "energyville_study" / "EVB01"

    # ---------------------------------------------------------------- 产物目录
    @property
    def release(self) -> Path:
        return self.rel

    # ---------------------------------------------------------------- 数据文件
    @property
    def seg_csv(self) -> Path:
        return self.data_v3 / "segments_100m.csv.gz"

    @property
    def trip_csv(self) -> Path:
        return self.data_v3 / "trip_index.csv"

    @property
    def freeze_md(self) -> Path:
        return self.data_v3 / "DATASET_FREEZE.md"

    # ---------------------------------------------------------------- 既有依赖（只读）
    def E_stops(self, name: str) -> Path:
        return self.models / "E_stops" / name

    def G_stoptiming(self, name: str) -> Path:
        return self.models / "G_stoptiming" / name

    def L_prefixstyle(self, name: str) -> Path:
        return self.models / "L_prefixstyle" / name

    def X_papercomp(self, name: str) -> Path:
        return self.models / "X_papercomp" / name

    def anchor_table(self, name: str = "PRED_MATRIX.csv") -> Path:
        """BB_pooled 的臂矩阵（75 臂逐趟预测 + y 基）—— 回归验收的证据源。"""
        return self.models / "BB_pooled" / name

    # ---------------------------------------------------------------- 模型 / 运行产物
    def model(self, name: str, *parts: str) -> Path:
        """既有模型目录（**只读输入**）—— ``P.model("SePT", "sept.py")``。"""
        return self.models.joinpath(name, *parts)

    def runs(self, name: str = "", *parts: str) -> Path:
        """本框架的**运行产物根**：``<发布树>/runs/<name>/...``（源仓一个字都不动）。"""
        base = self.release / "runs"
        return base.joinpath(name, *parts) if name else base

    def fw(self, name: str = "", *parts: str) -> Path:
        """本框架自身的产物根：``<发布树>/runs/_fw/<name>``（与模型运行隔离）。"""
        base = self.release / "runs" / "_fw"
        return base.joinpath(name, *parts) if name else base

    @property
    def artifacts(self) -> dict[str, ArtifactRoot]:
        return {
            "runs": ArtifactRoot("runs", self.runs(), "训练运行（ckpt / 日志 / 逐段预测）"),
            "fw": ArtifactRoot("fw", self.fw(), "统一框架自身产物（回归表 / dogfood）"),
            "eval": ArtifactRoot("eval", self.fw("eval"), "评估表：逐位回归对比"),
            "toy": ArtifactRoot("toy", self.fw("toy"), "玩具模型的即插即用证明"),
        }

    def artifact(self, root_name: str, *parts: str) -> Path:
        if root_name not in self.artifacts:
            raise KeyError("未知产物根 %r；可用 %s" % (root_name, sorted(self.artifacts)))
        return self.artifacts[root_name](*parts)

    # ---------------------------------------------------------------- 自检
    def check(self, level: str = "frozen") -> dict[str, Any]:
        """启动自检（**只读**）。

        ``frozen``：按【档位】校验 100 m 档 SHA16 == 298f65ca0dc5c2f5，并核对
                    canonical 行/趟数与三划分计数（值取自常量，不重算）。
        ``paths`` ：只检查关键路径存在性，不做哈希（用于还不想读 5 MB gz 的场合）。
        """
        info: dict[str, Any] = {"root": str(self.root), "source": self.source,
                                "release": str(self.release)}
        if level == "paths":
            info["missing"] = [str(p) for p in (self.seg_csv, self.trip_csv, self.freeze_md)
                               if not p.exists()]
            return info
        if level != "frozen":
            raise ValueError("paths.check(level=?) 只支持 'paths' | 'frozen'，实得 %r" % level)
        import hashlib
        h = hashlib.sha256()
        with open(self.seg_csv, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
        got = h.hexdigest()[:16]
        exp = None
        for line in self.freeze_md.read_text(encoding="utf-8").splitlines():
            cells = [c.strip().strip("`") for c in line.split("|")]
            if len(cells) >= 4 and re.fullmatch(r"100\s*m", cells[1]):
                for cell in reversed(cells):
                    if re.fullmatch(r"[0-9a-f]{16}", cell):
                        exp = cell
                        break
            if exp:
                break
        if exp is None:
            raise RuntimeError("未能从 %s 解析 100 m 档 SHA256 —— 拒绝运行" % self.freeze_md)
        if got != exp or got != SEG_SHA16_100M:
            raise RuntimeError("冻结哈希不匹配 100 m：实测 %s / 记录 %s / 常量 %s —— 拒绝运行"
                               % (got, exp, SEG_SHA16_100M))
        info.update({"sha16_100m": got, "freeze_table": exp, "frozen_ok": True,
                     "n_seg_expected": N_SEG_EXPECTED, "n_trip_expected": N_TRIP_EXPECTED,
                     "n_seg_by_split": dict(N_SEG_BY_SPLIT),
                     "n_trip_by_split": dict(N_TRIP_BY_SPLIT)})
        return info

    # ---------------------------------------------------------------- 迁移审计（可选）
    def scan_absolute_paths(self, pattern: str | None = None) -> list[dict]:
        """扫描仍在用绝对路径字面量的文件（**只读**；给"逐步切过来"提供清单）。

        默认 pattern 取 ``SEGMENTS_CANDIDATE``（本模块唯一来源），调用方也可显式给。
        返回 ``[{path, line, text}]``，按文件排序。不修改任何文件。
        """
        if pattern is None:
            pattern = SEGMENTS_CANDIDATE
        hits: list[dict] = []
        for p in sorted(self.root.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:                                   # pragma: no cover
                continue
            for i, line in enumerate(txt.splitlines(), 1):
                if pattern in line:
                    hits.append({"path": str(p.relative_to(self.root)), "line": i,
                                 "text": line.strip()[:160]})
        return hits

    def __repr__(self) -> str:
        return "Paths(root=%s, source=%s, release=%s)" % (self.root, self.source, self.release)


# 模块级默认实例（懒初始化：import 本身不碰磁盘，避免"只是 import 就报错"）
_P: Paths | None = None


def _default() -> Paths:
    global _P
    if _P is None:
        _P = Paths()
    return _P


class _LazyPaths:
    """``P`` —— 模块级默认 ``Paths`` 的懒代理（首次取属性才解析根目录）。"""

    def __getattr__(self, item: str) -> Any:
        return getattr(_default(), item)

    def __repr__(self) -> str:
        try:
            return repr(_default())
        except Exception as e:                                # noqa: BLE001
            return "Paths(<未解析: %s>)" % e


P = _LazyPaths()


def artifact(root_name: str, *parts: str) -> Path:
    """``P.artifact`` 的函数式别名（给不方便 import P 的脚本用）。"""
    return _default().artifact(root_name, *parts)


# --------------------------------------------------------------------------- CLI
def _main(argv: Iterable[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="paths —— 路径唯一来源自检")
    ap.add_argument("--where", action="store_true", help="只打印解析到的根与来源")
    ap.add_argument("--check", action="store_true", help="做冻结哈希自检")
    ap.add_argument("--scan", action="store_true", help="扫描仍用绝对路径字面量的文件数")
    a = ap.parse_args(list(argv) if argv is not None else None)
    p = Paths()
    print("repo_root   = %s" % p.root)
    print("source      = %s" % p.source)
    print("release     = %s" % p.release)
    print("candidates  =")
    for c in candidate_roots():
        print("    %-60s %s" % (c, "OK" if _looks_like_seg_dir(c) else "-"))
    print("seg_csv     = %s" % p.seg_csv)
    print("study       = %s" % p.study)
    for k, v in p.artifacts.items():
        print("artifact[%-5s] = %s" % (k, v))
    if a.check:
        info = p.check("frozen")
        print("frozen      = %s" % info)
    if a.scan:
        hits = p.scan_absolute_paths()
        n_files = len({h["path"] for h in hits})
        print("绝对路径字面量：%d 行 / %d 个文件" % (len(hits), n_files))
        by_dir: dict[str, int] = {}
        for h in hits:
            by_dir[h["path"].split("/")[0]] = by_dir.get(h["path"].split("/")[0], 0) + 1
        for k in sorted(by_dir, key=lambda x: -by_dir[x])[:15]:
            print("    %-24s %5d 行" % (k, by_dir[k]))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
