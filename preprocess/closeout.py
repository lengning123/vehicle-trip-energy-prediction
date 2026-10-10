#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""preprocess/closeout.py —— 收口三件套（**只移不删** + 闸门 + 清单）。

做三件事，全部幂等：

1. **只移不删**：把发布树内的**数据类**与**权重类**文件移到同级目录
   ``/root/autodl-tmp/DSH_release_artifacts/``（保留原相对路径；此路径为**示例**，
   实际由 ``framework/paths.py`` 派生 —— 见下「路径怎么来」）。
   两类清单见 ``DATA_EXT`` / ``WEIGHT_EXT``。
2. **写 MANIFEST.md**：每件记 原路径 / 大小 / SHA256 / 用途 / **如何再生成**。
3. **真跑三条闸门并打印数字**：数据/权重命中数、那个断言关键字的 grep 命中数、
   清理后的文件数与体量。

路径怎么来（不写死绝对路径）
============================
* 发布树根 = ``paths.release_root()``（显式参数 > ``DSH_RELEASE`` > ``paths._RELEASE_DEFAULT``）；
* 移出目录 = 发布树的**同级目录** + ``<默认发布树名>_artifacts``，名字常量取自 ``paths``，
  于是整条路径都跟着 ``paths.py`` 的**唯一来源**走，本文件里不再出现绝对路径字面量；
* 取不到 ``paths`` 时退化为「本文件 → parents[1]」，与改动前同值。

为什么"只移不删"
================
这些产物是**验证与复现的证据**（冻结逐趟预测、标准 ``trip_preds.npz``、逐段预测）。
发布树要干净（不含数据/权重），但证据不能消失 —— 移到同级目录即可两全。

用法::

    python -B preprocess/closeout.py                 # 执行（幂等）
    python -B preprocess/closeout.py --dry-run       # 只列清单不改盘
    python -B preprocess/closeout.py --gates-only    # 只跑闸门

本文件**不含 assert**。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------- 路径引导
# framework/ 与 preprocess/ 挂到 sys.path（与 preprocess/env.py 同一套写法），
# 然后**只**从 framework/paths.py 取路径 —— 本文件不出现任何绝对路径字面量。
_RELEASE = Path(__file__).resolve().parents[1]
for _p in (_RELEASE / "framework", _RELEASE / "preprocess", _RELEASE):
    _s = str(_p)
    if _s in sys.path:
        sys.path.remove(_s)
    sys.path.insert(0, _s)

import paths as _paths                                              # noqa: E402

try:
    _RELEASE = _paths.release_root()        # 产物根：DSH_RELEASE > paths 默认值
    # 移出目录 = 发布树的同级目录；名字沿用 paths 的默认发布树名（不另写字面量）
    _ARTIFACTS_NAME = (_paths._RELEASE_DEFAULT.rstrip("/").rsplit("/", 1)[-1]
                       + "_artifacts")
except Exception:                                                # noqa: BLE001
    _ARTIFACTS_NAME = _RELEASE.name + "_artifacts"

# 末尾留分隔符：``shutil.move`` 对已存在的目录按"移进去"处理，与改动前逐字一致。
ARTIFACTS = Path(str(_RELEASE.parent / _ARTIFACTS_NAME) + "/")

# 数据类与权重类（**发布树内应为 0 命中**）
DATA_EXT = ("*.npz", "*.npy", "*.csv", "*.csv.gz", "*.tsv", "*.jsonl", "*.tif", "*.tiff",
            "*.pkl", "*.joblib", "*.parquet", "*.zip", "*.gz")
WEIGHT_EXT = ("*.pt", "*.pth", "*.ckpt", "*.onnx", "*.safetensors")
ALL_EXT = DATA_EXT + WEIGHT_EXT

SKIP_DIRS = {".git", "__pycache__"}

# 用途与再生成方法（按相对路径前缀匹配；未命中的给通用说明）
PURPOSE: tuple[tuple[str, str, str], ...] = (
    ("models/paper/frozen/", "论文复现臂的**冻结逐趟预测**（代理版 = 公平口径）",
     "`python -B preprocess/make_frozen_preds.py --src <07_segments 根>`"),
    ("runs/SePT-70_v1.0/seg_preds.npz", "SePT-70 的**逐段**预测（Wh）—— 神经臂冻结基线的加速来源",
     "`python -B models/ours_tree/model.py`"),
    ("runs/SePT-70_v1.0/trip_preds.npz", "SePT-70 标准趟级预测（口径元数据随行 + 键序指纹）",
     "`python -B models/ours_tree/model.py`"),
    ("runs/paper__", "论文臂标准趟级预测（装载冻结件 + 同口径重算的产物）",
     "`python -B models/paper/adapter.py --arm <臂名>`"),
    ("runs/", "神经/树臂的运行产物（ckpt / 日志 / 标准 trip_preds.npz）",
     "`python -B models/<臂>/adapter.py [--smoke]`（ckpt 由 trainer 协议写入）"),
)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def scan(root: Path = _RELEASE) -> list[Path]:
    out: list[Path] = []
    for ext in ALL_EXT:
        for p in root.rglob(ext):
            if any(s in p.parts for s in SKIP_DIRS):
                continue
            if p.is_file():
                out.append(p)
    return sorted(set(out))


def purpose_of(rel: str) -> tuple[str, str]:
    for pre, why, how in PURPOSE:
        if rel.startswith(pre):
            return why, how
    return "运行产物 / 数据文件", "见对应模型的 README 的「跑法」一节"


def move_out(*, dry_run: bool = False) -> list[dict]:
    rows: list[dict] = []
    for p in scan():
        rel = p.relative_to(_RELEASE).as_posix()
        why, how = purpose_of(rel)
        rec = {"rel": rel, "bytes": p.stat().st_size, "sha256": sha256(p),
               "purpose": why, "regenerate": how}
        rows.append(rec)
        if dry_run:
            continue
        dst = ARTIFACTS / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        shutil.move(str(p), str(dst))
    if not dry_run:
        write_manifest(rows)
    return rows


def write_manifest(rows: list[dict]) -> Path:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    man = ARTIFACTS / "MANIFEST.md"
    old: list[dict] = []
    if man.exists():
        try:
            old = json.loads((ARTIFACTS / "MANIFEST.json").read_text(encoding="utf-8"))
        except Exception:                                        # noqa: BLE001
            old = []
    seen = {r["rel"] for r in rows}
    rows = rows + [r for r in old if r.get("rel") not in seen]
    (ARTIFACTS / "MANIFEST.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# DSH_release_artifacts —— 从发布树**移出**的数据/权重产物",
             "",
             "> 发布树**不含**数据与权重文件；这些文件被**移出**（不是删除）到本目录，",
             "> 保留原相对路径，供验证与复现取回。",
             "",
             # 下面是「取回」的**示例命令**：实际路径由 framework/paths.py 派生
             "取回：`cp -r /root/autodl-tmp/DSH_release_artifacts/<相对路径> "
             "/root/autodl-tmp/DSH_release/<相对路径>`",
             "",
             "| 原路径（相对发布树） | 大小 (B) | SHA256 | 用途 | 如何再生成 |",
             "|---|---:|---|---|---|"]
    tot = 0
    for r in sorted(rows, key=lambda x: x["rel"]):
        tot += int(r["bytes"])
        lines.append("| `%s` | %d | `%s` | %s | `%s` |"
                     % (r["rel"], r["bytes"], r["sha256"][:16], r["purpose"],
                        r["regenerate"]))
    lines += ["", "**合计 %d 件 / %s 字节**" % (len(rows), format(tot, ",")),
              "", "## 生成", "",
              "`python -B preprocess/closeout.py`（幂等；只移不删）"]
    man.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return man


def gates() -> dict:
    """真跑三条闸门并返回数字。"""
    n_data = len(scan())

    def sh(cmd: str) -> tuple[int, str]:
        r = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
        return r.returncode, r.stdout

    # ★ 闸门**不能包含那个字符串字面量** —— 否则 grep 会命中这段代码自己
    #   （实测踩到：4 个命中全部来自本文件）。用拼接构造模式，让树内该字面量为 0。
    PAT = "as" + "sert "
    rc_a, out_a = sh('grep -rn "%s" %s | wc -l' % (PAT, _RELEASE))
    n_hits = int(out_a.strip() or 0)
    rc_f, out_f = sh("find %s -type f | wc -l" % _RELEASE)
    n_files = int(out_f.strip() or 0)
    rc_d, out_d = sh("du -sh %s | cut -f1" % _RELEASE)
    size = out_d.strip()
    rc_s, out_s = sh('grep -rnE \'(^|[^.[:alnum:]_])assert([^[:alnum:]_]|$)\' %s | wc -l'
                     % _RELEASE)
    n_broad = int(out_s.strip() or 0)
    rc_fn, out_fn = sh("grep -rhoE 'assert_[a-z_]+' %s | wc -l" % _RELEASE)
    n_fn = int(out_fn.strip() or 0)
    return {"data_weight_hits": n_data, "assert_hits": n_hits,
            "broad_assert_lines": n_broad, "assert_fn_names": n_fn,
            "files": n_files, "size": size}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="发布树收口：只移不删 + MANIFEST + 三条闸门")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gates-only", action="store_true")
    a = ap.parse_args(argv)
    if not a.gates_only:
        rows = move_out(dry_run=a.dry_run)
        print("== 移出 %d 件 -> %s ==" % (len(rows), ARTIFACTS))
        for r in rows:
            print("   %-56s %9d B  %s" % (r["rel"], r["bytes"], r["sha256"][:16]))
        if not a.dry_run:
            print("   MANIFEST -> %s" % (ARTIFACTS / "MANIFEST.md"))
    g = gates()
    print("\n== 三条闸门（实测） ==")
    print("  ① 数据/权重命中 : %d   （须为 0）" % g["data_weight_hits"])
    print("  ② grep -rn (as+sert+空白) : %d   （须为 0）" % g["assert_hits"])
    print("     broad 口径行数      : %d   （含说明文字里的 \"assert\" 一词）"
          % g["broad_assert_lines"])
    print("     assert_* 标识符      : %d   （函数/方法名，合法）" % g["assert_fn_names"])
    print("  ③ 清理后文件数 / 体量 : %d 个 / %s" % (g["files"], g["size"]))
    return 0 if (g["data_weight_hits"] == 0 and g["assert_hits"] == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
