#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""seg_loader —— 逐段数据集【唯一】索引换算入口（根治「三索引空间错位」bug 家族）。

本模块存在的理由
================
逐段任务被迫同时维护**三个索引空间**，此前每个 agent 各写一套换算，
于是同一族错位 bug 反复重入（本会话至少 4 次：v2 GRU、Transformer 的 batch1/2/4）：

    ① canonical（规范全量行）  201,271 行，唯一键 (file, seg_idx)
    ② split 掩码               train 134,212 / validation 30,484 / future 36,575
    ③ sequence（按趟重排 + 定长截断 + padding）
                               L=200 若按"每趟只取前 200 段"截断 → 只覆盖 104,639 行（0.52）；
                               若按 stride=L 平铺（本模块口径）→ 覆盖 100%，行序 = canonical

四次事故的签名（本模块逐条给出拦截器）：

    ① IndexError: boolean index did not match indexed array along axis 0;
       size of axis is 104639 but size of corresponding boolean axis is 201271
    ② ValueError: Found input variables with inconsistent numbers of samples:
       [134212, 201271]
    ③ ValueError: The truth value of an array with more than one element is ambiguous
       （``min(nseg, L)`` 对数组误用内建 ``min``）
    ④ UserWarning: Using a target size (torch.Size([512])) that is different to the
       input size (torch.Size([512, 1]))          ← 静默广播，不报错却伤精度

本模块的契约
============
1. **只认行键 ``(file, seg_idx)``**；位置索引一律禁止跨空间使用，只在同一空间内
   （且已用 ``check_space`` 证明身份）才允许。
2. 每个对象携带 ``Space`` 标签（名字 + 尺寸）。任何「两块数据必须同空间」的操作都先过
   ``check_space``：不匹配就抛 ``SpaceMismatch``，消息里**同时打印两个空间的名称与尺寸**
   （例：``canonical(201,271) vs seq-subset(104,639)``）。
3. 序列空间必须能反查 canonical 行号：``Sequences.pos2row`` / ``row2pos``，
   并有往返校验 ``Sequences.check_roundtrip()``。
4. **掩码必须参与 loss 与聚合**：``masked_mse`` / ``masked_abs`` / ``trip_totals``
   只接受带 mask 的输入；聚合前 ``check_mask_counts()`` 断言
   ``mask.sum()`` 等于该趟入窗段数。
5. 防 ``min()/max()/and`` 误用：``safe_min`` / ``safe_max`` / ``safe_all`` / ``safe_any``
   对数组**立即响亮失败**并给出正确写法。
6. 防静默广播：``assert_shape`` 要求形状**严格相等**（不做广播推导），专打
   ``target(512,)`` vs ``pred(512,1)`` 这类只给 warning 不报错的错。

两条**实测**踩过的坑（都已进判据套件）
========================================
* ``build_sequences`` **只在给了 ``scaler`` 时才标准化**；不给就返回**原始量纲**。
  参考实现是内部标准化的，照抄其协议却忘了传 scaler，val WMAPE 会从 7.7000 退化到
  10.8467（同 seed/同网络/同批样本）。统计量必须来自 train。
* ``L`` 与 ``stride`` 决定覆盖率：平铺（``stride == L``）不丢行；"每趟只取前 L 段"
  会静默丢样本（``L=200`` 时全表 104,639/201,271 = 0.52）。两种口径必须写明。

快速用法（完整说明见 ``SEG_LOADER.md``）
========================================
    from seg_loader import load_canonical, build_sequences, FEATS_REALIZED
    ds = load_canonical()                      # 冻结校验 + 201,271 行规范表
    m  = ds.mask('train')                      # 134,212
    seq = build_sequences(ds, m, L=None, features=FEATS_REALIZED)   # L=None 不截断
    seq.check_mask_counts().check_roundtrip()

**规范表契约**（``load_canonical`` 返回的列，调用方不得自行重算）
    file, vehicle, split, seg_idx, cum_start_m, cum_end_m, seg_len_m, seg_dur_s,
    seg_energy_wh, mean_speed_kph, speed_std, speed_p95, stop_frac, ambient_c_med,
    pack_t_med, soc_start, soc_end, y_full          ← 冻结表 + 三划分标签（kWh 量化）
    veh2, trip_km, cum_frac                          ← 加载器集中派生的 plan 列
其中 ``veh2 = vehicle.str.contains("BEV2")``、``trip_km = 逐趟 max(cum_end_m)/1000``、
``cum_frac = cum_start_m/(trip_km*1000)`` —— 与 unified_v3 逐位同口径，但**只在这里算一次**。

配套：``lib/tests/test_alignment.py``（判据套件）、``lib/SEG_LOADER.md``（文档）、
     ``lib/verify/seg_ff_repro.py``（用本加载器复现 FF MLP 逐种子数字）
红线：只读 train/validation/future；**不读 calibration**；不改 ``data_v3`` 与任何既有产物。
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- 路径引导
# 本文件在 <release>/preprocess/seg_loader.py ⇒ framework = parents[1]/framework。
# 与 preprocess/env.py 同一套写法：把 framework/ 挂到 sys.path，再按裸模块名 import。
_RELEASE = Path(__file__).resolve().parents[1]
for _p in (_RELEASE / "framework", _RELEASE / "preprocess", _RELEASE):
    _s = str(_p)
    if _s in sys.path:
        sys.path.remove(_s)
    sys.path.insert(0, _s)

import paths as _paths                                              # noqa: E402

__all__ = [
    "S", "DATA_V3", "STUDY", "SEG_CSV", "TRIP_CSV", "FREEZE_MD", "SPLITS",
    "N_SEG_EXPECTED", "N_TRIP_EXPECTED", "N_SEG_BY_SPLIT", "N_TRIP_BY_SPLIT",
    "SEG_SHA16_100M", "FEATS_PLAN6", "FEATS_REALIZED4", "FEATS_REALIZED",
    "Space", "SpaceMismatch", "ShapeContractError", "SoftBooleanError", "Labelled",
    "check_space", "assert_shape", "safe_min", "safe_max", "safe_all", "safe_any",
    "RowKeys", "SplitMask", "SegDataset", "Sequences", "LossReport",
    "sha16", "freeze_100m_hash", "check_freeze",
    "load_canonical", "split_mask", "trip_ids_of", "trip_split_of", "build_sequences",
    "masked_mse", "masked_abs", "trip_totals", "wmape", "wmape_by_trip",
    "canonicalize_seq_pred", "manifest",
]

# --------------------------------------------------------------------------- 路径与冻结常量
# DECISION(改动①)：原为直接写死 ``S = Path("/root/autodl-tmp/DSH_experiments/07_segments")``。
# 现在按同一优先级解析，**先认证据**（否则会命中发布树自己）：
#   ① 证据文件判据（``data_v3/DATASET_FREEZE.md`` + ``segments_100m.csv.gz``）命中的候选；
#   ② 环境变量 ``DSH_SEG_DIR`` / ``DSH_REPO_ROOT`` / ``SEG_DIR``；
#   ③ ``paths.SEGMENTS_CANDIDATE``（云端默认位置 —— 绝对路径字面量只在 paths.py 定义一次）。
# 换机器 / 换挂载点：设 ``DSH_SEG_DIR`` 或改 ``framework/paths.py``，本文件一个字都不用改。
def _resolve_seg_root() -> Path:
    """段表仓根：证据优先，其次 env，最后"存在的候选"（不得抛，保持旧有的"只赋值"语义）。"""
    try:
        return _paths.repo_root(probe=True)                # ① 证据命中（推荐入口）
    except Exception:                                      # noqa: BLE001
        pass
    for _k in _paths._ENV_KEYS:                            # ② 显式 env
        _v = os.environ.get(_k)
        if _v:
            _p = Path(_v)
            return _p if _p.name == "07_segments" else _p / "07_segments"
    try:
        _seg = _paths.Paths(probe=False).seg               # ③ 退化：存在的候选
        if _seg.is_dir():
            return _seg
    except Exception:                                      # noqa: BLE001
        pass
    return Path(_paths.SEGMENTS_CANDIDATE)


S = _resolve_seg_root()
DATA_V3 = S / "data_v3"
# DECISION(改动②)：原为 ``Path("/root/autodl-tmp/reports/energyville_study/EVB01")``；
# 改为由 paths 的云端报告根派生（字面量只在 paths.py 出现一次，换挂载点自动跟随）。
STUDY = Path(_paths.REPORTS_CANDIDATE) / "energyville_study" / "EVB01"
SEG_CSV = DATA_V3 / "segments_100m.csv.gz"
TRIP_CSV = DATA_V3 / "trip_index.csv"
FREEZE_MD = DATA_V3 / "DATASET_FREEZE.md"

SPLITS = ("train", "validation", "future")     # 只有这三个；calibration 是红线

SEG_SHA16_100M = "298f65ca0dc5c2f5"            # 100 m 档；按档位取，不许取第一行
N_SEG_EXPECTED = 201_271
N_TRIP_EXPECTED = 1_147
N_SEG_BY_SPLIT = {"train": 134_212, "validation": 30_484, "future": 36_575}
N_TRIP_BY_SPLIT = {"train": 766, "validation": 188, "future": 193}

# 参考实现（unified_v3.py）的特征口径，供复现对照；本模块不预设任何实验特征集
FEATS_PLAN6 = ["seg_len_m", "ambient_c_med", "pack_t_med", "soc_start", "cum_frac", "veh2"]
FEATS_REALIZED4 = ["mean_speed_kph", "speed_std", "speed_p95", "stop_frac"]
FEATS_REALIZED = FEATS_PLAN6 + FEATS_REALIZED4


def sha16(path: os.PathLike | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def freeze_100m_hash(freeze_text: str) -> str | None:
    """按【档位】从 DATASET_FREEZE.md 取 100 m 那行的哈希。

    v2 曾误取文件里第一个哈希（50 m 档）导致拒跑 —— 按档位取是硬要求。
    """
    for line in freeze_text.splitlines():
        cells = [c.strip().strip("`") for c in line.split("|")]
        if len(cells) >= 4 and re.fullmatch(r"100\s*m", cells[1]):
            for cell in reversed(cells):
                if re.fullmatch(r"[0-9a-f]{16}", cell):
                    return cell
    return None


def check_freeze(seg_csv: Path = SEG_CSV, freeze_md: Path = FREEZE_MD) -> str:
    """按档位校验冻结哈希；不匹配 = 拒绝运行（响亮失败，不给凑合跑的机会）。"""
    exp = freeze_100m_hash(Path(freeze_md).read_text(encoding="utf-8"))
    got = sha16(seg_csv)
    if exp is None:
        raise RuntimeError("未能从 %s 解析 100 m 档 SHA256 —— 拒绝运行" % freeze_md)
    if got != exp:
        raise RuntimeError("冻结哈希不匹配 100 m: 实测 %s vs 冻结 %s —— 拒绝运行" % (got, exp))
    return got


# =========================================================================== 空间标签
@dataclass(frozen=True)
class Space:
    """一个索引空间的身份证：名字 + 尺寸 + 可选口径说明。

    名字不是装饰：所有跨空间报错都靠它把两个空间的名字与尺寸直接印进异常里。
    """

    name: str
    size: int
    note: str = ""

    def __str__(self) -> str:
        base = "%s(%s)" % (self.name, format(self.size, ","))
        return base + ("[%s]" % self.note if self.note else "")


class SpaceMismatch(ValueError):
    """两个数据块不同空间 / 长度不一致 —— 消息里必须同时出现两边空间名与尺寸。

    统一拦截签名 ①（boolean index did not match indexed array）与
    签名 ②（inconsistent numbers of samples）。
    """


class ShapeContractError(AssertionError):
    """形状不一致（含本该 (n,) 却给了 (n,1)）—— 拦截静默广播（签名 ④）。"""


class SoftBooleanError(TypeError):
    """把数组当标量用（``if arr:`` / ``arr and b`` / ``min(arr, L)``）—— 拦截签名 ③。"""


# --------------------------------------------------------------------------- 空间守恒检查
def _space_of(obj: Any) -> Space | None:
    return getattr(obj, "space", None)


def _len_of(obj: Any) -> int:
    """取对象的**空间尺寸**（不是基数）。

    带 ``.space`` 的对象一律用 ``space.size``：``SplitMask.__len__`` 是"被选中的行数"
    （基数），而它在空间检查里代表的是**整个 canonical 空间** —— 两者混用会让
    "掩码 + 全量数组"这种完全合法的组合被误判为长度不一致（本模块自查时抓到过）。
    """
    sp = getattr(obj, "space", None)
    if sp is not None and getattr(sp, "size", -1) >= 0:
        return int(sp.size)
    if isinstance(obj, pd.DataFrame):
        return len(obj)
    try:
        return len(obj)
    except TypeError:
        return -1


def _short_repr(obj: Any) -> str:
    if isinstance(obj, pd.DataFrame):
        return "DataFrame(rows=%d, cols=%d)" % (len(obj), obj.shape[1])
    if isinstance(obj, pd.Series):
        return "Series(len=%d, name=%r)" % (len(obj), obj.name)
    if isinstance(obj, Labelled):
        return "%s%s" % (obj.space, tuple(np.shape(obj.values)))
    try:
        return "array%s" % (tuple(np.shape(obj)),)
    except Exception:  # noqa: BLE001
        return type(obj).__name__


def check_space(*objs: Any, names: Sequence[str] | None = None,
                require: Space | None = None) -> Space:
    """断言若干对象属于同一索引空间且长度一致；否则抛 ``SpaceMismatch``。

    * 带 ``.space`` 标签的对象（SegDataset / SplitMask / Sequences / Labelled）：
      标签与**空间尺寸**必须相等（基数可以不同 —— 掩码本来就只选一部分行）；
    * 不带标签的裸数组（ndarray / Series / list）：只做**长度一致性**检查并继承已确定的
      空间，于是裸数组混进来也至少有长度保护；
    * ``require``：显式要求所有人都是这个空间（最严，推荐关键路径使用）。
    """
    items = list(objs)
    if not items:
        raise ValueError("check_space(): 至少需要一个对象")
    tags = [_space_of(o) for o in items]
    sizes = [_len_of(o) for o in items]
    labels = []
    for t, o in zip(tags, items):
        if t is None:
            labels.append(_short_repr(o))
        else:
            try:
                card = len(o)
            except TypeError:
                card = t.size
            extra = " 基数=%s" % format(card, ",") if card != t.size else ""
            labels.append("%s%s" % (t, extra))
    alias = list(names) if names and len(names) == len(items) else \
        ["arg%d" % i for i in range(len(items))]

    def _fail(why: str) -> None:
        lines = ["索引空间不匹配：%s" % why]
        for a, lab, sz in zip(alias, labels, sizes):
            lines.append("    %-18s 空间=%-34s 空间尺寸=%s" % (a, lab, format(sz, ",")))
        if require is not None:
            lines.append("    %-18s 空间=%s" % ("要求(require)", require))
        lines.append("  提示：不要用位置索引跨空间；先用 ds.mask()/build_sequences() 拿到"
                     "目标空间的掩码，或让所有入参来自同一个 SegDataset/Sequences。")
        raise SpaceMismatch("\n".join(lines))

    if require is not None:
        for t in tags:
            if t is not None and t != require:
                _fail("存在非 %s 的对象（该对象是 %s）" % (require, t))

    known = [t for t in tags if t is not None]
    if known:
        base = known[0]
        for t in known[1:]:
            if t != base:
                _fail("%s 与 %s 不是同一个索引空间" % (base, t))
    else:
        base = Space("anonymous", sizes[0] if sizes else -1, "无标签裸数组")

    uniq = sorted({s for s in sizes if s >= 0})
    if len(uniq) > 1:
        _fail("长度不一致（正是 inconsistent numbers of samples 的成因）")
    return base


def assert_shape(x: Any, shape: Any, name: str = "x", *, context: str = "") -> Any:
    """形状**严格**相等断言（不推导广播）。

    * ``shape`` 给 int   → 要求 ``np.shape(x) == (shape,)``
    * ``shape`` 给 tuple → 要求逐轴相等（``-1`` 表示该轴任意）

    典型拦截：``assert_shape(pred, (512,), 'pred')`` 在 ``pred.shape == (512, 1)`` 时报错 ——
    这条 warning 在生产里从来没人看，但它会静默改变损失口径。
    """
    got = tuple(np.shape(x))
    want = (shape,) if isinstance(shape, int) else tuple(shape)
    if got == want:
        return x
    if len(want) != len(got):
        def sq(t: Sequence[Any]) -> str:
            return "(" + ", ".join(str(w) for w in t) + ("," if len(t) == 1 else "") + ")"
        extra = ""
        gz = [g for g in got if g != 1]
        wz = [w for w in want if w != -1]
        if gz == wz or (len(got) == len(want) + 1 and got[-1] == 1) or \
           (len(want) == len(got) + 1 and want[-1] == 1):
            extra = ("\n  这是**广播陷阱**：形状只差一个长度为 1 的轴，计算不会报错但语义已变"
                     "（对照 UserWarning: Using a target size ... that is different to the "
                     "input size ...）。请显式 reshape/squeeze，勿依赖广播。")
        raise ShapeContractError(
            "形状契约失败（%s）：期望 %s，实得 %s%s%s"
            % (name, sq(want), sq(got), ("；context=" + context) if context else "", extra))
    for axis, (w, g) in enumerate(zip(want, got)):
        if w != -1 and w != g:
            raise ShapeContractError(
                "形状契约失败（%s）axis=%d：期望 %d，实得 %d（完整期望 %s，实得 %s）%s"
                % (name, axis, w, g, want, got, ("；context=" + context) if context else ""))
    return x


# --------------------------------------------------------------------------- 防 min/max/and 误用
def _is_vector(x: Any) -> bool:
    if isinstance(x, (np.ndarray, pd.Series, pd.DataFrame, list, tuple)):
        try:
            return int(np.size(x)) != 1
        except Exception:  # noqa: BLE001
            return True
    return False


def _soft_bool_msg(where: str, a: Any, b: Any) -> str:
    return (
        "%s：至少一个操作数是**数组**，内建 min/max/and/or 会做逐元素比较并抛\n"
        "    ValueError: The truth value of an array with more than one element is ambiguous\n"
        "  a=%s  b=%s\n"
        "  正确写法：\n"
        "    · 逐元素取界: np.minimum(a, b) / np.maximum(a, b)\n"
        "    · 标量上界  : safe_min(a, b)（本模块已显式断言 a 是标量）\n"
        "    · 布尔组合  : (a > 0) & (b > 0) —— 位运算，不是 and"
        % (where, _short_repr(a), _short_repr(b)))


def safe_min(a: Any, b: Any) -> Any:
    """``min(a, b)`` 的安全替代：**任一操作数不是标量就响亮失败**。

    历史事故：``lens = min(nseg, L)`` 里 ``nseg`` 是逐趟数组 →
    ValueError: The truth value of an array with more than one element is ambiguous。
    正解是 ``np.minimum(nseg, L)``。
    """
    if _is_vector(a) or _is_vector(b):
        raise SoftBooleanError(_soft_bool_msg("safe_min()", a, b))
    return a if a <= b else b


def safe_max(a: Any, b: Any) -> Any:
    """``max(a, b)`` 的安全替代（语义同 ``safe_min``）。"""
    if _is_vector(a) or _is_vector(b):
        raise SoftBooleanError(_soft_bool_msg("safe_max()", a, b))
    return a if a >= b else b


def safe_all(x: Any) -> bool:
    """``if x:`` 的安全替代：数组一律显式化（``np.all``）。"""
    if _is_vector(x):
        raise SoftBooleanError(
            "safe_all()：%s 是多元素数组，不能在布尔上下文里用。"
            "正解：np.all(x) / np.any(x) / (x > 0).all()。" % _short_repr(x))
    return bool(x)


def safe_any(x: Any) -> bool:
    """``safe_all`` 的 any 版本。"""
    if _is_vector(x):
        raise SoftBooleanError(
            "safe_any()：%s 是多元素数组，不能在布尔上下文里用。"
            "正解：np.any(x) / np.all(x) / (x > 0).any()。" % _short_repr(x))
    return bool(x)


# =========================================================================== 行键
@dataclass(frozen=True)
class RowKeys:
    """行键容器：唯一键 ``(file, seg_idx)`` —— canonical 空间里唯一合法的身份。

    任何「我这一版数组对应哪些行」的声明都必须落到这个类型上，
    这样 ``keys.equals(canonical)`` 才是可断言的（每次变换后的必做断言）。
    """

    files: np.ndarray
    seg_idx: np.ndarray

    def __post_init__(self) -> None:
        f = np.asarray(self.files, dtype=object)
        s = np.asarray(self.seg_idx, dtype=np.int64)
        if f.shape[0] != s.shape[0]:
            raise SpaceMismatch(
                "RowKeys 内部长度不一致：files(%s) vs seg_idx(%s)"
                % (format(f.shape[0], ","), format(s.shape[0], ",")))
        object.__setattr__(self, "files", f)
        object.__setattr__(self, "seg_idx", s)

    def __len__(self) -> int:
        return int(self.files.shape[0])

    def __repr__(self) -> str:
        return "RowKeys(n=%s)" % format(len(self), ",")

    @property
    def space(self) -> Space:
        return Space("canonical-keys", len(self), "(file, seg_idx)")

    @property
    def key_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"file": self.files, "seg_idx": self.seg_idx})

    def is_unique(self) -> bool:
        return not bool(self.key_frame.duplicated().any())

    def assert_unique(self, name: str = "keys") -> RowKeys:
        d = self.key_frame.duplicated()
        if bool(d.any()):
            dup = self.key_frame[d].head(5).to_dict("records")
            raise SpaceMismatch("行键不唯一（%s）：%s 个重复，例如 %s"
                                % (name, format(int(d.sum()), ","), dup))
        return self

    def equals(self, other: RowKeys) -> bool:
        """逐位相等（同序、同内容）。顺序不同请先各取 ``.sorted()``。"""
        if len(self) != len(other):
            raise SpaceMismatch(
                "RowKeys 比较：长度不同 %s vs %s（这正是 boolean index did not match "
                "indexed array 的成因之一）"
                % (format(len(self), ","), format(len(other), ",")))
        return bool((self.files == other.files).all() and (self.seg_idx == other.seg_idx).all())

    def assert_equals(self, other: RowKeys, name: str = "keys") -> None:
        if not self.equals(other):
            bad = np.flatnonzero((self.files != other.files) | (self.seg_idx != other.seg_idx))
            i = int(bad[0]) if bad.size else -1
            raise SpaceMismatch(
                "%s.equals(canonical) 失败：%s vs %s 行中首个不同 @pos %d：%s/%s vs %s/%s"
                % (name, format(len(self), ","), format(len(other), ","), i,
                   self.files[i], self.seg_idx[i], other.files[i], other.seg_idx[i]))

    def sorted(self) -> RowKeys:
        """按 (file, seg_idx) 字典序排序（用于顺序不同但集合相同的判定）。"""
        o = np.lexsort((self.seg_idx, self.files))
        return RowKeys(self.files[o], self.seg_idx[o])

    def sorted_equal(self, other: RowKeys) -> bool:
        a, b = self.sorted(), other.sorted()
        if len(a) != len(b):
            raise SpaceMismatch("RowKeys.sorted_equal：长度不同 %s vs %s"
                                % (format(len(a), ","), format(len(b), ",")))
        return a.equals(b)

    def take(self, idx: np.ndarray, name: str = "take") -> RowKeys:
        idx = np.asarray(idx)
        if idx.dtype == bool:
            if idx.shape[0] != len(self):
                raise SpaceMismatch(
                    "RowKeys.take 布尔掩码尺寸不符：掩码 %s vs keys %s —— 你多半把 split "
                    "掩码（canonical 空间）用在了别的空间上"
                    % (format(idx.shape[0], ","), format(len(self), ",")))
        elif idx.size and (idx.min() < 0 or idx.max() >= len(self)):
            raise SpaceMismatch("RowKeys.take 位置越界：范围 [%d, %d] vs n=%s"
                                % (idx.min(), idx.max(), format(len(self), ",")))
        return RowKeys(self.files[idx], self.seg_idx[idx])


@dataclass(frozen=True)
class Labelled:
    """带空间标签的数组（薄包装，只为了让 ``check_space`` 有东西可比）。"""

    space: Space
    values: np.ndarray

    def __len__(self) -> int:
        return int(np.shape(self.values)[0])


# =========================================================================== 规范表
class SegDataset:
    """规范全量表（canonical）：201,271 行，键 ``(file, seg_idx)``，行号 = 唯一位置索引。

    两条不变量（每次变换后都要断言）：

      * 行键唯一，且等于规范键序（``ds.assert_canonical(keys)``）；
      * 行序按 ``(file, seg_idx)`` 升序，同一趟的段**连续**（``trip_offsets``）。

    第二条让「序列空间 = 规范表的一段连续切片」这一性质可断言，也把所有位置算术锁死在
    规范行序上。
    """

    def __init__(self, df: pd.DataFrame, *, source: str = "unknown",
                 meta: Mapping[str, Any] | None = None):
        need = {"file", "seg_idx", "split", "seg_len_m", "seg_energy_wh"}
        missing = need - set(df.columns)
        if missing:
            raise SpaceMismatch("规范表缺列 %s；实得 %s" % (sorted(missing), list(df.columns)))
        self.df = df
        self.source = source
        self.meta = dict(meta or {})
        self.space = Space("canonical", len(df), "全量行 (file, seg_idx)")
        self.keys = RowKeys(df["file"].to_numpy(dtype=object),
                            df["seg_idx"].to_numpy(dtype=np.int64))
        self.keys.assert_unique("canonical")
        self.split = df["split"].to_numpy(dtype=object)
        bad = set(np.unique(self.split)) - set(SPLITS)
        if bad:
            raise SpaceMismatch(
                "split 出现未授权取值 %s —— canonical 只允许 %s（出现 calibration 即违反红线）"
                % (sorted(bad), list(SPLITS)))
        f = self.keys.files
        chg = np.flatnonzero(np.r_[True, f[1:] != f[:-1]])
        self.trip_offsets = chg
        self.trip_files = f[chg]
        self.trip_nseg = np.diff(np.r_[chg, len(f)]).astype(np.int64)
        self.trip_ids = np.repeat(np.arange(len(chg)), self.trip_nseg)

    # ---------------------------------------------------------------- 基本属性
    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return ("SegDataset(n=%s, trips=%s, source=%r)"
                % (format(len(self), ","), format(len(self.trip_files), ","), self.source))

    def col(self, name: str) -> np.ndarray:
        if name not in self.df.columns:
            raise SpaceMismatch("规范表无列 %r；可用列：%s" % (name, list(self.df.columns)))
        return self.df[name].to_numpy()

    def labelled(self, name: str) -> Labelled:
        return Labelled(self.space, self.col(name))

    def assert_canonical(self, keys: RowKeys, name: str = "keys") -> None:
        """每次变换后调用：``keys.equals(canonical)``（不等即抛，附首个不同点）。"""
        keys.assert_equals(self.keys, name=name)

    def trip_slice(self, trip_id: int) -> slice:
        """趟 → 规范行号切片（靠 ``trip_offsets`` 的连续性，O(1)）。"""
        t = int(trip_id)
        if not 0 <= t < len(self.trip_files):
            raise SpaceMismatch("trip_id 越界：%d vs 0..%d" % (t, len(self.trip_files) - 1))
        a = int(self.trip_offsets[t])
        return slice(a, a + int(self.trip_nseg[t]))

    # ---------------------------------------------------------------- 掩码
    def mask(self, which: str | Iterable[str] = "train") -> SplitMask:
        """canonical 空间的划分掩码（唯一官方来源）。"""
        wanted = (which,) if isinstance(which, str) else tuple(which)
        bad = set(wanted) - set(SPLITS)
        if bad:
            raise SpaceMismatch("split_mask 只允许 %s；收到 %s（**禁止** calibration —— 红线）"
                                % (list(SPLITS), sorted(bad)))
        if not wanted:
            raise SpaceMismatch("split_mask: 至少要一个 split")
        m = np.isin(self.split, np.asarray(wanted, dtype=object))
        return SplitMask(self, m, wanted)

    def manifest(self) -> dict:
        segs = {s: int((self.split == s).sum()) for s in SPLITS}
        trips = {s: int((self.split[self.trip_offsets] == s).sum()) for s in SPLITS}
        return {
            "source": self.source,
            "n_rows": int(len(self)),
            "n_trips": int(len(self.trip_files)),
            "by_split_rows": segs,
            "by_split_trips": trips,
            "n_keys_unique": bool(self.keys.is_unique()),
            "trip_nseg": {"min": int(self.trip_nseg.min()),
                          "max": int(self.trip_nseg.max()),
                          "median": float(np.median(self.trip_nseg)),
                          "sum": int(self.trip_nseg.sum())},
        }


class SplitMask:
    """canonical 空间上的一个划分掩码（带空间标签，因此不能误用到其它空间）。

    注意 ``len(mask)`` 是**基数**（被选中的行数），``mask.space.size`` 才是**空间尺寸**
    （canonical 全量行数）。跨空间检查用后者（见 ``_len_of``）。
    """

    def __init__(self, ds: SegDataset, mask: np.ndarray, which: tuple[str, ...]):
        self.ds = ds
        self.mask = np.asarray(mask, dtype=bool)
        self.which = tuple(which)
        self.space = ds.space
        assert_shape(self.mask, len(ds), "split_mask",
                     context="split=%s 的掩码必须覆盖全部 %s 规范行"
                             % ("+".join(which), format(len(ds), ",")))
        self.label = "+".join(which)

    def __len__(self) -> int:
        return int(self.mask.sum())

    def __repr__(self) -> str:
        return "SplitMask(%s: %s/%s)" % (self.label, format(len(self), ","),
                                         format(self.mask.shape[0], ","))

    @property
    def idx(self) -> np.ndarray:
        """被选中的规范行号（升序）。"""
        return np.flatnonzero(self.mask)

    def keys(self) -> RowKeys:
        k = self.ds.keys.take(self.mask, name="split_mask[%s]" % self.label)
        self.ds.assert_canonical(k, name="split_mask[%s].keys" % self.label)
        return k

    def sub(self, arr: np.ndarray, name: str = "array") -> np.ndarray:
        """把 canonical 空间的数组按本掩码取子集（含空间/长度检查）。"""
        check_space(self, arr, names=["mask[%s]" % self.label, name], require=self.space)
        assert_shape(arr, len(self.ds), name,
                     context="按 %s 掩码取子集的内容必须是 canonical 全量数组" % self.label)
        return np.asarray(arr)[self.mask]

    def trips(self) -> np.ndarray:
        return np.unique(self.ds.trip_ids[self.mask])

    def check(self, *, expected_rows: int | None = None,
              expected_trips: int | None = None) -> SplitMask:
        """计数自检：行数 / 趟数（默认用冻结期望值）。"""
        if expected_rows is None and self.label in N_SEG_BY_SPLIT:
            expected_rows = N_SEG_BY_SPLIT[self.label]
        if expected_trips is None and self.label in N_TRIP_BY_SPLIT:
            expected_trips = N_TRIP_BY_SPLIT[self.label]
        if expected_rows is not None and len(self) != expected_rows:
            raise SpaceMismatch("掩码 %s 行数 %s ≠ 期望 %s"
                                % (self.label, format(len(self), ","),
                                   format(expected_rows, ",")))
        if expected_trips is not None and len(self.trips()) != expected_trips:
            raise SpaceMismatch("掩码 %s 趟数 %s ≠ 期望 %s"
                                % (self.label, format(len(self.trips()), ","),
                                   format(expected_trips, ",")))
        return self


# =========================================================================== 序列空间
class Sequences:
    """定长窗口序列空间：``X[n_windows, L, F]`` / ``mask[n_windows, L]`` / ``trip_ids``。

    关键不变量
    ----------
    * ``pos2row``：``(n_windows, L)`` 的**规范行号**（-1 = padding / 越界）。
      它是序列空间与 canonical 之间**唯一**的换算表：禁止下游自行算术推导。
    * ``row2pos``：规范行号 → **首次**出现的 flat 位置（``-1`` 表示未覆盖）。
    * ``coverage``：每个规范行被多少个窗口覆盖；``weights = 1/coverage``。
      重叠窗口不归一会**静默重复加权** —— 这是「每行被算 N 次」的事故来源。
    * Padding 语义：``mask=0``、``pos2row=-1``、``X=pad_value``，
      **永不进入 loss/聚合**（见 ``masked_mse`` / ``masked_abs`` / ``trip_totals``）。
    * flat 一维长度是 ``n_cells``（= ``n_windows*L``，**含 padding**），
      **不是** ``mask.sum()``（去 padding 的有效格数）—— 混用这两个数是签名 ① 的成因。
      把 canonical 数组搬进序列空间请用 ``embed()``，不要手写 ``zeros+索引``。
    """

    def __init__(self, *, X: np.ndarray, mask: np.ndarray, trip_ids: np.ndarray,
                 pos2row: np.ndarray, offsets: np.ndarray, lengths: np.ndarray,
                 windows_per_trip: np.ndarray, stride: int, L: int,
                 features: Sequence[str], ds: SegDataset,
                 coverage: np.ndarray, truncated: np.ndarray,
                 selected: Any = None, dropped_tail_rows: int = 0, gap_rows: int = 0,
                 truncation_loss_rows: int = 0):
        self.X = np.asarray(X)
        self.mask = np.asarray(mask, dtype=bool)
        self.trip_ids = np.asarray(trip_ids, dtype=np.int64)
        self.pos2row = np.asarray(pos2row, dtype=np.int64)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.windows_per_trip = np.asarray(windows_per_trip, dtype=np.int64)
        self.stride = int(stride)
        self.L = int(L)
        self.features = tuple(features)
        self.ds = ds
        self.coverage = np.asarray(coverage, dtype=np.int64)
        self.truncated = np.asarray(truncated, dtype=bool)
        self.selected = selected
        self.dropped_tail_rows = int(dropped_tail_rows)
        self.gap_rows = int(gap_rows)
        self.truncation_loss_rows = int(truncation_loss_rows)
        self.meta: dict = {}
        self.space = Space("seq", int(self.mask.shape[0]),
                           "windows L=%d stride=%d" % (self.L, self.stride))
        self.flat_space = Space("seq-flat", int(self.mask.size),
                                "%s cells" % format(self.mask.size, ","))
        # ---- 不变量
        assert_shape(self.X, (self.mask.shape[0], self.L, len(self.features)), "X")
        assert_shape(self.mask, (self.pos2row.shape[0], self.L), "mask")
        assert_shape(self.trip_ids, self.mask.shape[0], "trip_ids")
        assert_shape(self.pos2row, (self.mask.shape[0], self.L), "pos2row")
        assert_shape(self.coverage, len(ds), "coverage")
        if self.mask.shape[0] and not np.array_equal(self.mask, self.pos2row >= 0):
            n_bad = int((self.mask != (self.pos2row >= 0)).sum())
            raise SpaceMismatch(
                "Sequences 不变量破坏：mask 与 (pos2row>=0) 有 %s 格不一致"
                "（padding 位置必须 mask=0 且 pos2row=-1）" % format(n_bad, ","))
        if self.mask.any():
            mx = int(self.pos2row[self.mask].max())
            if mx >= len(ds):
                raise SpaceMismatch("pos2row 越界：max=%s ≥ canonical n=%s"
                                    % (format(mx, ","), format(len(ds), ",")))
        self.row2pos = np.full(len(ds), -1, dtype=np.int64)
        if self.mask.any():
            flat_pos = np.flatnonzero(self.mask.ravel())
            rows = self.pos2row.ravel()[flat_pos]
            order = np.argsort(rows, kind="stable")
            rows_sorted = rows[order]
            starts = np.flatnonzero(np.r_[True, rows_sorted[1:] != rows_sorted[:-1]])
            self.row2pos[rows_sorted[starts]] = flat_pos[order][starts]

    # ---------------------------------------------------------------- 基础
    def __len__(self) -> int:
        return int(self.mask.shape[0])

    def __repr__(self) -> str:
        return ("Sequences(windows=%s, L=%d, F=%d, stride=%d, mask.sum=%s, covered_rows=%s)"
                % (format(len(self), ","), self.L, len(self.features), self.stride,
                   format(int(self.mask.sum()), ","),
                   format(int((self.coverage > 0).sum()), ",")))

    @property
    def flat_mask(self) -> np.ndarray:
        return self.mask.reshape(-1)

    def flat_shape(self) -> tuple[int, int]:
        """窗口网格 ``(n_windows, L)``（**二维**视图；要一维长度用 ``n_cells``）。"""
        return (int(self.mask.shape[0]), self.L)

    @property
    def n_cells(self) -> int:
        """flat 空间的一维长度 = ``n_windows * L``（**含 padding**）。

        ``masked_mse``/``trip_totals`` 的入参就用这个长度；不要把
        ``mask.sum()``（有效格数）当成它 —— 那正是 ① 家族的尺寸错配。
        """
        return int(self.mask.size)

    def embed(self, canonical_values: np.ndarray, *, fill: float = 0.0) -> np.ndarray:
        """canonical 数组 → **flat 序列向量**（长度 ``n_cells``；padding 处填 ``fill``）。

        这是"从规范空间搬一份预测/目标进序列空间"的**唯一**推荐写法：
        不要手工 ``np.zeros(N*L)`` 再按 ``pos2row`` 索引，那正是错位家族的手写换算。
        """
        a = np.asarray(canonical_values)
        assert_shape(a, len(self.ds), "canonical_values",
                     context="embed() 的入参必须是 canonical 全量数组（%s）"
                             % format(len(self.ds), ","))
        out = np.full(self.n_cells, float(fill), dtype=np.float64)
        fm = self.flat_mask
        out[fm] = np.asarray(a, dtype=np.float64)[self.pos2row.ravel()[fm]]
        return out

    def flat_to_rows(self, flat_idx: np.ndarray) -> np.ndarray:
        """flat 位置（窗口 i 第 t 格 = ``i*L+t``）→ 规范行号（-1 = padding）。"""
        f = np.asarray(flat_idx, dtype=np.int64)
        if f.size and (f.min() < 0 or f.max() >= self.mask.size):
            raise SpaceMismatch("flat 位置越界：[%d, %d] vs n_cells=%s"
                                % (f.min(), f.max(), format(self.mask.size, ",")))
        return self.pos2row.reshape(-1)[f]

    @property
    def weights(self) -> np.ndarray:
        """规范行 → 归一权重（``1/覆盖次数``；未覆盖为 0）。重叠窗口聚合必用。"""
        w = np.zeros(len(self.ds), dtype=np.float64)
        nz = self.coverage > 0
        w[nz] = 1.0 / self.coverage[nz]
        return w

    def flat_weights(self, *, canonical: bool = True) -> np.ndarray:
        """逐格权重（``(n_windows, L)`` 的 flat 版）；padding 位置权重为 0。

        ``canonical=True``（默认）= 按规范行归一（1/覆盖次数），于是重叠窗口下的加权平均
        与不重叠窗口的平均在数学上等价。
        """
        w = np.zeros(self.mask.size, dtype=np.float64)
        rows = self.pos2row.ravel()[self.flat_mask]
        wt = self.weights[rows] if canonical else np.ones(rows.size, dtype=np.float64)
        w[np.flatnonzero(self.flat_mask)] = wt
        return w

    def trip_lengths(self, trip_id: int) -> dict:
        """该趟在序列空间的账目：窗口数 / mask.sum / 该趟总段数。"""
        t = int(trip_id)
        loc = np.flatnonzero(self.trip_ids == t)
        if loc.size == 0:
            raise SpaceMismatch("trip_id=%d 不在本 Sequences 里（可能被 gap 规则整体剔除了）" % t)
        return {"trip_id": t,
                "file": str(self.ds.trip_files[t]),
                "windows": int(loc.size),
                "mask_sum": int(self.mask[loc].sum()),
                "n_seg_total": int(self.ds.trip_nseg[t])}

    def check_mask_counts(self, *, strict: bool | None = None) -> Sequences:
        """断言 ``mask.sum()`` 等于各趟入窗段数之和，并逐趟与 canonical 对账。

        ``strict``（默认：``stride == L`` 时为 True）额外要求**所有被覆盖的规范行覆盖次数
        一致** —— 不一致就意味着有行会被重复计入，即静默重复加权。不管 ``strict`` 取值，
        只要出现覆盖次数 > 1，``overlap_report()`` 都会把它数清楚（绝不静默）。
        """
        if strict is None:
            strict = (self.stride == self.L)
        total = int(self.mask.sum())
        per: dict[str, int] = {}
        for t in np.unique(self.trip_ids):
            loc = np.flatnonzero(self.trip_ids == t)
            got = int(self.mask[loc].sum())
            sl = self.ds.trip_slice(int(t))
            in_window = int(self.coverage[sl].sum())
            if got != in_window:
                raise SpaceMismatch(
                    "趟 %s：mask.sum=%s ≠ 该趟入窗段数（coverage 汇总）=%s"
                    % (self.ds.trip_files[int(t)], format(got, ","), format(in_window, ",")))
            per[str(self.ds.trip_files[int(t)])] = got
        if sum(per.values()) != total:
            raise SpaceMismatch("mask.sum()=%s ≠ Σ逐趟=%s"
                                % (format(total, ","), format(sum(per.values()), ",")))
        if strict:
            cov = self.coverage[self.coverage > 0]
            if cov.size and int(cov.min()) != int(cov.max()):
                rep = self.overlap_report()
                raise SpaceMismatch(
                    "覆盖次数不一致：被覆盖的规范行里既有 %d 次也有 %d 次（%s 行被重复覆盖）。\n"
                    "  ⇒ 重叠窗口（stride=%d < L=%d）下直接 sum/mean 会**静默重复加权**。\n"
                    "     · 要每行一次：stride=L（不重叠）；\n"
                    "     · 确实要重叠：聚合用 seq.flat_weights() / seq.weights 归一，"
                    "并在报告里写明每行被计入的次数分布 %s"
                    % (int(cov.min()), int(cov.max()),
                       format(int((self.coverage > 1).sum()), ","),
                       self.stride, self.L, rep))
        return self

    def overlap_report(self) -> dict:
        """重叠诊断：多少行被覆盖几次（把重复加权变成可读数字）。"""
        cov = self.coverage
        vals, cnts = np.unique(cov[cov > 0], return_counts=True)
        return {"hist": {int(v): int(c) for v, c in zip(vals, cnts)},
                "n_rows_overlapped": int((cov > 1).sum()),
                "max_overlap": int(cov.max()) if cov.size else 0}

    def check_roundtrip(self, *, allow_overlap: bool = False) -> Sequences:
        """序列位置 ↔ 规范行号往返一致。

        * ``flat -> rows -> flat`` 对**所有被 mask 的格子**逐位恒等（不重叠窗口）；
        * padding 格 ``pos2row == -1`` 且 ``row2pos`` 不回指；
        * ``stride < L``（重叠窗口）下不是恒等映射 —— 默认**响亮失败**，
          要重叠必须显式 ``allow_overlap=True``，并自行处理权重。
        """
        flat_pos = np.flatnonzero(self.flat_mask)
        rows = self.pos2row.ravel()[flat_pos]
        back = self.row2pos[rows]
        if not np.array_equal(back, flat_pos):
            if self.stride != self.L and allow_overlap:
                return self
            bad = np.flatnonzero(back != flat_pos)
            i = int(bad[0])
            rep = self.overlap_report()
            raise SpaceMismatch(
                "序列↔规范往返失败：%s 格不一致（首例 flat=%d (row=%d) 回指到 flat=%d）。\n"
                "  原因：stride=%d < L=%d 造成重叠，%s 个规范行被多次覆盖（hist=%s）。\n"
                "  ⇒ 若是有意重叠：check_roundtrip(allow_overlap=True)，聚合用 "
                "seq.flat_weights()；\n"
                "  ⇒ 若不是：把 stride 设成 L，或检查 pos2row 是否被下游改写。"
                % (format(bad.size, ","), int(flat_pos[i]), int(rows[i]), int(back[i]),
                   self.stride, self.L, format(rep["n_rows_overlapped"], ","), rep["hist"]))
        return self

    def check_coverage(self, *, require_full: bool = False) -> Sequences:
        """覆盖账：多少行进了窗口、多少行被截断丢掉。``require_full`` 时要求全覆盖。

        判据基数是**本次建序列所用掩码内的行数**（``meta['rows_in_mask']``），
        不是全表 —— 否则一次 train-only 的构建会被误判为"覆盖不完整"。
        """
        cov = self.coverage
        base = int(self.meta.get("rows_in_mask", int((cov > 0).sum())))
        if require_full and int((cov > 0).sum()) != base:
            lost = int(base - int((cov > 0).sum()))
            in_mask = np.isin(self.ds.trip_ids, np.unique(self.trip_ids))
            first = np.flatnonzero(in_mask & (cov == 0))
            raise SpaceMismatch(
                "覆盖不完整：掩码内 %s 行中有 %s 行未进窗口（L=%d；首例 canonical row=%d）。\n"
                "  ⇒ 若模型是展平口径（每行必须恰好一次），截断就是**静默丢样本**：\n"
                "     · 把 L 设为不小于最大趟段数（本数据集 max n_seg 见 ds.manifest()），或\n"
                "     · 显式声明该臂的覆盖率并把覆盖率写进报告。"
                % (format(base, ","), format(lost, ","), self.L,
                   int(first[0]) if first.size else -1))
        return self

    # ---------------------------------------------------------------- 取回
    def rows_of_window(self, i: int) -> np.ndarray:
        return self.pos2row[int(i)][self.mask[int(i)]]

    def flat_index_of(self, row: int) -> int:
        p = int(self.row2pos[int(row)])
        if p < 0:
            raise SpaceMismatch(
                "规范行 %s 未被任何窗口覆盖（coverage=0）—— 它在截断/间隔中被丢弃，"
                "不能从序列空间取回" % format(int(row), ","))
        return p

    def gather_trip(self, trip_id: int, flat_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """从 flat 预测里取出该趟的（值, 规范行号）：**padding 不参与**。"""
        f = np.asarray(flat_pred)
        if f.shape == self.flat_shape():
            f = f.reshape(-1)
        assert_shape(f, self.n_cells, "flat_pred",
                     context="flat 预测必须是 (n_windows*L,) 或 (n_windows, L)")
        loc = np.flatnonzero(self.trip_ids == int(trip_id))
        if loc.size == 0:
            raise SpaceMismatch("trip_id=%d 不在本 Sequences 里" % int(trip_id))
        vals, rows = [], []
        for i in loc:
            m = self.mask[i]
            if not bool(m.any()):
                continue
            vals.append(f[i * self.L: i * self.L + self.L][m])
            rows.append(self.pos2row[i][m])
        if not vals:
            return np.zeros(0), np.zeros(0, dtype=np.int64)
        return np.concatenate(vals), np.concatenate(rows)


# =========================================================================== 载入
_LABEL_CACHE: dict[str, pd.DataFrame] = {}


def _read_labels(study: Path = STUDY) -> pd.DataFrame:
    """只读 train/validation/future 三个 predictions 表并取 ``y_full``（**不读 calibration**）。"""
    key = str(study)
    if key in _LABEL_CACHE:
        return _LABEL_CACHE[key]
    frames = []
    for s in SPLITS:
        p = Path(study) / ("predictions_%s.csv" % s)
        d = pd.read_csv(p, usecols=["file", "split", "y_full"])
        got = set(d["split"].unique())
        if got != {s}:
            raise SpaceMismatch(
                "%s 里出现了非 %s 的划分：%s（**红线**：只允许 %s）"
                % (p.name, s, sorted(got), list(SPLITS)))
        frames.append(d)
    lab = pd.concat(frames, ignore_index=True).drop_duplicates("file")
    _LABEL_CACHE[key] = lab
    return lab


def load_canonical(*, seg_csv: Path = SEG_CSV, trip_csv: Path = TRIP_CSV,
                   freeze_md: Path = FREEZE_MD, study: Path = STUDY,
                   verify_freeze: bool = True, expect_counts: bool = True) -> SegDataset:
    """载入**规范全量表**（唯一入口）：冻结校验 → 段表 → 标签 → 一致性断言。

    返回 ``SegDataset``：行序 ``(file, seg_idx)`` 升序、同趟连续、键唯一，
    只含 train/validation/future（出现 calibration 直接抛）。
    """
    if verify_freeze:
        got = check_freeze(seg_csv, freeze_md)
        if got != SEG_SHA16_100M:
            raise SpaceMismatch(
                "100 m 档哈希 %s ≠ 记录常量 %s（DATASET_FREEZE.md 可能已变）"
                % (got, SEG_SHA16_100M))
    df = pd.read_csv(seg_csv)
    n0 = len(df)
    lab = _read_labels(Path(study))
    df = df.merge(lab[["file", "y_full"]], on="file", how="left")
    if len(df) != n0:
        raise SpaceMismatch("标签 merge 改变了行数：%s → %s（canonical 空间必须冻结）"
                            % (format(n0, ","), format(len(df), ",")))
    n_missing = int(df["y_full"].isna().sum())
    if n_missing:
        raise SpaceMismatch("%s 行的 y_full 缺失（标签未覆盖该趟）—— 拒绝静默丢行"
                            % format(n_missing, ","))
    df = df.sort_values(["file", "seg_idx"], kind="stable").reset_index(drop=True)
    # ---- 派生列：由加载器集中计算（禁止各 agent 各写一套，正是错位家族的另一半来源）
    #      veh2    : vehicle 含 BEV2 → 1/0（与 unified_v3 逐位同口径）
    #      trip_km : 该趟总里程 = 逐趟 max(cum_end_m)/1000
    #      cum_frac: cum_start_m / 该趟总里程（plan 特征之一）
    df["veh2"] = df["vehicle"].astype(str).str.contains("BEV2").astype(int)
    df["trip_km"] = df.groupby("file")["cum_end_m"].transform("max") / 1000.0
    df["cum_frac"] = df["cum_start_m"] / (df["trip_km"] * 1000.0).replace(0, np.nan)
    n_bad_frac = int((~np.isfinite(df["cum_frac"].to_numpy(float))).sum())
    if n_bad_frac:
        raise SpaceMismatch(
            "%d 行的 cum_frac 非有限（trip_km=0 的趟）—— 派生列异常，拒绝静默填值"
            % format(n_bad_frac, ","))
    ds = SegDataset(df, source=str(seg_csv),
                    meta={"sha16": SEG_SHA16_100M, "labels_from": str(study)})

    trip = pd.read_csv(trip_csv)
    idx = pd.Index(ds.trip_files, name="file")
    j = trip.set_index("file")["n_seg"].to_frame().join(
        pd.Series(ds.trip_nseg, index=idx, name="n_actual"))
    if j["n_actual"].isna().any():
        raise SpaceMismatch("段表有趟不在 trip_index 里：%s"
                            % j.index[j["n_actual"].isna()][:5].tolist())
    bad = j[j["n_seg"] != j["n_actual"]]
    if len(bad):
        raise SpaceMismatch("%d 趟的段数与 trip_index.n_seg 不符，例如 %s"
                            % (len(bad), bad.head(3).to_dict("index")))
    tmap = trip.set_index("file")["split"]
    ts = tmap.reindex(idx).to_numpy(dtype=object)
    seg_first = ds.split[ds.trip_offsets]
    if not np.array_equal(ts, seg_first):
        bad_i = np.flatnonzero(ts != seg_first)[:5]
        raise SpaceMismatch("趟级 split 与段表 split 不一致 @ %s"
                            % [(str(ds.trip_files[i]), ts[i], seg_first[i]) for i in bad_i])

    if expect_counts:
        man = ds.manifest()
        if man["n_rows"] != N_SEG_EXPECTED:
            raise SpaceMismatch("canonical 行数 %s ≠ 冻结期望 %s"
                                % (format(man["n_rows"], ","), format(N_SEG_EXPECTED, ",")))
        if man["n_trips"] != N_TRIP_EXPECTED:
            raise SpaceMismatch("canonical 趟数 %s ≠ 冻结期望 %s"
                                % (format(man["n_trips"], ","), format(N_TRIP_EXPECTED, ",")))
        for s in SPLITS:
            ds.mask(s).check()
    return ds


def split_mask(ds: SegDataset, which: str | Iterable[str] = "train") -> SplitMask:
    """canonical 空间划分掩码（唯一官方来源）。"""
    return ds.mask(which)


def trip_ids_of(ds: SegDataset, mask: SplitMask | np.ndarray) -> np.ndarray:
    """掩码内的趟号（canonical 空间）。"""
    m = mask.mask if isinstance(mask, SplitMask) else np.asarray(mask)
    if m.dtype != bool or m.shape != (len(ds),):
        raise SpaceMismatch(
            "trip_ids_of：掩码必须是 canonical(%s) 长度的布尔数组，实得 dtype=%s shape=%s"
            % (format(len(ds), ","), m.dtype, np.shape(m)))
    return np.unique(ds.trip_ids[m])


def trip_split_of(ds: SegDataset, trips: Sequence[int]) -> dict[int, str]:
    """趟号 → split 名；越界即抛。"""
    out = {}
    for t in trips:
        sl = ds.trip_slice(int(t))
        out[int(t)] = str(ds.split[sl][0])
    return out


# =========================================================================== 序列构建
def build_sequences(ds: SegDataset, mask: SplitMask | np.ndarray = "train", *,
                    L: int | None = 200, stride: int | None = None,
                    features: Sequence[str] | None = None, max_gap_rows: int = 0,
                    pad_value: float = 0.0, fillna: float | None = -999.0,
                    scaler: Mapping[str, Any] | None = None) -> Sequences:
    """把 canonical 的一个划分掩码窗口化成 ``(X[N,L,F], mask[N,L], trip_ids)``。

    参数
    ----
    mask        : canonical 空间掩码（``SplitMask`` 或等长布尔数组）—— 只接受 canonical。
    L           : 窗口长度；``None`` = 不截断（取最大趟段数）。显式给小 L 会截断，账目记在
                  ``truncation_loss_rows`` 与 ``coverage`` 里；``check_coverage()``
                  会把覆盖缺口摆到台面上。
    stride      : 窗口步长（默认 = L，即不重叠）。``stride < L`` 表示同一行被多次覆盖，
                  必须用 ``seq.flat_weights()`` 归一；``check_mask_counts()`` 会拦住。
    max_gap_rows: 允许跳过的连续缺行上限（默认 0 = 任何不连续都抛）。序列要求趟内段连续；
                  静默跳过会改掉序列的物理意义。
    fillna      : 非有限特征的填充值（默认 -999，与参考实现一致）；``None`` = 不填。
    scaler      : ``{"mu":..., "sd":...}``；**给了才标准化**（返回 ``(X-mu)/sd``）。

                  ⚠️ 不给 ``scaler`` 时 ``seq.X`` 是**原始量纲**（与冻结表同量纲）。
                  参考实现 unified_v3 的标准化是它内部做的 —— 照抄它的协议却忘了传
                  scaler，网络就会吃原始量纲，实测 val WMAPE 从 7.7000 退化到 10.8467。
                  统计量**只能来自 train 掩码**：
                  ``mu = raw[tr].mean(0)；sd = where(raw[tr].std(0)>=1e-9, ..., 1.0)``。
    """
    if features is None:
        raise SpaceMismatch(
            "build_sequences 必须显式给 features（禁止按列位置取数：列序变过，"
            "而位置索引正是本 bug 家族的根源）")
    feats = list(features)
    miss = [c for c in feats if c not in ds.df.columns]
    if miss:
        raise SpaceMismatch("features 缺列 %s；可用列 %s" % (miss, list(ds.df.columns)))
    if len(set(feats)) != len(feats):
        raise SpaceMismatch("features 有重复列：%s" % feats)
    if isinstance(mask, SplitMask):
        check_space(mask, ds, names=["mask", "ds"], require=ds.space)
        m = mask.mask
        label = mask.label
    else:
        m = np.asarray(mask)
        if m.dtype != bool or m.shape != (len(ds),):
            raise SpaceMismatch(
                "build_sequences 的 mask 必须是 canonical(%s) 的布尔数组，实得 dtype=%s "
                "shape=%s —— 若你手里是 sequence/子集空间的掩码，请先回到 canonical 重建"
                % (format(len(ds), ","), m.dtype, np.shape(m)))
        label = "custom"
    L = int(ds.trip_nseg.max()) if L is None else int(L)
    if L <= 0:
        raise SpaceMismatch("L 必须为正整数，实得 %d" % L)
    stride = L if stride is None else int(stride)
    if stride <= 0:
        raise SpaceMismatch("stride 必须为正整数，实得 %d" % stride)

    rows_in_mask = int(m.sum())
    sel_trips = np.unique(ds.trip_ids[m])
    if sel_trips.size == 0:
        raise SpaceMismatch("掩码 %s 为空（0 行）—— 没有可建序列的趟" % label)

    F = len(feats)
    Xc = ds.df[feats].astype(float).to_numpy(dtype=np.float64)
    assert_shape(Xc, (len(ds), F), "X_canonical")
    finite = np.isfinite(Xc).all(axis=1)
    bad_in_mask = int((~finite & m).sum())
    Xc_f = Xc if fillna is None else np.where(finite[:, None], Xc, float(fillna))

    # ---- 趟内连续性：掩码选中的趟必须整趟入选（否则序列被切开）
    gap_rows = 0
    for t in sel_trips:
        sl = ds.trip_slice(int(t))
        n_in = int(m[sl].sum())
        n_tot = int(ds.trip_nseg[int(t)])
        if n_in != n_tot:
            missing = int(n_tot - n_in)
            if missing > max_gap_rows:
                raise SpaceMismatch(
                    "趟 %s 在掩码 %s 内只有 %s/%s 段（缺 %s 段）。序列要求趟内连续 ——\n"
                    "  · 若是**故意**的：把 max_gap_rows 设成允许上限并接受序列被切开；\n"
                    "  · 若只想用完整的趟：先把掩码收敛到趟级（trip-level）再建序列。"
                    % (ds.trip_files[int(t)], label, format(n_in, ","),
                       format(n_tot, ","), format(missing, ",")))
            gap_rows += missing

    # ---- 逐趟建窗（同趟连续；窗口内位置 = canonical 行序）
    starts_per_trip: list[tuple[int, np.ndarray, np.ndarray]] = []
    for t in np.flatnonzero(np.isin(np.arange(len(ds.trip_files)), sel_trips)):
        n_here = int(ds.trip_nseg[int(t)])
        n_win = int(np.ceil(n_here / stride))
        s = np.arange(n_win, dtype=np.int64) * stride
        e = np.minimum(s + L, n_here)
        base = int(ds.trip_offsets[int(t)])
        rows_abs = np.flatnonzero(m[ds.trip_slice(int(t))]) + base
        if rows_abs.size:
            hits = ((rows_abs[:, None] >= base + s[None, :]) &
                    (rows_abs[:, None] < base + e[None, :])).any(axis=0)
        else:
            hits = np.zeros(n_win, dtype=bool)
        if bool(hits.any()):
            starts_per_trip.append((int(t), s[hits], e[hits]))
    if not starts_per_trip:
        raise SpaceMismatch("掩码 %s 内没有任何窗口含有效行" % label)
    N = int(sum(len(s) for _, s, _ in starts_per_trip))

    X = np.full((N, L, F), float(pad_value), dtype=np.float64)
    seq_mask = np.zeros((N, L), dtype=bool)
    pos2row = np.full((N, L), -1, dtype=np.int64)
    trip_ids = np.zeros(N, dtype=np.int64)
    lengths = np.zeros(N, dtype=np.int64)
    offsets = np.zeros(N + 1, dtype=np.int64)
    truncated = np.zeros(len(ds.trip_files), dtype=bool)
    truncation_loss_rows = 0
    dropped_tail_rows = 0

    w = 0
    for t, s_arr, e_arr in starts_per_trip:
        base = int(ds.trip_offsets[t])
        n_here = int(ds.trip_nseg[t])
        covered = np.zeros(n_here, dtype=bool)
        for s, e in zip(s_arr.tolist(), e_arr.tolist()):
            gl = base + np.arange(s, e, dtype=np.int64)
            sel = m[gl]
            if not bool(sel.any()):
                continue
            k = gl.size
            trip_ids[w] = t
            pos2row[w, :k] = gl
            seq_mask[w, :k] = sel
            X[w, :k, :] = Xc_f[gl]
            covered[np.arange(s, e)[sel]] = True
            lengths[w] = int(sel.sum())
            w += 1
            offsets[w] = offsets[w - 1] + L
        lost = int(n_here - covered.sum())
        dropped_tail_rows += lost
        if lost:
            truncated[t] = True
            truncation_loss_rows += lost
    if w != N:
        raise SpaceMismatch("窗口计数不自洽：实际写入 %d ≠ 预算 %d" % (w, N))

    coverage = np.bincount(pos2row[seq_mask], minlength=len(ds)).astype(np.int64) \
        if bool(seq_mask.any()) else np.zeros(len(ds), dtype=np.int64)

    if scaler is not None:
        mu = np.asarray(scaler["mu"], dtype=np.float64)
        sd = np.asarray(scaler["sd"], dtype=np.float64)
        assert_shape(mu, F, "scaler.mu")
        assert_shape(sd, F, "scaler.sd")
        if not bool(np.all(sd > 0)):
            raise SpaceMismatch("scaler.sd 必须全为正（零方差列会让 z-score 变成 inf）")
        X = (X - mu) / sd
        X[~seq_mask] = float(pad_value)

    seq = Sequences(X=X.astype(np.float32), mask=seq_mask, trip_ids=trip_ids,
                    pos2row=pos2row, offsets=offsets, lengths=lengths,
                    windows_per_trip=np.array([len(s) for _, s, _ in starts_per_trip],
                                              dtype=np.int64),
                    stride=stride, L=L, features=feats, ds=ds, coverage=coverage,
                    truncated=truncated,
                    selected=mask if isinstance(mask, SplitMask) else None,
                    dropped_tail_rows=dropped_tail_rows, gap_rows=gap_rows,
                    truncation_loss_rows=truncation_loss_rows)
    seq.meta = {
        "split": label, "rows_in_mask": rows_in_mask,
        "rows_covered": int((coverage > 0).sum()),
        "rows_not_covered": int(rows_in_mask - int((coverage > 0).sum())),
        "bad_feature_rows_in_mask": bad_in_mask,
        "max_gap_rows": max_gap_rows, "fillna": fillna,
        "scaler": None if scaler is None else {"mu": np.asarray(scaler["mu"]).tolist(),
                                               "sd": np.asarray(scaler["sd"]).tolist()},
    }
    return seq


# =========================================================================== mask-aware loss / 聚合
@dataclass
class LossReport:
    """一次 mask-aware 损失的结果（把分母是多少永远摆在脸上）。"""

    value: float
    n_rows: int            # 参与计算的非 padding 行数（= mask.sum()）
    n_cells: int           # 全部格子数（含 padding）—— 暴露忘了乘 mask
    n_rows_canonical: int  # 去重后的规范行数（重叠窗口下 < n_rows）
    label: str = ""

    def __str__(self) -> str:
        return ("%s=%.6f (n_rows=%s/%s cells, canonical=%s)"
                % (self.label or "loss", self.value, format(self.n_rows, ","),
                   format(self.n_cells, ","), format(self.n_rows_canonical, ",")))


def _flat_mask_of(seq: Sequences, weight: np.ndarray | None) -> np.ndarray:
    fm = seq.flat_mask.astype(np.float64)
    if weight is None:
        return fm
    w = np.asarray(weight, dtype=np.float64)
    assert_shape(w, seq.mask.size, "weight",
                 context="逐格权重必须是 (n_windows*L,)；要按规范行归一时用 "
                         "seq.flat_weights()")
    return fm * w


def _check_loss_args(seq: Sequences, pred: np.ndarray, target: np.ndarray,
                     label: str) -> tuple[np.ndarray, np.ndarray]:
    """形状契约：``pred``/``target`` 必须是 ``(N,L)`` **或** flat ``(N*L,)``；一律返回 flat。

    两条纪律各治一类事故：

    * ``(N,L,1)`` 这类多一维形状 → ``ShapeContractError``（静默广播入口，签名 ④）；
    * 尺寸等于 ``mask.sum()``（去 padding 的子集）→ ``SpaceMismatch``（签名 ①）。
    """
    want2d = seq.flat_shape()
    n = seq.n_cells
    flat = []
    for nm, arr in (("pred", pred), ("target", target)):
        a = np.asarray(arr)
        if a.shape == want2d:
            pass
        elif a.shape == (n,):
            pass
        elif a.size == n and a.size != int(seq.mask.sum()):
            raise ShapeContractError(
                "%s: %s 形状 %s 与序列空间 %s 不符（元素数相同但形状不同）。\n"
                "  ⇒ 这正是**静默广播陷阱**的形状家族（size 对得上、shape 对不上）："
                "旧的 torch 代码只给一条 UserWarning 然后照算。请显式 reshape。"
                % (label, nm, a.shape, want2d))
        else:
            raise SpaceMismatch(
                "%s: %s 形状 %s 与序列空间 %s 不符（该空间 flat 尺寸 = %s = n_windows×L，"
                "**含 padding**）。\n"
                "  ⇒ 若 %s 的长度是 %s（= mask.sum()，即去 padding 后的有效格数），"
                "你手里是 **seq-subset** 空间而不是 **seq** 空间 —— 这就是"
                "「canonical 201,271 vs seq-subset 104,639」错位的标准形态。\n"
                "     正解：用 seq.embed(规范数组) 或 seq.flat_mask 索引回去。"
                % (label, nm, a.shape, want2d, format(n, ","),
                   nm, format(int(seq.mask.sum()), ",")))
        flat.append(a.astype(np.float64).reshape(-1))
    if flat[0] is flat[1]:
        raise SpaceMismatch("%s: pred 与 target 是同一个对象 —— 检查是否漏了反变换" % label)
    return flat[0], flat[1]


def masked_mse(seq: Sequences, pred: np.ndarray, target: np.ndarray, *,
               weight: np.ndarray | None = None, label: str = "masked_mse") -> LossReport:
    """**mask-aware** MSE：padding 恒不贡献（分母只数被 mask 的格子）。

    ``pred``/``target`` 必须与 ``seq.mask`` 同形（``(N,L)``；也接受 flat ``(N*L,)``）——
    一律过 ``assert_shape``，**不允许** ``(N,L,1)`` 这类多一维的形状（静默广播入口）。
    """
    yp, yt = _check_loss_args(seq, pred, target, label)
    w = _flat_mask_of(seq, weight)
    den = float(w.sum())
    if den <= 0:
        raise SpaceMismatch("%s: mask 求和为 0 —— 没有有效样本，loss 无定义" % label)
    # 一律在 **flat 空间**上算：权重天然是 (N*L,)，二维入参先展平
    # （不展平的话 numpy 会用 (N,L) 与 (N*L,) 做广播并直接报错/静默错算）
    val = float((((yp.reshape(-1) - yt.reshape(-1)) ** 2) * w).sum() / den)
    rows = seq.pos2row.ravel()[seq.flat_mask]
    return LossReport(val, int(seq.flat_mask.sum()), int(seq.flat_mask.size),
                      int(np.unique(rows).size), label)


def masked_abs(seq: Sequences, pred: np.ndarray, target: np.ndarray, *,
               weight: np.ndarray | None = None, label: str = "masked_abs") -> LossReport:
    """**mask-aware** MAE（同 ``masked_mse`` 的形状与掩码纪律）。"""
    yp, yt = _check_loss_args(seq, pred, target, label)
    w = _flat_mask_of(seq, weight)
    den = float(w.sum())
    if den <= 0:
        raise SpaceMismatch("%s: mask 求和为 0 —— 没有有效样本" % label)
    val = float((np.abs(yp.reshape(-1) - yt.reshape(-1)) * w).sum() / den)
    rows = seq.pos2row.ravel()[seq.flat_mask]
    return LossReport(val, int(seq.flat_mask.sum()), int(seq.flat_mask.size),
                      int(np.unique(rows).size), label)


def trip_totals(seq: Sequences, pred: np.ndarray, *, weight: np.ndarray | None = None,
                trips: Sequence[int] | None = None) -> pd.DataFrame:
    """按趟聚合序列预测（**padding 不计入**；重叠窗口按 ``weight`` 归一）。

    返回列：``trip_id, file, split, n_seg, n_rows, total``。
    ``n_seg`` 是 trip_index 口径的该趟总段数，``n_rows`` 是本趟实际入窗行数；
    两者不等即说明有截断/间隔 —— 这时**必须**把它写进报告
    （``check_mask_counts`` / ``check_coverage`` 会先替你拦一次）。
    """
    yp = np.asarray(pred)
    if yp.shape == seq.flat_shape():
        yp = yp.reshape(-1)
    if yp.shape != (seq.n_cells,):
        raise SpaceMismatch(
            "trip_totals: pred 形状 %s ≠ 序列空间（%s / %s；flat 尺寸 %s）"
            % (yp.shape, seq.space, seq.flat_space, format(seq.n_cells, ",")))
    yp = yp.astype(np.float64)
    w = _flat_mask_of(seq, weight)
    valid = seq.flat_mask
    if not bool(valid.any()):
        raise SpaceMismatch("trip_totals: 序列空间没有任何有效格（mask 全 0）")
    vv = (yp * w)[valid]
    tt = np.repeat(seq.trip_ids, seq.L)[valid]
    order = np.argsort(tt, kind="stable")
    vv, tt = vv[order], tt[order]
    want = np.unique(tt) if trips is None else np.asarray(list(trips), dtype=np.int64)
    lo = np.searchsorted(tt, want)
    hi = np.searchsorted(tt, want, side="right")
    rows = []
    for t, a, b in zip(want.tolist(), lo.tolist(), hi.tolist()):
        sl = seq.ds.trip_slice(t)
        rows.append({
            "trip_id": int(t),
            "file": str(seq.ds.trip_files[t]),
            "split": str(seq.ds.split[sl][0]),
            "n_seg": int(seq.ds.trip_nseg[t]),
            "n_rows": int(b - a),
            "total": float(vv[a:b].sum()),
        })
    return pd.DataFrame(rows)


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """WMAPE(%) = Σ|ŷ−y| / Σ|y| × 100（趟级口径，与 unified_v3.wmape 一致）。"""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise SpaceMismatch("wmape: y_true%s vs y_pred%s —— 两个口径的行数必须一致"
                            "（inconsistent numbers of samples 家族）"
                            % (y_true.shape, y_pred.shape))
    den = float(np.abs(y_true).sum())
    if den == 0:
        raise SpaceMismatch("wmape: Σ|y_true| = 0，分母为零")
    return float(np.abs(y_pred - y_true).sum() / den * 100.0)


_YFULL: dict[str, float] | None = None


def _yfull_wh() -> dict[str, float]:
    global _YFULL
    if _YFULL is None:
        t = pd.read_csv(TRIP_CSV, usecols=["file", "y_full"])
        _YFULL = dict(zip(t["file"].astype(str), (t["y_full"] * 1000.0).astype(float)))
    return _YFULL


def wmape_by_trip(seq: Sequences, pred: np.ndarray, *, weight: np.ndarray | None = None,
                  y_true_wh: Mapping[str, float] | np.ndarray | None = None,
                  split: str | None = None, y_base: str = "quantized") -> dict:
    """按趟 WMAPE（mask-aware）。

    ``y_base``：

      * ``quantized``（默认 = 公布口径）：``trip_index.y_full``（**kWh、float32 量化**）×1000；
      * ``exact``：``trip_index.sum_seg_wh``（精确 Wh）。

    两个基不可混引（CORRECTIONS 附 6a）：量化基与精确基在趟级最大差 167.2 Wh，会造成
    0.01–0.08 pp 的系统差。返回字典里始终带 ``y_base``，便于随结论一起报。
    """
    tot = trip_totals(seq, pred, weight=weight)
    if split is not None:
        if split not in SPLITS:
            raise SpaceMismatch("wmape_by_trip: split 只允许 %s" % list(SPLITS))
        tot = tot[tot["split"] == split].reset_index(drop=True)
    # ★ 静默陷阱：某趟在序列空间**一行都没进**（整趟被 L 截断/被间隔剔除）时，
    #   它的 total = 0，会被当成"预测 0 Wh"计入 WMAPE —— 这是把"缺样本"当成"预测为零"。
    empty = tot[tot["n_rows"] == 0]
    if len(empty):
        raise SpaceMismatch(
            "wmape_by_trip: %d 趟在序列空间没有任何有效行（例：%s），"
            "其 total=0 会被误当成\"预测 0 Wh\"混进 WMAPE。\n"
            "  ⇒ 检查 L/stride 是否把这些趟整趟截掉了（seq.check_coverage()），"
            "或显式声明只用哪些趟评估。"
            % (len(empty), empty["file"].head(3).tolist()))
    if y_true_wh is None:
        if y_base == "exact":
            t = pd.read_csv(TRIP_CSV, usecols=["file", "sum_seg_wh"])
            mp = dict(zip(t["file"].astype(str), t["sum_seg_wh"].astype(float)))
            tag = "trip_index.sum_seg_wh [精确 Wh 基]"
        else:
            mp = _yfull_wh()
            tag = "trip_index.y_full(kWh,float32)*1000 [量化 y 基]"
        y = tot["file"].map(mp).to_numpy(dtype=np.float64)
    elif isinstance(y_true_wh, Mapping):
        y = tot["file"].map(y_true_wh).to_numpy(dtype=np.float64)
        tag = "custom mapping"
    else:
        y = np.asarray(y_true_wh, dtype=np.float64).reshape(-1)
        if y.shape[0] != len(tot):
            raise SpaceMismatch("wmape_by_trip: y_true 长度 %s ≠ 趟数 %s"
                                % (format(y.shape[0], ","), format(len(tot), ",")))
        tag = "custom array"
    if not bool(np.isfinite(y).all()):
        raise SpaceMismatch("wmape_by_trip: 有 %d 趟的 y_true 缺失（NaN）"
                            % int((~np.isfinite(y)).sum()))
    return {"wmape": wmape(y, tot["total"].to_numpy()), "n_trips": int(len(tot)),
            "y_base": tag, "split": split}


def canonicalize_seq_pred(seq: Sequences, pred: np.ndarray,
                          *, weights: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """flat 序列预测 → ``(规范行号, 规范化预测, 覆盖次数)``，供与 canonical 实现逐位对拍。

    ``weights=True``（默认）时按覆盖次数**平均**（等价于不重叠窗口下的一行一次）；
    ``False`` 时直接求和（会把重叠行算 N 次 —— 只用于显式诊断）。
    """
    yp = np.asarray(pred)
    if yp.shape == seq.flat_shape():
        yp = yp.reshape(-1)
    if yp.shape != (seq.n_cells,):
        raise SpaceMismatch("canonicalize_seq_pred: pred 形状 %s ≠ 序列空间（%s；flat 尺寸 %s）"
                            % (yp.shape, seq.space, format(seq.n_cells, ",")))
    rows = seq.pos2row.ravel()[seq.flat_mask]
    vals = yp[seq.flat_mask].astype(np.float64)
    order = np.argsort(rows, kind="stable")
    rows, vals = rows[order], vals[order]
    uniq, inv, cnt = np.unique(rows, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=vals, minlength=uniq.size)
    return uniq, (sums / cnt if weights else sums), cnt


def manifest(ds: SegDataset | None = None,
             seqs: Mapping[str, Sequences] | None = None) -> dict:
    """给报告用的一页清单：空间名、尺寸、覆盖、截断、间隔账目。"""
    out: dict[str, Any] = {"spaces": {}, "sequences": {}}
    if ds is not None:
        out["spaces"]["canonical"] = ds.manifest()
        out["spaces"]["masks"] = {s: len(ds.mask(s)) for s in SPLITS}
    for k, sq in (seqs or {}).items():
        cov = sq.coverage
        out["sequences"][k] = {
            "space": str(sq.space), "windows": len(sq), "L": sq.L, "F": len(sq.features),
            "stride": sq.stride, "mask_sum": int(sq.mask.sum()),
            "mask_cells": int(sq.mask.size),
            "padding_frac": round(1 - float(sq.mask.sum()) / float(sq.mask.size), 6),
            "covered_rows": int((cov > 0).sum()),
            "overlap": sq.overlap_report(),
            "truncated_trips": int(sq.truncated.sum()),
            "truncation_loss_rows": sq.truncation_loss_rows,
            "gap_rows": sq.gap_rows,
            "bad_feature_rows_in_mask": sq.meta.get("bad_feature_rows_in_mask"),
        }
    return out
