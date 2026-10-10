#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""schema —— 统一框架的 **标准数据契约**（第 1 层，只加不改）。

一句话
======
> 结果文件里**没有索引、也没有单位**。本模块把这两样都变成**拒绝读取**的硬门。

它与既有 ``lib/preds_io.py`` 的分工
===================================
``preds_io`` 守的是**读任意历史结果文件**（CSV/npz，键靠推断、单位靠外部声明，重在
"读得进来"）。本模块守的是**新框架自己产出的标准契约**（键是**固定**的、单位是**写死**
的、口径是**强制声明**的），重在"**写得出去、且写不错**"。

三层契约（每层都有单位与键的显式声明）
======================================

① 趟级预测 ``trip_preds.npz`` / `.csv`（**主交付格式**）
    ``file`` (str) · ``split`` (str) · ``y_wh`` (float64, **Wh**) ·
    ``pred_wh`` (float64, **Wh**) · ``phi`` (float64, 可选, 前缀比例) ·
    口径元数据 ``info_condition`` / ``target_caliber`` / ``units``

② 段级预测 ``seg_preds.npz``
    ``file`` (str) · ``seg_idx`` (int64) · ``split`` · ``y_wh`` (**Wh**) ·
    ``pred_wh`` (**Wh**) —— 键 = ``(file, seg_idx)``（= canonical 唯一键）

③ 特征矩阵 ``features.npz``
    ``file`` · ``seg_idx`` · ``X`` (float32/float64, ``(n_seg, n_feat)``) ·
    ``feature_names`` · ``split`` · ``units``（**逐特征**单位声明）

强制声明的两个口径维度
======================
* ``info_condition``  ∈ {``plan-time``, ``realized``, ``physics-calibrated``}
* ``target_caliber``  ∈ {``y_full``, ``y_rem@phi``}

这两个维度**必须显式给出**：前者决定"在线信息能否拿到"，后者决定"预测的是哪一段能量"。
把不同取值的臂排进同一张榜 = **混排**（本会话已见：realized 臂与 plan-time 臂被放在
同一张表里比较）。本模块的 ``check_homogeneous`` 就是拦这个的。

单位纪律（Wh vs kWh）
=====================
``Wh`` 是**唯一内部量纲**。任何 kWh 输入必须在**门口**乘 1000 并留痕（``to_wh``）；
禁止"猜测列的量纲"。``assert_unit_ratio`` 用 ``Σp/Σy ∈ [0.5, 2]`` 拦 k×1000 / ÷1000
这一类量纲崩坏（与既有 ``preds_io.assert_units``、``BB_pooled`` 的门禁同口径）。

红线
====
只读 train/validation/future；``split`` 出现 ``calibration`` 一律抛（本模块的 ``_check_split``）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "SchemaError", "UnitError", "CaliberError", "KeyOrderError",
    "INFO_CONDITIONS", "TARGET_CALIBERS", "UNIT_WH", "UNIT_KWH", "UNITS",
    "GE1KWH", "UNIT_LO", "UNIT_HI",
    "FieldSpec", "Schema",
    "TRIP_PREDS", "SEG_PREDS", "FEATURES",
    "TripPreds", "SegPreds", "FeatureMatrix",
    "to_wh", "assert_unit_ratio", "check_homogeneous", "phi_label",
    "make_trip_preds", "make_seg_preds", "meta_block", "read_meta",
]

# --------------------------------------------------------------------------- 口径常量
INFO_CONDITIONS = ("plan-time", "realized", "physics-calibrated")
TARGET_CALIBERS = ("y_full", "y_rem@phi")

UNIT_WH = "Wh"
UNIT_KWH = "kWh"
UNITS = (UNIT_WH, UNIT_KWH)
KWH_TO_WH = 1000.0

GE1KWH = 1000.0                  # MAPE(y≥1kWh) 的门槛（Wh）
UNIT_LO, UNIT_HI = 0.5, 2.0       # Σp/Σy 的合法带（与 preds_io / BB_pooled 同口径）

# 契约要求的**元数据槽位**（不算"多余键"）：单位 / 口径两维 / 自由元数据
META_KEYS = frozenset({"units", "info_condition", "target_caliber", "meta", "y_basis",
                       "y_full_wh", "phi_label", "model"})

# y 基（**唯一合法来源**：trip_index.y_full 为 kWh、float32 量化；×1000 得 Wh，不取整）
Y_BASIS_MAIN = "trip_index.y_full(kWh, float32 量化) × 1000 = Wh（不取整）"
Y_BASIS_EXACT = "Σ seg_energy_wh（≡ trip_index.sum_seg_wh，精确 Wh）"


# --------------------------------------------------------------------------- 异常
class SchemaError(ValueError):
    """契约违反（缺键 / 多键 / 形状不符 / 非有限值）—— 消息里必须给出可用键与实得键。"""


class UnitError(SchemaError):
    """单位不合法（未声明 / 不在白名单 / 量纲比越界）—— 拦截 kWh 冒充 Wh。"""


class CaliberError(SchemaError):
    """口径不合法或**混排**（info_condition / target_caliber 缺省、取值非法、同表不一致）。"""


class KeyOrderError(SchemaError):
    """键序不符（行序即身份：结果文件里没有索引，所以"第 i 行"必须被钉死）。"""


# --------------------------------------------------------------------------- 字段规格
@dataclass(frozen=True)
class FieldSpec:
    """一个字段的完整契约：名字 + 类型 + **单位** + 可空 + 语义。

    ``unit=None`` 表示"不是物理量"（如 file / split / 索引），**不是**"单位不重要"。
    """

    name: str
    dtype: str                       # "str" | "int64" | "float64" | "float32" | "object"
    unit: str | None = None
    required: bool = True
    note: str = ""

    def cast(self, v: Any) -> np.ndarray:
        if self.dtype == "str":
            return np.asarray(v).astype(str)
        if self.dtype == "object":
            return np.asarray(v, dtype=object)
        if self.dtype == "int64":
            a = np.asarray(v)
            if a.dtype.kind == "f" and not np.all(np.isfinite(a)):
                raise SchemaError("字段 %r 期望 int64，实得含非有限值的浮点" % self.name)
            return a.astype(np.int64)
        if self.dtype in ("float64", "float32"):
            a = np.asarray(v, dtype=np.float64)
            return a.astype(np.float32) if self.dtype == "float32" else a
        raise SchemaError("未知 dtype %r（字段 %s）" % (self.dtype, self.name))


@dataclass(frozen=True)
class Schema:
    """一张标准表的契约：键 + 字段 + 单位 + 口径维度。"""

    name: str
    key: tuple[str, ...]
    fields: tuple[FieldSpec, ...]
    caliber_dims: tuple[str, ...] = ("info_condition", "target_caliber")
    note: str = ""

    # ---------------------------------------------------------------- 查询
    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields)

    @property
    def required(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields if f.required)

    def spec(self, name: str) -> FieldSpec:
        for f in self.fields:
            if f.name == name:
                return f
        raise SchemaError("schema %r 无字段 %r；可用：%s" % (self.name, name, list(self.names)))

    def unit_of(self, name: str) -> str | None:
        return self.spec(name).unit

    def units(self, names: Iterable[str] | None = None) -> dict[str, str | None]:
        want = set(names) if names is not None else set(self.names)
        return {f.name: f.unit for f in self.fields if f.name in want}

    # ---------------------------------------------------------------- 校验
    def check_keys(self, data: Mapping[str, Any], *, allow_extra: bool = True,
                   where: str = "") -> None:
        miss = [k for k in self.required if k not in data]
        if miss:
            raise SchemaError(
                "schema %r 缺必需键 %s%s\n  实得键：%s\n  提示：标准契约的键是**固定**的；"
                "缺键不许猜、不许按位置补 —— 请用 make_%s() 生成。"
                % (self.name, miss, ("（%s）" % where) if where else "", sorted(data), self.name))
        if not allow_extra:
            # 只对**数据字段**判多键；units / info_condition / target_caliber / meta 是
            # 契约要求的**元数据槽位**（见 META_KEYS），不算多余。
            extra = [k for k in data if k not in self.names and k not in META_KEYS]
            if extra:
                raise SchemaError("schema %r 出现未登记键 %s（允许清单：%s）"
                                  % (self.name, extra, list(self.names) + sorted(META_KEYS)))

    def check_units_declared(self, data: Mapping[str, Any]) -> dict[str, str]:
        """**单位必须显式**：声明缺失或不在白名单 → 抛。

        声明位置：``data['units']`` 为 ``{字段名: 单位}`` 的映射（逐列）。
        """
        if "units" not in data:
            raise UnitError(
                "schema %r 缺 'units' 声明 —— 结果文件不带单位，**必须外部显式声明**。\n"
                "  本 schema 里带量纲的字段：%s\n"
                "  正解：units={'%s': 'Wh', ...}"
                % (self.name,
                   {f.name: f.unit for f in self.fields if f.unit},
                   next(f.name for f in self.fields if f.unit)))
        u = data["units"]
        if not isinstance(u, Mapping):
            raise UnitError("schema %r 的 units 必须是 {字段名: 单位} 映射，实得 %r"
                            % (self.name, type(u).__name__))
        for f in self.fields:
            if f.unit is None:
                continue
            got = u.get(f.name)
            if got is None:
                raise UnitError("schema %r 字段 %r 未声明单位（应为 %r）—— 拒绝读取"
                                % (self.name, f.name, f.unit))
            if str(got) not in UNITS:
                raise UnitError("schema %r 字段 %r 的单位 %r 不在白名单 %s"
                                % (self.name, f.name, got, list(UNITS)))
            if str(got) != f.unit:
                raise UnitError(
                    "schema %r 字段 %r 的单位声明 %r ≠ 契约要求的 %r。\n"
                    "  ⇒ 标准契约只收 %s；kWh 请在**门口**乘 1000 后声明 Wh（schema.to_wh）。"
                    % (self.name, f.name, got, f.unit, f.unit))
        return {k: str(v) for k, v in u.items()}

    def check_caliber(self, data: Mapping[str, Any], *, required: bool = True) -> dict[str, str]:
        """**口径两维必须显式声明**（缺省即抛；取值非法即抛）。"""
        out: dict[str, str] = {}
        for dim in self.caliber_dims:
            v = data.get(dim)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                if required:
                    raise CaliberError(
                        "schema %r 未声明 %r —— 该维度**强制声明**（防止不同口径的臂混排）。\n"
                        "  合法取值：%s"
                        % (self.name, dim,
                           INFO_CONDITIONS if dim == "info_condition" else TARGET_CALIBERS))
                continue
            v = str(v)
            allowed = INFO_CONDITIONS if dim == "info_condition" else TARGET_CALIBERS
            if v not in allowed:
                raise CaliberError("schema %r 的 %s=%r 非法；合法取值 %s"
                                   % (self.name, dim, v, list(allowed)))
            out[dim] = v
        return out

    def check_split(self, arr: Any, where: str = "split") -> np.ndarray:
        a = np.asarray(arr).astype(str)
        bad = sorted(set(a.tolist()) - {"train", "validation", "future"})
        if bad:
            raise SchemaError(
                "%s 出现未授权取值 %s —— 只允许 train/validation/future。\n"
                "  ★ 出现 calibration 即违反红线（本框架任何一层都不读 calibration 标签）。"
                % (where, bad))
        return a

    def check_lengths(self, data: Mapping[str, Any], where: str = "") -> int:
        lens = {k: int(np.shape(v)[0]) for k, v in data.items()
                if k not in ("units", "meta", "feature_names", "info_condition",
                             "target_caliber", "phi_label") and np.ndim(v) >= 1}
        if not lens:
            raise SchemaError("schema %r 没有任何数组字段%s" % (self.name, where))
        uniq = sorted(set(lens.values()))
        if len(uniq) != 1:
            raise SchemaError(
                "schema %r 各字段长度不一致%s：%s\n"
                "  ⇒ 这正是 'Found input variables with inconsistent numbers of samples' 家族；"
                "一律按**键**对齐，不许按位置补齐。"
                % (self.name, ("（%s）" % where) if where else "",
                   {k: v for k, v in lens.items()}))
        return uniq[0]

    def check_finite(self, data: Mapping[str, Any], names: Sequence[str] | None = None) -> None:
        for nm in (names or [f.name for f in self.fields
                             if f.dtype.startswith("float")]):
            a = np.asarray(data[nm], dtype=np.float64)
            n_bad = int((~np.isfinite(a)).sum())
            if n_bad:
                raise SchemaError("schema %r 字段 %r 含 %d 个非有限值（NaN/Inf）—— 拒绝静默丢行"
                                  % (self.name, nm, n_bad))

    def validate(self, data: Mapping[str, Any], *, where: str = "",
                 allow_extra: bool = False) -> "Schema":
        self.check_keys(data, allow_extra=allow_extra, where=where)
        self.check_units_declared(data)
        self.check_caliber(data)
        self.check_lengths(data, where=where)
        if "split" in self.names:
            self.check_split(data["split"], where="%s.split" % (self.name if not where else where))
        return self


# --------------------------------------------------------------------------- 三张契约
TRIP_PREDS = Schema(
    name="trip_preds",
    key=("file",),
    fields=(
        FieldSpec("file", "str", None, True, "趟键（唯一）"),
        FieldSpec("split", "str", None, True, "train/validation/future"),
        FieldSpec("y_wh", "float64", UNIT_WH, True, "趟级真值（Wh）—— y 基必须随行声明"),
        FieldSpec("pred_wh", "float64", UNIT_WH, True, "趟级预测（Wh）"),
        FieldSpec("phi", "float64", None, False, "前缀比例 φ（target_caliber=y_rem@phi 时必填）"),
    ),
    note="趟级预测主交付格式；键 = file；单位 Wh。",
)

SEG_PREDS = Schema(
    name="seg_preds",
    key=("file", "seg_idx"),
    fields=(
        FieldSpec("file", "str", None, True, "趟键"),
        FieldSpec("seg_idx", "int64", None, True, "段内序号（与 canonical 同源）"),
        FieldSpec("split", "str", None, True, "train/validation/future"),
        FieldSpec("y_wh", "float64", UNIT_WH, True, "段级真值（Wh）"),
        FieldSpec("pred_wh", "float64", UNIT_WH, True, "段级预测（Wh）"),
    ),
    note="段级预测；键 = (file, seg_idx) = canonical 唯一键。",
)

FEATURES = Schema(
    name="features",
    key=("file", "seg_idx"),
    fields=(
        FieldSpec("file", "str", None, True, "趟键"),
        FieldSpec("seg_idx", "int64", None, True, "段内序号"),
        FieldSpec("split", "str", None, True, "train/validation/future"),
        FieldSpec("X", "float32", None, True, "(n_seg, n_feat)；逐特征单位见 units"),
        FieldSpec("feature_names", "object", None, True, "长度 = n_feat 的列名（**顺序即口径**）"),
    ),
    note="特征矩阵；禁止按列位置取数（列序变过），一律按 feature_names。",
)


# --------------------------------------------------------------------------- 单位工具
def to_wh(values: Any, unit: str, *, name: str = "values") -> np.ndarray:
    """把显式声明单位的数组换算到 **Wh**（唯一内部量纲）。

    ``unit`` **必须**给：不猜、不自动。返回新数组，不改入参。
    """
    u = str(unit)
    if u not in UNITS:
        raise UnitError("%s 的单位 %r 不在白名单 %s —— 拒绝猜测量纲" % (name, u, list(UNITS)))
    a = np.asarray(values, dtype=np.float64)
    return a * KWH_TO_WH if u == UNIT_KWH else a


def assert_unit_ratio(pred_wh: Any, y_wh: Any, *, lo: float = UNIT_LO, hi: float = UNIT_HI,
                      scope: str = "all", name: str = "pred") -> float:
    """``Σp/Σy ∈ [lo, hi]`` 单位门（**越界即抛**，消息给出换算建议）。

    作用域 ``scope`` 只影响报错文案；分母**必须与分子同一作用域**（本会话已见"分母作用域"
    事故：拿全量 Σy 当某一划分的分母）。
    """
    p = np.asarray(pred_wh, dtype=np.float64).reshape(-1)
    y = np.asarray(y_wh, dtype=np.float64).reshape(-1)
    if p.shape != y.shape:
        raise SchemaError("assert_unit_ratio: pred%s vs y%s 长度不一致（%s 作用域）"
                          % (p.shape, y.shape, scope))
    den = float(np.abs(y).sum())
    if den <= 0:
        raise UnitError("assert_unit_ratio: Σ|y| = 0（%s 作用域）—— 单位门无定义" % scope)
    r = float(np.abs(p).sum() / den)
    if not (lo <= r <= hi):
        hint = ""
        if r < lo / 4:
            hint = "（Σp/Σy≈%.4g：预测疑为 **kWh 而 y 是 Wh**，差 1000×）" % r
        elif r > hi * 4:
            hint = "（Σp/Σy≈%.4g：预测疑为 **Wh 而 y 是 kWh**，差 1000×）" % r
        raise UnitError(
            "单位门失败（%s，作用域=%s）：Σp/Σy = %.6g 越出 [%g, %g] %s\n"
            "  正解：在门口用 schema.to_wh(values, 'kWh') 显式换算，并逐列传 units。"
            % (name, scope, r, lo, hi, hint))
    return r


def phi_label(target_caliber: str) -> str:
    """``y_rem@0.6`` → ``'0.6'``（用于分组键与文件名）；``y_full`` → ``'full'``。"""
    s = str(target_caliber)
    if s == "y_full":
        return "full"
    if s.startswith("y_rem@"):
        return s.split("@", 1)[1]
    raise CaliberError("无法解析 target_caliber=%r（合法：%s）" % (s, list(TARGET_CALIBERS)))


def _caliber_ok(dim: str, value: str) -> bool:
    """口径取值合法性：``target_caliber`` 允许 ``y_rem@<任意 φ>``（φ 网格不写死在这里）。"""
    if dim != "target_caliber":
        return value in INFO_CONDITIONS
    if value in TARGET_CALIBERS:
        return True
    if value.startswith("y_rem@"):
        try:
            float(value.split("@", 1)[1])
            return True
        except ValueError:
            return False
    return False


def normalize_caliber(dim: str, value: str) -> str:
    """把 ``y_rem@0.6`` 归一到 ``y_rem@phi``（**同一族**）—— 用于"同口径"比较与分组。

    ★ 为什么需要它：``target_caliber`` 的取值域里写的是 ``y_rem@phi``，
      但实际数据带的是具体 φ（``y_rem@0.2`` / ``y_rem@0.6`` …）。
      **逐条校验**必须接受具体 φ；而**判断"是不是同一口径族"**时必须把它们归一，
      否则同一族的两个臂会被误判成"混排"。
    """
    v = str(value)
    if dim == "target_caliber" and v.startswith("y_rem@"):
        return "y_rem@phi"
    return v


def check_homogeneous(rows: Sequence[Mapping[str, Any]], *,
                      dims: Sequence[str] = ("info_condition", "target_caliber"),
                      where: str = "榜单") -> dict[str, str]:
    """**防混排**：同一张榜 / 同一张表里，口径维度必须唯一。

    ``rows`` 是若干条记录的元数据（dict）。返回该表的唯一口径组合；
    出现两个不同取值 → ``CaliberError`` 并列出各自的臂名（若有 ``arm``/``name`` 字段）。
    """
    out: dict[str, str] = {}
    for dim in dims:
        vals: dict[str, list[str]] = {}
        for i, r in enumerate(rows):
            raw = r.get(dim)
            if raw is None:
                raise CaliberError(
                    "%s 第 %d 条记录缺 %s —— 该维度**强制声明**（缺省即视为混排风险，直接拒绝）"
                    % (where, i, dim))
            raw = str(raw)
            allowed = INFO_CONDITIONS if dim == "info_condition" else TARGET_CALIBERS
            if not _caliber_ok(dim, raw):
                raise CaliberError("%s 第 %d 条的 %s=%r 非法；合法取值 %s"
                                   % (where, i, dim, raw, list(allowed)))
            v = normalize_caliber(dim, raw)
            tag = str(r.get("arm") or r.get("name") or r.get("model") or ("#%d" % i))
            vals.setdefault(v, []).append(tag)
        if len(vals) > 1:
            detail = "；".join("%s ← %s" % (k, ", ".join(v[:6])) for k, v in sorted(vals.items()))
            raise CaliberError(
                "口径混排（%s，维度 %s）：同表出现 %d 个不同取值 —— %s\n"
                "  ⇒ 不同 %s 的臂**不可排进同一张榜**（如 plan-time 与 realized 的输入信息量不同，"
                "数字不可直接比）。\n  正解：按 %s 分组各出一张榜。"
                % (where, dim, len(vals), detail, dim, dim))
        out[dim] = next(iter(vals))
    return out


# --------------------------------------------------------------------------- 容器
@dataclass
class TripPreds:
    """趟级预测 + 口径元数据（标准契约对象）。"""

    file: np.ndarray
    split: np.ndarray
    y_wh: np.ndarray
    pred_wh: np.ndarray
    phi: np.ndarray | None = None
    info_condition: str = ""
    target_caliber: str = ""
    units: dict[str, str] = field(default_factory=dict)
    y_basis: str = Y_BASIS_MAIN
    model: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.file = np.asarray(self.file).astype(str)
        self.split = np.asarray(self.split).astype(str)
        self.y_wh = np.asarray(self.y_wh, dtype=np.float64)
        self.pred_wh = np.asarray(self.pred_wh, dtype=np.float64)
        self.phi = None if self.phi is None else np.asarray(self.phi, dtype=np.float64)
        if not self.units:
            self.units = {"y_wh": UNIT_WH, "pred_wh": UNIT_WH}
        n = len(self.file)
        for nm, a in (("split", self.split), ("y_wh", self.y_wh), ("pred_wh", self.pred_wh)):
            if len(a) != n:
                raise SchemaError("TripPreds 字段 %s 长度 %d ≠ file 长度 %d" % (nm, len(a), n))
        if self.phi is not None and len(self.phi) != n:
            raise SchemaError("TripPreds.phi 长度 %d ≠ file 长度 %d" % (len(self.phi), n))
        dup = pd.Index(self.file).duplicated()
        if bool(dup.any()):
            raise SchemaError("TripPreds.file 不唯一：%d 个重复，例如 %s"
                              % (int(dup.sum()), self.file[dup][:5].tolist()))
        if self.target_caliber == "y_full" and self.phi is not None:
            raise CaliberError("target_caliber='y_full' 时不应带 phi（实得 %d 个）—— 口径自相矛盾"
                               % len(self.phi))
        if str(self.target_caliber).startswith("y_rem@") and self.phi is None:
            raise CaliberError("target_caliber=%r 必须带 phi（切点比例）" % self.target_caliber)

    # ---------------------------------------------------------------- 视图
    def mask(self, split: str) -> np.ndarray:
        return self.split == str(split)

    def select(self, split: str) -> "TripPreds":
        m = self.mask(split)
        return TripPreds(file=self.file[m], split=self.split[m], y_wh=self.y_wh[m],
                         pred_wh=self.pred_wh[m],
                         phi=None if self.phi is None else self.phi[m],
                         info_condition=self.info_condition,
                         target_caliber=self.target_caliber, units=dict(self.units),
                         y_basis=self.y_basis, model=self.model, extra=dict(self.extra))

    @property
    def caliber(self) -> dict[str, str]:
        return {"info_condition": self.info_condition, "target_caliber": self.target_caliber}

    # ---------------------------------------------------------------- 校验
    def validate(self) -> "TripPreds":
        data: dict[str, Any] = {"file": self.file, "split": self.split, "y_wh": self.y_wh,
                                "pred_wh": self.pred_wh, "units": self.units,
                                "info_condition": self.info_condition,
                                "target_caliber": self.target_caliber}
        if self.phi is not None:
            data["phi"] = self.phi
        TRIP_PREDS.validate(data, where="TripPreds(%s)" % (self.model or "?"))
        if bool((self.y_wh <= 0).any()):
            n = int((self.y_wh <= 0).sum())
            raise SchemaError("TripPreds.y_wh 有 %d 个 ≤ 0 的趟 —— MAPE 系列定义不成立（拒绝静默）"
                              % n)
        assert_unit_ratio(self.pred_wh, self.y_wh, scope="all", name="trip_preds")
        return self

    # ---------------------------------------------------------------- 序列化
    def to_npz(self, path: Any, *, compress: bool = True) -> Any:
        """按固定键序落盘（**键序 = 行序 = 身份**）；口径元数据随行。

        注意：``file`` 是 ``np.savez`` 的**形参名**，直接当 kwarg 会与之冲突
        （``TypeError: savez_compressed() got multiple values for argument 'file'``），
        所以数组名统一用 ``file_``，``from_npz`` 再映射回 ``file``。
        """
        import json
        from pathlib import Path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {"file_": self.file.astype(object), "split": self.split,
                                  "y_wh": self.y_wh, "pred_wh": self.pred_wh}
        if self.phi is not None:
            arrays["phi"] = self.phi
        arrays["meta"] = np.array(json.dumps(self.meta_block(), ensure_ascii=False))
        (np.savez_compressed if compress else np.savez)(p, **arrays)
        return p

    def meta_block(self) -> dict[str, Any]:
        return meta_block(info_condition=self.info_condition,
                          target_caliber=self.target_caliber, units=self.units,
                          y_basis=self.y_basis, model=self.model,
                          n=len(self.file), extra=self.extra)

    @classmethod
    def from_npz(cls, path: Any) -> "TripPreds":
        z = np.load(path, allow_pickle=True)
        m = read_meta(z)
        fkey = "file_" if "file_" in z.files else "file"
        return cls(file=z[fkey], split=z["split"], y_wh=z["y_wh"], pred_wh=z["pred_wh"],
                   phi=(z["phi"] if "phi" in z.files else None),
                   info_condition=m["info_condition"], target_caliber=m["target_caliber"],
                   units=m["units"], y_basis=m.get("y_basis", Y_BASIS_MAIN),
                   model=m.get("model", ""), extra=m.get("extra", {}))

    def to_frame(self) -> pd.DataFrame:
        d = {"file": self.file, "split": self.split, "y_wh": self.y_wh, "pred_wh": self.pred_wh}
        if self.phi is not None:
            d["phi"] = self.phi
        return pd.DataFrame(d)

    def __len__(self) -> int:
        return int(len(self.file))

    def __repr__(self) -> str:
        return ("TripPreds(n=%d, model=%r, %s/%s, Σy=%.1f Wh)"
                % (len(self), self.model, self.info_condition, self.target_caliber,
                   float(self.y_wh.sum())))


@dataclass
class SegPreds:
    """段级预测 + 口径元数据（标准契约对象）。"""

    file: np.ndarray
    seg_idx: np.ndarray
    split: np.ndarray
    y_wh: np.ndarray
    pred_wh: np.ndarray
    info_condition: str = ""
    target_caliber: str = "y_full"
    units: dict[str, str] = field(default_factory=dict)
    model: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.file = np.asarray(self.file).astype(str)
        self.seg_idx = np.asarray(self.seg_idx, dtype=np.int64)
        self.split = np.asarray(self.split).astype(str)
        self.y_wh = np.asarray(self.y_wh, dtype=np.float64)
        self.pred_wh = np.asarray(self.pred_wh, dtype=np.float64)
        if not self.units:
            self.units = {"y_wh": UNIT_WH, "pred_wh": UNIT_WH}
        n = len(self.file)
        for nm, a in (("seg_idx", self.seg_idx), ("split", self.split),
                      ("y_wh", self.y_wh), ("pred_wh", self.pred_wh)):
            if len(a) != n:
                raise SchemaError("SegPreds 字段 %s 长度 %d ≠ file 长度 %d" % (nm, len(a), n))
        k = pd.DataFrame({"file": self.file, "seg_idx": self.seg_idx})
        dup = k.duplicated()
        if bool(dup.any()):
            raise SchemaError("SegPreds 键 (file, seg_idx) 不唯一：%d 个重复" % int(dup.sum()))

    def validate(self) -> "SegPreds":
        SEG_PREDS.validate({"file": self.file, "seg_idx": self.seg_idx, "split": self.split,
                            "y_wh": self.y_wh, "pred_wh": self.pred_wh, "units": self.units,
                            "info_condition": self.info_condition,
                            "target_caliber": self.target_caliber}, where="SegPreds")
        assert_unit_ratio(self.pred_wh, self.y_wh, scope="all", name="seg_preds")
        return self

    def trip_totals(self, *, key: str = "file") -> pd.DataFrame:
        """段级 → 趟级求和（**只按键分组**，不按位置）。"""
        d = pd.DataFrame({"file": self.file, "split": self.split,
                          "y_wh": self.y_wh, "pred_wh": self.pred_wh})
        g = d.groupby(key, sort=False, as_index=False).agg(
            split=("split", "first"), y_wh=("y_wh", "sum"), pred_wh=("pred_wh", "sum"))
        return g

    def to_npz(self, path: Any) -> Any:
        import json
        from pathlib import Path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # ★ ``file`` 是 np.savez 的形参名 —— 数组名统一用 ``file_``（同 TripPreds.to_npz）
        np.savez_compressed(
            p, file_=self.file.astype(object), seg_idx=self.seg_idx, split=self.split,
            y_wh=self.y_wh, pred_wh=self.pred_wh,
            meta=np.array(json.dumps(meta_block(
                info_condition=self.info_condition, target_caliber=self.target_caliber,
                units=self.units, model=self.model, n=len(self.file), extra=self.extra),
                ensure_ascii=False)))
        return p

    @classmethod
    def from_npz(cls, path: Any) -> "SegPreds":
        z = np.load(path, allow_pickle=True)
        m = read_meta(z)
        fkey = "file_" if "file_" in z.files else "file"
        return cls(file=z[fkey], seg_idx=z["seg_idx"], split=z["split"], y_wh=z["y_wh"],
                   pred_wh=z["pred_wh"], info_condition=m["info_condition"],
                   target_caliber=m["target_caliber"], units=m["units"],
                   model=m.get("model", ""), extra=m.get("extra", {}))

    def __len__(self) -> int:
        return int(len(self.file))


@dataclass
class FeatureMatrix:
    """特征矩阵 + **逐特征**单位声明（禁止按列位置取数）。"""

    file: np.ndarray
    seg_idx: np.ndarray
    split: np.ndarray
    X: np.ndarray
    feature_names: Sequence[str]
    units: dict[str, str | None] = field(default_factory=dict)
    info_condition: str = "plan-time"
    target_caliber: str = "y_full"
    source: str = ""

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X)
        self.feature_names = list(self.feature_names)
        if self.X.ndim != 2 or self.X.shape[1] != len(self.feature_names):
            raise SchemaError("FeatureMatrix.X 形状 %s 与 feature_names 长度 %d 不符"
                              % (self.X.shape, len(self.feature_names)))
        if len(set(self.feature_names)) != len(self.feature_names):
            raise SchemaError("FeatureMatrix.feature_names 有重复列名")
        if not np.isfinite(self.X).all():
            raise SchemaError("FeatureMatrix.X 含非有限值 —— 请在 preprocess 阶段显式填缺")

    def column(self, name: str) -> np.ndarray:
        try:
            return self.X[:, self.feature_names.index(name)]
        except ValueError:
            raise SchemaError("FeatureMatrix 无列 %r；可用：%s"
                              % (name, self.feature_names[:12])) from None

    def select(self, names: Sequence[str]) -> np.ndarray:
        idx = []
        for nm in names:
            if nm not in self.feature_names:
                raise SchemaError("FeatureMatrix 缺列 %r（select 的列必须存在）" % nm)
            idx.append(self.feature_names.index(nm))
        return self.X[:, idx]

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


# --------------------------------------------------------------------------- 元数据
def meta_block(*, info_condition: str, target_caliber: str,
               units: Mapping[str, str], y_basis: str = Y_BASIS_MAIN,
               model: str = "", n: int = 0,
               extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """标准元数据块（随 npz 一起落盘）—— 口径两维 + 单位 + y 基 + 单位门结果。"""
    ic, tc = str(info_condition), str(target_caliber)
    if ic not in INFO_CONDITIONS:
        raise CaliberError("meta_block: info_condition=%r 非法；合法 %s" % (ic, list(INFO_CONDITIONS)))
    if not _caliber_ok("target_caliber", tc):
        raise CaliberError("meta_block: target_caliber=%r 非法；合法 %s"
                           "（y_rem@ 后须是可解析的 φ，如 y_rem@0.2）"
                           % (tc, list(TARGET_CALIBERS)))
    for k, v in units.items():
        if v is not None and str(v) not in UNITS:
            raise UnitError("meta_block: units[%r]=%r 不在白名单 %s" % (k, v, list(UNITS)))
    return {
        "schema": "trip_preds/seg_preds v1",
        "info_condition": ic,
        "target_caliber": tc,
        "phi_label": phi_label(tc),
        "units": {str(k): (None if v is None else str(v)) for k, v in units.items()},
        "y_basis": str(y_basis),
        "model": str(model),
        "n": int(n),
        "generated_by": "lib/{paths,schema,registry,eval_trip,trainer,infer}.py",
        "extra": dict(extra or {}),
        # ★ 键序指纹提到顶层：meta 是"我这份产物是什么"的自证，键序属于其中，
        #   埋在 extra 里会让只看顶层元数据的人以为没有指纹（实测踩到）。
        "key_sha16": (dict(extra or {}).get("key_sha16") or ""),
    }


def read_meta(npz_or_dict: Any) -> dict[str, Any]:
    """从 npz（或已 load 的映射）里读出标准元数据块；缺 'meta' 即抛（口径不明 = 拒绝用）。"""
    import json
    obj = npz_or_dict
    if hasattr(obj, "files"):
        if "meta" not in obj.files:
            raise CaliberError(
                "npz 缺 'meta' 块 —— 口径（info_condition/target_caliber）与单位都不明，拒绝读取。\n"
                "  正解：用 infer.py 产出标准 trip_preds.npz（写入方负责声明）。\n"
                "  实得键：%s" % list(getattr(obj, "files", [])))
        raw = obj["meta"]
    elif isinstance(obj, Mapping) and "meta" in obj:
        raw = obj["meta"]
    else:
        raise CaliberError("read_meta: 入参里没有 'meta'（实得 %r）" % type(obj).__name__)
    if isinstance(raw, np.ndarray) and raw.dtype.kind in "OUS":
        raw = raw.reshape(-1)[0]
    return json.loads(str(raw)) if isinstance(raw, (str, bytes, np.str_)) else dict(raw)


# --------------------------------------------------------------------------- 构造入口
def make_trip_preds(file: Any, split: Any, y_wh: Any, pred_wh: Any, *,
                    info_condition: str, target_caliber: str, phi: Any = None,
                    units: Mapping[str, str] | None = None, model: str = "",
                    y_wh_unit: str = UNIT_WH, pred_wh_unit: str = UNIT_WH,
                    y_basis: str = Y_BASIS_MAIN, validate: bool = True,
                    extra: Mapping[str, Any] | None = None) -> TripPreds:
    """构造标准趟级预测（**单位必须显式**；kWh 会自动换算到 Wh 并留痕）。"""
    u = dict(units or {})
    y = to_wh(y_wh, y_wh_unit, name="y")
    p = to_wh(pred_wh, pred_wh_unit, name="pred")
    u.setdefault("y_wh", UNIT_WH)
    u.setdefault("pred_wh", UNIT_WH)
    tp = TripPreds(file=file, split=split, y_wh=y, pred_wh=p, phi=phi,
                   info_condition=info_condition, target_caliber=target_caliber,
                   units=u, y_basis=y_basis, model=model, extra=dict(extra or {}))
    return tp.validate() if validate else tp


def make_seg_preds(file: Any, seg_idx: Any, split: Any, y_wh: Any, pred_wh: Any, *,
                   info_condition: str, target_caliber: str = "y_full",
                   units: Mapping[str, str] | None = None, model: str = "",
                   y_wh_unit: str = UNIT_WH, pred_wh_unit: str = UNIT_WH,
                   validate: bool = True) -> SegPreds:
    u = dict(units or {})
    u.setdefault("y_wh", UNIT_WH)
    u.setdefault("pred_wh", UNIT_WH)
    sp = SegPreds(file=file, seg_idx=seg_idx, split=split,
                  y_wh=to_wh(y_wh, y_wh_unit, name="y"),
                  pred_wh=to_wh(pred_wh, pred_wh_unit, name="pred"),
                  info_condition=info_condition, target_caliber=target_caliber,
                  units=u, model=model)
    return sp.validate() if validate else sp


def frame_to_trip_preds(df: pd.DataFrame, *, info_condition: str, target_caliber: str,
                        file_col: str = "file", split_col: str = "split",
                        y_col: str = "y_wh", pred_cols: Sequence[str] = ("pred_wh",),
                        phi_col: str | None = "phi", y_unit: str = UNIT_WH,
                        unit_map: Mapping[str, str] | None = None,
                        model: str = "") -> dict[str, TripPreds]:
    """一张宽表 → 每个预测列一个 ``TripPreds``（**按键对齐**；不按位置）。"""
    if file_col not in df.columns or split_col not in df.columns:
        raise SchemaError("frame_to_trip_preds: 缺键列 %r/%r；实得 %s"
                          % (file_col, split_col, list(df.columns)))
    um = dict(unit_map or {})
    out: dict[str, TripPreds] = {}
    for c in pred_cols:
        if c not in df.columns:
            raise SchemaError("frame_to_trip_preds: 预测列 %r 不在表里；可用 %s"
                              % (c, list(df.columns)))
        out[c] = make_trip_preds(
            df[file_col].to_numpy(), df[split_col].to_numpy(),
            df[y_col].to_numpy(float), df[c].to_numpy(float),
            info_condition=info_condition, target_caliber=target_caliber,
            phi=(df[phi_col].to_numpy(float) if (phi_col and phi_col in df.columns) else None),
            model=model or c, y_wh_unit=um.get(y_col, y_unit),
            pred_wh_unit=um.get(c, y_unit))
    return out
