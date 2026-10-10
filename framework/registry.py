#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""registry —— 模型注册表（统一框架第 2 层，只加不改）。

它解决什么
==========
审计实测：全树 ``Registry`` / ``get_model`` / ``def train(`` **0 命中**；``def fit(`` 有
**4 套互斥签名**（``fit(feats)`` / ``fit(X, y)`` / ``fit(cfg, T, seed)`` / ``fit(model)``）。
于是"跑一个模型"这件事没有统一入口：每个模型自带一套调用约定，评估口径也随之漂移。

本模块把"一个模型"收敛成**两个函数 + 两个强制声明的口径维度**：

    register(name, info_condition, target_caliber, train_fn, predict_fn, **info)
    m = get(name); info = list_models()

强制声明的两个维度（**缺省即抛**）
==================================
* ``info_condition`` ∈ {``plan-time``, ``realized``, ``physics-calibrated``}
* ``target_caliber`` ∈ {``y_full``, ``y_rem@phi``}

理由：前者决定"出发前能否拿到这些输入"，后者决定"预测的是全程能量还是切点后的剩余能量"。
两者不同的臂**数字不可直接比** —— 本会话已见 realized 臂与 plan-time 臂被排进同一张榜。
``leaderboard()`` 因此**强制分组**，混排会抛 ``CaliberError``。

**发布树的便利入口**：``register_model(pkg)`` 把"导入一个模型包的 ``register_spec()``"
标准化；各臂的 ``register_spec`` 只写自己的名字/口径/两个函数。

minimal 用法（这就是"新增一个模型"的全部）
=========================================
    def train_fn(ctx):                       # ctx: TrainContext
        coef = fit_something(ctx.data.flat_X, ctx.data.flat_y)
        return ModelHandle(params=coef)
    def predict_fn(handle, ctx=None):
        return {"trip_wh": ...}              # 逐趟 Wh
    register("toy-linear", "plan-time", "y_full", train_fn, predict_fn, version="0.1")

红线
====
只加不改：本模块不 import 任何既有模型模块（避免副作用），只登记调用约定。
本文件**不含 assert**：一切自检走显式 ``raise``。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from schema import (INFO_CONDITIONS, TARGET_CALIBERS, CaliberError, SchemaError,
                    _caliber_ok, normalize_caliber)

__all__ = [
    "ModelSpec", "ModelHandle", "Registry", "RegistryError", "AlreadyRegistered",
    "UnknownModel", "REGISTRY", "register", "get", "list_models", "unregister",
    "describe", "leaderboard", "familes", "validate_caliber", "register_model",
    "load_package", "load_all",
]


class RegistryError(KeyError):
    """注册表错误基类。"""


class AlreadyRegistered(RegistryError):
    """重名注册（默认拒绝；``override=True`` 才覆盖，且**必须**留痕）。"""


class UnknownModel(RegistryError):
    """请求的模型未注册 —— 消息里必须给出**可用模型清单**（不猜、不静默回退）。"""


@dataclass
class ModelHandle:
    """``train_fn`` 的返回值：**训练好的东西** + 可选产物路径 + 元数据。

    ``params`` 是任意模型自己的载荷（dict / ndarray / torch state_dict / 路径字符串…），
    框架只要求它可被 ``predict_fn`` 消费；``ckpt_dir`` / ``log_csv`` 供 ``trainer`` 记账。
    """

    params: Any = None
    ckpt_dir: Any = None
    log_csv: Any = None
    history: Any = None
    n_params: int | None = None
    protocol: str = "native"
    meta: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return ("ModelHandle(protocol=%r, n_params=%s, ckpt_dir=%s)"
                % (self.protocol, self.n_params, self.ckpt_dir))


@dataclass
class ModelSpec:
    """一个已注册模型的登记项（**口径两维是必填字段**，不是可选装饰）。"""

    name: str
    info_condition: str
    target_caliber: str
    train_fn: Callable[..., Any]
    predict_fn: Callable[..., Any]
    version: str = ""
    family: str = ""
    notes: str = ""
    features: Sequence[str] = ()
    build_fn: Callable[..., Any] | None = None
    eval_fn: Callable[..., Any] | None = None
    source: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_caliber(self.info_condition, self.target_caliber, where="模型 %r" % self.name)
        if not callable(self.train_fn):
            raise RegistryError("模型 %r 的 train_fn 不可调用（实得 %r）"
                                % (self.name, type(self.train_fn).__name__))
        if not callable(self.predict_fn):
            raise RegistryError("模型 %r 的 predict_fn 不可调用（实得 %r）"
                                % (self.name, type(self.predict_fn).__name__))

    @property
    def caliber(self) -> dict[str, str]:
        return {"info_condition": self.info_condition, "target_caliber": self.target_caliber}

    def to_row(self) -> dict[str, Any]:
        return {"arm": self.name, "name": self.name, "version": self.version,
                "family": self.family, "info_condition": self.info_condition,
                "target_caliber": self.target_caliber, "n_features": len(self.features),
                "has_build_fn": self.build_fn is not None,
                "has_eval_fn": self.eval_fn is not None, "notes": self.notes,
                "source": self.source}

    def __repr__(self) -> str:
        return ("ModelSpec(%r v%s, %s/%s, family=%r)"
                % (self.name, self.version or "?", self.info_condition,
                   self.target_caliber, self.family))


def validate_caliber(info_condition: str, target_caliber: str, *, where: str = "") -> None:
    """口径两维的**唯一校验入口**（缺省/非法一律抛）。

    ``target_caliber`` 的取值域是 ``{y_full, y_rem@phi}``，但**实际数据带的是具体 φ**
    （``y_rem@0.2`` / ``y_rem@0.6`` …）。因此这里用 ``schema._caliber_ok`` 做**逐条**校验
    （接受 ``y_rem@<任意可解析的 φ>``），而"是不是同一口径族"的**归一化**
    （``y_rem@0.6`` → ``y_rem@phi``）由 ``schema.normalize_caliber`` 负责 ——
    两者分工明确，混起来就会出现"同一族的两个臂被误判成混排"。
    """
    if info_condition is None or str(info_condition) == "":
        raise CaliberError(
            "%s 未声明 info_condition —— **强制声明**。合法取值 %s\n"
            "  它决定「出发前能否拿到这些输入」：plan-time / realized / physics-calibrated "
            "的臂数字不可直接比。" % (where or "register()", list(INFO_CONDITIONS)))
    if not _caliber_ok("info_condition", str(info_condition)):
        raise CaliberError("%s 的 info_condition=%r 非法；合法取值 %s"
                           % (where or "register()", info_condition, list(INFO_CONDITIONS)))
    if target_caliber is None or str(target_caliber) == "":
        raise CaliberError(
            "%s 未声明 target_caliber —— **强制声明**。合法取值 %s\n"
            "  它决定「预测的是哪一段能量」：y_full（全程）/ y_rem@phi（切点后剩余）。"
            % (where or "register()", list(TARGET_CALIBERS)))
    if not _caliber_ok("target_caliber", str(target_caliber)):
        raise CaliberError("%s 的 target_caliber=%r 非法；合法取值 %s"
                           "（y_rem@ 后须是可解析的 φ，如 y_rem@0.2）"
                           % (where or "register()", target_caliber, list(TARGET_CALIBERS)))


class Registry:
    """模型注册表（进程内；``register`` / ``get`` / ``list_models``）。

    设计取舍：**注册不 import 模型代码**。``train_fn`` / ``predict_fn`` 由调用方（模型的
    适配脚本）提供，注册表只管登记与校验 —— 这样"注册一个模型"永远不会因为某个重依赖
    （torch / lightgbm）没装而整体失败。
    """

    def __init__(self) -> None:
        self._specs: dict[str, ModelSpec] = {}
        self._order: list[str] = []

    # ---------------------------------------------------------------- 写
    def register(self, name: str, info_condition: str, target_caliber: str,
                 train_fn: Callable[..., Any] | None = None,
                 predict_fn: Callable[..., Any] | None = None, *,
                 override: bool = False, **info: Any) -> ModelSpec:
        """登记一个模型。``name`` 唯一；重名默认**拒绝**（``override=True`` 才覆盖）。

        额外字段（全部可选）：``version`` / ``family`` / ``notes`` / ``features`` /
        ``build_fn`` / ``eval_fn`` / ``source`` / ``extra``。
        """
        nm = str(name).strip()
        if not nm:
            raise RegistryError("register(): name 不能为空")
        if nm in self._specs and not override:
            old = self._specs[nm]
            raise AlreadyRegistered(
                "模型 %r 已注册（%s/%s，source=%s）—— 拒绝静默覆盖。\n"
                "  正解：改名，或显式 register(..., override=True)（会留痕）。"
                % (nm, old.info_condition, old.target_caliber, old.source or "?"))
        missing = [k for k, v in (("train_fn", train_fn), ("predict_fn", predict_fn))
                   if v is None]
        if missing:
            raise RegistryError(
                "register(%r) 缺 %s —— 即插即用的**全部要求**就是这两个函数：\n"
                "    train_fn(ctx) -> ModelHandle\n"
                "    predict_fn(handle, ctx) -> 逐段/逐趟预测（Wh）\n"
                "  两个都没有 = 只是登记了名字，跑不起来。" % (nm, missing))
        spec = ModelSpec(name=nm, info_condition=info_condition,
                         target_caliber=target_caliber, train_fn=train_fn,
                         predict_fn=predict_fn, **info)
        self._specs[nm] = spec
        if nm not in self._order:
            self._order.append(nm)
        return spec

    def unregister(self, name: str) -> None:
        nm = str(name)
        if nm not in self._specs:
            raise UnknownModel("unregister: 未注册 %r；可用：%s" % (nm, self._order))
        del self._specs[nm]
        self._order.remove(nm)

    def clear(self) -> None:
        self._specs.clear()
        self._order.clear()

    # ---------------------------------------------------------------- 读
    def get(self, name: str) -> ModelSpec:
        nm = str(name)
        if nm not in self._specs:
            raise UnknownModel("未注册模型 %r。已注册 %d 个：\n%s"
                               % (nm, len(self._order),
                                  "\n".join("    %s" % self._row(k) for k in self._order)))
        return self._specs[nm]

    def __contains__(self, name: object) -> bool:
        return str(name) in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def names(self) -> list[str]:
        return list(self._order)

    def list_models(self) -> list[dict[str, Any]]:
        """全部已注册模型的摘要（口径两维在列，便于一眼看出谁跟谁能比）。"""
        return [self._specs[k].to_row() for k in self._order]

    def table(self) -> Any:
        import pandas as pd
        return pd.DataFrame(self.list_models())

    def describe(self, name: str) -> str:
        s = self.get(name)
        lines = ["模型 %s%s" % (s.name, (" v" + s.version) if s.version else ""),
                 "  信息条件 : %s" % s.info_condition,
                 "  目标口径 : %s" % s.target_caliber,
                 "  家族     : %s" % (s.family or "-"),
                 "  特征数   : %d" % len(s.features),
                 "  来源     : %s" % (s.source or "-"),
                 "  说明     : %s" % (s.notes or "-")]
        if s.features:
            lines.append("  特征     : %s%s"
                         % (", ".join(s.features[:8]), " …" if len(s.features) > 8 else ""))
        return "\n".join(lines)

    def _row(self, name: str) -> str:
        s = self._specs[name]
        return "%-28s %-19s %-11s %s" % (s.name, s.info_condition, s.target_caliber,
                                         s.family or "-")

    def leaderboard(self, entries: Iterable[Mapping[str, Any]], *,
                    metric: str = "wmape", split: str = "pooled") -> Any:
        """把评估结果排成榜 —— **强制按口径两维分组**；同组内混排即抛。

        ``entries`` 每项至少含 ``arm``（或 ``name``）、``info_condition``、``target_caliber``、
        以及指标列（默认 ``<split>_<metric>``，也接受裸 ``<metric>``）。
        """
        import pandas as pd
        rows = [dict(e) for e in entries]
        for r in rows:
            if not r.get("info_condition") or not r.get("target_caliber"):
                nm = r.get("arm") or r.get("name") or "?"
                spec = self._specs.get(str(nm))
                if spec is not None:
                    r.setdefault("info_condition", spec.info_condition)
                    r.setdefault("target_caliber", spec.target_caliber)
            if not r.get("info_condition") or not r.get("target_caliber"):
                raise CaliberError(
                    "leaderboard: 记录 %r 缺口径维度，且它不在注册表里 —— 拒绝入榜（防混排）。"
                    % (r.get("arm") or r.get("name") or "?"))
        from schema import check_homogeneous
        check_homogeneous(rows, where="leaderboard(metric=%s)" % metric)
        df = pd.DataFrame(rows)
        col = _resolve_metric_col(df, metric, split)
        if col is None:
            raise SchemaError(
                "leaderboard: 表里没有可用的指标列（要 %r / %r / 任何 *_%s / %s）。\n"
                "  实得列：%s\n"
                "  ⇒ 榜单不猜列名，但也别因为「我只算了 val」就判死刑："
                "按 <split>_<metric> → 任意 <split>_<metric> → <metric> 的顺序找。"
                % ("%s_%s" % (split, metric), metric, metric, metric, list(df.columns)))
        return df.sort_values(col, kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- 模块级单例
REGISTRY = Registry()


def register(name: str, info_condition: str, target_caliber: str,
             train_fn: Callable[..., Any] | None = None,
             predict_fn: Callable[..., Any] | None = None, **kw: Any) -> ModelSpec:
    """模块级注册入口（转发 ``REGISTRY.register``）。"""
    return REGISTRY.register(name, info_condition, target_caliber, train_fn, predict_fn, **kw)


def get(name: str) -> ModelSpec:
    """按名取模型（未注册即抛，并列出可用模型）。"""
    return REGISTRY.get(name)


def list_models() -> list[dict[str, Any]]:
    """全部已注册模型摘要。"""
    return REGISTRY.list_models()


def unregister(name: str) -> None:
    REGISTRY.unregister(name)


def describe(name: str) -> str:
    return REGISTRY.describe(name)


def leaderboard(entries: Iterable[Mapping[str, Any]], **kw: Any) -> Any:
    return REGISTRY.leaderboard(entries, **kw)


def familes() -> list[str]:
    """已注册模型用到的家族名（拼写保留，便于 grep）。"""
    seen: list[str] = []
    for r in REGISTRY.list_models():
        f = r.get("family") or "-"
        if f not in seen:
            seen.append(f)
    return seen


# --------------------------------------------------------------------------- 发布树便利入口
def load_package(module_name: str) -> Any:
    """导入一个模型包并调用它的 ``register_spec()``（**只加不改**的注册入口）。

    ``module_name`` 形如 ``models.itransformer.adapter``；导入前会先把发布树根挂上
    ``sys.path``，因此从任意工作目录调用都能成立。
    """
    import importlib
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for d in (root, root / "framework", root / "preprocess"):
        s = str(d)
        if s not in sys.path:
            sys.path.insert(0, s)
    mod = importlib.import_module(module_name)
    fn = getattr(mod, "register_spec", None)
    if fn is None:
        raise RegistryError("模块 %r 没有 register_spec() —— 即插即用要求它存在且返回 ModelSpec"
                            % module_name)
    spec = fn()
    if not isinstance(spec, ModelSpec):
        # 允许 register_spec 只做副作用（在模块 import 期已 register）而返回 None
        if spec is not None:
            raise RegistryError("模块 %r 的 register_spec() 返回了 %r，不是 ModelSpec"
                                % (module_name, type(spec).__name__))
        nm = getattr(mod, "NAME", None)
        if nm is None or str(nm) not in REGISTRY:
            raise RegistryError("模块 %r 的 register_spec() 既没返回 ModelSpec，"
                                "也没有在 import 期注册 NAME（%r）" % (module_name, nm))
        spec = REGISTRY.get(str(nm))
    return spec


def load_all(module_names: Sequence[str] | None = None) -> list[Any]:
    """把发布树里全部适配器注册进表（``run.py --list`` / 榜单脚本用）。"""
    mods = list(module_names or [
        "models.ours_tree.model",
        "models.ours_personal.adapter",
        "models.itransformer.adapter",
        "models.timexer.adapter",
        "models.paper.adapter",
    ])
    out, failed = [], []
    for m in mods:
        try:
            out.append(load_package(m))
        except Exception as e:                                    # noqa: BLE001
            failed.append((m, "%s: %s" % (type(e).__name__, e)))
    if failed:
        raise RegistryError("load_all：%d 个适配器注册失败：\n%s"
                            % (len(failed), "\n".join("    %s -> %s" % f for f in failed)))
    return out


# --------------------------------------------------------------------------- 榜单列解析
_SPLIT_ORDER = ("pooled", "val", "dev")


def _resolve_metric_col(df: Any, metric: str, split: str) -> str | None:
    """找排序用的指标列：``<split>_<metric>`` → 任意 ``<split>_<metric>`` → ``<metric>``。

    ★ 为什么要退让：只算了 val 的臂（如单划分评估）不该因为"没有 pooled 列"就报错 ——
      那不是口径问题，是本次评估的作用域问题。但**绝不猜非指标列**（列名必须
      以 ``_<metric>`` 结尾或恰等于 ``metric``）。
    """
    want = "%s_%s" % (split, metric)
    if want in df.columns:
        return want
    for sp in _SPLIT_ORDER:
        c = "%s_%s" % (sp, metric)
        if c in df.columns:
            return c
    for c in df.columns:
        if str(c).endswith("_%s" % metric) or str(c) == metric:
            return str(c)
    return None
