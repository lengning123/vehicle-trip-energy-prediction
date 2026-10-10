#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""itransformer.adapter —— **plan-time iTransformer**（倒置注意力，纯外生，无个性化）。

口径声明（**必读**）
====================
* ``info_condition = "plan-time"``：只用 70 项 plan-time 特征（realized 项 **0** 个）。
* ``target_caliber = "y_full"``：趟级全程能量（Wh）。
* 单位：Wh。
* 与源树的重大差异（**必须写明，否则数字会被误引**）：
  源树 ``models/P_transformer`` 的 iTransformer 臂是 **realized 口径**
  （``FEATS_REALIZED``，实测段时长/车速/分位数/停车比），公布 8.1017 / 5.8037
  （val/dev，另一套 y 基），**不能**与本臂比。本适配器实现的是**同架构的 plan-time 版本**：
  输入换成 70 项 plan-time 特征，口径随之变成 plan-time / y_full。

架构（iTransformer 的"倒置"）
=============================
不把"时间步"当 token，而把**变量**当 token、把时间当特征维：

    (B, L, F)  --等里程分箱 24 箱-->  (B, F, 24)
               --per-var Linear-->    (B, F, d)
               --变量维自注意力-->     (B, F, d)   ← 注意力在**变量之间**做
               --均值池化-->          (B, d) --> head --> Δ

这正是 iTransformer 论文的核心（"inverted"：注意力学变量间相关，而非时间依赖）。
本臂的 head 输出的是**相对冻结基线的对数倍率** ``Δ``：

    pred = base_full · exp(clamp(Δ, ±0.7))

``base_full`` 是冻结 SePT-70 的趟级预测（= 本发布树 ``ours_tree`` 的产物），
因此 ``Δ=0`` ⇒ 预测严格等于基线。于是"神经侧到底加了多少"是一个**可直接读出的数**。

已知局限
========
* 本臂的基线与 ``ours_tree`` 共享同一份冻结树 ⇒ 两臂**不独立**；比较的是
  "树 vs 树+神经修正"，不是"树 vs 另一个独立模型"。
* 序列被压成 24 个等里程箱 ⇒ **箱内的时间顺序被丢弃**。这是有意为之的口径选择
  （源树的同构做法），但它意味着本臂不适于声称"捕捉了段间动态"。
* 段数上限 ``Lmax=2012`` 来自冻结段表；更长的趟会被右侧截断（本数据里没有）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REL = Path(__file__).resolve().parents[2]
for _p in (_REL, _REL / "framework", _REL / "preprocess"):
    _s = str(_p)
    if _s in sys.path:
        sys.path.remove(_s)
    sys.path.insert(0, _s)

import nn_train as NT                                                # noqa: E402

NAME = "iTransformer-plan-inverted"
FAMILY = "neural-exo"
SOURCE = "models/P_transformer (run.py --archs itrans) — 本适配器为 plan-time 变体"


def make_spec(*, seed: int = 42, quick: bool = False, d: int = 24, heads: int = 4,
              layers: int = 2, policy: str = "plan70", smoke: bool = False) -> NT.NNSpec:
    return NT.NNSpec(
        name=NAME, family=FAMILY, arch="itransformer", policy=policy, base="tree_full",
        use_personal=False, use_cross=False, phi=0.0,
        d=d, heads=heads, layers=layers, d_ff=2 * d, dropout=0.1,
        lr=2e-3, weight_decay=1e-4,
        max_steps=(400 if quick else 1200), batch=64, eval_every=100, smoke=bool(smoke),
        patience_epochs=6, seed=int(seed),
        notes="倒置注意力（变量当 token）+ 冻结树基线残差；纯 plan-time，无个性化",
        source=SOURCE,
        extra={"n_bins": 24, "inverted": True,
               "upstream_realized_arm": {"val": 8.1017, "dev": 5.8037,
                                         "info_condition": "realized",
                                         "note": "源树真身；口径不同，不可与本臂比"}})


def run(*, seed: int = 42, quick: bool = False) -> dict:
    return NT.train_neural(make_spec(seed=seed, quick=quick),
                           repro_check=not quick).to_row()


def register_spec():
    """把本臂登记进统一注册表（``registry.load_package`` 的即插即用入口）。

    ★ 与 ``train_neural`` **共用 ``NN.train.register_nn_spec``** —— 只写一份注册代码，
      避免"同名两口径"。``extra['kind']='deferred'`` 表示这是**延迟执行**登记：
      真正跑训练时由 ``train_neural`` 覆盖同一条目（override，留痕）。
    """
    return NT.register_nn_spec(make_spec(), source=SOURCE)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="plan-time iTransformer（倒置注意力）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--d", type=int, default=24)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--quick", action="store_true", help="少步数（仍会跑一轮完整协议）")
    ap.add_argument("--smoke", action="store_true", help="**接口冒烟**：30 步 / 1 种子，只证明链路通，不看数字")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    sp = make_spec(seed=a.seed, quick=a.quick, d=a.d, layers=a.layers, smoke=a.smoke)
    o = NT.train_neural(sp, repro_check=False).to_row()
    if a.json:
        print(json.dumps(o, ensure_ascii=False, indent=2, default=str))
    return 0 if o["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
