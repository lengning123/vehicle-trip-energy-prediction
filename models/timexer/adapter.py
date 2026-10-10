#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""timexer.adapter —— **AA_prefixxer**（TimeXer 式前缀交叉注意力）。

口径声明（**必读**）
====================
* ``info_condition = "realized"``：外生 70 项是 plan-time；**个性化分支吃实测前缀**。
  与 ``models/AA_prefixxer`` 的自我声明一致（"hybrid：外生 plan-time / 内生 realized，
  且窗口上界 ≤ 切点"）。
* ``target_caliber = "y_full"``（默认，φ=0）。
  源树默认跑的是 ``y_rem@φ``（φ∈{0.2,0.4,0.6}）；本适配器默认 φ=0 以给出一个
  与 ``ours_tree`` **可对齐的整趟格子**，同时保留 ``--phi`` 复现 φ>0 的剩余口径。
* 单位：Wh。
* 因果纪律：前缀 patch 只覆盖 ``seg < prefix_len``，``prefix_len`` = 切点之前的段数。

架构（TimeXer 的两个关键点）
============================
1. **内生序列单独编码**：前缀段级序列 → ≤64 个非重叠 patch → patch 维自注意力
   （``PatchEncoder``）。这条分支是"个性化"的载体。
2. **内生当 query、外生当 context 做跨注意力**（``CrossFuse``）：
   每个"变量 token"去问"前缀里哪些 patch 与我相关"，得到逐变量的个性化修正。
   这正是 TimeXer 把 endogenous 序列接进 exogenous 变量的做法。

::

    pred = base · exp( clamp( Δ( cross_attn(x_exo ← x_prefix) ), ±0.7 ) )

**个性化开关** ``--personal {on,off}``
=====================================
* ``on`` ：完整 TimeXer 双分支（patch 自注意力 + 前缀→变量跨注意力 + 拼接头）。
* ``off``：内生分支与跨注意力**都不建** ⇒ 退化为纯外生倒置注意力（= plan-time 基线
  + 神经修正）。两态之差 = 前缀信息的直接测量。

已知局限
========
* 前缀用 **100 m 段级**特征，源树用 **1 Hz** patch（``P=max(30 s, ⌈T/64⌉)``）。
  粒度更粗 ⇒ 本臂**不是**源树 ``AA_prefixxer`` 数字的复现，只是**同机制**的可跑实现。
* φ>0 时有效趟数会因空后缀而减少（``cut_ok`` 掩码），各 φ 的有效趟数不同。
* 源树在 φ=0.6 上对基线的最好成绩是 +0.97 pp（CI [+0.281, +1.715]）；
  φ=0.2/0.4 上**没有**稳定增益 —— 别把 φ=0.6 的结论外推到全部 φ。
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

NAME_ON = "AA_prefixxer-timexer-ON"
NAME_OFF = "AA_prefixxer-timexer-OFF"
FAMILY = "neural-prefix-cross"
SOURCE = "models/AA_prefixxer (aa_prefixxer_20_model.py, class PrefixXer)"


def make_spec(*, personal: bool = True, phi: float = 0.0, seed: int = 42,
              quick: bool = False, policy: str = "plan70", smoke: bool = False) -> NT.NNSpec:
    base = "tree_rem" if phi > 0 else "tree_full"
    return NT.NNSpec(
        name=(NAME_ON if personal else NAME_OFF),
        family=FAMILY, arch="timexer", policy=policy, base=base,
        use_personal=bool(personal), use_cross=True, phi=float(phi),
        info_override="realized",          # ← 前缀分支吃实测 ⇒ 整臂是 realized
        d=24, heads=4, layers=2, d_ff=48, dropout=0.1,
        lr=2e-3, weight_decay=1e-4,
        max_steps=(400 if quick else 1200), batch=64, eval_every=100, smoke=bool(smoke),
        patience_epochs=6, seed=int(seed),
        notes=("TimeXer 式：前缀 patch 自注意力 + 外生←内生跨注意力；个性化=%s；φ=%g"
               % ("ON" if personal else "OFF", phi)),
        source=SOURCE,
        extra={"n_bins": 24, "patch_max": 64, "cross_attention": True,
               "personalization_switch": "use_personal",
               "upstream_note": "源树用 1 Hz patch；本适配器用 100 m 段级 patch（同机制，非复现）"})


def run(*, personal: bool = True, phi: float = 0.0, seed: int = 42,
        quick: bool = False, smoke: bool = False) -> dict:
    sp = make_spec(personal=personal, phi=phi, seed=seed, quick=quick, smoke=smoke)
    return NT.train_neural(sp, repro_check=False).to_row()


def register_spec():
    """登记本臂（默认态 = 个性化 ON、φ=0.0 整趟格）到统一注册表。"""
    return NT.register_nn_spec(make_spec(personal=True, phi=0.0), source=SOURCE)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AA_prefixxer / TimeXer 式前缀交叉注意力")
    ap.add_argument("--personal", choices=("on", "off"), default="on")
    ap.add_argument("--phi", type=float, default=0.0, choices=(0.0, 0.2, 0.4, 0.6))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="少步数（仍会跑一轮完整协议）")
    ap.add_argument("--smoke", action="store_true", help="**接口冒烟**：30 步 / 1 种子，只证明链路通，不看数字")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    o = run(personal=(a.personal == "on"), phi=a.phi, seed=a.seed, quick=a.quick, smoke=a.smoke)
    if a.json:
        print(json.dumps(o, ensure_ascii=False, indent=2, default=str))
    return 0 if o["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
