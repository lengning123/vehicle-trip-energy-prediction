#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ours_personal.adapter —— **JJ_pathway 前缀条件化**（残差骨架 + 逐趟个性化门）。

口径声明（**必读**）
====================
* ``info_condition = "realized"``
  外生侧 70 项特征是 **plan-time**；个性化侧吃的是窗口 ``[s0, cut]`` 内的**实测**轨迹。
  只要有实测信息进来，这条臂就**不是** plan-time —— 把它和 plan-time 臂排进同一张榜是错的。
* ``target_caliber = "y_rem@0.2"``（默认 φ=0.2）
  预测的是**切点之后**的剩余能量；φ=0 时退化为 ``y_full``。
* 单位：Wh（输入输出都是 Wh；kWh 只在读表时 ×1000 一次）。
* 因果纪律：窗口上界 = 切点 ``cut_m``，**切点之后的任何真实值都不参与**。
  ``prefix_len`` 由 ``features.build_cuts`` 的 ``n_pre`` 给出，不手写算术。

模型（与源树 ``models/JJ_pathway`` 同构）
========================================
::

    pred = base_rem · exp( clamp( Δ(v_prefix, x_exo), ±0.7 ) )

* ``base_rem``：**冻结的 SePT-70 plan-time 段级预测**在**后缀段**上求和（不训练）。
* ``Δ``：把前缀 patch 序列（≤64 块自注意力）与外生 24 箱表征拼起来过一个头。
* 与源树的差别：源树还学一个逐趟门 ``γ = σ(w·[G, ctx])``；本适配器把门并入 ``Δ`` 的
  输出尺度（``log_scale``），因为二者的可观测效果一致（``γ→0`` ⇔ ``Δ→0``），
  而单参数化少一组超参、更容易逐位复现。

**个性化开关** ``--personal {on,off}``
=====================================
* ``on``  ：建个性化分支（前缀 patch 自注意力 + 融合），这是本臂的定义形态。
* ``off`` ：**整条分支不建**（不是置零、不是 dropout）⇒ 网络退化为纯外生模型，
  即"plan-time 基线 + 神经修正"。两态之差就是**前缀信息的直接测量**。

已知局限
========
* 逐趟标签用 ``y_rem@φ`` 的**精确 Wh 基**（``Σ 后缀 seg_energy_wh``），与 SePT-70 的
  ``y_full`` 量化基**不是同一套基**；跨基比较只能看 WMAPE（无量纲），不能比 MAE。
* 短趟 + 大 φ 会切出空后缀（源协议标 ``cut_ok=False``）⇒ 这些趟**从训练与评估里剔除**
  （``ok`` 掩码），因此各 φ 的有效趟数不同（φ=0.2 时 184 趟量级）。
* 本臂**不含** 1 Hz 原始轨迹：前缀特征用的是 100 m 段级特征序列（同源、同键序、
  可完全从发布树的输入重建）。源树用 1 Hz patch，那是**更细**的输入 ⇒
  本适配器的绝对数字**不可**当作源树 ``JJ_pathway`` 数字引用。
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

NAME_ON = "JJ_pathway-prefix-personal-ON"
NAME_OFF = "JJ_pathway-prefix-personal-OFF"
FAMILY = "neural-personal"
SOURCE = "models/JJ_pathway (REPORT.md) + scripts/jj_pathway_20_model.py"


def make_spec(*, personal: bool = True, phi: float = 0.2, seed: int = 42,
              policy: str = "plan70", quick: bool = False, smoke: bool = False) -> NT.NNSpec:
    """构造本臂的规格（`personal` 就是那个**开关**）。"""
    return NT.NNSpec(
        name=(NAME_ON if personal else NAME_OFF),
        family=FAMILY, arch="prefix", policy=policy, base="tree_rem",
        use_personal=bool(personal), use_cross=False, phi=float(phi),
        info_override="realized",          # ← 个性化分支吃实测前缀 ⇒ 整臂是 realized
        d=24, heads=4, layers=2, d_ff=48, dropout=0.1,
        lr=2e-3, weight_decay=1e-4,
        max_steps=(400 if quick else 1200), batch=64, eval_every=100, smoke=bool(smoke),
        patience_epochs=6, seed=int(seed),
        notes=("残差骨架 base_rem·exp(clamp Δ) + 前缀 patch 条件化；个性化=%s；φ=%g"
               % ("ON" if personal else "OFF", phi)),
        source=SOURCE,
        extra={"phi_grid": [0.2, 0.4, 0.6], "patch_max": 64, "n_bins": 24,
               "personalization_switch": "use_personal"})


def run(*, personal: bool = True, phi: float = 0.2, seed: int = 42,
        quick: bool = False, smoke: bool = False) -> dict:
    spec = make_spec(personal=personal, phi=phi, seed=seed, quick=quick, smoke=smoke)
    r = NT.train_neural(spec, repro_check=False)
    return r.to_row()


def register_spec():
    """登记本臂（默认态 = 个性化 ON、φ=0.2）到统一注册表。

    与 ``train_neural`` 共用 ``register_nn_spec`` —— 注册代码只有一份。
    """
    return NT.register_nn_spec(make_spec(personal=True, phi=0.2), source=SOURCE)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="JJ_pathway 前缀条件化（个性化开关）")
    ap.add_argument("--personal", choices=("on", "off"), default="on",
                    help="个性化分支开关（on/off；off = 退化为纯外生）")
    ap.add_argument("--phi", type=float, default=0.2, choices=(0.0, 0.2, 0.4, 0.6))
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
