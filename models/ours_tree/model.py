#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ours_tree.model —— **SePT-70 v1.0**（段级 plan-time 梯度提升树，发布树干净版）。

它是什么
========
逐 100 m 段预测能量（Wh）的 ``HistGradientBoostingRegressor``，再**按趟求和**得趟级预测。
70 项特征**全部 plan-time**（realized 项 0 个），口径 ``plan-time / y_full``。

它与源树的关系（**只读，不改一个字**）
======================================
源实现是 ``07_segments/models/SePT/sept.py``（v1.0，380 行）。发布树**不复制**它，而是用
``framework/preprocess.py`` 里同一份冻结列清单与**同一套装配顺序**重建特征；差别只有一处、
且是刻意的：

    源实现把 ``G_stoptiming/g2_features_100m_trmap.csv.gz`` 的 29 列 ``t_*`` **并进来但不用**
    （只为与 ``M_final`` 保持同一 merge 顺序）。本模块**同样读它并核对哈希**，因为
    "读它"这件事本身是依赖门的一部分；但**不把它并入宽表** —— 那 29 列不进 70 特征。
    结果：[0 列进特征]，对预测**逐位无影响**。

冻结常量（逐位可复现的前提，改一个数字就不再与公布值可比）
==========================================================
* 超参：``loss=absolute_error, lr=0.05, max_iter=500, max_leaf_nodes=31,
  min_samples_leaf=20, l2=1.0, early_stopping=False, random_state=42``
* 训练集：**全部 134,212 个 train 段**（不做任何子集筛选；误筛会静默恶化约 0.78 pp）
* y 基：``seg_features.y_full``(kWh, float32 量化) × 1000 = Wh
* 锚点：val WMAPE **10.0605030031** / dev **8.5513990875**（容差 3e-6）

已知局限
========
* 这是最优**树**模型，不是全场最优（plan-time 口径下它是最优之一）。
* y 基用的是**量化 kWh 列**×1000；换精确 Wh（``Σ seg_energy_wh``）会得到 10.0786 / 8.4808，
  是**另一套基**，两套数不可混引。
* 树模型没有 ckpt：``fit`` 是单次确定性拟合（``random_state=42``、无早停），
  因此"推理"必然重跑同一份 ``fit``；发布树的 4.6 KB 级确定性由此而来。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

# --- 发布树引导（必须在 import framework 之前） -------------------------------
import sys as _sys
_REL = Path(__file__).resolve().parents[2]
for _p in (_REL, _REL / "framework", _REL / "preprocess"):
    _s = str(_p)
    if _s in _sys.path:
        _sys.path.remove(_s)
    _sys.path.insert(0, _s)

import dataio as DIO                                                 # noqa: E402
import eval_trip as ET                                               # noqa: E402
import features as FT                                                # noqa: E402
import infer as INF                                                  # noqa: E402
import paths as pth                                                  # noqa: E402
import registry as reg                                               # noqa: E402
import trainer as TR                                                 # noqa: E402

NAME = "SePT-70_v1.0"
VERSION = "1.0"
FAMILY = "tree"
DATE = "2026-10-11"

INFO_CONDITION = "plan-time"
TARGET_CALIBER = "y_full"

# 冻结超参（与源实现 TREE 字典逐字一致）
TREE: dict[str, Any] = {"loss": "absolute_error", "learning_rate": 0.05, "max_iter": 500,
                        "max_leaf_nodes": 31, "min_samples_leaf": 20,
                        "l2_regularization": 1.0, "early_stopping": False,
                        "random_state": 42}
SEED = 42
N_TRAIN_SEG_EXPECTED = 134_212

Y_BASIS = "seg_features.y_full(kWh, float32 量化) × 1000 = Wh"

# 公布锚点（逐位复现值；容差 3e-6）
ANCHOR = {"val": 10.0605030031, "dev": 8.5513990875}
ANCHOR_TOL = 3e-6
# 源实现自报的精确复现值（本模块的交叉核对用）
UPSTREAM = {"val": 10.060503003124914, "dev": 8.551399087500657}

# 源实现（只读；用于交叉核对，**不是**运行依赖）
SOURCE_PY = "models/SePT/sept.py"

# ckpt/ 的说明（树臂无参数文件，是设计如此）
_CKPT_NOTE = """# `ckpt/` 说明（SePT-70，树臂）

本目录**没有参数文件**，这是设计如此，不是漏了。

SePT-70 是单次确定性拟合的梯度提升树（`HistGradientBoostingRegressor`，
`random_state=42`、`early_stopping=False`、`max_iter=500`）：

* **没有早停** ⇒ 不存在 `best_val.pt`（"最好的那一步"这个概念不适用）；
* **没有可微参数** ⇒ 不存在 `last_*.pt` 的环形保留；
* 逐位复现靠的是 `../model_card.json` 里的**冻结超参 + 冻结特征顺序 + 冻结数据哈希**
  （`data_v3/segments_100m.csv.gz` SHA16 `298f65ca0dc5c2f5`），而不是靠权重文件。

因此本臂的"推理"必然重跑同一份 `fit`（约 8 秒 / 134,212 段 / 70 特征）。
把树导出成文件反而会引入 sklearn 版本与精度差异，**故意不做**。

跑法：`python -B models/ours_tree/model.py`
"""


class OursTreeError(RuntimeError):
    """SePT-70 适配层错误。"""


def _hgb(**kw: Any) -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor
    p = dict(TREE)
    p.update(kw)
    return HistGradientBoostingRegressor(**p)


# --------------------------------------------------------------------------- 训练
def train_fn(ctx: TR.TrainContext) -> reg.ModelHandle:
    """在**全部 train 段**上拟合 SePT-70（单次确定性拟合，无早停 / 无 LR 调度）。

    框架契约：返回 :class:`registry.ModelHandle`；``meta['pred_wh']`` 必须是**趟级**预测
    （单位门在同一作用域上算），``meta['rerun']`` 是"同种子重跑一遍"的回调。
    """
    t0 = time.time()
    pack = DIO.load_pack("plan70", P=pth.Paths())
    seg_split = pack.df["split"].to_numpy(dtype=object).astype(str)
    is_tr = seg_split == "train"
    n_tr = int(is_tr.sum())
    if n_tr != N_TRAIN_SEG_EXPECTED:
        raise OursTreeError("train 段数 %d ≠ %d（训练视图被动过？）—— 拒绝拟合"
                            % (n_tr, N_TRAIN_SEG_EXPECTED))
    if pack.n_feat != 70:
        raise OursTreeError("特征数 %d ≠ 70 —— SePT-70 的定义被改动了" % pack.n_feat)

    net = _hgb()
    net.fit(pack.raw[is_tr], pack.y_seg_wh[is_tr])
    net.n_features_in_ = pack.n_feat
    seg_wh = np.asarray(net.predict(pack.raw), dtype=np.float64)
    trip_wh = np.bincount(pack.flat_trip, weights=seg_wh, minlength=pack.n_trips)

    ctx.log("[%s] 拟合完成：%d train 段 / %d 特征 / %d 趟（%.1fs）"
            % (NAME, n_tr, pack.n_feat, pack.n_trips, time.time() - t0))
    ctx.ckpt.log({"epoch": 1, "step": 1, "train_loss": "", "holdout_mse": "",
                  "val_wmape": "", "dev_wmape": "", "lr": "",
                  "pred_true_ratio": round(float(trip_wh.sum() / pack.y_full_wh.sum()), 6),
                  "note": "native-fit（单次确定性 HGBR，无早停；ckpt 无参数文件）"})
    h = reg.ModelHandle(params=net, protocol="native", n_params=None,
                        ckpt_dir=ctx.ckpt.ck, log_csv=ctx.ckpt.log_csv)
    h.meta = {
        "pack": pack, "seg_wh": seg_wh, "pred_wh": trip_wh,
        "probes": {}, "allow_dead": (),
        "rerun": lambda: np.bincount(
            pack.flat_trip,
            weights=np.asarray(_hgb().fit(pack.raw[is_tr], pack.y_seg_wh[is_tr])
                               .predict(pack.raw), dtype=np.float64),
            minlength=pack.n_trips),
        "note": "SePT-70：段级 HGBR → 按趟求和；协议里的 holdout/warmup/cosine 不适用",
    }
    # 落一份"可携带"的模型卡（树模型无参数文件，落超参 + 特征顺序即等价）
    card = {"name": NAME, "version": VERSION, "family": FAMILY, "date": DATE,
            "info_condition": INFO_CONDITION, "target_caliber": TARGET_CALIBER,
            "y_basis": Y_BASIS, "features": list(pack.feats), "n_features": pack.n_feat,
            "hyperparams": dict(TREE), "seed": SEED,
            "train_segments": n_tr, "n_segments": pack.n_seg, "n_trips": pack.n_trips,
            "split": pack.info["trip_by_split"], "anchor": dict(ANCHOR),
            "anchor_tol": ANCHOR_TOL, "source_py": SOURCE_PY,
            "ckpt_note": "树模型无参数文件：ckpt/ 下只有 train_log.csv 与 train_manifest.json；"
                         "复现靠本卡的冻结超参 + 冻结特征顺序 + 冻结数据哈希。",
            "deps_sha16": {k: FT.sha16(pth.Paths().seg / k) for k in FT.DEP_HASHES}}
    (Path(ctx.out_dir) / "model_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    ck = Path(ctx.out_dir) / "ckpt"
    ck.mkdir(parents=True, exist_ok=True)
    (ck / "README.md").write_text(_CKPT_NOTE, encoding="utf-8")
    return h


# --------------------------------------------------------------------------- 推理
def predict_fn(handle: reg.ModelHandle, ctx: Any = None, **kw: Any) -> dict[str, Any]:
    """``ModelHandle`` → ``{"trip_wh": ..., "seg_wh": ...}``（Wh；与训练同一份特征）。"""
    m = getattr(handle, "meta", None) or {}
    if "pack" not in m or "seg_wh" not in m:
        raise OursTreeError(
            "handle 里没有 pack/seg_wh —— 本臂的 predict_fn 只接受**本框架 train_fn 产出的**"
            "handle（SePT-70 不落参数文件，跨进程传递请改用 --model ours_tree 全链重跑）。")
    pack = m["pack"]
    seg = np.asarray(m["seg_wh"], dtype=np.float64)
    if seg.shape[0] != pack.n_seg:
        raise OursTreeError("seg_wh 长度 %d ≠ 段数 %d" % (seg.shape[0], pack.n_seg))
    return {"trip_wh": np.asarray(m["pred_wh"], dtype=np.float64), "seg_wh": seg}


def register_spec(*, override: bool = True) -> reg.ModelSpec:
    """注册进统一注册表（**两个口径维度是强制的**，缺一即抛）。"""
    return reg.register(
        NAME, INFO_CONDITION, TARGET_CALIBER, train_fn, predict_fn,
        version=VERSION, family=FAMILY,
        notes="段级 HGBR（70 项 plan-time 特征）→ 按趟求和；对应源实现 %s" % SOURCE_PY,
        features=tuple(DIO.FEATS_PLAN70),
        source=SOURCE_PY, override=override)


# --------------------------------------------------------------------------- 全链
def run(*, seed: int = 42, quick: bool = False, verbose: bool = True) -> dict[str, Any]:
    """发布树内全链：``registry → trainer（native） → infer → eval_trip``。

    ``quick=True`` 时只取前若干趟做冒烟（**不看锚点**，锚点只在全量下有意义）。
    """
    sp = register_spec()
    pack = DIO.load_pack("plan70")
    if quick:
        raise OursTreeError("ours_tree 不支持 --quick：树的锚点只在全量 134,212 train 段上有意义"
                            "（截断会改变拟合，数字不可比）。要冒烟请跑 --stage smoke。")
    out_dir = pth.Paths().runs(NAME)
    res = TR.train(sp, data=TR.prepare(), mode="native", protocol=TR.Protocol(),
                   out_dir=out_dir, seed=seed, repro_check=False)
    h = res.handle
    inf = INF.predict_to_npz(predict_fn, h, None, out_dir=out_dir,
                             files=pack.trips, split=pack.split, y_wh=pack.y_full_wh,
                             info_condition=INFO_CONDITION, target_caliber=TARGET_CALIBER,
                             model=NAME, y_basis=Y_BASIS,
                             trip_of_row=pack.flat_trip, mask=None)
    m_val = ET.eval_trip(inf.trip.select("validation"), info_condition=INFO_CONDITION,
                         target_caliber=TARGET_CALIBER, splits=("val",))
    m_dev = ET.eval_trip(inf.trip.select("future"), info_condition=INFO_CONDITION,
                         target_caliber=TARGET_CALIBER, splits=("dev",))
    m_all = ET.eval_trip(inf.trip, info_condition=INFO_CONDITION,
                         target_caliber=TARGET_CALIBER)
    got = {"val": float(m_val.metric("WMAPE", "val")), "dev": float(m_dev.metric("WMAPE", "dev"))}
    diffs = {k: abs(got[k] - ANCHOR[k]) for k in ANCHOR}
    worst = max(diffs.values())
    verdict = "PASS" if worst <= ANCHOR_TOL else "FAIL"
    out = {
        "model": NAME, "family": FAMILY, "seed": int(seed),
        "info_condition": INFO_CONDITION, "target_caliber": TARGET_CALIBER,
        "y_basis": Y_BASIS,
        "n_seg": pack.n_seg, "n_trips": pack.n_trips, "n_feat": pack.n_feat,
        "trip_by_split": pack.info["trip_by_split"],
        "val_wmape": got["val"], "dev_wmape": got["dev"],
        "pooled_wmape": float(m_all.metric("WMAPE", "pooled")),
        "val_mae_wh": float(m_val.metric("MAE", "val")),
        "val_bias_wh": float(m_val.metric("bias", "val")),
        "val_rmse_wh": float(m_val.metric("RMSE", "val")),
        "val_mape_ge1kwh": float(m_val.metric("MAPE_ge1kWh", "val")),
        "unit_ratio_val": float(m_val.unit_ratio.get("val", float("nan"))),
        "anchor": dict(ANCHOR), "anchor_tol": ANCHOR_TOL, "abs_diff": diffs,
        "upstream_published": dict(UPSTREAM), "verdict": verdict,
        "npz": str(inf.npz), "key_sha16": inf.meta.get("key_sha16"),
        "pipeline": "registry → trainer(native) → infer → eval_trip",
        "n_params": res.n_params, "unit_ratio_train": {k: float(v) for k, v in
                                                       res.unit_ratio.items()},
    }
    rep = pth.Paths().runs(NAME) / "eval_report.json"
    rep.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print("\n★ %s：%s" % (NAME, verdict))
        print("   锚点 %s" % ANCHOR)
        print("   实测 val=%.10f dev=%.10f |Δ| 最大 %.3e（容差 %g）"
              % (got["val"], got["dev"], worst, ANCHOR_TOL))
        print("   六指标（val）：MAPE(y>=1kWh)=%.4f MAE=%.2f Wh bias=%.2f Wh RMSE=%.2f Wh"
              % (out["val_mape_ge1kwh"], out["val_mae_wh"], out["val_bias_wh"],
                 out["val_rmse_wh"]))
        print("   pooled WMAPE = %.6f | 单位比 Σp/Σy = %.6f"
              % (out["pooled_wmape"], out["unit_ratio_val"]))
        print("   标准产物 %s（key_sha16=%s）" % (out["npz"], out["key_sha16"]))
        print("   报告 %s" % rep)
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="SePT-70（树）· 发布树全链")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", action="store_true", help="只打印 JSON")
    a = ap.parse_args(argv)
    o = run(seed=a.seed, verbose=not a.json)
    if a.json:
        print(json.dumps(o, ensure_ascii=False, indent=2))
    return 0 if o["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
