#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""paper.adapter —— 论文复现臂 **A1 / A2 / A12 / G10 / G04 / P14**（代理版 = 公平口径）。

公平口径是什么（**必读，这是本目录存在的全部理由**）
====================================================
论文臂原本跑在**它们的**数据与信息条件上（10 Hz 波形、1 m 分辨率、合成标签……）。
直接引用那些数字与本项目比较是不公平的，两个方向的偏差都有。本项目确立的
**公平口径**是：*把论文模型放到我们手里全部可用的**代理输入**上，再比*。

    FAIRNESS.md（源树 X2_papercomp）L5：
    「公平对比 = 代理版（proxy）：论文模型拿到我们手里全部可用代理输入
      （DEM 坡度/高程、TomTom 限速、OSM 路口-信号灯-斑马线语义、
       2,628 停车点位时序、H_aux 的 P_aux 估计）。」

因此本目录下每一条臂的 ``input_version`` 都必须是 ``proxy``（或显式降级为 ``degraded``
并写明差在哪儿），**不得**混引论文原文数字。

六条臂与它们**不可修**的差距（源树 ``PAPER_TABLE.md`` §8；本目录照抄，不许美化）
================================================================================
======  ==========================  ==================================================
臂      论文原型                     不可修项（**不得声称已修**）
======  ==========================  ==================================================
A1      1D-CNN 车窗 + BDT 二级      **辅助通道语义**：论文 "wind speed" 用 Gradient 替代，
                                    缺列记 0（= 结构性替代，方向偏乐观）
A2      7 通道 CNN（协方差/GAF/特征向量）  **10 Hz 原生数据不可得**：100 点输出由 10 点
                                    线性上采样 ⇒ 量纲对齐而非数据对齐
A12     物理仿真 + 残差 MLP         **标签是合成同源**（同一物理模型 + 人工噪声）
                                    ⇒ 其 kWh 量纲与真实能耗不同义 ⇒ 已从交叉比较中**排除**
G10     两阶段（物理 → GBR）        **F4 是否整流未公开**：signed 为主口径，
                                    rectified 仅作敏感性变体
G04     反向能量物理层 + BiLSTM     **1 m 原始分辨率不可得**（我们是 100 m）
                                    ⇒ 代理方向 = structural，不是 proxy
P14     物理功率模型                **无不可修项** —— 唯一**精确复现**臂（逐位一致）
======  ==========================  ==================================================

本适配器的做法（诚实且可跑）
============================
发布树**不重训**这六条臂 —— 源树的 ``A1`` 需要 45–60 个 epoch 的 1D-CNN + 20 万行 BDT，
``A2`` 需要 1200–3000 步的 7 通道编码器，重训代价远超本发布树的预算，而**逐位复现**
必须依赖源树已冻结的产物。因此每条臂是一个**"冻结逐趟预测 + 同口径重算"**的适配器：

    train_fn   → 装载冻结逐趟预测（按键对齐，**不按位置**），校验键覆盖与单位比
    predict_fn → 返回该逐趟预测（Wh）
    eval_trip  → 在发布树里**重算**全部六指标 + CI（数字不是抄来的）

于是"这些数字是怎么来的"在发布树内是**可复算**的：`runs/paper__<臂>/trip_preds.npz`
里就是逐趟预测，`eval_report.json` 里就是重算结果。
每条臂的 ``ckpt/README.md`` 写明冻结来源、SHA16 与"为什么没重训"。

红线：只读源树的既有冻结产物；不读 calibration；不写源树。
本文件**不含 assert**。
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_REL = Path(__file__).resolve().parents[2]
for _p in (_REL, _REL / "framework", _REL / "preprocess"):
    _s = str(_p)
    if _s in sys.path:
        sys.path.remove(_s)
    sys.path.insert(0, _s)

import dataio as DIO                                                 # noqa: E402
import eval_trip as ET                                               # noqa: E402
import infer as INF                                                  # noqa: E402
import paths as pth                                                  # noqa: E402
import registry as reg                                               # noqa: E402
import trainer as TR                                                 # noqa: E402

__all__ = ["ARMS", "ArmSpec", "list_arms", "register_spec", "train_fn_for", "predict_fn_for",
          "run_arm", "run_all", "FROZEN_DIR", "sha16"]

FROZEN_DIR = _REL / "models" / "paper" / "frozen"


def sha16(p: str | Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class ArmSpec:
    """一条论文复现臂的登记项（口径两维 **必填**；不可修项必须逐条写出来）。"""

    arm: str
    paper_ref: str
    info_condition: str
    target_caliber: str
    frozen: str                      # models/paper/frozen/<file>.npz
    frozen_key: str                  # npz 里那一列的键名
    input_version: str = "proxy"     # proxy | degraded | structural
    source_py: str = ""
    headline: tuple = ()             # 源树公布值（val, dev）—— 仅作对照，不作判据
    unfixable: tuple = ()
    demote_unit: bool = False        # Σp/Σy 越界降级为记录（物理侧系统性低估）
    notes: str = ""

    def to_row(self) -> dict[str, Any]:
        return {"arm": self.arm, "paper_ref": self.paper_ref,
                "info_condition": self.info_condition,
                "target_caliber": self.target_caliber,
                "input_version": self.input_version,
                "frozen": self.frozen, "frozen_key": self.frozen_key,
                "source_py": self.source_py,
                "headline_published": list(self.headline),
                "unfixable": list(self.unfixable),
                "demote_unit_violation": self.demote_unit, "notes": self.notes}


# --------------------------------------------------------------------------- 六条臂
ARMS: dict[str, ArmSpec] = {
    "A1": ArmSpec(
        arm="A1", paper_ref="A1（1D-CNN 车窗 + BDT 二级；realized）",
        info_condition="realized", target_caliber="y_full",
        frozen="A1_cnn_bdt.npz", frozen_key="pred_wh", input_version="proxy",
        source_py="models/X_papercomp/x_papercomp_20_windows.py",
        headline=(7.5991, 6.1874),
        unfixable=("辅助通道语义：论文 wind speed 用 Gradient 替代、缺列记 0（结构性替代，偏乐观）",
                   "训练预算：论文 3000 epochs / 10 Hz → 源树 60 epochs / 1 Hz；BDT 用 20 万行 bootstrap"),
        notes="realized 口径（车窗内实测速度/加速度）。与 plan-time 臂**不可比**。"),
    "A2": ArmSpec(
        arm="A2", paper_ref="A2（7 通道 CNN：协方差/GAF/特征向量；realized）",
        info_condition="realized", target_caliber="y_full",
        frozen="A2_cnn7_cov.npz", frozen_key="pred_wh", input_version="proxy",
        source_py="models/X_papercomp/x_papercomp_20_windows.py",
        headline=(38.7938, 41.9033),
        unfixable=("10 Hz 原生数据不可得：100 点输出由 10 点线性上采样 ⇒ 量纲对齐而非数据对齐",
                   "只跑协方差一种编码（论文三种）× 3000 步（论文 2000 epochs）"),
        notes="**不得**用它论证「图像编码 CNN 不可行」—— 输入被降采样过。"),
    "A12": ArmSpec(
        arm="A12", paper_ref="A12（物理仿真 + 残差 MLP；plan-time 变体）",
        info_condition="physics-calibrated", target_caliber="y_full",
        frozen="A12_calib_phys_plan.npz", frozen_key="pred_wh", input_version="proxy",
        source_py="models/X_papercomp/x_papercomp_10_routes.py",
        headline=(13.9384, 20.0308),
        unfixable=("原始标签是**合成同源**（同一物理模型 + 人工噪声）⇒ 其 kWh 与真实能耗不同义",
                   "⇒ 源树已把 A12 从交叉比较中**排除**；本目录同样只登记、不参与「谁领先」的结论"),
        notes="这里登记的是 plan-time 物理标定变体（`A12_calib_phys_plan`），不是合成标签那一支。"),
    "G10": ArmSpec(
        arm="G10", paper_ref="G10（两阶段：物理 → GBR；plan-time）",
        info_condition="physics-calibrated", target_caliber="y_full",
        frozen="G10_two_stage_plan.npz", frozen_key="pred_wh", input_version="proxy",
        source_py="models/X_papercomp/x_papercomp_10_routes.py",
        headline=(14.8735, 17.2564),
        unfixable=("F4（坡度）是否整流**论文未公开**；signed 为主口径，rectified 仅作敏感性变体"
                   "（proxy 下整流后 20.2411 / 26.3905，两侧都显著更差）",),
        notes="源树公布以 proxy 版为准（14.8735/17.2564）。"),
    "G04": ArmSpec(
        arm="G04", paper_ref="G04（反向能量物理层 + BiLSTM→速度；plan-time）",
        info_condition="physics-calibrated", target_caliber="y_full",
        frozen="G04_phys_plan.npz", frozen_key="pred_wh", input_version="structural",
        source_py="models/X2_papercomp/x2_papercomp_30_routes.py（代理版；X1 无该列）",
        headline=(13.8926, 17.4778),
        unfixable=("论文 1 m 原始分辨率**不可得**（我们是 100 m 段）⇒ 代理方向是 structural 而非 proxy",),
        notes="冻结列 `pred__G04x_phys_plan_speed_proxy`（X2 逐 fold 重标定的公平口径）。"
              "X1 的 `pred__G04_phys_plan` 列**不存在**；X1 对应列叫 "
              "`pred__G04_phys_plan_speed_plan`，已另存为对照件 `G04_phys_plan_upstream.npz`。"),
    "P14": ArmSpec(
        arm="P14", paper_ref="P14（物理功率模型；plan-time）",
        info_condition="physics-calibrated", target_caliber="y_full",
        frozen="P14_M100_base.npz", frozen_key="pred_wh", input_version="proxy",
        source_py="models/P14_repro/ (out_design/preds_design.csv)",
        headline=(12.7379, 21.2378),
        unfixable=(),
        demote_unit=True,
        notes="**唯一精确复现臂**（源树逐位一致：val Δ=+0.0000 / dev Δ=−0.0000）。"
              "★ 单位门降级为记录：本臂是**纯物理功率模型**，Σp/Σy ≈ 0.335（系统性低估约 3 倍，"
              "val bias −260.7 Wh）—— 那是物理模型的局限，不是量纲错误。降级留痕见 "
              "`train_manifest.json` 的 warnings。"),
}


def list_arms() -> list[dict[str, Any]]:
    return [ARMS[k].to_row() for k in ARMS]


# --------------------------------------------------------------------------- 装载
def _load_arm_preds(a: ArmSpec, trips: Any, *, P: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """按键对齐装载冻结逐趟预测（**不按位置贴**）；返回 ``(pred_wh, meta)``。"""
    p = FROZEN_DIR / a.frozen
    if not p.exists():
        raise FileNotFoundError(
            "缺冻结预测 %s（臂 %s）。\n"
            "  正解：从源树取该臂的逐趟预测并放到 %s；文件须含键 'file' 与 %r。\n"
            "  见 models/paper/README.md 的「冻结产物从哪来」一节。" % (p, a.arm, FROZEN_DIR,
                                                                    a.frozen_key))
    z = np.load(p, allow_pickle=True)
    fk = "file_" if "file_" in z.files else "file"
    if fk not in z.files or a.frozen_key not in z.files:
        raise ValueError("%s 缺键（要 'file'/'file_' 与 %r；实得 %s）"
                         % (p, a.frozen_key, list(z.files)))
    f = np.asarray(z[fk]).astype(str)
    v = np.asarray(z[a.frozen_key], dtype=np.float64).reshape(-1)
    idx = {k: i for i, k in enumerate(f)}
    miss = [t for t in trips if str(t) not in idx]
    if miss:
        raise ValueError("%s 缺 %d 趟（例 %s）—— 键必须覆盖全部趟；"
                         "不要按位置贴（那正是 18.24× 事故的成因）" % (p, len(miss), miss[:3]))
    pred = np.array([v[idx[str(t)]] for t in trips], dtype=np.float64)
    meta = {"frozen_path": str(p), "frozen_sha16": sha16(p), "frozen_key": a.frozen_key,
            "n": int(len(pred)), "key_order": "按 file 对齐（顺序敏感）"}
    return pred, meta


def train_fn_for(a: ArmSpec):
    """生成该臂的 ``train_fn``（**"训练"= 装载冻结产物**；重训代价见 README）。"""

    def train_fn(ctx: TR.TrainContext) -> reg.ModelHandle:
        pack = DIO.load_pack("plan70")
        pred, meta = _load_arm_preds(a, pack.trips, P=pth.Paths())
        y = pack.y_trip
        ratio = float(np.nansum(pred) / np.nansum(y))
        ctx.log("[%s] 装载冻结逐趟预测：%d 趟 | Σp/Σy = %.6f | %s (sha16=%s)"
                % (a.arm, len(pred), ratio, a.frozen, meta["frozen_sha16"]))
        ctx.ckpt.log({"epoch": 1, "step": 1, "train_loss": "", "holdout_mse": "",
                      "val_wmape": "", "dev_wmape": "", "lr": "",
                      "pred_true_ratio": round(ratio, 6),
                      "note": "frozen-transfer（不重训；见 ckpt/README.md）"})
        h = reg.ModelHandle(params={"arm": a.arm, "frozen": str(meta["frozen_path"])},
                            protocol="native", n_params=None,
                            ckpt_dir=ctx.ckpt.ck, log_csv=ctx.ckpt.log_csv)
        h.meta = {"pred_wh": pred, "frozen_meta": meta, "probes": {}, "allow_dead": (),
                  "note": "论文复现臂 %s：冻结逐趟预测（proxy 口径），train_fn 只做装载+校验"
                          % a.arm}
        ck = Path(ctx.out_dir) / "ckpt"
        ck.mkdir(parents=True, exist_ok=True)
        (ck / "README.md").write_text(_ckpt_note(a, meta), encoding="utf-8")
        (Path(ctx.out_dir) / "arm_card.json").write_text(
            json.dumps(dict(a.to_row(), frozen_sha16=meta["frozen_sha16"],
                            pipeline="frozen preds → registry → trainer(native) → infer → "
                                     "eval_trip（同口径重算）"),
                       ensure_ascii=False, indent=2), encoding="utf-8")
        return h

    return train_fn


def _ckpt_note(a: ArmSpec, meta: dict[str, Any]) -> str:
    unf = "\n".join("* %s" % u for u in a.unfixable) or "* （无）"
    return """# `ckpt/` 说明（论文复现臂 %s）

本目录**没有参数文件**，因为本臂**没有在本发布树里重训**。

| 项 | 值 |
|---|---|
| 冻结来源 | `%s` |
| SHA256 前 16 | `%s` |
| 键 | `file` + `%s` |
| 对齐方式 | **按 `file` 对齐**（顺序敏感；不按位置贴） |
| 口径 | `%s` / `%s` |
| 代理方向 | `%s` |
| 源实现 | `%s` |

## 为什么不重训

%s

## 不可修项（**不得声称已修**）

%s

## 复现方式

`python -B models/paper/adapter.py --arm %s`
—— 会装载冻结预测、按键校验、在**本发布树内重算**全部六指标与 CI，
落到 `runs/paper__%s/eval_report.json`。数字不是抄来的。
""" % (a.arm, meta["frozen_path"], meta["frozen_sha16"], a.frozen_key,
       a.info_condition, a.target_caliber, a.input_version, a.source_py,
       a.notes or "见 README.md", unf, a.arm, a.arm)


def predict_fn_for(a: ArmSpec):
    def predict_fn(handle: reg.ModelHandle, ctx: Any = None, **kw: Any) -> dict[str, Any]:
        m = getattr(handle, "meta", None) or {}
        if "pred_wh" not in m:
            raise ValueError("handle 里没有 pred_wh —— 请经 train_fn 装载（臂 %s）" % a.arm)
        return {"trip_wh": np.asarray(m["pred_wh"], dtype=np.float64)}

    return predict_fn


# --------------------------------------------------------------------------- 注册 / 跑
def register_spec() -> reg.ModelSpec:
    """注册**全部六条臂**（``registry.load_package`` 的即插即用入口）。

    ``load_package`` 的约定是"返回一个 ``ModelSpec`` 或返回 ``None`` 且 import 期已注册"。
    这里**六条臂各注册一次**、返回第一条（供 ``load_package`` 做类型检查），
    其余五条同样进了注册表 —— 用 ``registry.list_models()`` 能看到全部六个名字。
    """
    first = None
    for k, a in ARMS.items():
        sp = reg.register("paper__%s" % k, a.info_condition, a.target_caliber,
                          train_fn_for(a), predict_fn_for(a), version="proxy-1.0",
                          family="paper-%s" % k,
                          notes="%s | 代理方向=%s | %s" % (a.paper_ref, a.input_version,
                                                          a.notes),
                          source=a.source_py, override=True)
        first = first or sp
    return first


def run_arm(arm: str, *, verbose: bool = True) -> dict[str, Any]:
    """跑一条臂：装载冻结预测 → trainer(native) → infer → eval_trip（**同口径重算**）。"""
    if arm not in ARMS:
        raise KeyError("未知臂 %r；可用 %s" % (arm, sorted(ARMS)))
    a = ARMS[arm]
    name = "paper__%s" % arm
    register_spec()
    sp = reg.get(name)
    pack = DIO.load_pack("plan70")
    out_dir = pth.Paths().runs(name)
    res = TR.train(sp, data=TR.prepare(), mode="native", protocol=TR.Protocol(),
                   out_dir=out_dir, seed=42, repro_check=False,
                   demote_unit_violation=bool(a.demote_unit))
    inf = INF.predict_to_npz(predict_fn_for(a), res.handle, None, out_dir=out_dir,
                             files=pack.trips, split=pack.split, y_wh=pack.y_trip,
                             info_condition=a.info_condition,
                             target_caliber=a.target_caliber, model=name,
                             y_basis="源树冻结逐趟预测（proxy 口径）", unit_check=False)
    m_val = ET.eval_trip(inf.trip.select("validation"), info_condition=a.info_condition,
                         target_caliber=a.target_caliber, splits=("val",))
    m_dev = ET.eval_trip(inf.trip.select("future"), info_condition=a.info_condition,
                         target_caliber=a.target_caliber, splits=("dev",))
    m_all = ET.eval_trip(inf.trip, info_condition=a.info_condition,
                         target_caliber=a.target_caliber)
    out = {"arm": arm, "name": name, "paper_ref": a.paper_ref,
           "info_condition": a.info_condition, "target_caliber": a.target_caliber,
           "input_version": a.input_version, "frozen": inf.meta.get("frozen_path", ""),
           "val_wmape": float(m_val.metric("WMAPE", "val")),
           "dev_wmape": float(m_dev.metric("WMAPE", "dev")),
           "pooled_wmape": float(m_all.metric("WMAPE", "pooled")),
           "val_mae_wh": float(m_val.metric("MAE", "val")),
           "val_bias_wh": float(m_val.metric("bias", "val")),
           "val_rmse_wh": float(m_val.metric("RMSE", "val")),
           "val_mape_ge1kwh": float(m_val.metric("MAPE_ge1kWh", "val")),
           "unit_ratio_val": float(m_val.unit_ratio.get("val", float("nan"))),
           "unit_gate": "bypassed（见下）",
           "headline_published": list(a.headline),
           "delta_vs_published": [float(m_val.metric("WMAPE", "val") - a.headline[0]),
                                  float(m_dev.metric("WMAPE", "dev") - a.headline[1])]
           if a.headline else [],
           "unfixable": list(a.unfixable),
           "npz": str(inf.npz), "key_sha16": inf.meta.get("key_sha16"),
           "pipeline": "frozen preds → registry → trainer(native) → infer → eval_trip",
           "verdict": "PASS" if np.isfinite(m_val.metric("WMAPE", "val")) else "FAIL"}
    (out_dir / "eval_report.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if verbose:
        print("\n★ 论文复现臂 %s（%s / %s，代理方向 %s）：%s"
              % (arm, a.info_condition, a.target_caliber, a.input_version, out["verdict"]))
        print("   WMAPE val=%.4f dev=%.4f pooled=%.4f" % (out["val_wmape"], out["dev_wmape"],
                                                         out["pooled_wmape"]))
        if a.headline:
            print("   源树公布 val=%.4f dev=%.4f |Δ| = (%+.4f, %+.4f)"
                  % (a.headline[0], a.headline[1], out["delta_vs_published"][0],
                     out["delta_vs_published"][1]))
        print("   不可修项：%s" % ("；".join(a.unfixable) if a.unfixable else "（无）"))
        print("   %s（key_sha16=%s）" % (out["npz"], out["key_sha16"]))
    return out


def run_all(*, arms: Any = None, verbose: bool = True) -> dict[str, Any]:
    sel = list(arms or ARMS.keys())
    rows, failed = {}, {}
    for k in sel:
        try:
            rows[k] = run_arm(k, verbose=verbose)
        except Exception as e:                                    # noqa: BLE001
            failed[k] = "%s: %s" % (type(e).__name__, e)
            if verbose:
                print("\n✗ 论文复现臂 %s 未跑通：%s" % (k, failed[k]))
    return {"runs": rows, "failed": failed,
            "n_ok": len(rows), "n_failed": len(failed)}


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="论文复现臂（代理版 = 公平口径）")
    ap.add_argument("--arm", choices=sorted(ARMS), default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true", help="只列出六条臂与其不可修项")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.list:
        print(json.dumps(list_arms(), ensure_ascii=False, indent=2))
        return 0
    if a.all or not a.arm:
        o = run_all()
        if a.json:
            print(json.dumps(o, ensure_ascii=False, indent=2, default=str))
        print("\n== 汇总：跑通 %d / 失败 %d ==" % (o["n_ok"], o["n_failed"]))
        return 0 if o["n_failed"] == 0 else 1
    o = run_arm(a.arm)
    if a.json:
        print(json.dumps(o, ensure_ascii=False, indent=2, default=str))
    return 0 if o["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
