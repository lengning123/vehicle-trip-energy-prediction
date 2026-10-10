# `ckpt/` 说明（论文复现臂 P14）

本目录**没有参数文件**，因为本臂**没有在本发布树里重训**。

| 项 | 值 |
|---|---|
| 冻结来源 | `/root/autodl-tmp/DSH_release/models/paper/frozen/P14_M100_base.npz` |
| SHA256 前 16 | `cc68448b2e87ddea` |
| 键 | `file` + `pred_wh` |
| 对齐方式 | **按 `file` 对齐**（顺序敏感；不按位置贴） |
| 口径 | `physics-calibrated` / `y_full` |
| 代理方向 | `proxy` |
| 源实现 | `models/P14_repro/ (out_design/preds_design.csv)` |

## 为什么不重训

**唯一精确复现臂**（源树逐位一致：val Δ=+0.0000 / dev Δ=−0.0000）。★ 单位门降级为记录：本臂是**纯物理功率模型**，Σp/Σy ≈ 0.335（系统性低估约 3 倍，val bias −260.7 Wh）—— 那是物理模型的局限，不是量纲错误。降级留痕见 `train_manifest.json` 的 warnings。

## 不可修项（**不得声称已修**）

* （无）

## 复现方式

`python -B models/paper/adapter.py --arm P14`
—— 会装载冻结预测、按键校验、在**本发布树内重算**全部六指标与 CI，
落到 `runs/paper__P14/eval_report.json`。数字不是抄来的。
