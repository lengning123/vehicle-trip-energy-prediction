# `ckpt/` 说明（论文复现臂 A1）

本目录**没有参数文件**，因为本臂**没有在本发布树里重训**。

| 项 | 值 |
|---|---|
| 冻结来源 | `/root/autodl-tmp/DSH_release/models/paper/frozen/A1_cnn_bdt.npz` |
| SHA256 前 16 | `8913382c00eb0df0` |
| 键 | `file` + `pred_wh` |
| 对齐方式 | **按 `file` 对齐**（顺序敏感；不按位置贴） |
| 口径 | `realized` / `y_full` |
| 代理方向 | `proxy` |
| 源实现 | `models/X_papercomp/x_papercomp_20_windows.py` |

## 为什么不重训

realized 口径（车窗内实测速度/加速度）。与 plan-time 臂**不可比**。

## 不可修项（**不得声称已修**）

* 辅助通道语义：论文 wind speed 用 Gradient 替代、缺列记 0（结构性替代，偏乐观）
* 训练预算：论文 3000 epochs / 10 Hz → 源树 60 epochs / 1 Hz；BDT 用 20 万行 bootstrap

## 复现方式

`python -B models/paper/adapter.py --arm A1`
—— 会装载冻结预测、按键校验、在**本发布树内重算**全部六指标与 CI，
落到 `runs/paper__A1/eval_report.json`。数字不是抄来的。
