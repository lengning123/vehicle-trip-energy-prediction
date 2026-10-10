# `ckpt/` 说明（论文复现臂 G04）

本目录**没有参数文件**，因为本臂**没有在本发布树里重训**。

| 项 | 值 |
|---|---|
| 冻结来源 | `/root/autodl-tmp/DSH_release/models/paper/frozen/G04_phys_plan.npz` |
| SHA256 前 16 | `146dfb60b6f3416a` |
| 键 | `file` + `pred_wh` |
| 对齐方式 | **按 `file` 对齐**（顺序敏感；不按位置贴） |
| 口径 | `physics-calibrated` / `y_full` |
| 代理方向 | `structural` |
| 源实现 | `models/X2_papercomp/x2_papercomp_30_routes.py（代理版；X1 无该列）` |

## 为什么不重训

冻结列 `pred__G04x_phys_plan_speed_proxy`（X2 逐 fold 重标定的公平口径）。X1 的 `pred__G04_phys_plan` 列**不存在**；X1 对应列叫 `pred__G04_phys_plan_speed_plan`，已另存为对照件 `G04_phys_plan_upstream.npz`。

## 不可修项（**不得声称已修**）

* 论文 1 m 原始分辨率**不可得**（我们是 100 m 段）⇒ 代理方向是 structural 而非 proxy

## 复现方式

`python -B models/paper/adapter.py --arm G04`
—— 会装载冻结预测、按键校验、在**本发布树内重算**全部六指标与 CI，
落到 `runs/paper__G04/eval_report.json`。数字不是抄来的。
