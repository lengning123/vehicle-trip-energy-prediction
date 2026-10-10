# `ckpt/` 说明（论文复现臂 A12）

本目录**没有参数文件**，因为本臂**没有在本发布树里重训**。

| 项 | 值 |
|---|---|
| 冻结来源 | `/root/autodl-tmp/DSH_release/models/paper/frozen/A12_calib_phys_plan.npz` |
| SHA256 前 16 | `e13d7e98e1db7e32` |
| 键 | `file` + `pred_wh` |
| 对齐方式 | **按 `file` 对齐**（顺序敏感；不按位置贴） |
| 口径 | `physics-calibrated` / `y_full` |
| 代理方向 | `proxy` |
| 源实现 | `models/X_papercomp/x_papercomp_10_routes.py` |

## 为什么不重训

这里登记的是 plan-time 物理标定变体（`A12_calib_phys_plan`），不是合成标签那一支。

## 不可修项（**不得声称已修**）

* 原始标签是**合成同源**（同一物理模型 + 人工噪声）⇒ 其 kWh 与真实能耗不同义
* ⇒ 源树已把 A12 从交叉比较中**排除**；本目录同样只登记、不参与「谁领先」的结论

## 复现方式

`python -B models/paper/adapter.py --arm A12`
—— 会装载冻结预测、按键校验、在**本发布树内重算**全部六指标与 CI，
落到 `runs/paper__A12/eval_report.json`。数字不是抄来的。
