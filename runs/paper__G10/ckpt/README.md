# `ckpt/` 说明（论文复现臂 G10）

本目录**没有参数文件**，因为本臂**没有在本发布树里重训**。

| 项 | 值 |
|---|---|
| 冻结来源 | `/root/autodl-tmp/DSH_release/models/paper/frozen/G10_two_stage_plan.npz` |
| SHA256 前 16 | `5444bf7553c6c6ad` |
| 键 | `file` + `pred_wh` |
| 对齐方式 | **按 `file` 对齐**（顺序敏感；不按位置贴） |
| 口径 | `physics-calibrated` / `y_full` |
| 代理方向 | `proxy` |
| 源实现 | `models/X_papercomp/x_papercomp_10_routes.py` |

## 为什么不重训

源树公布以 proxy 版为准（14.8735/17.2564）。

## 不可修项（**不得声称已修**）

* F4（坡度）是否整流**论文未公开**；signed 为主口径，rectified 仅作敏感性变体（proxy 下整流后 20.2411 / 26.3905，两侧都显著更差）

## 复现方式

`python -B models/paper/adapter.py --arm G10`
—— 会装载冻结预测、按键校验、在**本发布树内重算**全部六指标与 CI，
落到 `runs/paper__G10/eval_report.json`。数字不是抄来的。
