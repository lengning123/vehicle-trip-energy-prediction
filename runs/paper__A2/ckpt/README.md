# `ckpt/` 说明（论文复现臂 A2）

本目录**没有参数文件**，因为本臂**没有在本发布树里重训**。

| 项 | 值 |
|---|---|
| 冻结来源 | `/root/autodl-tmp/DSH_release/models/paper/frozen/A2_cnn7_cov.npz` |
| SHA256 前 16 | `a80e31cd077b8aa9` |
| 键 | `file` + `pred_wh` |
| 对齐方式 | **按 `file` 对齐**（顺序敏感；不按位置贴） |
| 口径 | `realized` / `y_full` |
| 代理方向 | `proxy` |
| 源实现 | `models/X_papercomp/x_papercomp_20_windows.py` |

## 为什么不重训

**不得**用它论证「图像编码 CNN 不可行」—— 输入被降采样过。

## 不可修项（**不得声称已修**）

* 10 Hz 原生数据不可得：100 点输出由 10 点线性上采样 ⇒ 量纲对齐而非数据对齐
* 只跑协方差一种编码（论文三种）× 3000 步（论文 2000 epochs）

## 复现方式

`python -B models/paper/adapter.py --arm A2`
—— 会装载冻结预测、按键校验、在**本发布树内重算**全部六指标与 CI，
落到 `runs/paper__A2/eval_report.json`。数字不是抄来的。
