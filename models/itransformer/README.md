# `itransformer/` —— **plan-time iTransformer**（倒置注意力）

## 口径声明

| 维度 | 取值 |
|---|---|
| `info_condition` | **`plan-time`**（70 项特征，realized 项 0） |
| `target_caliber` | **`y_full`**（趟级全程能量） |
| 单位 | **Wh** |
| 架构 | **倒置（inverted）**：不把时间步当 token，而把**变量**当 token、时间当特征维 |
| 骨架 | `pred = base_full · exp(clamp(Δ, ±0.7))`，`base_full` = 冻结 SePT-70 趟级预测 |

> ⚠️ **与源树的重大差异（必须写明）**：源树 `models/P_transformer` 的 iTransformer 臂是
> **realized 口径**（`FEATS_REALIZED`：段时长 / 车速 / 分位数 / 停车比），公布
> **8.1017 / 5.8037**（val/dev，另一套 y 基），比较对象是 `REF_FF_MLP_realized`（7.1750/6.8827）。
> **那两个数字与本节任何数字不可互换**：信息条件不同、y 基不同。
> 本适配器实现的是**同架构的 plan-time 版本**（输入换成 70 项 plan-time 特征）。
> 源树自己的结论是「序列注意力在本数据上无稳定增益，倒置形式只改变误差结构」
> —— 本臂不主张相反的结论。

## 架构

```
(B, L, F) --等里程分箱 24 箱--> (B, F, 24)      # 变长序列 → 定长 token
          --per-var Linear-->   (B, F, d)
          --变量维自注意力 ×2--> (B, F, d)      # ★ 注意力在**变量之间**做（inverted）
          --均值池化-->          (B, d)
          --head--> Δ  →  pred = base · exp(clamp(Δ, ±0.7))
```

注意力在变量维而非时间维 —— 这是 iTransformer 论文的核心主张（"inverted"：注意力学
**变量间相关**，而非时间依赖）。

## 跑法

```bash
python -B models/itransformer/adapter.py                # 正式口径（默认）
python -B models/itransformer/adapter.py --quick        # 少步数（仍走完整协议）
python -B models/itransformer/adapter.py --smoke        # **接口冒烟**：30 步 / 1 种子
python -B models/itransformer/adapter.py --d 32 --layers 2 --seed 43
```

`--smoke` 只证明「注册可用 / 链路通 / 能产出标准 `trip_preds.npz`」，
**不追求收敛、不追求精度**；`smoke=true` 会写进 `eval_report.json`，
所以"这个数字是冒烟来的"永远可查。

## ckpt 指针

正式口径走 `mode="protocol"`：`runs/iTransformer-plan-inverted/ckpt/` 下有
`best_val.pt`（**早停只看 train 内部 holdout** `tr[::7][:110]`）与 `last_1..5.pt`。
推理**默认只用 `best_val.pt`**（early stop 选出来的那一个）；改用 `last_*` 必须显式指定并留痕。

## 已知局限

1. **与 `ours_tree` 共享同一份冻结基线** ⇒ 两臂**不独立**。比较的是
   "树 vs 树 + 神经修正"，不是"树 vs 另一个独立模型"。Δ 因此可解释为"神经侧在这一骨架上加了多少"。
2. **序列被压成 24 个等里程箱** ⇒ 箱内的时间顺序被丢弃。这是有意的口径选择
   （与源树 `HH_fusion` / `JJ_pathway` 的 24 箱同构），但意味着本臂**不宜**声称"捕捉了段间动态"。
3. **`Lmax = 2012`** 来自冻结段表（最长趟的段数）；更长的趟会被右侧截断（本数据里没有）。
4. **同种子重训逐位一致未被自动验证**：框架的 `repro_check` 只重跑 `eval_fn`（**不是**重训），
   证明不了重训级复现。`eval_report.json` 的 `meta.determinism` 里如实写了这一点，
   并给出已证的**推理路径确定性**（同权重两次前向逐位一致）。
   要证重训级复现：用同 `--seed` 重跑一次，比对 `trip_preds.npz` 的 `pred_wh` 是否逐位相同。
5. **CPU/GPU 都跑**（框架自动选 `cuda` 若可用）。GPU 上
   `torch.use_deterministic_algorithms(True, warn_only=True)` 的可用性会影响末位一致性。
