# `timexer/` —— **AA_prefixxer**（TimeXer 式前缀交叉注意力）

## 口径声明

| 维度 | 取值 |
|---|---|
| `info_condition` | **`realized`**（外生 70 项是 plan-time；**个性化分支吃实测前缀**） |
| `target_caliber` | **`y_full`**（φ=0，默认）/ `y_rem@φ`（`--phi 0.2|0.4|0.6`） |
| 单位 | **Wh** |
| 因果纪律 | 前缀 patch **只**覆盖 `seg < prefix_len`（切点之前）；`prefix_len` 由 `features.build_cuts` 的 `n_pre` 给出，**不手写算术** |

> 源树 `models/AA_prefixxer` 的自我声明与本目录一致：**hybrid** —— 外生 plan-time /
> 内生 realized，且**窗口上界 ≤ 切点**（"全程未使用切点之后的任何真实值"）。

## 架构（TimeXer 的两个关键点）

```
外生: (B,L,70) --24 箱--> (B,70,24) --per-var Linear + 变量维自注意力-->  (B,70,d)  ┐
                                                                                    ├─ 融合
内生: (B,L,70) --前缀 patchify(≤64)--> (B,P,70) --patch 维自注意力-->     (B,P,d)   ┘
                                          ↓
           ★ 跨注意力：外生**变量 token 当 query**、内生前缀 patch 当 key/value
                          （"每个变量去问：前缀里哪些 patch 与我相关"）
                                          ↓
              pred = base · exp(clamp(Δ, ±0.7))
```

1. **内生序列单独编码**（`PatchEncoder`）：前缀段级序列 → ≤64 个非重叠 patch →
   patch 维自注意力。这是"个性化"的载体。
2. **内生当 key/value、外生当 query 做跨注意力**（`CrossFuse`）：这正是 TimeXer 把
   endogenous 序列接进 exogenous 变量的用法。

**个性化开关** `--personal {on,off}`：

* `on` ：完整双分支（patch 自注意力 + 前缀→变量跨注意力 + 拼接头）；
* `off`：内生分支与跨注意力**都不建**（不是置零、不是 dropout）⇒ 退化为纯外生倒置注意力。

## 跑法

```bash
python -B models/timexer/adapter.py --personal on  --phi 0.0 --smoke   # 接口冒烟（30 步）
python -B models/timexer/adapter.py --personal on  --phi 0.0           # 正式口径
python -B models/timexer/adapter.py --personal off --phi 0.0           # 关掉个性化
python -B models/timexer/adapter.py --personal on  --phi 0.6           # y_rem@0.6
```

## ckpt 指针

正式口径 ⇒ `runs/AA_prefixxer-timexer-ON/ckpt/{best_val.pt, last_1..5.pt}` + `train_log.csv`。
推理默认只用 `best_val.pt`。

## 已知局限

1. **前缀是 100 m 段级，不是 1 Hz。** 源树用 1 Hz patch（`P = max(30 s, ⌈T/64⌉)`，上限 64 块）；
   本适配器用**同一趟的段级特征序列**（同源、同键序、可从输入完全重建）。
   粒度更粗 ⇒ **不是**源树 `AA_prefixxer` 数字的复现，只是**同机制**的可跑实现。
   它的绝对数字**不可**当作源树数字引用。
2. **φ 之间的结论不可外推。** 源树在 φ=0.6 上对基线的最好成绩是 +0.97 pp
   （CI [+0.281, +1.715]），而 φ=0.2 / 0.4 上**没有**稳定增益（甚至略差）。
3. **φ>0 时有效趟数随 φ 减少**（短趟切出空后缀 ⇒ `cut_ok=False` ⇒ 从训练与评估中剔除），
   因此不同 φ 的趟数不同，**不可直接横比样本量**。
4. **`MAPE` 列口径**：`MAPE_ge1kWh` 的分母**只数 `y ≥ 1000 Wh` 的趟**；本臂的
   `y_rem@φ` 在 φ=0.6 时大量趟的剩余能量 < 1 kWh ⇒ 该列的子集很小，CI 很宽，
   **主指标用 WMAPE**（能量加权）。
5. **同种子重训逐位一致未被自动验证**（同 `itransformer/README.md` 第 4 条）。
