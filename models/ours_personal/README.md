# `ours_personal/` —— **JJ_pathway 前缀条件化**（个性化是一个开关）

## 口径声明

| 维度 | 取值 |
|---|---|
| `info_condition` | **`realized`** |
| `target_caliber` | **`y_rem@0.2`**（默认；`--phi 0.4|0.6` 可换，`--phi 0` ⇒ `y_full`） |
| 单位 | **Wh**（`Σ 后缀 seg_energy_wh`，精确基） |
| 因果纪律 | 窗口上界 = 切点；切点之后的任何真实值都不参与 |
| 骨架 | `pred = base_rem · exp(clamp(Δ, ±0.7))` |

**为什么是 `realized`**：外生侧 70 项是 plan-time，但个性化侧吃窗口 `[s0, cut]` 内的
**实测**轨迹 ⇒ 只要有实测信息进来，这条臂就**不是** plan-time。
把它和 plan-time 臂排进同一张榜是错的（`registry.leaderboard()` 会直接抛）。

## 模型（与源树 `models/JJ_pathway` 同构）

```
base_rem  = 冻结 SePT-70 plan-time 段级预测 在**后缀段**上求和      （不训练）
Δ         = head( [ 外生 24 箱表征 ⊕ 前缀 patch 表征 ⊕ 二者逐元素积] )
pred      = base_rem · exp( clamp(Δ, ±0.7) )
```

* ``base_rem`` 冻结 ⇒ **Δ=0 时预测严格等于基线**，于是"前缀信息值多少"是**直接测量**。
* 与源树的差别：源树另学一个逐趟门 ``γ = σ(w·[G, ctx])``；本适配器把门并入 ``Δ`` 的
  输出尺度（``log_scale``）。两者可观测效果一致（``γ→0`` ⇔ ``Δ→0``），单参数化更易复现。

## ⭐ 个性化开关

```bash
python -B models/ours_personal/adapter.py --personal on  --phi 0.2   # 本臂的定义形态
python -B models/ours_personal/adapter.py --personal off --phi 0.2   # 退化为纯外生
python -B models/ours_personal/adapter.py --personal on  --phi 0.2 --smoke   # 接口冒烟
```

* `on` ：建个性化分支（前缀 patch 自注意力 + 拼接融合）；
* `off`：**整条分支不建**（不是置零、不是 dropout）⇒ 网络退化为
  「plan-time 外生 + 神经修正」。

两态之差 = **前缀信息的直接测量**（同种子、同划分、同基线，唯一变量就是这个开关）。
注册名相应为 `JJ_pathway-prefix-personal-ON` / `...-OFF`，两态**各自是一条独立臂**。

## ckpt 指针

`runs/JJ_pathway-prefix-personal-{ON,OFF}/ckpt/{best_val.pt, last_1..5.pt}` + `train_log.csv`
+ `train_manifest.json`（含 `protocol` 行与 `unit_ratio`）。推理默认只用 `best_val.pt`。

## 已知局限

1. **前缀是 100 m 段级，不是 1 Hz。** 源树用 1 Hz 真实车速/加速度/停车/驻停时长 →
   `P = max(30 s, ⌈T/64⌉)` 上限 64 块。本适配器用段级特征序列（同源、同键序、可完全重建）。
   ⇒ **不是**源树 `JJ_pathway` 数字的复现，只是**同机制**的可跑实现。
2. **y 基不同**：本臂用 `y_rem@φ` 的**精确 Wh 基**（`Σ 后缀 seg_energy_wh`），
   与 SePT-70 的 `y_full` **量化基**不是一套。跨基只能比无量纲的 WMAPE。
3. **有效趟数随 φ 减少**（短趟 + 大 φ ⇒ 空后缀 ⇒ `cut_ok=False` ⇒ 剔除）。
   φ=0.2/0.4/0.6 的趟数不同，源树记录为 184 趟量级（canonical 是 188/193）。
4. **注册表口径守卫**：`(plan-time, y_rem@φ)` 这个组合被 `dataio.Policy` **显式拒绝**
   （"没有已冻结的对照锚点，拒绝悄悄跑一个无法归位的格子"）。本臂通过
   `info_override="realized"` 声明 realized 口径 —— 特征集（70 项）与信息条件
   （realized）是**两个正交维度**，必须能分开声明。
5. **同种子重训逐位一致未被自动验证**（同 `itransformer/README.md` 第 4 条）。
6. **Δ 的可解释性有限**：`Δ` 是 log 倍率且被 clamp 在 ±0.7 ⇒ 预测的倍率被限制在
   `[e^-0.7, e^0.7] ≈ [0.50, 2.01]`。若某趟真实剩余能量与基线差 3 倍以上，本臂**结构上够不到**。
