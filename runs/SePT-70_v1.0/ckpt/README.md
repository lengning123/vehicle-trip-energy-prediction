# `ckpt/` 说明（SePT-70，树臂）

本目录**没有参数文件**，这是设计如此，不是漏了。

SePT-70 是单次确定性拟合的梯度提升树（`HistGradientBoostingRegressor`，
`random_state=42`、`early_stopping=False`、`max_iter=500`）：

* **没有早停** ⇒ 不存在 `best_val.pt`（"最好的那一步"这个概念不适用）；
* **没有可微参数** ⇒ 不存在 `last_*.pt` 的环形保留；
* 逐位复现靠的是 `../model_card.json` 里的**冻结超参 + 冻结特征顺序 + 冻结数据哈希**
  （`data_v3/segments_100m.csv.gz` SHA16 `298f65ca0dc5c2f5`），而不是靠权重文件。

因此本臂的"推理"必然重跑同一份 `fit`（约 8 秒 / 134,212 段 / 70 特征）。
把树导出成文件反而会引入 sklearn 版本与精度差异，**故意不做**。

跑法：`python -B models/ours_tree/model.py`
