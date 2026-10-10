# `ours_tree/` —— SePT-70（段级 plan-time 梯度提升树）

> **Segment-level / Plan-time / Tree / 70 特征**，v1.0。
> 身份：原 **A5 臂**（源树 `models/M_final/` 的主结果臂）的**命名固化**。

## 口径声明

| 维度 | 取值 |
|---|---|
| `info_condition` | **`plan-time`**（70 项特征**全部**在出发前可得；realized 项 **0** 个） |
| `target_caliber` | **`y_full`**（趟级全程能量） |
| 单位 | **Wh**（`y_wh` / `pred_wh` 恒为 Wh；kWh 只在读表时 ×1000 一次） |
| 聚合 | 段级预测（逐 100 m）→ **按趟求和** |
| y 基 | `seg_features.y_full`（kWh，**float32 量化**）× 1000 = Wh |

## 主结果（冻结，**逐位可复现**）

| 划分 | 本树实测 | 源树公布 | \|Δ\| |
|---|---:|---:|---:|
| val WMAPE | `10.060503003124914` | `10.0605030031` | **2.49e-11** |
| dev WMAPE | `8.551399087500657` | `8.5513990875` | **6.57e-13** |
| pooled WMAPE | `9.308800282915419` | — | — |

容差 `3e-6`；实测与源树自报的精确复现值**逐位相同**。

六指标（val）：`MAPE(y≥1kWh)` = 10.1145 %，`MAE` = 354.42 Wh，
`bias` = **−163.53 Wh**（系统性低估），`RMSE` = 858.02 Wh，`Σp/Σy` = 0.953582。

## 跑法

```bash
python -B models/ours_tree/model.py            # 全链 + 锚点门，约 8 秒
python -B models/ours_tree/model.py --json     # 只出 JSON
```

链路（**禁止旁路**）：`registry.register → trainer.train(mode="native") → infer.predict_to_npz
→ eval_trip.eval_trip`。

产物：`runs/SePT-70_v1.0/{trip_preds.npz, trip_preds.meta.json, seg_preds.npz,
model_card.json, train_log.csv, train_manifest.json, eval_report.json, ckpt/README.md}`。

## ckpt 指针说明

**本臂没有参数文件**，这是设计如此：

* 单次确定性拟合（`HistGradientBoostingRegressor`，`random_state=42`，
  `early_stopping=False`，`max_iter=500`）⇒ **没有早停**、**没有"最好的那一步"**；
* 逐位复现靠 `model_card.json` 里的**冻结超参 + 冻结特征顺序 + 冻结数据哈希**
  （`data_v3/segments_100m.csv.gz` SHA16 `298f65ca0dc5c2f5`），不是靠权重文件；
* 因此"推理"必然重跑同一份 `fit`（约 8 秒 / 134,212 段）。把树导出成文件反而会引入
  sklearn 版本与精度差异，**故意不做**。

`runs/SePT-70_v1.0/ckpt/` 下只有 `README.md`（这一段的副本）。

## 70 项特征的构成（**位置与 schema 说明**）

| 块 | 列数 | 列名 | 来源文件 |
|---|---:|---|---|
| `PLAN6` | 6 | `seg_len_m`, `ambient_c_med`, `pack_t_med`, `soc_start`, `cum_frac`, `veh2` | `data_v3/segments_100m.csv.gz` |
| **`TT17`** | **17** | **`tt_sl_wmean`, `tt_sl_min`, `tt_sl_max`, `tt_urban_frac`, `tt_seg_speed_kph`, `tt_seg_dur_s`, `tt_cong_ratio`, `tt_speed_deficit`, `tt_dur_s`, `tt_len_m`, `tt_trip_speed_kph`, `tt_delay_s`, `tt_traffic_len_m`, `tt_n_turn`, `tt_n_turn60`, `tt_turn_deg`, `tt_turn_deg_per_km`** | `models/E_stops/seg_features_100m.csv.gz`（**TomTom 派生**） |
| `OSM35` | 35 | 全部 `osm_*`，**去掉** `osm_stoplike_500` / `osm_crossing_500` / `osm_junction_500`（同一信息在 100/250/ndist 上已重复表达） | `models/E_stops/seg_features_100m.csv.gz` |
| `TIME5` | 5 | `dep_hour`, `dep_dow`, `dep_is_weekend`, `dep_hour_sin`, `dep_hour_cos` | 前者来自 E_stops；**后 2 列在装配时派生** |
| `trip_km` | 1 | `trip_km` | `models/E_stops/seg_features_100m.csv.gz` |
| `STOP6` | 6 | `p_stop`, `pred_stop_frac`, `pred_n_ep`, `p_ep`, `pred_dwell_s`, `pred_stop_frac_plan` | `models/E_stops/seg_stop_pred_100m.csv.gz` |
| **合计** | **70** | | |

### ⚠️ TomTom 派生特征（**需用户自备**）

`TT17` 这 17 列**全部来自 TomTom**：`tt_sl_*` = `sections.speedLimit`（km/h）；
`tt_seg_speed_kph` = `progressPoints` 子段速度；`tt_urban_frac` = 城市路占比；
`tt_seg_dur_s` = `Σ ds / v_TT`（严格 plan-time 时间基）；`tt_n_turn*` / `tt_turn_deg*` =
路由路径的转弯几何。覆盖率：段级 0.9741 / 趟级 0.9974（限速子集 0.6895）。

**本发布树不含 TomTom 的取数脚本、不含 API key、不含原始调用记录。**
这 17 列的**名字、位置（在 70 列中的顺序）与语义**如上表，
**取值需要用户自备**：把它们以同名列放进 `models/E_stops/seg_features_100m.csv.gz` 即可。
只保留另外 53 列也能跑通链路，但模型需重训且**与本文公布数字不可比**。

## 已知局限

1. **它是最优的"树"，不是全场最优。** plan-time 口径下它是最优之一；源树另有 realized
   口径的前馈 MLP 2×64（7.2827 / 6.8606），**两者信息条件不同、不可直接比**。
2. **y 基是量化基**（`y_full` 列，float32 量化 ×1000）。换精确 Wh（`Σ seg_energy_wh`）
   会得到 `10.0786 / 8.4808` —— **另一套基，两套数不可混引**。
3. **系统性低估**：val `bias` = −163.5 Wh（`Σp/Σy` = 0.954）。
4. **`ambient_c_med` / `pack_t_med` / `soc_start` 三列在同名于两个来源表**
   （`segments_100m` 与 `seg_features_100m`）。本树的装配顺序与源实现一致
   （先基表、后特征表且**只并入 base 里没有的列**）⇒ 这三列解析到**基表**。
   对复现无影响（哈希门钉死了输入），但重实现时容易搞错方向。
5. **`G_stoptiming/g2_features_100m_trmap.csv.gz` 的 29 列 `t_*` 一列都不进 70 特征。**
   读它 + 核对哈希是"依赖门"的一部分（与源实现同序）；源实现并入后又不用，
   本树**不并入**（对预测逐位无影响）。
