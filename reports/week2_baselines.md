# 第 2 周阶段记录：无学习 baseline

已实现 `scripts/evaluate_baselines.py`。它读取 `trip_index.csv`，对每个 trip 重采样到 1 Hz，并评估：

- persistence：未来每个采样点取历史最后一个能耗值；
- history mean：未来每个采样点取过去 60 秒均值。

评估目标是未来 10/30/60 秒累计能耗，指标为 MAE、RMSE、masked-MAPE。结果写入 `runs/baselines/metrics.json`，该目录已被 Git 忽略。

## Test 首轮结果（累计能耗）

| 预测时域 | Persistence MAE / RMSE | History mean MAE / RMSE |
|---|---:|---:|
| 10 s | 0.006413 / 0.198756 | **0.005458 / 0.065618** |
| 30 s | 0.020903 / 0.591542 | **0.013429 / 0.132651** |
| 60 s | 0.042839 / 1.214542 | **0.023657 / 0.224584** |

样本数分别为 2,106,458、2,012,362、1,878,324。masked-MAPE 受接近零能耗目标影响较大，因此当前以 MAE/RMSE 作为主要比较依据。
