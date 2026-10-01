# 第 2 周阶段记录：预处理与切分协议

## 已完成

- 新增 `src/ev_energy/preprocess.py`：按实际时间戳将单个 trip 重采样为 1 Hz，输出电流、SOC、电压均值、积分能耗和 `observed` 标志。
- 能耗按 `功率(kW) × 时间(h)` 积分，使用相邻时间戳间隔；不把不规则采样行强行解释为 1 秒。
- 新增 `scripts/build_trip_index.py`：对 32,552 个 trip 按起始时间排序，固定划分为 train 22,787、validation 4,883、test 4,882。
- 已通过 2 个单元测试，覆盖官方公式和 500 ms 不规则采样积分。

## 产物

`runs/protocol/trip_index.csv` 为本地生成文件，包含 `veh_id/trip/start_ms/end_ms/rows/split`，已被 Git 忽略，不会把原始数据索引提交到仓库。

## 下一步

在本周剩余工作中，将按该索引提取固定窗口，先实现 persistence、历史均值和线性回归可比 baseline，再记录首轮 MAE/RMSE/Masked-MAPE。
