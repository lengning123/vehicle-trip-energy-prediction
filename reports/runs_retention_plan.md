# `runs/` 最终保留与清理清单

## 2026-09-25 已执行清理（当前权威状态）

用户已授权并实际完成本地与云端旧实验清理。本地删除约2.03 GiB；云端含早期上传缓存共删除约16.88 GiB，可用空间约22 GiB。最新保留目录、删除清单、恢复边界见[清理结果与保留清单](cleanup_20260925/清理结果与保留清单.md)。V3数据、论文对照、个性化结果、原始数据和重建代码均保留。

**以下内容仅是此前的历史清单，其“必须保留V2”“尚未删除”等表述已经被本次记录替代。**

## 2026-09-22 更新：V3优先，旧清单仅作历史记录

2026-09-23补充：必须保留`paper_v3_comparison/`的正式结果与队列状态；云端`route_paper_v3/`与`paper_power_v3_*/`是当前训练依赖，训练结束前不要清理。各协议的实际工况数组采用硬链接共享，目录表观大小不可简单相加为磁盘实际占用。`paper_v3_power_smoke/`与`paper_v3_a2_smoke/`为测试产物；`paper_v3_stage1_results.tgz`是结果传输副本。仍未执行删除。

下文“最终”“无泄露”“正式结果”均指当时V2阶段，不代表已通过后续原始数据审计。V3发现混入HEV、长采样缺口插值及研究对象边界问题，旧结果不可直接替代当前主实验。

必须保留：`route_segments_v3_full/`（含manifest与守恒验证）、`v3_experiments/`（配置、划分、检查点、预测与评测）、`v3_a1_route_bridge/`（探索性证据）、`route_raw_audit_v3/`（原始审计）。云端强树模型joblib尚未全部同步，应保留云端副本；云端gap10/gap60敏感性缓存也暂保留。

`route_v3_smoke/`、`v3_model_smoke/`、`v3_gpu_smoke/` 属于调试产物，确认正式结果完整后可清理。`v3_results_20260922.tgz` 为传输副本，确认解压结果完整后可清理。旧V1/V2及五篇复现目前仍保留用于追溯，不按本清单自动删除。本轮没有执行删除。体积数字以下方历史盘点为准，不是当前实时占用。

盘点时间：2026-09-20，P0–P4 全部结果同步并通过 SHA-256 校验后。当前本地 `runs/` 约 **2.01 GB**。本清单只给出建议，**没有执行删除**。

## A. 必须保留：最终数据与协议（约 596 MB）

| 路径 | 体积 | 用途 |
|---|---:|---|
| `route_eved_v2/` | 576.66 MB | 无泄露路线级 V2 主缓存；已替代旧版 `route_eved/` |
| `route_segments_v2_50m/` | 19.65 MB | 最终选定 50 m 路段缓存，训练、评估和场景切片共同使用 |
| `p0_protocol_audit.json` | <0.01 MB | 距离、温度、分组互斥、标签长尾和有限值审计证据 |

25 m、100 m 的完整缓存没有下载到本地；对应模型指标已经保留，因 50 m 最优，无需再保存大缓存。

## B. 必须保留：论文正式结果与检查点（约 78 MB）

### 三项 V2 协议复现

- `a12_eved_v2_deployable/`
- `g10_eved_v2_deployable/`
- `g04_eved_v2_route/`
- `route_strong_baselines_v2_trip/`

### V1 主模型与结构消融

- `dual_path_v1_p0/`
- `dual_path_v1_p0_nohistory/`
- `dual_path_v1_p0_25m/`
- `dual_path_v1_p0_100m/`
- `dual_path_v1_p0_maskedgn/`

### V2 主模型、融合和工况误差分解

- `v2_trip_fused_mg_s42/` 至 `v2_trip_fused_mg_s46/`
- `v2_trip_fused_mg_ensemble/`
- `v2_trip_direct_s42/`
- `v2_trip_mechanism_s42/`
- `v2_trip_mechanism_mg_s42/`
- `v2_trip_fused_s42/`
- `v2_trip_mechanism_rule/`
- `v2_trip_mechanism_oracle/`

### 泛化、few-shot 与不确定性

- `v2_vehicle_f0/` 至 `v2_vehicle_f4/`
- `v2_route_f0/` 至 `v2_route_f4/`
- `v2_time/`
- `v2_vehicle_f0_fewshot/` 至 `v2_vehicle_f4_fewshot/`
- `route_baselines_v2_vehicle_f0/` 至 `route_baselines_v2_vehicle_f4/`
- `route_baselines_v2_route_f0/` 至 `route_baselines_v2_route_f4/`
- `route_baselines_v2_time/`

这些目录共同支撑五随机种子、vehicle/route 五折、时间外推、0/1/3/5/10-shot 和 90% conformal 区间，正式论文完成前不建议删。

## C. 可以立即清除：已被 V2 替代或重复归档（约 1.33 GB）

| 路径 | 体积 | 原因 |
|---|---:|---|
| `route_eved/` | 1,231.51 MB | 旧版路线缓存，包含已修复的距离/温度协议问题；已由更小的 `route_eved_v2/` 完整替代 |
| `route_segments_50m/` | 26.69 MB | 旧版 V1 路段缓存；已由 `route_segments_v2_50m/` 替代 |
| `p0_p4_results.tgz` | 75.47 MB | 云端结果传输归档；已解压，且本地/云端 SHA-256 均为 `1437680ec7bac363405f33962607b1968c80f0d79e02e55c953f74389a582fc3` |
| `a1_eved_strict.log` | <0.01 MB | 已退役训练日志 |
| `a1_paper01_b2048_bench.log/.err` | <0.01 MB | 旧批大小性能测试日志 |
| `modi_eved_memmap_paper01.log/.err` | <0.01 MB | 已删除旧缓存留下的孤立日志 |

清除 C 类后，`runs/` 预计从约 **2.01 GB** 降至约 **0.67 GB**。

## D. 论文定稿后可选压缩（约 28 MB）

五个 `v2_vehicle_f*_fewshot/` 内的 `predictions_k*.npz` 占用约 28 MB。正式论文出图和误差审计期间建议保留；定稿后若只需要汇总数值，可只保留每个目录的 `metrics.json`，删除这些预测数组。

## E. 不在当前 `runs/` 中的旧目录

早期清单中的 `modi_eved_memmap_paper01/`、`modi_eved_full/`、各类 `*smoke*`、旧 A1/A2 重复训练、GRU/TCN/linear 简化基线目前已不在本地 `runs/` 中，无需再次处理。旧 A1/A2 的正式证据仍位于 `reports/enhanced_reproduction/source_data/`，不要把报告源数据误删。

## 最终建议

现在最稳妥的操作是只清除 C 类六组目标，保留 A、B、D。若需要我实际执行删除，应再次明确授权；删除前还会按绝对路径复核目标，避免触及最终 V2 数据和模型。
