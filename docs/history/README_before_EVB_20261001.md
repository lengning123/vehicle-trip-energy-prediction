# 整理前README：历史eVED研究背景

2026-10-01归档。原正文保留，链接改为从本目录解析；当前主线请读[根README](../../README.md)。本文件中的运行缓存/模型属于历史本地产物。

# eVED 整车能耗预测复现与模型研究

## 当前研究版本（2026-09-26）

2026-09-26：三组状态递推×三种子9次实验全部完成。整程MAE：并行0.1854±0.0034、隐状态0.1768±0.0013、显式状态0.1785±0.0010 kWh。显式反馈暂未超过隐状态；耗时低估主要集中在含停车路段。详见[实验结论与后续设计](../../reports/state_rollout_v3/实验结论与后续设计.md)、[完整对照表](../../reports/state_rollout_v3/三组状态递推对照报告.md)及[状态反馈诊断](../../reports/state_rollout_v3/状态反馈诊断.md)。

2026-09-25：已清理本地与云端废弃V1/V2训练产物，见[清理清单](../../reports/cleanup_20260925/清理结果与保留清单.md)。基于相同V3数据启动并行分段、隐状态递归、显式状态递归三组×三种子对照，见[冻结协议与启动记录](../../reports/state_rollout_v3/实验协议与启动记录.md)。队列入口为`run_state_rollout_v3.py`。

2026-09-24：五篇论文方法的 V3 同协议对照矩阵已完成并核验。无历史提出模型三种子 MAE 0.1927±0.0016 kWh，与验证集选出的强树基线 0.1911 kWh 接近，不能声称整程任务 SOTA。新车 K=10 个性化模型三种子 MAE 0.2284±0.0089 kWh，但相对 G10 直接回归和强树的双层区间均跨零。见[最终结论](../../reports/paper_v3_comparison/最终结论与下一步.md)、[公平对照报告](../../reports/paper_v3_comparison/论文模型公平对照报告.md)、[三种子确认](../../reports/paper_v3_comparison/个性化三种子确认.md)及[执行记录](../../reports/v3_study/论文模型公平对照执行记录.md)。

完整 eVED 发布包已核验：54 个周文件、22,436,808 行。纯电车只有3辆，不足以支撑原定跨车辆五折及10-shot设计，按用户条件采用 **EV+PHEV 电池净耗电**，不表示PHEV油电总能耗。

当前 V3 缓存 `runs/route_segments_v3_full` 包含3,759条行程（438 EV、3,321 PHEV），排除HEV，保留净回收行程。主实验已完成消融、三种子、车辆/OD五折、时间外推与新车少样本评测。详见 [V3全量实验报告](../../reports/v3_study/V3全量实验报告.md) 和 [数据审计执行记录](../../reports/v3_audit/V3执行记录.md)。

V3入口：`build_route_v3.py`、`train_dual_path_v3.py`、`complete_v3_study.py`、`evaluate_v3.py`、`report_v3_study.py`（均在 `scripts/`）。旧V1/V2与五篇原任务复现保留为历史实验；其总体、标签和划分不同，不得与V3直接作模型性能提升比较。五篇方法现已按统一 V3 行程任务完成适配训练，但这不等于原论文数据与指标的逐项严格复现。

仓库只保留可重复使用的数据管线、五个论文复现实验、统一评测与提出模型。原始数据和训练产物位于 `data/`、`runs/`，均不进入 Git。

## 五个复现实验

| 编号 | 方法 | 数据/训练入口 |
|---|---|---|
| A1 | MODI CNN-BDT 功率预测 | `prepare_modi_eved.py`、`train_a1_eved.py` |
| A2 | GAF/CNN7 功率预测 | `prepare_modi_eved.py`、`train_a2_eved.py` |
| A12 | 物理能耗+残差 MLP | `build_route_eved.py`、`train_a12_eved.py` |
| G10 | Tesla 驾驶特征两阶段/直接回归 | `build_route_eved.py`、`train_g10_eved.py` |
| G04 | 路线感知 BiLSTM+反向物理能耗 | `build_route_eved.py`、`train_g04_eved.py` |

A1/A2 另保留严格版与加强特征版，并通过 `run_strict_enhanced_ablation_cloud.sh` 做同划分消融。A12/G10/G04 的统一入口是 `run_a12_g10_g04_cloud.sh`。

## 核心目录

```text
src/ev_energy/                 可复用数据与提出模型模块
scripts/audit_eved.py          原始 eVED 审计
scripts/prepare_modi_eved.py   A1/A2 严格缓存
scripts/build_eved_enhanced.py A1/A2 加强特征缓存
scripts/build_route_eved.py    A12/G10/G04 统一路线缓存
scripts/build_route_segments.py提出模型 50 m 路段缓存
scripts/train_*.py             独立训练入口
tests/                         公式、结构与泄露边界回归测试
reports/                       已完成的复现报告和图表
```

## 数据协议

eVED 电池功率按 `-HV Battery Current × HV Battery Voltage` 计算，并使用实测时间间隔积分。P0 修订后的 V2 路线数据包含 3,129 条行程，主 trip-disjoint 为 2180/470/479；另提供 vehicle/route 五折和时间外推协议。路线距离来自匹配 GPS 几何，温度只取出发观测。实际未来速度、电流、电压、未来 SOC、实际时长和行程结束统计量不得进入可部署模型。

## 第一版提出模型

`train_dual_path_v1.py` 实现车辆历史个性化的双路径模型：

```text
50 m 路线序列 -> TCN -> 速度头 -> 可微车辆物理 -> 残差头 -> 机理能耗
                     \-> 路线池化 + 直接能耗头 ---------> 直接能耗
过去训练行程 -> 匿名车辆历史编码 -> 个性化物理参数与融合门控
```

构建缓存：

```powershell
python scripts/build_route_segments.py `
  --route-data runs/route_eved_v2 `
  --out runs/route_segments_v2_50m `
  --segment-m 50
```

训练：

```powershell
python scripts/train_dual_path_v1.py `
  --data-dir runs/route_segments_v2_50m `
  --out runs/dual_path_v1_p0 `
  --normalization layer
```

历史向量只使用当前行程之前且属于训练集的行程。vehicle-disjoint 下验证/测试车辆历史强制为空，避免把未知车辆标签泄露进个性化特征。

早期报告中的 0.1656 kWh 使用了已被 P0 修复的旧距离、温度和 padding 协议，不能再作为最终主结果。严格 V2 下，50 m LayerNorm V1 测试 MAE 为 0.2052 kWh，去历史为 0.2261 kWh；旧报告仅保留为研发记录。

## P0–P4 提出模型与完整评测

`train_dual_path_v2.py` 增加多工况辅助头、时序车辆历史注意力、可学习驱动/再生效率、不确定性门控，以及 trip/vehicle/route/time 四种协议：

```powershell
python scripts/train_dual_path_v2.py `
  --data-dir runs/route_segments_v2_50m `
  --out runs/v2_trip_fused_mg_s42 `
  --protocol trip --mode fused --random-history `
  --normalization masked_group --seed 42
```

五随机种子单模型 MAE 为 0.2138±0.0048 kWh，等权集成为 0.2033 kWh；vehicle-disjoint 五折为 0.2831±0.0620 kWh，route-disjoint 五折为 0.2215±0.0297 kWh，时间外推为 0.1740 kWh。P0–P4 的审计、全部消融、few-shot、90% conformal 区间和 9 张图见 [`reports/p0_p4/P0-P4_完整实验报告.md`](../../reports/p0_p4/P0-P4_完整实验报告.md)。

`runs/` 的逐项保留/清理建议见 [`reports/runs_retention_plan.md`](../../reports/runs_retention_plan.md)。
