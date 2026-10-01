# EV-B代码导航

代码沿用原位置以保持跨脚本导入和冻结代码SHA；不复制为第二套可修改训练入口。EV-B02/code_snapshot是当时执行的只读归档。

## 执行代码

| 文件 | 职责 | 依赖/说明 |
|---|---|---|
| [energyville_evb01_data.py](../../scripts/energyville_evb01_data.py) | R05候选→时点、路线、状态、历史、划分和合同 | numpy/pandas；不拟合 |
| [run_energyville_evb01.py](../../scripts/run_energyville_evb01.py) | B0及R/S/H/O固定树、选模、预测、报告 | 数据适配模块、sklearn/joblib/threadpoolctl |
| [verify_energyville_evb01.py](../../scripts/verify_energyville_evb01.py) | 独立哈希、因果权限、指标/周块和可选模型重放 | 模型重放需要本地joblib与匹配环境 |
| [energyville_evb02_data.py](../../scripts/energyville_evb02_data.py) | 事件释放、在线历史、共同支持、空间检索 | EV-B01模块、scipy.spatial.cKDTree；不拟合 |
| [run_energyville_evb02.py](../../scripts/run_energyville_evb02.py) | 固定模型重放及恰好三Oracle拟合 | 两轮数据模块、EV-B01评分；不重训S/H/O |
| [energyville_evb02_report.py](../../scripts/energyville_evb02_report.py) | 指标与诊断生成实验报告 | 读取产物，不拟合 |
| [repair_energyville_evb02_replay.py](../../scripts/repair_energyville_evb02_replay.py) | CSV舍入导致树阈值变化的零fit修复 | 原共享列/训练历史恢复，留初版与修订记录 |
| [verify_energyville_evb02.py](../../scripts/verify_energyville_evb02.py) | 同模型/权限/指标/合同独立复核 | 需要原S/H/O及三个新Oracle模型 |

## 检查与数据重建

- [EV-B01测试](../../tests/test_energyville_evb01.py)：空间特征、桶结束时点、划分、历史权限、评分。
- [EV-B02测试](../../tests/test_energyville_evb02.py)：事件释放/校准排除/未结束记录不能影响较早查询。
- [R05信号重建](../../reports/energyville_label_rebuild_20260930/data_pipeline/rebuild_recorded_sessions.py)。
- [独立量测审计](../../reports/energyville_label_rebuild_20260930/label_audit/audit_labels.py)。
- [标签与路线关联](../../reports/energyville_label_rebuild_20260930/data_pipeline/integrate_labels_and_routes.py)。
- [充电字段复核](../../reports/energyville_label_rebuild_20260930/data_pipeline/recheck_charger_fields.py)。
- [最终视图与manifest](../../reports/energyville_label_rebuild_20260930/data_pipeline/finalize_views.py)。
- [R06逐行补查](../../reports/energyville_study/EVB01_review_20260930/review_predictions.py)：事后描述，不训练。
- [经理验收](../../reports/energyville_label_rebuild_20260930/manager_acceptance.py)：原发布量测与处理合同抽查。

原check_context.py、update_research_context.py、update_research_review.py是有副作用的一次性研究文档更新器，不属于复现训练入口，不在全量脚本扫描中逐个执行。

## 阅读顺序

先读根README与四研究文件，然后系列索引、任务对应冻结协议，再读适配→运行→验收。修复代码只用于对应数值问题，不能以重新fit掩盖原舍入差异。
