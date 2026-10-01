# EV-B实验系列索引

更新：2026-10-01。本目录是系列导航，不替代冻结报告。代码保留scripts原路径，历史快照与执行哈希保持原身份。

## 项目目标与当前判断

目标为已知BEV、给定路线、合法起始观测与已完成历史条件下的整程净电预算。先判断数据和信息哪里不足，再决定算法；Oracle与可部署输入分开。

EV-B02之后的当前判断：及时历史可得但相同H没有稳定开发收益，耗时单块弱，coverage可能代理采样/空间与隐藏条件。优先采样、几何与条件支持；S仍是合法参考。详见[研究解读](EVB02/EVB02_研究结论与下一步.md)和[根决策记录](../../decision_log.md)。

## 按实验浏览

| 阶段 | 问题 | 状态/计算 | 主阅读入口 |
|---|---|---|---|
| R05 | 全量原始会话、温度截断和可信标签 | 完成数据重构，未训练 | [管理验收](../energyville_label_rebuild_20260930/研究经理验收与方向.md) |
| EV-B01 | 合法状态与历史的信息价值 | 完成，4次固定fit、B0/R/S/H/O | [报告](EVB01/EVB01_实验报告.md) |
| R06 | 实验报告充分性、偏差与历史支持 | 完成，无新fit | [复核](EVB01_review_20260930/报告充分性与下一阶段判断.md) |
| EV-B02 | 在线历史与time/motion/quality分解 | 完成，控制重放0fit、新Oracle恰好3fit | [报告](EVB02/EVB02_实验报告.md) |
| 下一轮 | coverage组成、几何与合法条件支持 | 方向待冻结，未启动 | [最新判断](EVB02/EVB02_研究结论与下一步.md)、[todo](../../todo.md) |

## 共同成绩与可比较范围

单位kWh，MAE；共同1272 ID、766训练/188验证/125未用校准/193开发参考。历史H的观察权限变化单列。

| 组 | 验证 | 开发参考 | 输入性质 |
|---|---:|---:|---|
| EV-B01 S / EV-B02 S-fixed | 0.3548 | 0.5513 | 合法固定参考 |
| H-frozen | 0.3893 | 0.5653 | 训练期冻结历史 |
| EV-B02 H-online | 0.3473 | 0.5618 | 更早已结束记录在线释放，权重不更新 |
| O-combined | 0.3641 | 0.4510 | 未来Oracle |
| EV-B02 O-time | 0.3757 | 0.5593 | 未来Oracle |
| EV-B02 O-motion | 0.3915 | 0.5154 | 未来Oracle |
| EV-B02 O-quality | 0.3605 | 0.4739 | 未来Oracle |

更低MAE不保证低估尾风险更低。H-online与三个单块均未过预先冻结的完整投入门槛。开发分数用于研究，不是新的独立未来确认。

## 协议、代码与研究快照

- EV-B01：[冻结协议](EVB01/protocol_frozen.md)、[运行说明](EVB01/运行与交付说明.md)、[输入todo](EVB01/todo_input_snapshot.md)、[执行后研究快照](EVB01/research_context_snapshot/)。
- R06：[统计](EVB01_review_20260930/supplemental_diagnostics.json)、[完成todo快照](EVB01_review_20260930/todo_EVB01_completed_snapshot.md)。
- EV-B02：[冻结协议](EVB02/protocol_frozen.md)、[复算入口](EVB02/README.md)、[运行代码快照](EVB02/code_snapshot/)、[执行前研究](EVB02/research_before_execution/)、[执行后研究](EVB02/research_after_execution/)。
- 当前理解：[problem](../../problem.md)、[hypotheses](../../hypotheses.md)、[experiments](../../experiments.md)、[decision_log](../../decision_log.md)、[todo](../../todo.md)。
- 代码导航：[CODE_MAP](../../docs/evb/CODE_MAP.md)；运行导航：[REPRODUCTION](../../docs/evb/REPRODUCTION.md)。

快照里的相对路径保持原项目根语义，不把快照误作当前任务。模型/数据哈希用于绑定当时执行版本；Git正常化文本行尾可能改变某些Windows文档的字节哈希，不能因此改写原manifest。需要严格重放原模型时使用原云端产物或按新目录重建。

## Git中的产物与本地产物

Git保存脚本、测试、Markdown、JSON合同/指标、CSV逐行评测/来源，以及图。原始大ZIP、重建信号ZIP/GZ、R05批量标签CSV和模型joblib不随Git提交，仍留本地/原云端。新clone复算前先按数据说明获取并重建输入；指标阅读不依赖模型文件。
