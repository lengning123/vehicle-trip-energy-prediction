# Vehicle Trip Energy Prediction

基于公开车辆日志研究整程能耗预测。当前主线是 **EnergyVille 两辆已知纯电车的整条记录净电量预算**，历史 eVED/V4 油电任务保留为独立研究背景。

## 核心目标

在车辆尚未完成给定路线时，只利用计划路线代理、当时已观测状态和已经完成的历史，估计走完这条路线需要的电量，服务出行电量预算、途中早期更新和低估风险判断。

研究目标是弄清**任务定义、量测边界、可用信息和工况覆盖中，究竟什么限制了预测表现**。通过少量可区分假设的对照实验决定下一步数据与机制工作，而不是持续搜索更大的网络。

业务对象始终是整程预算。有可信量测和实际价值时可研究整程分量，但当前CAN字段不能认证纯牵引或纯HVAC，因此EV-B系列使用电池端带符号净电量。

## 从这里开始

| 需要了解什么 | 入口 |
|---|---|
| EV-B系列做了什么、学到什么 | [系列总索引](reports/energyville_study/README.md) |
| 项目当前问题理解 | [problem.md](problem.md) |
| 当前根因假设及证据 | [hypotheses.md](hypotheses.md) |
| 实验知识记录 | [experiments.md](experiments.md) |
| 为什么选择或暂缓一个方向 | [decision_log.md](decision_log.md) |
| 实验执行方案与状态 | [todo.md](todo.md) |
| 何时建/切分支、执行、验收及合并main | [实验与Git分支管理总则](EXPERIMENT_GIT_POLICY.md) |
| 代码分工与依赖关系 | [代码导航](docs/evb/CODE_MAP.md) |
| 环境、数据重建、运行与验收 | [复现指南](docs/evb/REPRODUCTION.md) |
| 数据来源、许可与Git保管范围 | [数据说明](docs/evb/DATA.md) |

每轮工作先读四个研究文件，再读todo和最近报告。汇报必须区分“做了什么、学到了什么、因此下一步做什么”。根文件是当前理解，实验目录中的协议/代码/研究快照是当时版本。

后续修改从codex/分支开始，执行前冻结提交、合同与数据身份，完成相应验收后再合并main。可信负结果可以合并；替换当前基线另需通过事前效果与风险门槛。多窗口使用独立目录，运行中不切换分支。新方案与验收采用[模板](docs/evb/templates/EXPERIMENT_PLAN.md)，Agent入口见[AGENTS.md](AGENTS.md)。

## 当前进度（2026-10-01整理，研究结论截止EV-B02）

| 实验 | 目的 | 最重要的发现 |
|---|---|---|
| [EV-B01](reports/energyville_study/EVB01/EVB01_实验报告.md) | 固定树比较路线、起始状态、历史与未来Oracle | 状态S验证较路线R改善20.2%，未来仅1.1%且区间跨零；未来运行统计有诊断价值 |
| [EV-B01复核/R06](reports/energyville_study/EVB01_review_20260930/报告充分性与下一阶段判断.md) | 检查长程偏差与历史可得性 | 37条长程中32条高估；原历史中位104.9天，可取得更近的历史，但精度收益未证 |
| [EV-B02](reports/energyville_study/EVB02/EVB02_实验报告.md) | 同H模型在线历史重放；拆耗时、运动和覆盖Oracle | 在线历史验证改善10.8%，开发仅0.6%；覆盖代理更值得审查，三个单块均未过完整门槛 |

当前保留EV-B01验证选择的S作为合法预算参考。下一研究方向是采样/几何定义、覆盖代理和条件路线表示；没有已经启动的EV-B03、区间或网络搜索队列。E5继续搁置。[最新资源判断](reports/energyville_study/EVB02/EVB02_研究结论与下一步.md)

## 任务与评价边界

- 数据为EnergyVille V2公开1秒均值日志，2辆BEV，1396驾驶源，EV-B共同候选1272条。源记录没有认证真实ON/OFF；实走GPS是给定计划路线的离线代理。
- 主量来自累计放电减累计充电，并以有观测的IV补边；有符号、kWh。内部交叉一致性不等于独立校准电表真值。
- EV-B01训练766、验证188、校准125、原未来193。后者已用于研究设计，EV-B02起称开发参考，不再是全新独立测试；125校准终局量尚未消费。
- 特征截至首个可信包温桶结束的t_cut。EV-B02整程预算在t_cut+1秒输出，使原前缀可用；suffix标签边界保持原t_cut，不称严格出发前任务。
- 实际未来耗时、运动与覆盖只作Oracle。其收益不等于可部署收益、交通因果或理论误差下界。
- 本库没有比已评开发数据更晚的源；只有两个已知车，不宣称陌生车普适。旧eVED/PHEV目标、总体和成绩不能与新BEV分数直接比较提升。

## 目录

    docs/evb/                         代码、数据与复现导航
    requirements/evb.txt              EV-B核心依赖
    problem.md / hypotheses.md        当前理解与假设
    experiments.md / decision_log.md  实验知识与选择理由
    todo.md                           执行方案及完成状态
    scripts/*energyville*             EV-B数据适配、训练、修复、验收
    tests/test_energyville_evb*.py     因果时点/公式/重放检查
    reports/energyville_study/         EV-B冻结协议、报告、指标、预测与快照
    reports/energyville_label_rebuild_20260930/
                                      R05数据处理、标签合同及量测审计代码
    reports/literature_review_20260930/ 文献证据与研究方向
    docs/history/                     整理前README及旧研究背景

运行方法见[复现指南](docs/evb/REPRODUCTION.md)。原始ZIP、重建信号包、模型权重、本地云登录配置和论文全文保留在本地，不进入本次Git归档；来源与哈希、重建代码、实验合同、指标和逐行评测结果一并保存。
