# 本仓库的研究协作入口

开始工作前阅读[实验与Git分支管理总则](EXPERIMENT_GIT_POLICY.md)，再读problem.md、hypotheses.md、experiments.md、decision_log.md、todo.md和最近报告。

- main用于已审方案、已验收证据和可靠代码。修改从codex/分支开始，满足总则对应门槛后才合并；可信负结果可以归档，合并不等于替换基线。
- 多窗口写入使用独立clone/worktree。任何训练/验证进程正在使用的目录都不得切分支、更新代码或覆盖产物。
- 运行前提交并冻结execution_commit、合同、数据/模型哈希、权限与预算。执行、尝试、修复和验收身份分别保存；不回写历史，不用未来信息选模或清洗部署输入。
- 有科学证据的工作更新四研究文件和todo，解释“做了什么→学到了什么→下一步做什么”。纯文档/基础设施无新证据时注明研究结论未变。
- 科学验收不要求机械重训。采用与改动相关的检查；保留125 calibration当前权限及193 development身份，原EV-B01/02冻结版本不覆盖。
- 本总则提供执行与验收规则，不新增用户逐次确认要求；遵守用户已有授权，不自行启动尚未授权的研究队列。

模板入口：[方案](docs/evb/templates/EXPERIMENT_PLAN.md)、[运行身份](docs/evb/templates/RUN_RECORD.json)、[合并验收](docs/evb/templates/MERGE_REVIEW.md)。
