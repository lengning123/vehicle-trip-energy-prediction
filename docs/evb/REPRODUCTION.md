# EV-B环境、复现与验收

## 先区分阅读与重新执行

阅读已完成报告/指标/逐行CSV不需要训练。原始数据和joblib未上传，新clone不能立即对保存模型作重放；先获取输入、另建复现实验产物。现有本地/云端已完成目录只做只读验收，不再次fit或重建覆盖。

环境来自两轮runtime.json：Python3.12.3、numpy2.3.2、pandas3.0.5、sklearn1.9.1，CPU4线程。核心依赖见[requirements/evb.txt](../../requirements/evb.txt)。其他依赖没有完整原版本锁，安装后另存pip freeze，不能称已完全锁定环境。

在隔离环境可执行：

    python -m pip install -r requirements/evb.txt
    python tests/test_energyville_evb01.py
    python tests/test_energyville_evb02.py

准备代码需要pandas显式纳秒API，保存模型需要匹配sklearn环境。本项目整理没有升级本地包或启动新训练。

## 第一步：在新clone重建R05输入

原包下载与校验见[数据说明](DATA.md)。在项目根目录执行，脚本会写R05目录，已经有原验收产物时应使用新的项目副本，不能覆盖原目录：

    python reports/energyville_label_rebuild_20260930/data_pipeline/rebuild_recorded_sessions.py
    python reports/energyville_label_rebuild_20260930/label_audit/audit_labels.py
    python reports/energyville_label_rebuild_20260930/data_pipeline/integrate_labels_and_routes.py
    python reports/energyville_label_rebuild_20260930/data_pipeline/recheck_charger_fields.py
    python reports/energyville_label_rebuild_20260930/data_pipeline/finalize_views.py

检查artifact_manifest和integrated_dataset_summary，并依据[验收规则](../../reports/energyville_label_rebuild_20260930/验收规则.md)确认源哈希、1396来源、1272剩余候选、符号、计数器/IV窗口与缺口。新环境重建应另存输入/版本/差异，不伪称必然与旧字节hash完全相同。

check_context.py、update_research_context.py、update_research_review.py是有副作用的一次性上下文更新器，**不是**上述数据/模型流程的一环。

## 第二步：EV-B01独立复现

以下以新输出目录EVB01_replay为例；若存在已冻结结果，换新唯一目录：

    python scripts/energyville_evb01_data.py --data reports/energyville_label_rebuild_20260930/data_pipeline --todo reports/energyville_study/EVB01/todo_input_snapshot.md --out reports/energyville_study/EVB01_replay
    python scripts/run_energyville_evb01.py --out reports/energyville_study/EVB01_replay
    python scripts/verify_energyville_evb01.py --out reports/energyville_study/EVB01_replay --models

固定4次fit，不打开新的模型网格。todo使用当时输入快照，不用根目录“已完成”的todo替代冻结协议。验证选模先锁定、再看未来错误；新复现不能把已知开发集重新包装成独立测试。

若已有原模型与输入，只读验收可按[EV-B01运行说明](../../reports/energyville_study/EVB01/运行与交付说明.md)使用现有目录。

## 第三步：EV-B02独立复现

这里prior指上一步产生的可读模型和合同；若使用原prior，则应先核查其文件与环境：

    python scripts/energyville_evb02_data.py --prior reports/energyville_study/EVB01_replay --data reports/energyville_label_rebuild_20260930/data_pipeline --todo reports/energyville_study/EVB02/research_before_execution/todo.md --out reports/energyville_study/EVB02_replay
    python scripts/run_energyville_evb02.py --prior reports/energyville_study/EVB01_replay --out reports/energyville_study/EVB02_replay
    python scripts/repair_energyville_evb02_replay.py --prior reports/energyville_study/EVB01_replay --out reports/energyville_study/EVB02_replay
    python scripts/verify_energyville_evb02.py --prior reports/energyville_study/EVB01_replay --out reports/energyville_study/EVB02_replay

阶段A重放固定模型，不更新权重；阶段B恰好三fit。修复入口零fit，恢复精确共享列/训练历史，保留初版结果。历史事件须end+1秒≤查询cut，同车，125cal标签永不释放。

## 原版本哈希与Git

冻结实验目录和执行代码使用.gitattributes的-text规则保存字节，防止Windows自动行尾转换破坏原hash。根研究文件继续正常文本管理。已有completion/delivery/input manifest不改写；新文档不追记成当年训练输入。

Git不包含joblib/原ZIP/批量R05CSV，完整验收需要这些本地产物。只复制Git文件运行原delivery验收会缺外部文件，这是归档范围限制，不能宣称所有模型已从Git重放。

## 本轮整理验证

核查新导航链接、上传文件范围、Python语法、大小、敏感内容和原EV-B报告/代码字节是否保持；没有重复模型训练。仓库归档清单另见docs/evb/git_archive_manifest.json。
