# EV-B02交付与复算入口

本轮已经完成，不要在已完成目录再次fit。完整结果：[主报告](EVB02_实验报告.md)；资源判断：[研究结论](EVB02_研究结论与下一步.md)。根研究文件已更新D14/R07。

## 当前产物与口径

- `completion.json`：修正后的核心实验文件hash；`verification_cloud.json`：独立保存模型/指标/许可复算；`verification_local.json`：同步及图片检查。
- `protocol_frozen.md/run_config.json/input_manifest.json`：执行前协议及输入SHA。四研究文件/todo的执行前后快照分别在research_before_execution与research_after_execution。
- `predictions_validation.csv/predictions_development.csv`：最终7组；`predictions_stageA_*`：固定控制及在线H。开发是旧future193条，不是新独立测试。
- `model_O-{time,motion,quality}.joblib`：仅三新增模型；原S/H/O保留在EVB01，不复制改写其权重。
- `history_*`/`released_history_records.json`：事件来源/特征/快照；125calibration只有元数据引用，没有终局能量进入释放/拟合/评分。
- `conditional_support/reweighted_description/route_history_coverage/route_history_pairs/geometry_diagnostics`：描述诊断，没有事后修正预测或删除样本。
- `initial_replay_outputs.zip/replay_numerical_revision.json`：初版在线CSV舍入问题及零fit修复。最终预测前共享R/S/thermal直接取原features.csv；训练H也直接恢复原值。验证/开发结果未实质变化。
- `code_snapshot`：执行/报告/修复/验收代码和测试；模型所需云环境版本在runtime.json。没有本地fit或环境升级。

## 既有云环境的复算顺序

云目录`/root/autodl-tmp`，Python`/root/miniconda3/bin/python`；云原始输入`data/energyville_evb01_r05`，旧冻结结果`reports/energyville_study/EVB01`。只读验收当前完成目录：

```bash
cd /root/autodl-tmp
/root/miniconda3/bin/python scripts/verify_energyville_evb02.py --prior reports/energyville_study/EVB01 --out reports/energyville_study/EVB02
```

重新实现验证时必须使用**新的唯一输出目录**，不得覆盖本轮：先`energyville_evb02_data.py --prior ... --data ... --out NEW_DIR --todo research_before_execution/todo.md`，然后`run_energyville_evb02.py --prior ... --out NEW_DIR`，再`repair_energyville_evb02_replay.py --prior ... --out NEW_DIR`（零fit，恢复精确共享输入），最后独立verify。这只是复算说明，不授权新的云fit；正式3fit上限不因修复重复。

原执行入口与修复入口分别保留SHA，初版结果归档；修复后completion重新登记核心结果，新增文档/代码快照由最终delivery_manifest登记。没有E5、区间、地图或其他自动训练队列。
