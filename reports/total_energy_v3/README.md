# V3 total-energy and congestion experiment

Status: all 24 formal runs completed. On 2026-09-28, checkpoints, predictions, logs, configurations and audits were synchronized locally, the report was independently regenerated, and the assertion test script passed.

Deliverables: [full tables](results/T0_T1总能耗实验报告.md), [interpretation and completion checklist](results/结果解读与完成清单.md), plus summary.json and diagnostics.json in results/. No test-driven tuning or additional training was performed.

Frozen protocol: [冻结实验协议](冻结实验协议.md). This supersedes the pending choices in the initial protocol note.

T0 uses estimated battery-plus-fuel energy. T1 additionally uses a coarse congestion indicator derived from the evaluated trip: an ex-post information experiment, not a predeparture deployable result.

Original V3 trips, partitions, battery labels and completed-history rules remain frozen.
