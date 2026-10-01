"""Finalize explicit full-source/remaining-source candidate views and manifest."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parent
m=pd.read_csv(BASE/'recorded_session_master.csv')
s=json.loads((BASE/'summary.json').read_text(encoding='utf-8'))
mapping={'moving_with_external_charging':'moving_with_charger_signal','charging_stationary_record':'stationary_with_charger_signal'}
m['classification']=m.classification.replace(mapping)
for name in ['all_session_inventory.csv','driving_source_records.csv']:
 frame=pd.read_csv(BASE/name);frame['classification']=frame.classification.replace(mapping)
 frame.to_csv(BASE/name,index=False,encoding='utf-8-sig')
s['classification_counts']={mapping.get(k,k):v for k,v in s['classification_counts'].items()}
s['external_charging_evidence_records_is_broad_signal_suspicion_not_confirmation']=True
(BASE/'summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
t=json.loads((BASE/'integrated_dataset_summary.json').read_text(encoding='utf-8'))
def b(x):return x.fillna(False).astype(str).str.lower().isin(['true','1'])
main=b(m.main_net_battery_route_candidate)
prefix_length=m.temperature_cut_dropped_duration_s
coverage=m.temperature_cut_dropped_iv_observed_time_s.div(prefix_length).where(prefix_length.gt(0),1)
available=b(m.full_source_hybrid_available)&coverage.ge(.99)&m.temperature_cut_remaining_rows.ge(2)
m['temperature_cut_prefix_iv_coverage']=coverage
m['temperature_cut_remaining_net_candidate_available']=available
m['temperature_cut_remaining_net_candidate_kwh']=(m.full_source_hybrid_net_kwh-m.temperature_cut_dropped_iv_observed_net_kwh).where(available)
m['temperature_cut_remaining_boundary_start_utc']=m.temperature_cut_prediction_utc
m['temperature_cut_remaining_boundary_end_utc']=m.end_utc
m['temperature_cut_remaining_label_method']='full-source counter plus observed IV edges minus observed prefix IV; approximate measurement chain'
m['temperature_cut_remaining_route_plausible_km']=(m.gps_path_plausible_km-m.temperature_cut_dropped_gps_plausible_km).where(available)
m['temperature_cut_remaining_initial_temp_age_s']=0.0
m['temperature_cut_remaining_is_original_predeparture_task']=False
m['temperature_cut_remaining_training_contract_frozen']=False
m['charger_evidence_status']=np.where(b(m.confirmed_external_charging_from_counter_or_joint_direct),'counter_or_joint_direct_external_evidence',
    np.where(b(m.uncertain_charger_signal_or_mode_overlap),'uncertain_signal_or_mode_overlap','no_external_evidence'))
re=pd.read_csv(BASE/'charger_field_recheck.csv').set_index('file')
m['possible_stale_charger_fields']=m.file.map((re.cp_alone.eq(0)&re.robust_negative_batt.eq(0))).fillna(False)
m.to_csv(BASE/'recorded_session_master.csv',index=False,encoding='utf-8-sig')
m.loc[main].to_csv(BASE/'main_net_battery_route_candidates.csv',index=False,encoding='utf-8-sig')
m.loc[b(m.main_view_5pct_counter_quantization_sufficient)].to_csv(BASE/'main_net_battery_counter_precision_5pct.csv',index=False,encoding='utf-8-sig')
m.loc[b(m.main_view_internal_consistency_checked)].to_csv(BASE/'main_net_battery_internal_consistency_checked.csv',index=False,encoding='utf-8-sig')
m.loc[main&b(m.full_source_hybrid_available)].to_csv(BASE/'full_source_hybrid_net_candidates.csv',index=False,encoding='utf-8-sig')
m.loc[main&available].to_csv(BASE/'temperature_cut_remaining_net_candidates.csv',index=False,encoding='utf-8-sig')
m.loc[main&~b(m.uncertain_charger_signal_or_mode_overlap)].to_csv(BASE/'main_net_battery_excluding_uncertain_charge_sensitivity.csv',index=False,encoding='utf-8-sig')
t['temperature_cut_remaining_net_main_candidates']=int((main&available).sum())
t['temperature_cut_remaining_net_main_ge_10km']=int((main&available&m.temperature_cut_remaining_route_plausible_km.ge(10)).sum())
t['temperature_cut_remaining_accounting_identity_max_abs_kwh']=float((m.loc[available,'full_source_hybrid_net_kwh']-m.loc[available,'temperature_cut_dropped_iv_observed_net_kwh']-m.loc[available,'temperature_cut_remaining_net_candidate_kwh']).abs().max())
t['possible_stale_charger_field_records']=int(b(m.possible_stale_charger_fields).sum())
t['main_excluding_uncertain_charger_ge_10km']=int((main&~b(m.uncertain_charger_signal_or_mode_overlap)&m.net_route_path_plausible_km.ge(10)).sum())
t['remaining_net_label_is_IV_edge_approximation']=True
t['timestamp_timezone_contract']='Explicit source suffix retained per record. Parsed UTC is computational normalization; naive logger timezone not certified, no historical-weather joins.'
(BASE/'integrated_dataset_summary.json').write_text(json.dumps(t,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
readme=f"""# EnergyVille 发布会话重构与可关联候选数据

本目录由公开 V2 ZIP 实际处理产生：全 {s['source_sessions']} 会话、{s['source_rows']:,} 行；原ZIP只读，MD5仍为 {s['source_zip_md5']}。发布件是1秒均值CSV，不是全量未处理CAN。全部1396驾驶source保留，不以首行温度、10km、或单个>2秒缺口删源记录。

## 主要结果

- 首桶有包温477条，但全部1396条在内部都有可信包温；首次延迟中位1秒、95分位3秒、最大6秒。桶结束后可用，因此截断损失中位2秒、95分位4秒；路程损失95分位10.70m，观测IV净能量损失95分位3.459Wh。不是宣称冷启动损耗为零。
- 主计数器窗口与运动/路线关联候选 {t['main_net_battery_route_candidates']} 条，其中 >=10km {t['main_ge_10km']} 条；回收首温缺失样本 {t['main_recovered_despite_no_first_temperature']} 条。
- 同窗IV覆盖>=99%且差值满足预设 .02kWh + 3%|E| 的内部一致性子集 {t['main_internal_consistency_checked_candidates']} 条；兼具计数器保守量化界<=5% {t['main_internal_consistency_and_5pct_quantization_candidates']} 条。该规则筛量测可用性，不是模型误差，也不是独立真值精度。
- full-source hybrid 候选 {t['full_source_hybrid_main_candidates']} 条：计数器共同窗口净量 + 完全可观测前后有符号IV边缘；仅端点同步、两边覆盖>=99%才提供，保留独立候选名称。
- 首温桶结束后的剩余source净电量候选 {t['temperature_cut_remaining_net_main_candidates']} 条（其中 >=10km {t['temperature_cut_remaining_net_main_ge_10km']}），使用 full-source hybrid 减观测前缀IV，目标是剩余路线预算，不能称原始出发前任务。
- 宽充电字段/模式重叠86条只是疑似；计数器或联合直接规则标10条外充迹象。主view仍保留70条待确认信号/模式归属记录，排除这些的敏感性view为1289条。短促ChargeLine电流/电压残留不机械等于充电。
- 28个source与其他模式时间轴重叠：13同IV副本，5存在同桶IV冲突；驾驶route目标拥有驾驶source区间一次，停车/充电副本只作历史与边界证据，冲突从主view排除。

## 可直接读取的文件

| 文件 | 含义 |
|---|---|
| all_session_inventory.csv | 全2310源文件去向、起止、模式、时间/量测统计 |
| driving_source_records.csv | 1396源记录、运动/疑似外充/重叠/温度/缺口，不删source |
| recorded_session_master.csv | 一对一合并独立label审计，源整程、计数器窗口、温度截断均保留；主入口 |
| main_net_battery_route_candidates.csv | 主计数器窗口候选，路程/起终点/过去热快照已对齐该窗口 |
| main_net_battery_internal_consistency_checked.csv | 同窗计数器与IV一致性子集 |
| main_net_battery_counter_precision_5pct.csv | 量化尺度适合百分比评价的子集，不代表预测误差<=5% |
| full_source_hybrid_net_candidates.csv | 原source完整区间的counter+IV边缘候选，非纯counter真值 |
| temperature_cut_remaining_net_candidates.csv | 首温桶结束之后的剩余source预算候选，采用IV边缘近似 |
| temperature_truncation_views.csv | 1396首温裁剪索引及损失诊断；不是独立可训练数据 |
| strict_start_input_snapshots.csv | source起点前已结束桶的历史包温及年龄；首桶SOC/外温列标明并非出发前合法输入 |
| driving_signal_views_1s.zip | 1396原始发布值+质量标记，无未来温度回填；里程哨兵另给clean列 |
| net_battery_route_views_1s.zip | 对齐净电量counter窗口的GPS几何，实际时间仅评估元数据 |
| observed_thermal_history.csv.gz | 所有模式可观测包温、时间与source来源，重叠同桶去重 |
| *_candidate_view.csv | 八种counter组合候选均留存，除battery_net外未另生成几何对齐，不作为训练ready |
| summary.json / integrated_dataset_summary.json | 全源与最终候选统计/合同 |

## 标签与边界合同

主label为电池端净支出 ΔTotalDischarge − ΔTotalCharge；实际端点和as-of年龄在 battery_net_counter_sample_times 中。main net路线仅使用 battery_net_window_start/end 区间，不能把少几秒的标签配整条source路线。source起止、漏边seconds、丢掉的路程均单列。无ON/OFF，不能声称真实出发至熄火已还原。

T/R是多路复用的BMS累计计数器，源内首次/末次往往错开；原始观测与候选全部保留，但T不是已证明纯牵引电功、D−T不是已证明独立HVAC。单个counter自然跨度delta列仅诊断，不能把不同端点直接相减作为组合标签。

IV只积分连续<=2秒且两端有效的观测间隔，原source_iv总和从不冒充跨缺口完整能量。计数器可跨短日志缺口记录累积，但路径缺失、标签区间归属仍需质量flag；65条主候选有>2秒缺口，分层报告。>60秒和counter边界不符合规则者保留候选层，没有删原数据。

温度桶Timestamp代表1秒均值桶开始，最早Timestamp+1秒才可用。严格起点快照只查桶已结束的过去，保存age，不把过期数日温度称当前温度；24个跨模式同桶包温冲突另有统计。src第一桶SOC/外温只是量测诊断，不是已获合法出发快照。

原文本明确时区后缀与naive时间分别标记。_utc为计算归一轴，并不认证naive logger的真实UTC；不能用它猜匹配历史天气或时段机制。

## 复现与尚待冻结

依次运行 rebuild_recorded_sessions.py、兄弟label_audit/audit_labels.py、integrate_labels_and_routes.py、recheck_charger_fields.py、finalize_views.py；以上脚本只读原ZIP，不执行作者脚本。

本目录完成实际量测重构与候选数据交付，没有训练模型。四项训练前合同仍需冻结：路线属性映射、合法SOC/外温快照、时间/路线隔离划分、仅训练期拟合的预处理。全部主view显式 model_training_ready=False，不能将几百万秒或1359条记录解释为多车普适性能证明。
"""
(BASE/'数据处理说明.md').write_text(readme,encoding='utf-8')
manifest={'files':[],'schema':'all source CSV records retained; candidate views distinct from frozen training set'}
for p in sorted(BASE.iterdir()):
 if p.is_file() and p.name not in ['artifact_manifest.json','observed_thermal_history.csv']:
  manifest['files'].append({'name':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(BASE/'artifact_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
assert len(m)==1396 and m.file.is_unique and not b(m.main_view_temperature_required).any()
assert t['temperature_cut_remaining_accounting_identity_max_abs_kwh']<1e-12
print(json.dumps({k:t[k] for k in ['main_net_battery_route_candidates','main_ge_10km','full_source_hybrid_main_candidates','temperature_cut_remaining_net_main_candidates','temperature_cut_remaining_net_main_ge_10km']},ensure_ascii=False),flush=True)

