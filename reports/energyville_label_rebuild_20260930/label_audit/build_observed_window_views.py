"""Post-process label audit into explicit observed-counter-window views."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from audit_labels import LABELS
OUT=Path(__file__).resolve().parent
def qs(s):
 s=pd.to_numeric(s,errors='coerce').dropna()
 return {str(k):float(v) for k,v in s.quantile([0,.05,.5,.95,1]).items()} if len(s) else {}
def main():
 all_df=pd.read_csv(OUT/'all_mode_label_audit.csv')
 views=[]
 for _,r in all_df.iterrows():
  for label in LABELS:
   value=r.get(label+'_kwh')
   available=bool(pd.notna(value))
   row={'file':r['file'],'zip_path':r['zip_path'],'vehicle':r['vehicle'],'mode':r['mode'],'label':label,'kwh':value,'observed_counter_window_available':available,
   'whole_source_record_screen':r.get(label+'_tier'),'screen_reason':r.get(label+'_reason'),
   'window_start_utc':r.get(label+'_window_start_utc'),'window_end_utc':r.get(label+'_window_end_utc'),
   'counter_sample_times':r.get(label+'_counter_sample_times'),'source_start_offset_s':r.get(label+'_start_offset_s'),'source_end_offset_s':r.get(label+'_end_offset_s'),
   'max_endpoint_asynchrony_s':r.get(label+'_max_endpoint_asynchrony_s'),'visible_edge_abs_iv_kwh':r.get(label+'_visible_edge_abs_iv_kwh'),
   'quantization_conservative_kwh':r.get(label+'_quantization_conservative_kwh'),'quantization_relative_bound':r.get(label+'_quantization_relative_bound'),
   'iv_window_coverage':r.get(label+'_iv_window_coverage'),'source_long_gap_count':r.get('gap_gt_60s_count'),'source_external_charge_evidence':r.get('external_charge_any_evidence'),
   'counter_negative_steps':r.get(label+'_negative_steps')}
   row['counter_window_monotonic']=bool(available and row['counter_negative_steps']==0)
   # Window availability is not completeness of a physical trip, even with a monotonic lifetime counter.
   row['trip_attribution_needs_review']=bool(r.get('gap_gt_60s_count',0)>0 or r.get('external_charge_any_evidence',False))
   views.append(row)
 pd.DataFrame(views).to_csv(OUT/'observed_counter_window_labels.csv',index=False,encoding='utf-8-sig')
 dr=all_df[all_df['mode']=='driving sessions'].copy()
 strong=dr.battery_net_tier.eq('strong_internal_candidate')
 coverage=dr.battery_net_iv_window_coverage.ge(.99)
 dr['battery_net_iv_crosscheck_engineering_pass']=coverage & dr.battery_net_iv_minus_counter_kwh.abs().le(.02+.03*dr.battery_net_kwh.abs())
 dr['battery_net_counter_and_iv_screen']=strong & dr.battery_net_iv_crosscheck_engineering_pass
 dr['battery_net_quantization_relative_5pct_pass']=dr.battery_net_kwh.abs().ge(.08-1e-8)
 dr['non_drive_residual_same_window_T_over_D']=dr.non_drive_residual_component_drive_delta_kwh/dr.non_drive_residual_component_discharge_delta_kwh.replace(0,np.nan)
 dr.to_csv(OUT/'source_label_audit.csv',index=False,encoding='utf-8-sig')
 joined=dr.merge(pd.read_csv(OUT.parents[1]/'public_dataset_audit_20260930/energyville_candidate_trip_table.csv')[['file','gps_path_km']],on='file',validate='one_to_one')
 res=dr.non_drive_residual_kwh
 summary={'driving_files':len(dr),'observed_counter_window_count':{label:int(dr[label+'_kwh'].notna().sum()) for label in LABELS},
 'same_window_non_drive':{'valid_files':int(res.notna().sum()),'T_over_D_quantiles':qs(dr.non_drive_residual_same_window_T_over_D),'exact_equal_tol_1e_8_count':int(res.abs().le(1e-8).sum()),'within_4wh_count':int(res.abs().le(.004+1e-8).sum()),'within_10wh_count':int(res.abs().le(.01+1e-8).sum()),'positive_gt_20wh_count':int(res.gt(.02+1e-8).sum()),'negative_lt_minus_4wh_count':int(res.lt(-.004-1e-8).sum()),'residual_kwh_quantiles':qs(res),'endpoint_asynchrony_quantiles':qs(dr.non_drive_residual_max_endpoint_asynchrony_s)},
 'drive_net_asynchrony_quantiles':qs(dr.bms_drive_net_max_endpoint_asynchrony_s),
 'drive_net_actual_start_missing_seconds_quantiles':qs(dr.bms_drive_net_start_offset_s),
 'drive_net_actual_end_missing_seconds_quantiles':qs(dr.bms_drive_net_end_offset_s),
 'drive_net_visible_edge_iv_quantiles':qs(dr.bms_drive_net_visible_edge_abs_iv_kwh),
 'battery_net':{'strong_internal_candidate_count':int(strong.sum()),'counter_and_IV_screen_count':int(dr.battery_net_counter_and_iv_screen.sum()),'strong_with_5pct_quantization_count':int((strong&dr.battery_net_quantization_relative_5pct_pass).sum()),'strong_ge_10km_count':int((joined.battery_net_tier.eq('strong_internal_candidate')&joined.gps_path_km.ge(10)).sum()),'per_vehicle':{v:{'strong_count':int(g.battery_net_tier.eq('strong_internal_candidate').sum()),'counter_and_IV_count':int(g.battery_net_counter_and_iv_screen.sum()),'strong_ge_10km_count':int((g.battery_net_tier.eq('strong_internal_candidate')&g.gps_path_km.ge(10)).sum())} for v,g in joined.groupby('vehicle')}},
 'IV_crosscheck_threshold':{'absolute_kwh':.02,'relative_to_abs_counter':.03,'coverage':.99,'meaning':'engineering consistency screen, not physical measurement error bound'},
 'near_zero_counts':{label:{'abs_le_4wh':int(dr[label+'_kwh'].abs().le(.004+1e-8).sum()),'abs_le_10wh':int(dr[label+'_kwh'].abs().le(.01+1e-8).sum()),'negative_lt_minus_4wh':int(dr[label+'_kwh'].lt(-.004-1e-8).sum())} for label in LABELS}}
 (OUT/'observed_window_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()


