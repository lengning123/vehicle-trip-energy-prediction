from pathlib import Path
import json, zipfile, io, hashlib
import pandas as pd
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
D=HERE/'data_pipeline'
m=pd.read_csv(D/'recorded_session_master.csv')
s=json.loads((D/'integrated_dataset_summary.json').read_text(encoding='utf-8'))
a=pd.read_csv(HERE/'label_audit/source_label_audit.csv')
b=pd.read_csv(D/'driving_source_records.csv')
flag=lambda x:x.fillna(False).astype(str).str.lower().isin(['true','1'])
checks={}
def ck(name,value):
    checks[name]=bool(value)
    assert value,name
ck('all1396_unique_and_joined',len(m)==1396 and m.file.is_unique and set(m.file)==set(a.file)==set(b.file))
main=flag(m.main_net_battery_route_candidate)
ck('main_count_matches_summary',int(main.sum())==s['main_net_battery_route_candidates'])
ck('no_temperature_gate',not flag(m.main_view_temperature_required).any())
ck('all_temperature_available_within_6s',flag(b.temperature_available_anywhere).all() and b.first_temperature_delay_s.max()==6)
ck('temperature_cut_bucket_finished',((pd.to_datetime(b.temperature_cut_prediction_utc,utc=True)-pd.to_datetime(b.first_temperature_bucket_utc,utc=True)).dt.total_seconds()==1).all())
ck('positive_source_elapsed_real_seconds',m.duration_s.median()>100)
for tag,available,cutoff in [('source','strict_start_previous_temp_available_utc','start_utc'),('counter','net_window_past_temp_available_utc','battery_net_window_start_utc')]:
    ii=m[available].notna()
    ck(tag+'_historical_temperature_is_past',(pd.to_datetime(m.loc[ii,available],utc=True)<=pd.to_datetime(m.loc[ii,cutoff],utc=True)).all())
ck('net_label_identity_all',(m.battery_net_kwh-(m.battery_discharge_kwh-m.battery_charge_kwh)).abs().max()<1e-8)
ck('main_no_external_evidence',not flag(m.loc[main,'main_view_external_charge_evidence']).any())
ck('main_no_conflicting_iv_overlap',m.loc[main,'overlap_conflicting_iv_rows'].eq(0).all())
ck('main_route_counter_boundaries_equal',m.loc[main,'net_route_label_start_utc'].eq(m.loc[main,'battery_net_window_start_utc']).all() and m.loc[main,'net_route_label_end_utc'].eq(m.loc[main,'battery_net_window_end_utc']).all())
h=main&flag(m.full_source_hybrid_available)
ck('hybrid_count_matches',int(h.sum())==s['full_source_hybrid_main_candidates'])
ck('hybrid_accounting',(m.loc[h,'full_source_hybrid_net_kwh']-m.loc[h,'battery_net_kwh']-m.loc[h,'full_source_hybrid_prefix_iv_net_kwh']-m.loc[h,'full_source_hybrid_suffix_iv_net_kwh']).abs().max()<1e-8)
ck('hybrid_ends_observed_and_counters_synchronous',m.loc[h,'full_source_hybrid_prefix_iv_coverage'].ge(.99).all() and m.loc[h,'full_source_hybrid_suffix_iv_coverage'].ge(.99).all() and flag(m.loc[h,'full_source_hybrid_counter_endpoints_synchronous']).all())
orig=ROOT/'reports/public_dataset_audit_20260930/sources/energyville_V2.zip'
ck('original_zip_unmodified',hashlib.md5(orig.read_bytes()).hexdigest()=='cc2d63758b411c0f8d4b61bdaa1fff1c')
samples=[]
selected=m.loc[main].sort_values('battery_net_kwh').iloc[[0,len(m.loc[main])//2,-1]]
with zipfile.ZipFile(orig) as z,zipfile.ZipFile(D/'driving_signal_views_1s.zip') as signals,zipfile.ZipFile(D/'net_battery_route_views_1s.zip') as routes:
    ck('all1396_signal_csvs',set(signals.namelist())==set(m.file))
    ck('all1396_route_csvs',set(routes.namelist())==set(m.file))
    for _,r in selected.iterrows():
        raw=pd.read_csv(io.BytesIO(z.read(r.zip_path)))
        ts=pd.to_datetime(raw.Timestamp,utc=True)
        manual=0
        for name,col,coef in [('discharge','TotalDischargeKWh3D2',1),('charge','TotalChargeKWh3D2',-1)]:
            q=json.loads(r.battery_net_counter_sample_times)[name]
            first=raw.loc[ts.eq(pd.Timestamp(q['start_utc'])),col].iloc[0]
            last=raw.loc[ts.eq(pd.Timestamp(q['end_utc'])),col].iloc[0]
            manual+=coef*(last-first)
        ck('raw_counter_'+r.file,abs(manual-r.battery_net_kwh)<1e-8)
        clean=pd.read_csv(io.BytesIO(signals.read(r.file)))
        pd.testing.assert_frame_equal(raw[raw.columns],clean[raw.columns],check_dtype=False,rtol=1e-9,atol=1e-10)
        ck('retained_original_columns_'+r.file,set(raw.columns).issubset(set(clean.columns)) and len(raw)==len(clean))
        route=pd.read_csv(io.BytesIO(routes.read(r.file)))
        rt=pd.to_datetime(route.Timestamp,utc=True)
        ck('route_bounds_'+r.file,rt.min()>=pd.Timestamp(r.battery_net_window_start_utc) and rt.max()<=pd.Timestamp(r.battery_net_window_end_utc))
        samples.append(dict(file=r.file,source_duration_s=float((ts.iloc[-1]-ts.iloc[0]).total_seconds()),counter_net_kwh=float(manual),label_net_kwh=float(r.battery_net_kwh),route_rows=len(route)))
for view,n in [('main_net_battery_route_candidates.csv',int(main.sum())),('main_net_battery_internal_consistency_checked.csv',s['main_internal_consistency_checked_candidates']),('full_source_hybrid_net_candidates.csv',int(h.sum()))]:
    v=pd.read_csv(D/view)
    ck('view_'+view,len(v)==n and v.file.is_unique and set(v.file).issubset(set(m.file)))
remaining=main&flag(m.temperature_cut_remaining_net_candidate_available)
ck('remaining_count_matches',int(remaining.sum())==s['temperature_cut_remaining_net_main_candidates'])
ck('remaining_prefix_covered',m.loc[remaining,'temperature_cut_prefix_iv_coverage'].ge(.99).all())
ck('remaining_accounting_identity',(m.loc[remaining,'full_source_hybrid_net_kwh']-m.loc[remaining,'temperature_cut_dropped_iv_observed_net_kwh']-m.loc[remaining,'temperature_cut_remaining_net_candidate_kwh']).abs().max()<1e-8)
ck('remaining_new_cut_is_observed_temp_end',m.loc[remaining,'temperature_cut_remaining_boundary_start_utc'].eq(m.loc[remaining,'temperature_cut_prediction_utc']).all())
ck('remaining_not_mislabeled_predeparture',not flag(m.loc[remaining,'temperature_cut_remaining_is_original_predeparture_task']).any())
rview=pd.read_csv(D/'temperature_cut_remaining_net_candidates.csv')
ck('remaining_export_matches',len(rview)==int(remaining.sum()) and set(rview.file)==set(m.loc[remaining,'file']))
report=dict(status='passed',checks=checks,samples=samples,main_count=int(main.sum()),full_source_hybrid_count=int(h.sum()),scope='source integrity, independent endpoint recalculation and data contract; no prediction accuracy tested')
(HERE/'manager_acceptance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
