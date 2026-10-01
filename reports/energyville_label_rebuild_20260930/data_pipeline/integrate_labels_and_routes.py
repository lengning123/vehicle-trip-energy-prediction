"""Merge independent label audit with signal/boundary reconstruction.
Counter-window route geometry and causal thermal snapshots use matching endpoints.
Source record labels are retained separately. No prediction model is trained.
"""
from pathlib import Path
import io,json,zipfile,hashlib
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parent
NAT=np.iinfo(np.int64).min
LABELS=['battery_net','battery_discharge','battery_charge','bms_drive_gross','bms_regen','bms_drive_net','non_drive_residual','external_charge_balance']
def utc(x):return str(pd.Timestamp(int(x),tz='UTC'))
def ns(x):return pd.Timestamp(x).value
def hav(lat,lon):
    a,b=np.radians(lat),np.radians(lon);h=np.sin(np.diff(a)/2)**2+np.cos(a[:-1])*np.cos(a[1:])*np.sin(np.diff(b)/2)**2
    return 6371.0088*2*np.arctan2(np.sqrt(np.clip(h,0,1)),np.sqrt(np.clip(1-h,0,1)))
def bools(s):return s.fillna(False).astype(str).str.lower().isin(['true','1'])
def qs(s):
    s=pd.to_numeric(s,errors='coerce').dropna()
    return {str(q):float(s.quantile(q)) for q in [0,.05,.5,.95,1]} if len(s) else {}
src=pd.read_csv(BASE/'driving_source_records.csv')
lab=pd.read_csv(BASE.parent/'label_audit/source_label_audit.csv')
assert src.file.is_unique and lab.file.is_unique and set(src.file)==set(lab.file)
for c in ['zip_path','vehicle','mode','start_utc','end_utc']:
    assert src.set_index('file')[c].sort_index().equals(lab.set_index('file')[c].sort_index()),c
common=[c for c in src if c in lab and c!='file']
lab=lab.rename(columns={c:c+'_audit' for c in common})
master=src.merge(lab,on='file',validate='one_to_one',how='left')
thermal=pd.read_csv(BASE/'observed_thermal_history.csv.gz')
thermal_by={v:g.sort_values('timestamp_ns').reset_index(drop=True) for v,g in thermal.groupby('vehicle')}
signal=zipfile.ZipFile(BASE/'driving_signal_views_1s.zip')
routezip=zipfile.ZipFile(BASE/'net_battery_route_views_1s.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6)
for i,r in master.iterrows():
    if pd.isna(r.battery_net_window_start_utc):continue
    start,end=ns(r.battery_net_window_start_utc),ns(r.battery_net_window_end_utc)
    d=pd.read_csv(io.BytesIO(signal.read(r.file)))
    tn=pd.to_datetime(d.Timestamp,utc=True).dt.as_unit('ns').astype('int64').to_numpy()
    keep=(tn>=start)&(tn<=end);a=d.loc[keep].copy();at=tn[keep]
    lat=pd.to_numeric(a.GPSLatitude04F,errors='coerce').to_numpy();lon=pd.to_numeric(a.GPSLongitude04F,errors='coerce').to_numpy()
    valid=np.isfinite(lat)&np.isfinite(lon)&(abs(lat)<=90)&(abs(lon)<=180)&~((lat==0)&(lon==0))
    ids=np.flatnonzero(valid)
    route=dict(net_route_rows=len(a),net_route_gps_valid_fraction=float(valid.mean()) if len(a) else 0,
         net_route_unique_points=len(np.unique(np.column_stack([lat[valid],lon[valid]]),axis=0)),
         net_route_label_start_utc=r.battery_net_window_start_utc,net_route_label_end_utc=r.battery_net_window_end_utc,
         net_route_path_plausible_km=np.nan,net_route_gps_gap_chord_km=0,net_route_implausible_links=0,
         net_route_first_gps_offset_s=(at[ids[0]]-start)/1e9 if len(ids) else np.nan,
         net_route_last_gps_offset_s=(end-at[ids[-1]])/1e9 if len(ids) else np.nan,
         net_route_physical_ON_OFF_known=False,net_route_future_path_is_planned_route_proxy=True,
         net_route_future_timestamps_are_training_features=False)
    if len(ids):
        route.update(net_route_start_latitude=float(lat[ids[0]]),net_route_start_longitude=float(lon[ids[0]]),
                     net_route_end_latitude=float(lat[ids[-1]]),net_route_end_longitude=float(lon[ids[-1]]))
    if len(ids)>1:
        links=hav(lat[ids],lon[ids]);dt=np.diff(at[ids])/1e9
        good=(dt>0)&(np.divide(links,dt,out=np.full_like(links,np.inf),where=dt>0)*3600<=200)
        route.update(net_route_path_plausible_km=float(links[good].sum()),
                     net_route_gps_gap_chord_km=float(links[good&(dt>2)].sum()),net_route_implausible_links=int((~good).sum()))
    for key,val in route.items():master.loc[i,key]=val
    # Preserve a distinct full-source candidate by adding only observed IV edges.
    power=pd.to_numeric(d.battery_power_kw,errors='coerce').to_numpy()
    dt=np.diff(tn)/1e9
    observed=np.isfinite(power[:-1])&np.isfinite(power[1:])&(dt>0)&(dt<=2)
    prefix=tn[1:]<=start
    suffix=tn[:-1]>=end
    prefix_length=(start-tn[0])/1e9
    suffix_length=(tn[-1]-end)/1e9
    prefix_coverage=float(dt[observed&prefix].sum()/prefix_length) if prefix_length>0 else 1.0
    suffix_coverage=float(dt[observed&suffix].sum()/suffix_length) if suffix_length>0 else 1.0
    energy=np.where(observed,(power[:-1]+power[1:])/2*dt/3600,0)
    prefix_energy=float(energy[prefix].sum());suffix_energy=float(energy[suffix].sum())
    synchronous=bool(r.battery_net_max_endpoint_asynchrony_s==0)
    available=bool(prefix_coverage>=.99 and suffix_coverage>=.99 and synchronous and r.battery_net_tier in ['A_recorded_window','strong_internal_candidate'])
    for key,val in dict(full_source_hybrid_net_kwh=r.battery_net_kwh+prefix_energy+suffix_energy if available else np.nan,
        full_source_hybrid_available=available,full_source_hybrid_prefix_iv_net_kwh=prefix_energy,
        full_source_hybrid_suffix_iv_net_kwh=suffix_energy,full_source_hybrid_prefix_iv_coverage=prefix_coverage,
        full_source_hybrid_suffix_iv_coverage=suffix_coverage,full_source_hybrid_counter_endpoints_synchronous=synchronous,
        full_source_hybrid_boundary_start_utc=r.start_utc,full_source_hybrid_boundary_end_utc=r.end_utc,
        full_source_hybrid_edge_method='signed observed IV trapezoids; 2-second max; distinct from pure counter label').items():
        master.loc[i,key]=val
    history=thermal_by[r.vehicle];ht=history.timestamp_ns.to_numpy()
    pos=np.searchsorted(ht,start-1_000_000_000,side='right')-1
    if pos>=0:
        p=history.iloc[pos]
        for key,val in dict(net_window_past_temp_available_utc=utc(p.timestamp_ns+1_000_000_000),
             net_window_past_temp_bucket_utc=utc(p.timestamp_ns),net_window_past_temp_age_s=(start-p.timestamp_ns-1_000_000_000)/1e9,
             net_window_past_temp_min=p.min_temperature,net_window_past_temp_max=p.max_temperature,net_window_past_temp_source=p.file).items():
            master.loc[i,key]=val
    routeexport=a[['Timestamp','GPSLatitude04F','GPSLongitude04F','GPS_valid','Odometer_clean_km']].copy()
    routeexport['timestamp_is_evaluation_metadata_not_predictor']=True
    routeexport['source_record_id']=r.file
    routezip.writestr(r.file,routeexport.to_csv(index=False).encode('utf-8'))
    if (i+1)%300==0:print('Aligned label routes',i+1,flush=True)
routezip.close()
# Cross-mode duplicates are owned by driving for the route task. Other modes supply history only.
driving_overlap=[]
for _,r in master.iterrows():
    details=json.loads(r.overlap_details_json)
    driving_overlap.append(any(o['mode']=='driving sessions' for o in details))
master['overlapping_driving_record_intervals']=driving_overlap
master['cross_mode_duplicate_owner']='driving_source_for_route_task; other_modes_for_history_only'
master['label_route_window_aligned']=master.net_route_rows.fillna(0).ge(2)
master['main_label_boundary_is_full_physical_trip']=False
master['suspected_charger_signal_or_mode_overlap']=bools(master.external_charging_evidence)
master['confirmed_external_charging_from_counter_or_joint_direct']=bools(master.external_charge_any_evidence)
master['uncertain_charger_signal_or_mode_overlap']=master.suspected_charger_signal_or_mode_overlap&~master.confirmed_external_charging_from_counter_or_joint_direct
master['main_view_external_charge_evidence']=master.confirmed_external_charging_from_counter_or_joint_direct
strong=master.battery_net_tier.isin(['A_recorded_window','strong_internal_candidate'])
route=(bools(master.movement_evidence)&master.net_route_gps_valid_fraction.ge(.95)&master.net_route_unique_points.ge(2)&master.net_route_path_plausible_km.gt(0))
main=strong&route&~master.main_view_external_charge_evidence&~master.overlapping_driving_record_intervals&master.overlap_conflicting_iv_rows.eq(0)&master.label_route_window_aligned
master['main_net_battery_route_candidate']=main
master['main_view_temperature_required']=False
master['main_view_ready_for_model_training']=False
master['training_remaining_contract']='freeze route mapping and legitimate SOC/ambient snapshots, temporal/route split, train-only preprocessing before fitting'
master['main_view_5pct_counter_quantization_sufficient']=main&bools(master.battery_net_quantization_sufficient_5pct)
master['main_view_internal_consistency_checked']=main&master.battery_net_iv_window_coverage.ge(.99)&master.battery_net_iv_minus_counter_kwh.abs().le(.02+.03*master.battery_net_kwh.abs())
master['main_view_internal_consistency_and_5pct_quantization']=master.main_view_internal_consistency_checked&master.main_view_5pct_counter_quantization_sufficient
reasons=[]
for i,r in master.iterrows():
    why=[]
    if not strong.iloc[i]:why.append('counter_window_not_strong_internal_candidate')
    if not route.iloc[i]:why.append('no_movement_or_insufficient_aligned_route')
    if r.main_view_external_charge_evidence:why.append('external_charging_evidence')
    if r.overlapping_driving_record_intervals:why.append('overlapping_driving_record_requires_ownership')
    if r.overlap_conflicting_iv_rows>0:why.append('conflicting_cross_mode_IV')
    if not r.label_route_window_aligned:why.append('label_route_window_not_aligned')
    reasons.append(';'.join(why) or 'included')
master['main_view_inclusion_reason']=reasons
master.to_csv(BASE/'recorded_session_master.csv',index=False,encoding='utf-8-sig')
master.loc[main].to_csv(BASE/'main_net_battery_route_candidates.csv',index=False,encoding='utf-8-sig')
master.loc[main&bools(master.full_source_hybrid_available)].to_csv(BASE/'full_source_hybrid_net_candidates.csv',index=False,encoding='utf-8-sig')
master.loc[master.main_view_5pct_counter_quantization_sufficient].to_csv(BASE/'main_net_battery_counter_precision_5pct.csv',index=False,encoding='utf-8-sig')
master.loc[master.main_view_internal_consistency_checked].to_csv(BASE/'main_net_battery_internal_consistency_checked.csv',index=False,encoding='utf-8-sig')
manifest={}
for label in LABELS:
    tier=master[label+'_tier']
    view=master[['file','vehicle','zip_path','start_utc','end_utc','movement_evidence','main_view_external_charge_evidence',
                 'main_net_battery_route_candidate']+[c for c in master if c.startswith(label+'_')]].copy()
    view['label_semantics_confirmed_as_pure_traction_or_HVAC']=False
    view['counter_window_route_geometry_aligned']=label=='battery_net'
    view.to_csv(BASE/(label+'_candidate_view.csv'),index=False,encoding='utf-8-sig')
    manifest[label]=dict(records=len(view),tier_counts=tier.value_counts().to_dict(),route_geometry_aligned=(label=='battery_net'))
summary=dict(source_driving_records=len(master),source_record_join='one_to_one_by_file_with_checked_source_boundaries',
    master_file='recorded_session_master.csv',label_tier_counts=master.battery_net_tier.value_counts().to_dict(),
    main_net_battery_route_candidates=int(main.sum()),full_source_hybrid_main_candidates=int((main&bools(master.full_source_hybrid_available)).sum()),full_source_hybrid_main_ge_10km=int((main&bools(master.full_source_hybrid_available)&master.gps_path_plausible_km.ge(10)).sum()),main_ge_10km=int((main&master.net_route_path_plausible_km.ge(10)).sum()),
    confirmed_external_charging_records=int(master.confirmed_external_charging_from_counter_or_joint_direct.sum()),
    uncertain_charger_signal_or_mode_overlap_records=int(master.uncertain_charger_signal_or_mode_overlap.sum()),
    main_with_uncertain_charger_signal_or_mode_overlap=int((main&master.uncertain_charger_signal_or_mode_overlap).sum()),
    main_excluding_uncertain_charger_signal_sensitivity=int((main&~master.uncertain_charger_signal_or_mode_overlap).sum()),
    main_by_vehicle=master.loc[main,'vehicle'].value_counts().to_dict(),
    main_internal_consistency_checked_candidates=int(master.main_view_internal_consistency_checked.sum()),
    main_internal_consistency_and_5pct_quantization_candidates=int(master.main_view_internal_consistency_and_5pct_quantization.sum()),
    main_internal_consistency_checked_ge_10km=int((master.main_view_internal_consistency_checked&master.net_route_path_plausible_km.ge(10)).sum()),
    main_precision_5pct_candidates=int(master.main_view_5pct_counter_quantization_sufficient.sum()),
    main_precision_5pct_ge_10km=int((master.main_view_5pct_counter_quantization_sufficient&master.net_route_path_plausible_km.ge(10)).sum()),
    main_with_first_source_bucket_temperature=int((main&bools(master.temperature_in_first_bucket)).sum()),
    main_with_thermal_anywhere=int((main&bools(master.temperature_available_anywhere)).sum()),
    main_thermal_cut_nonempty=int((main&master.temperature_cut_remaining_rows.ge(2)).sum()),
    main_counter_source_start_offset_s_quantiles=qs(master.loc[main,'battery_net_start_offset_s']),
    main_counter_source_end_offset_s_quantiles=qs(master.loc[main,'battery_net_end_offset_s']),
    main_route_source_distance_minus_window_distance_km_quantiles=qs(master.loc[main,'gps_path_plausible_km']-master.loc[main,'net_route_path_plausible_km']),
    main_iv_counter_difference_kwh_quantiles=qs(master.loc[main&master.battery_net_iv_window_coverage.ge(.99),'battery_net_iv_minus_counter_kwh']),
    main_counter_iv_ge_99pct_records=int((main&master.battery_net_iv_window_coverage.ge(.99)).sum()),
    main_view_reason_counts=master.main_view_inclusion_reason.value_counts().to_dict(),
    main_recovered_despite_no_first_temperature=int((main&~bools(master.temperature_in_first_bucket)).sum()),
    main_gap_gt_2s_records=int((main&master.gap_gt_2s_count.gt(0)).sum()),
    cross_mode_duplicate_owner='Driving task uses driving source intervals once; overlapping parking/charging copies never become extra training targets.',
    counter_route_views_ready=True,model_training_ready=False,
    missing_training_contract='Legitimate SOC/ambient snapshots, route mapping, temporal/route split and train-only preprocessing must be frozen.',
    candidate_label_manifest=manifest)
(BASE/'integrated_dataset_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
assert len(master)==1396 and master.file.is_unique
assert not master.loc[main,'main_view_external_charge_evidence'].any()
known=master.net_window_past_temp_available_utc.dropna()
assert (pd.to_datetime(known,utc=True)<=pd.to_datetime(master.loc[known.index,'battery_net_window_start_utc'],utc=True)).all()
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

