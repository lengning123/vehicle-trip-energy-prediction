"""Process all EnergyVille published 1-second mean records. No raw CAN reconstruction.
Source ZIP unchanged; temperature optional; label qualification separate.
"""
from pathlib import Path
import io,json,hashlib,zipfile
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parent
PROJECT=BASE.parents[2]
ZIP=PROJECT/'reports/public_dataset_audit_20260930/sources/energyville_V2.zip'
COUNTERS={'discharge':'TotalDischargeKWh3D2','charge':'TotalChargeKWh3D2','drive':'BMS_kwhDriveDischargeTotal','regen':'BMS_kwhRegenChargeTotal'}
NAT=np.iinfo(np.int64).min
def num(df,name):
    return pd.to_numeric(df[name],errors='coerce').to_numpy(dtype=float) if name in df else np.full(len(df),np.nan)
def utc(ns):
    return str(pd.Timestamp(int(ns),tz='UTC')) if ns is not None and ns!=NAT else None
def haversine(lat,lon):
    a,b=np.radians(lat),np.radians(lon)
    h=np.sin(np.diff(a)/2)**2+np.cos(a[:-1])*np.cos(a[1:])*np.sin(np.diff(b)/2)**2
    return 6371.0088*2*np.arctan2(np.sqrt(np.clip(h,0,1)),np.sqrt(np.clip(1-h,0,1)))
def cdelta(x,tn,cut=None):
    ids=np.flatnonzero(np.isfinite(x)&(tn<=cut if cut is not None else True))
    return (float(x[ids[-1]]-x[ids[0]]),int(tn[ids[0]]),int(tn[ids[-1]])) if len(ids)>1 else (None,None,None)
def read(z,name):
    df=pd.read_csv(io.BytesIO(z.read(name)))
    ts=pd.to_datetime(df.Timestamp,utc=True,errors='coerce').dt.as_unit('ns').astype('int64').to_numpy()
    return df,ts
def qs(s):
    s=pd.to_numeric(s,errors='coerce').dropna()
    return {str(q):float(s.quantile(q)) for q in [0,.05,.5,.95,1]} if len(s) else {}
BASE.mkdir(parents=True,exist_ok=True)
z=zipfile.ZipFile(ZIP)
names=sorted(n for n in z.namelist() if n.endswith('.csv') and 'sessions/' in n)
records,temps,charges,drive_cache=[],[],[],{}
cz=zipfile.ZipFile(BASE/'driving_signal_views_1s.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6)
for no,name in enumerate(names):
    df,alltn=read(z,name);file=name.rsplit('/',1)[-1];mode=name.split('/')[-2];vehicle=file.split('_')[0]
    validtime=alltn!=NAT;missingtime=int((~validtime).sum());df=df.loc[validtime].copy();tn=alltn[validtime]
    if not len(tn):
        records.append(dict(file=file,zip_path=name,vehicle=vehicle,mode=mode,source_rows=len(alltn),classification='no_valid_timestamp'))
        continue
    t0,t1=int(tn[0]),int(tn[-1]);dt=np.diff(tn)/1e9;duration=(t1-t0)/1e9
    power=num(df,'BattVoltage132')*num(df,'RawBattCurrent132')/1000
    ivok=np.isfinite(power[:-1])&np.isfinite(power[1:])&(dt>0)&(dt<=2)
    iv=np.where(ivok,(power[:-1]+power[1:])/2*dt/3600,0)
    ivd=np.where(ivok,(np.maximum(power[:-1],0)+np.maximum(power[1:],0))/2*dt/3600,0)
    ivc=np.where(ivok,(np.maximum(-power[:-1],0)+np.maximum(-power[1:],0))/2*dt/3600,0)
    lo,hi=num(df,'BMSminPackTemperature'),num(df,'BMSmaxPackTemperature')
    tempok=np.isfinite(lo)&np.isfinite(hi)&(lo>=-40)&(hi<=100)&(lo<=hi)
    if tempok.any():
        ids=np.flatnonzero(tempok)
        temps.append(pd.DataFrame(dict(vehicle=vehicle,timestamp_ns=tn[ids],min_temperature=lo[ids],max_temperature=hi[ids],file=file,mode=mode)))
    cp,cc,cv,dc=[num(df,c) for c in ['ChargeLinePower264','ChargeLineCurrent264','ChargeLineVoltage264','FC_dcCurrent']]
    active=(cp>.5)|((cc>.5)&(cv>50))|(dc>.5)
    if active.any():
        ids=np.flatnonzero(active)
        charges.append(pd.DataFrame(dict(vehicle=vehicle,timestamp_ns=tn[ids],file=file,mode=mode)))
    speed=num(df,'DI_uiSpeed');move=np.isfinite(speed)&(speed>1)
    oraw=num(df,'Odometer3B6');obad=np.isfinite(oraw)&((oraw>=4294967)|(oraw<0));odo=oraw.copy();odo[obad]=np.nan
    oi=np.flatnonzero(np.isfinite(odo));odiff=np.diff(odo[oi]) if len(oi)>1 else np.array([])
    ododelta=float(odo[oi[-1]]-odo[oi[0]]) if len(oi)>1 else None
    lat,lon=num(df,'GPSLatitude04F'),num(df,'GPSLongitude04F')
    gok=np.isfinite(lat)&np.isfinite(lon)&(abs(lat)<=90)&(abs(lon)<=180)&~((lat==0)&(lon==0));gi=np.flatnonzero(gok)
    if len(gi)>1:
        links=haversine(lat[gi],lon[gi]);gdt=np.diff(tn[gi])/1e9
        plausible=(gdt>0)&(np.divide(links,gdt,out=np.full_like(links,np.inf),where=gdt>0)*3600<=200)
        continuous=plausible&(gdt<=2)
        rawpath=float(links.sum());path=float(links[plausible].sum());contpath=float(links[continuous].sum())
        gapchord=float(links[plausible&(gdt>2)].sum());jumps=int((~plausible).sum());motionlinks=int(((links>.01)&plausible).sum())
    else:
        links=np.array([]);plausible=continuous=np.array([],dtype=bool)
        rawpath=path=contpath=gapchord=None;jumps=motionlinks=0
    counters={k:num(df,c) for k,c in COUNTERS.items()}
    r=dict(file=file,zip_path=name,vehicle=vehicle,mode=mode,source_rows=len(alltn),valid_timestamp_rows=len(tn),
        missing_timestamp_rows=missingtime,start_utc=utc(t0),end_utc=utc(t1),start_ns=t0,end_ns=t1,duration_s=duration,
        nonpositive_time_steps=int((dt<=0).sum()),max_gap_s=float(dt.max()) if len(dt) else 0,
        gap_gt_2s_count=int((dt>2).sum()),gap_gt_2s_time_s=float(dt[dt>2].sum()),
        gap_gt_60s_count=int((dt>60).sum()),gap_gt_60s_time_s=float(dt[dt>60].sum()),
        iv_observed_time_s=float(dt[ivok].sum()),iv_observed_time_fraction=float(dt[ivok].sum()/duration) if duration>0 else None,
        iv_observed_net_kwh=float(iv.sum()),iv_observed_discharge_kwh=float(ivd.sum()),iv_observed_charge_kwh=float(ivc.sum()),
        charge_active_own_rows=int(active.sum()),charge_active_own_first_utc=utc(tn[active][0]) if active.any() else None,
        charge_active_own_last_utc=utc(tn[active][-1]) if active.any() else None,
        speed_valid_rows=int(np.isfinite(speed).sum()),speed_positive_motion_rows=int(move.sum()),
        odometer_sentinel_rows=int(obad.sum()),odometer_valid_rows=len(oi),odometer_filtered_delta_km=ododelta,
        odometer_negative_steps=int((odiff<-.0011).sum()),
        odometer_first_offset_s=(tn[oi[0]]-t0)/1e9 if len(oi) else None,odometer_last_offset_s=(t1-tn[oi[-1]])/1e9 if len(oi) else None,
        gps_valid_fraction=float(gok.mean()),gps_unique_points=len(np.unique(np.column_stack((lat[gok],lon[gok])),axis=0)),
        gps_path_unfiltered_km=rawpath,gps_path_plausible_km=path,gps_continuous_observed_km=contpath,
        gps_gap_chord_km=gapchord,gps_implausible_links=jumps,gps_motion_links_gt_10m=motionlinks,
        temperature_valid_rows=int(tempok.sum()),temperature_in_first_bucket=bool(tempok[0]),temperature_available_anywhere=bool(tempok.any()),
        soc_in_first_bucket=float(num(df,'SOCave292')[0]),ambient_in_first_bucket=float(num(df,'VCRIGHT_tempAmbientRaw')[0]),
        source_record_boundary='first_valid_published_timestamp_to_last_valid_published_timestamp',physical_on_off_boundary_known=False)
    for k,x in counters.items():
        delta,first,last=cdelta(x,tn)
        r.update({f'{k}_counter_individual_span_delta_kwh':delta,f'{k}_counter_first_utc':utc(first),f'{k}_counter_last_utc':utc(last),
                  f'{k}_counter_first_offset_s':(first-t0)/1e9 if first else None,f'{k}_counter_last_offset_s':(t1-last)/1e9 if last else None})
    movement=bool(move.any() or (ododelta is not None and ododelta>.1) or (path is not None and path>.2 and motionlinks>0))
    r['movement_evidence']=movement
    r['classification']=('moving_with_charger_signal' if active.any() else 'moving_recorded_session') if movement else ('stationary_with_charger_signal' if active.any() else 'no_confirmed_movement')
    if tempok.any():
        j=np.flatnonzero(tempok)[0];tempns=int(tn[j]);cut=tempns+1_000_000_000;later=np.flatnonzero(tn>=cut)
        r.update(dict(first_temperature_bucket_utc=utc(tempns),first_temperature_delay_s=(tempns-t0)/1e9,
            temperature_cut_prediction_utc=utc(cut),temperature_cut_dropped_duration_s=min(duration,(cut-t0)/1e9),
            temperature_cut_remaining_rows=len(later),temperature_cut_remaining_duration_s=(t1-tn[later[0]])/1e9 if len(later) else 0,
            temperature_cut_min_temperature=float(lo[j]),temperature_cut_max_temperature=float(hi[j]),
            temperature_cut_dropped_iv_observed_net_kwh=float(iv[tn[1:]<=cut].sum()),
            temperature_cut_dropped_iv_observed_discharge_kwh=float(ivd[tn[1:]<=cut].sum()),
            temperature_cut_dropped_iv_observed_charge_kwh=float(ivc[tn[1:]<=cut].sum()),
            temperature_cut_dropped_iv_observed_time_s=float(dt[ivok&(tn[1:]<=cut)].sum()),
            temperature_cut_dropped_gps_plausible_km=float(links[plausible&(tn[gi[1:]]<=cut)].sum()) if len(gi)>1 else 0,
            temperature_cut_dropped_gps_continuous_km=float(links[continuous&(tn[gi[1:]]<=cut)].sum()) if len(gi)>1 else 0))
        for k,x in counters.items():
            delta,first,last=cdelta(x,tn,cut)
            r.update({f'temperature_cut_dropped_{k}_counter_delta_kwh':delta,f'temperature_cut_dropped_{k}_counter_first_utc':utc(first),f'temperature_cut_dropped_{k}_counter_last_utc':utc(last)})
    r['source_timestamp_timezone_explicit']=bool(str(df.Timestamp.iloc[0]).endswith('+00:00') or str(df.Timestamp.iloc[0]).endswith('Z'))
    r['parsed_utc_is_computational_axis_not_logger_timezone_certification']=True
    records.append(r)
    if mode=='driving sessions':
        drive_cache[file]=(tn,power)
        clean=df.copy();clean['GPS_valid']=gok;clean['Odometer_valid']=~obad&np.isfinite(oraw);clean['Odometer_clean_km']=odo
        clean['battery_power_kw']=power;clean['temperature_pair_valid']=tempok;clean['external_charging_bucket']=active
        clean['seconds_since_previous_bucket']=np.r_[np.nan,dt];clean['source_record_id']=file
        clean['signal_bucket_available_utc']=pd.to_datetime(tn+1_000_000_000,utc=True)
        cz.writestr(file,clean.to_csv(index=False).encode('utf-8'))
    if (no+1)%250==0:print(f'Processed {no+1}/{len(names)} source sessions',flush=True)
cz.close()
inventory=pd.DataFrame(records)
inventory.to_csv(BASE/'all_session_inventory.csv',index=False,encoding='utf-8-sig')
driving=inventory.loc[inventory['mode'].eq('driving sessions')].copy()
thermal=pd.concat(temps,ignore_index=True);charging=pd.concat(charges,ignore_index=True)
charge_by={f:g for f,g in charging.groupby('file')}
thermal.sort_values(['vehicle','timestamp_ns','file'],inplace=True)
conflicts=thermal.groupby(['vehicle','timestamp_ns'])[['min_temperature','max_temperature']].nunique().max(axis=1).gt(1)
thermal=thermal.drop_duplicates(['vehicle','timestamp_ns'],keep='last')
thermal.to_csv(BASE/'observed_thermal_history.csv.gz',index=False,encoding='utf-8-sig',compression='gzip')
thermal_by={v:g.reset_index(drop=True) for v,g in thermal.groupby('vehicle')}
for idx,r in driving.iterrows():
    previous=thermal_by[r.vehicle]
    prior_pos=np.searchsorted(previous.timestamp_ns.to_numpy(),r.start_ns-1_000_000_000,side='right')-1
    if prior_pos>=0:
        p=previous.iloc[prior_pos];age=(r.start_ns-p.timestamp_ns-1_000_000_000)/1e9
        for key,val in {'strict_start_previous_temp_bucket_utc':utc(p.timestamp_ns),'strict_start_previous_temp_available_utc':utc(p.timestamp_ns+1_000_000_000),
            'strict_start_previous_temp_age_s':age,'strict_start_previous_temp_min':p.min_temperature,'strict_start_previous_temp_max':p.max_temperature,
            'strict_start_previous_temp_source':p.file,'strict_start_previous_temp_source_mode':p['mode']}.items():driving.loc[idx,key]=val
    other=inventory.loc[(inventory.vehicle==r.vehicle)&(inventory.file!=r.file)&(inventory.start_ns<r.end_ns)&(inventory.end_ns>r.start_ns)]
    descriptions=[];same_iv=bad_iv=charge_overlap=0;tn,power=drive_cache[r.file]
    for _,o in other.iterrows():
        start=max(int(r.start_ns),int(o.start_ns));end=min(int(r.end_ns),int(o.end_ns))
        odf,otn=read(z,o.zip_path);op=num(odf,'BattVoltage132')*num(odf,'RawBattCurrent132')/1000
        common,ri,oi=np.intersect1d(tn,otn,return_indices=True);valid=np.isfinite(power[ri])&np.isfinite(op[oi])
        close=np.isclose(power[ri],op[oi],rtol=1e-7,atol=1e-7)&valid
        equal=int(close.sum());bad=int((valid&~close).sum());same_iv+=equal;bad_iv+=bad
        active=charge_by.get(o.file,pd.DataFrame(columns=['timestamp_ns']))
        active=active.loc[(active.timestamp_ns>=start)&(active.timestamp_ns<=end)]
        charge_overlap+=len(active)
        descriptions.append(dict(other_file=o.file,mode=o['mode'],overlap_s=(end-start)/1e9,common_buckets=len(common),same_iv_buckets=equal,conflicting_iv_buckets=bad,active_charging_buckets_in_other=len(active)))
    for key,val in dict(overlap_count=len(other),overlap_details_json=json.dumps(descriptions,ensure_ascii=False),overlap_same_iv_rows=same_iv,
        overlap_conflicting_iv_rows=bad_iv,charge_overlap_active_buckets=charge_overlap,external_charging_evidence=bool(r.charge_active_own_rows>0 or charge_overlap>0),
        duplicate_published_mode_rows=bool(same_iv>0 and bad_iv==0)).items():driving.loc[idx,key]=val
driving['temperature_required_for_label']=False
driving['route_candidate']=driving.movement_evidence.astype(bool)&driving.gps_valid_fraction.ge(.95)&driving.gps_unique_points.ge(2)
driving['route_continuity_review_required']=driving.gap_gt_2s_count.gt(0)|driving.gps_implausible_links.gt(0)
driving['source_record_retained']=True
driving['whole_record_label_boundary']='unchanged_source_record_first_to_last_published_timestamp'
driving.to_csv(BASE/'driving_source_records.csv',index=False,encoding='utf-8-sig')
inventory.to_csv(BASE/'all_session_inventory.csv',index=False,encoding='utf-8-sig')
tc=['file','vehicle','start_utc','end_utc','duration_s','temperature_available_anywhere','temperature_in_first_bucket','first_temperature_bucket_utc','first_temperature_delay_s']+[c for c in driving if c.startswith('temperature_cut_')]
views=driving[tc].copy();views['view_target']='remaining_recorded_session_after_temperature_bucket_has_ended'
views.to_csv(BASE/'temperature_truncation_views.csv',index=False,encoding='utf-8-sig')
sc=['file','vehicle','start_utc','temperature_in_first_bucket','soc_in_first_bucket','ambient_in_first_bucket']+[c for c in driving if c.startswith('strict_start_')]
snap=driving[sc].copy();snap['prediction_time_utc']=snap.start_utc;snap['first_bucket_values_are_predeparture_features']=False
snap.to_csv(BASE/'strict_start_input_snapshots.csv',index=False,encoding='utf-8-sig')
temp=driving.loc[driving.temperature_available_anywhere]
summary=dict(release_definition='published 1 s mean-resampled CSV sessions; not all raw CAN files',source_zip=str(ZIP),source_zip_md5=hashlib.md5(ZIP.read_bytes()).hexdigest(),
    source_sessions=len(inventory),source_rows=int(inventory.source_rows.sum()),source_mode_counts=inventory['mode'].value_counts().to_dict(),source_read_errors=0,
    driving_source_records_retained=len(driving),driving_vehicles=driving.vehicle.value_counts().to_dict(),classification_counts=driving.classification.value_counts().to_dict(),
    movement_evidence_records=int(driving.movement_evidence.sum()),external_charging_evidence_records=int(driving.external_charging_evidence.sum()),
    own_external_charging_records=int(driving.charge_active_own_rows.gt(0).sum()),observed_active_charging_mode_overlap_records=int(driving.charge_overlap_active_buckets.gt(0).sum()),
    overlapping_source_records=int(driving.overlap_count.gt(0).sum()),exact_same_iv_mode_overlap_records=int(driving.duplicate_published_mode_rows.sum()),conflicting_iv_mode_overlap_records=int(driving.overlap_conflicting_iv_rows.gt(0).sum()),
    long_gap_gt_2s_records=int(driving.gap_gt_2s_count.gt(0).sum()),long_gap_gt_60s_records=int(driving.gap_gt_60s_count.gt(0).sum()),
    route_candidates_unfiltered_by_label=int(driving.route_candidate.sum()),route_ge_10km_records=int(driving.gps_path_plausible_km.ge(10).sum()),odometer_sentinel_records=int(driving.odometer_sentinel_rows.gt(0).sum()),
    temperature_first_bucket_records=int(driving.temperature_in_first_bucket.sum()),temperature_anywhere_records=len(temp),temperature_missing_entire_records=int((~driving.temperature_available_anywhere).sum()),
    temperature_truncated_nonempty_records=int(driving.temperature_cut_remaining_rows.fillna(0).ge(2).sum()),temperature_delay_s_quantiles=qs(temp.first_temperature_delay_s),
    temperature_dropped_duration_s_quantiles=qs(temp.temperature_cut_dropped_duration_s),temperature_dropped_gps_km_quantiles=qs(temp.temperature_cut_dropped_gps_plausible_km),
    temperature_dropped_iv_net_kwh_quantiles=qs(temp.temperature_cut_dropped_iv_observed_net_kwh),temperature_delay_le_2s_records=int(temp.first_temperature_delay_s.le(2).sum()),
    temperature_delay_le_5s_records=int(temp.first_temperature_delay_s.le(5).sum()),temperature_delay_gt_60s_records=int(temp.first_temperature_delay_s.gt(60).sum()),
    temperature_total_dropped_duration_s=float(temp.temperature_cut_dropped_duration_s.sum()),temperature_total_dropped_gps_plausible_km=float(temp.temperature_cut_dropped_gps_plausible_km.sum()),
    temperature_total_dropped_iv_observed_net_kwh=float(temp.temperature_cut_dropped_iv_observed_net_kwh.sum()),temperature_dropped_iv_is_partial_when_gaps=True,
    strict_start_previous_thermal_any_age_records=int(driving.strict_start_previous_temp_age_s.notna().sum()),strict_start_previous_thermal_age_s_quantiles=qs(driving.strict_start_previous_temp_age_s),
    strict_start_previous_thermal_age_le_60s_records=int(driving.strict_start_previous_temp_age_s.le(60).sum()),strict_start_previous_thermal_age_le_600s_records=int(driving.strict_start_previous_temp_age_s.le(600).sum()),
    strict_start_previous_thermal_age_le_3600s_records=int(driving.strict_start_previous_temp_age_s.le(3600).sum()),overlapping_thermal_bucket_conflicts=int(conflicts.sum()),
    parameters=dict(gps_plausible_speed_max_kph=200,moving_odometer_min_km=.1,moving_gps_min_km=.2,gps_motion_link_min_km=.01,speed_positive_threshold_unit_unspecified=1,
        line_charge_power_min_kw=.5,line_charge_current_min_a=.5,line_charge_voltage_min_v=50,temperature_min_C=-40,temperature_max_C=100,iv_max_bridged_interval_s=2,signal_bucket_availability_delay_s=1),
    label_qualification='Independent audit required; observed IV sums and individual counter spans are diagnostics, not automatically whole-session labels.')
(BASE/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
assert len(inventory)==2310 and len(driving)==1396
assert driving.file.is_unique and driving.source_record_retained.all() and not driving.temperature_required_for_label.any()
known=driving.strict_start_previous_temp_available_utc.dropna()
assert (pd.to_datetime(known,utc=True)<=pd.to_datetime(driving.loc[known.index,'start_utc'],utc=True)).all()
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

