"""Read published EnergyVille ZIP and summarize measured sessions; no author code execution.
These are candidate labels/inputs, not an accepted training dataset.
"""
from pathlib import Path
import zipfile, io,json,collections,hashlib
import numpy as np,pandas as pd
base=Path(__file__).parent;src=base/'sources';out=base/'decoded';out.mkdir(exist_ok=True)
z=zipfile.ZipFile(src/'energyville_V2.zip')
records=[];schema=collections.Counter(); errors=[]; examples={}
fields=['BattVoltage132','RawBattCurrent132','GPSLatitude04F','GPSLongitude04F','BMSminPackTemperature','BMSmaxPackTemperature','VCRIGHT_tempAmbientRaw','SOCave292','DI_uiSpeed','Odometer3B6','TotalDischargeKWh3D2','TotalChargeKWh3D2','UI_batteryPreconditioningRequest']
for index,name in enumerate(n for n in z.namelist() if n.endswith('.csv') and 'sessions/' in n):
    mode=name.split('/')[-2];filename=name.split('/')[-1];vehicle=filename.split('_')[0]
    try:
        data=z.read(name);df=pd.read_csv(io.BytesIO(data));schema[(mode,tuple(df.columns))]+=1
        t=pd.to_datetime(df['Timestamp'],utc=True); dt=t.diff().dt.total_seconds();duration=float((t.iloc[-1]-t.iloc[0]).total_seconds())
        r={'file':filename,'zip_path':name,'vehicle':vehicle,'mode':mode,'rows':len(df),'start_utc':str(t.iloc[0]),'end_utc':str(t.iloc[-1]),'duration_s':duration,'max_gap_s':float(dt.max()) if len(df)>1 else None,'nonpositive_dt':int((dt<=0).sum()),'missing_timestamps':int(t.isna().sum()),'gap_gt_2s_intervals':int(dt.gt(2).sum()),'gap_gt_2s_time_s':float(dt[dt.gt(2)].sum())}
        if mode not in examples:
            examples[mode]=filename;(out/f'energyville_example_{mode.replace(" ","_")}.csv').write_bytes(data)
        for c in fields:
            if c in df:
                v=pd.to_numeric(df[c],errors='coerce');r[c+'_valid_ratio']=float(v.notna().mean());r[c+'_start']=float(v.iloc[0]) if pd.notna(v.iloc[0]) else None
            else:r[c+'_valid_ratio']=0.;r[c+'_start']=None
        v=pd.to_numeric(df['BattVoltage132'],errors='coerce');a=pd.to_numeric(df['RawBattCurrent132'],errors='coerce');power=v*a/1000
        valid=power.notna() & power.shift().notna() & dt.gt(0) & dt.le(2)
        # Diagnostic trapezoidal integral only for intervals <= 2 s on the nominal 1 s grid; never bridge long gaps.
        r['iv_valid_time_ratio']=float(dt[valid].sum()/duration) if duration>0 else None
        r['candidate_iv_net_kwh']=float(((power+power.shift())/2*dt/3600).where(valid).sum()) if valid.any() else None
        r['power_kw_min']=float(power.min());r['power_kw_max']=float(power.max())
        for c in ['TotalDischargeKWh3D2','TotalChargeKWh3D2','Odometer3B6']:
            x=pd.to_numeric(df[c],errors='coerce').dropna() if c in df else pd.Series(dtype=float)
            r[c+'_delta']=float(x.iloc[-1]-x.iloc[0]) if len(x)>=2 else None
            r[c+'_negative_steps']=int((x.diff() < -1e-6).sum())
            r[c+'_first_timestamp']=str(t.loc[x.index[0]]) if len(x) else None
            r[c+'_last_timestamp']=str(t.loc[x.index[-1]]) if len(x) else None
        dis=r['TotalDischargeKWh3D2_delta'];cha=r['TotalChargeKWh3D2_delta'];r['candidate_counter_net_kwh']=dis-cha if dis is not None and cha is not None else None
        if r['candidate_counter_net_kwh'] is not None and r['candidate_iv_net_kwh'] is not None:r['iv_minus_counter_kwh']=r['candidate_iv_net_kwh']-r['candidate_counter_net_kwh']
        lat=pd.to_numeric(df.get('GPSLatitude04F',pd.Series(dtype=float)),errors='coerce');lon=pd.to_numeric(df.get('GPSLongitude04F',pd.Series(dtype=float)),errors='coerce')
        validgps=lat.between(-90,90)&lon.between(-180,180)&(lat.ne(0)|lon.ne(0))
        r['gps_valid_ratio']=float(validgps.mean()) if len(df) else 0
        xy=df.loc[validgps,['GPSLatitude04F','GPSLongitude04F']].drop_duplicates() if validgps.any() else pd.DataFrame()
        r['gps_unique_points']=len(xy)
        if validgps.sum()>1:
            la=np.radians(lat[validgps].to_numpy());lo=np.radians(lon[validgps].to_numpy());h=np.sin(np.diff(la)/2)**2+np.cos(la[:-1])*np.cos(la[1:])*np.sin(np.diff(lo)/2)**2
            r['gps_path_km']=float((6371*2*np.arctan2(np.sqrt(h),np.sqrt(np.maximum(1-h,0)))).sum())
        else:r['gps_path_km']=None
        records.append(r)
    except Exception as e:errors.append({'file':filename,'error':str(e)})
    if (index+1)%200==0:print('Read sessions',index+1,flush=True)
df=pd.DataFrame(records);df.to_csv(base/'energyville_session_audit.csv',index=False,encoding='utf-8-sig')
summary={'rows':len(df),'errors':errors,'modes':{},'schemas':[{'mode':m,'columns':list(cols),'files':n} for (m,cols),n in schema.items()]}
for mode,g in df.groupby('mode'):
    s={'files':len(g),'vehicles':g.vehicle.value_counts().to_dict(),'start_min':g.start_utc.min(),'end_max':g.end_utc.max(),'records':int(g.rows.sum()),'max_gap_s':float(g.max_gap_s.max()),'nonpositive_dt_files':int(g.nonpositive_dt.gt(0).sum()),'gap_gt_2s_files':int(g.gap_gt_2s_intervals.gt(0).sum()),'missing_timestamp_files':int(g.missing_timestamps.gt(0).sum()),'start_fields_present':{c:int(g[c+'_start'].notna().sum()) for c in fields}}
    for c in ['duration_s','gps_path_km','gps_valid_ratio','gps_unique_points','candidate_iv_net_kwh','iv_valid_time_ratio','candidate_counter_net_kwh','iv_minus_counter_kwh','Odometer3B6_delta']:
        s[c+'_quantiles']=g[c].quantile([0,.05,.5,.95,1]).replace({np.nan:None}).to_dict()
    s['gps_path_ge_10km']=int(g.gps_path_km.ge(10).sum())
    s['counter_any_negative_steps']=int((g.TotalDischargeKWh3D2_negative_steps.gt(0)|g.TotalChargeKWh3D2_negative_steps.gt(0)).sum())
    summary['modes'][mode]=s
# Temporal association audit: preceding observed session must end before current session starts.
trips=df[df['mode']=='driving sessions'].copy();alltime=df.copy();alltime['start']=pd.to_datetime(alltime.start_utc);alltime['end']=pd.to_datetime(alltime.end_utc)
for i,row in trips.iterrows():
    prior=alltime[(alltime.vehicle==row.vehicle)&(alltime.end<=pd.Timestamp(row.start_utc))&(alltime.file!=row.file)].sort_values('end')
    if len(prior):
        last=prior.iloc[-1];trips.loc[i,'previous_observed_file']=last.file;trips.loc[i,'previous_observed_mode']=last['mode'];trips.loc[i,'previous_observed_gap_s']=(pd.Timestamp(row.start_utc)-last['end']).total_seconds()
    trips.loc[i,'overlap_with_other_session']=bool(((alltime.vehicle==row.vehicle)&(alltime.file!=row.file)&(alltime.start < pd.Timestamp(row.end_utc))&(alltime.end > pd.Timestamp(row.start_utc))).any())
trips['q_time_grid']=trips.max_gap_s.le(2)&trips.nonpositive_dt.eq(0)&trips.missing_timestamps.eq(0)
trips['q_iv_coverage']=trips.iv_valid_time_ratio.ge(.99)
trips['q_gps']=trips.gps_valid_ratio.ge(.99)&trips.gps_unique_points.ge(2)
trips['q_initial_thermal']=trips.BMSminPackTemperature_start.notna()&trips.BMSmaxPackTemperature_start.notna()
trips['q_initial_soc_ambient']=trips.SOCave292_start.notna()&trips.VCRIGHT_tempAmbientRaw_start.notna()
trips['q_no_overlap']=~trips.overlap_with_other_session.astype(bool)
trips['q_candidate_common_subset']=trips[[c for c in trips if c.startswith('q_') and c!='q_candidate_common_subset']].all(axis=1)
summary['provisional_quality_counts']={c:int(trips[c].sum()) for c in trips if c.startswith('q_')}
summary['provisional_common_subset_vehicles']=trips.loc[trips.q_candidate_common_subset,'vehicle'].value_counts().to_dict()
summary['provisional_common_subset_ge_10km']=int(trips.loc[trips.q_candidate_common_subset,'gps_path_km'].ge(10).sum())
trips.to_csv(base/'energyville_candidate_trip_table.csv',index=False,encoding='utf-8-sig')
summary['trip_previous_observed_modes']=trips.previous_observed_mode.value_counts().to_dict();summary['overlapping_driving_files']=int(trips.overlap_with_other_session.sum())
(base/'energyville_actual_data_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print('Done',len(df),'sessions',len(trips),'driving files','errors',len(errors),flush=True)
