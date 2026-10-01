"""Independent EnergyVille EV-B01 preparation; no model fit and no eVED cache."""
from __future__ import annotations
import argparse,hashlib,json,platform,zipfile
from pathlib import Path
import numpy as np
import pandas as pd

SECOND=1_000_000_000
INPUTS=('temperature_cut_remaining_net_candidates.csv','recorded_session_master.csv','driving_signal_views_1s.zip','observed_thermal_history.csv.gz','integrated_dataset_summary.json')
TREE_CONFIG=dict(loss='absolute_error',learning_rate=.05,max_iter=500,max_leaf_nodes=31,min_samples_leaf=20,l2_regularization=1.,early_stopping=False,random_state=42)


def write_json(path,obj):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def ns(values):
    return pd.to_datetime(values,utc=True).dt.as_unit('ns').astype('int64').to_numpy()


def flag(values):
    return values.fillna(False).astype(str).str.lower().isin(['true','1','1.0'])


def haversine(lat,lon):
    a,b=np.radians(lat),np.radians(lon);h=np.sin(np.diff(a)/2)**2+np.cos(a[:-1])*np.cos(a[1:])*np.sin(np.diff(b)/2)**2
    return 6371.0088*2*np.arctan2(np.sqrt(np.clip(h,0,1)),np.sqrt(np.clip(1-h,0,1)))


def valid_gps(lat,lon):
    return np.isfinite(lat)&np.isfinite(lon)&(abs(lat)<=90)&(abs(lon)<=180)&~((lat==0)&(lon==0))


def geometry_features(lat,lon,D):
    good=valid_gps(lat,lon);lat=np.asarray(lat)[good];lon=np.asarray(lon)[good]
    if not len(lat):raise ValueError('no remaining valid route positions')
    keep=np.r_[True,(np.diff(lat)!=0)|(np.diff(lon)!=0)];lat=lat[keep];lon=lon[keep]
    straight=float(haversine(np.array([lat[0],lat[-1]]),np.array([lon[0],lon[-1]]))[0])
    angle=np.arctan2((lon[-1]-lon[0])*np.cos(np.radians((lat[-1]+lat[0])/2)),lat[-1]-lat[0])
    out=dict(distance_km=float(D),start_lat=float(lat[0]),start_lon=float(lon[0]),end_lat=float(lat[-1]),end_lon=float(lon[-1]),straight_km=straight,
        route_to_straight_ratio=D/straight if straight>1e-6 else np.nan,ratio_undefined=float(straight<=1e-6),direction_sin=float(np.sin(angle)),direction_cos=float(np.cos(angle)))
    length=np.r_[0,np.cumsum(haversine(lat,lon))]*1000
    if length[-1]>0:
        points=np.r_[np.arange(0,length[-1],50.),length[-1]];la=np.interp(points,length,lat);lo=np.interp(points,length,lon)
        headings=np.arctan2(np.diff(lo)*np.cos(np.radians((la[:-1]+la[1:])/2)),np.diff(la));turns=abs(np.diff(np.unwrap(headings)))
    else:turns=np.array([]);points=np.array([0.])
    for name,v in [('turn_abs_mean',np.mean(turns) if len(turns) else 0.),('turn_abs_std',np.std(turns) if len(turns) else 0.),
        ('turn_abs_p90',np.quantile(turns,.9) if len(turns) else 0.),('turn_abs_sum',np.sum(turns))]:out[name]=float(v)
    return out,dict(unfiltered_geometry_km=float(length[-1]/1000),unique_geometry_points=len(lat),spatial_sample_points=len(points))


def od_key(r):
    def cell(lat,lon):return int(np.floor(lon*111320*np.cos(np.radians(51.))/250)),int(np.floor(lat*111320/250))
    a=cell(r['start_lat'],r['start_lon']);b=cell(r['end_lat'],r['end_lon'])
    return f'{a[0]}:{a[1]}>{b[0]}:{b[1]}|{int(r["distance_km"]//1)}'


def latest_state(d,times,cut,file):
    ended=times+SECOND<=cut;out={};prov=[]
    def last(col,valid=None):
        v=pd.to_numeric(d[col],errors='coerce').to_numpy(float);ix=np.flatnonzero(ended&np.isfinite(v)&(valid(v) if valid else True))
        return (float(v[ix[-1]]),int(times[ix[-1]])) if len(ix) else (np.nan,None)
    def add(name,v,t,source):
        out[name]=v;out[name+'_missing']=float(not np.isfinite(v));out[name+'_age_s']=(cut-t-SECOND)/SECOND if t is not None else np.nan
        if t is not None:
            assert t+SECOND<=cut;prov.append(dict(file=file,feature=name,source=source,bucket_ns=t,available_ns=t+SECOND,cut_ns=cut))
    v,t=last('SOCave292',lambda x:(x>=0)&(x<=100));add('soc',v,t,'SOCave292')
    v,t=last('VCRIGHT_tempAmbientRaw');source='VCRIGHT_tempAmbientRaw'
    if t is None:v,t=last('VCFRONT_tempAmbient');source='VCFRONT_tempAmbient'
    add('ambient_c',v,t,source);out['ambient_vcfront_source']=float(source=='VCFRONT_tempAmbient') if t is not None else np.nan
    lo=pd.to_numeric(d.BMSminPackTemperature,errors='coerce').to_numpy(float);hi=pd.to_numeric(d.BMSmaxPackTemperature,errors='coerce').to_numpy(float)
    ix=np.flatnonzero(ended&np.isfinite(lo)&np.isfinite(hi)&(lo>=-40)&(hi<=100)&(lo<=hi));t=int(times[ix[-1]]) if len(ix) else None
    add('pack_min_c',float(lo[ix[-1]]) if len(ix) else np.nan,t,'BMSminPackTemperature');add('pack_max_c',float(hi[ix[-1]]) if len(ix) else np.nan,t,'BMSmaxPackTemperature')
    out['pack_range_c']=out['pack_max_c']-out['pack_min_c'];out['pack_range_c_missing']=float(not np.isfinite(out['pack_range_c']))
    v,t=last('UI_batteryPreconditioningRequest',lambda x:(x>=0)&(x<=1));add('preconditioning_request',v,t,'UI_batteryPreconditioningRequest')
    out['soc_times_pack_min']=out['soc']*out['pack_min_c'];out['soc_times_pack_min_missing']=float(not np.isfinite(out['soc_times_pack_min']))
    return out,prov


def oracle_features(times,lat,lon,cut,end):
    sel=(times>=cut)&(times<=end)&valid_gps(lat,lon);tt=times[sel];la=lat[sel];lo=lon[sel]
    dt=np.diff(tt)/SECOND;ds=haversine(la,lo)*1000;v=np.divide(ds,dt,out=np.full_like(ds,np.nan),where=dt>0);good=(dt>0)&(dt<=2)&(v*3.6<=200)
    observed=float(dt[good].sum());duration=(end-cut)/SECOND;distance=float(ds[good].sum())
    v2=float(np.sum(ds[good]*v[good]**2)/distance) if distance>0 else np.nan;stop=float(dt[good&(v<.5)].sum()/observed) if observed>0 else np.nan
    return dict(actual_remaining_duration_s=duration,actual_gps_distance_weighted_v2=v2,actual_observed_stop_fraction=stop,
        actual_motion_coverage=observed/duration if duration>0 else np.nan,actual_motion_missing=float(observed==0),actual_v2_missing=float(not np.isfinite(v2)),actual_stop_missing=float(not np.isfinite(stop)))


def temporal_split(c):
    dates=pd.to_datetime(c.t_cut,utc=True).dt.strftime('%Y-%m-%d');daily=dates.value_counts().sort_index();cum=daily.cumsum().to_numpy();days=daily.index.to_numpy();boundaries=[]
    for f in (.60,.75,.85):
        ix=int(np.argmin(abs(cum-f*len(c))))
        if ix>=len(days)-1:raise ValueError('too few complete dates')
        boundaries.append(str(days[ix+1]))
    assert len(set(boundaries))==3
    split=np.select([dates<boundaries[0],dates<boundaries[1],dates<boundaries[2]],['train','validation','calibration'],default='future');purged=[]
    for early,later in zip(('train','validation','calibration'),('validation','calibration','future')):
        first=int(c.loc[split==later,'cut_ns'].min());bad=(split==early)&(c.end_ns.to_numpy()+SECOND>first)
        for i in np.flatnonzero(bad):purged.append(dict(file=c.file.iloc[i],original_split=early,reason='source_end+1s exceeds next partition start',next_partition_cut_ns=first))
        split[bad]='purged'
    return split,dict(boundary_dates=boundaries,method='closest complete-date cumulative count at60/75/85%; ties earlier',purged=purged)


def historical_features(c,route,pool,desc,thermal):
    features=[];prov=[];tp=[]
    for i,row in c.iterrows():
        prior=pool[(pool.vehicle==row.vehicle)&(pool.end_ns+SECOND<=row.cut_ns)&(pool.file!=row.file)].sort_values(['end_ns','file']).tail(10)
        ix=prior.index.to_numpy();p=desc.loc[ix];key=route.loc[i,'od_key'];same=prior[route.loc[ix,'od_key'].to_numpy()==key];s=desc.loc[same.index]
        out=dict(history_count=len(prior),history_missing=float(not len(prior)),history_recent_age_s=(row.cut_ns-int(prior.end_ns.max())-SECOND)/SECOND if len(prior) else np.nan,
            history_net_wh_per_km=1000*p.energy_kwh.sum()/p.distance_km.sum() if len(prior) and p.distance_km.sum()>0 else np.nan,
            history_wh_per_km_median=float((1000*p.energy_kwh/p.distance_km.where(p.distance_km>0)).median()) if len(prior) else np.nan,
            history_distance_mean=float(p.distance_km.mean()) if len(prior) else np.nan,history_duration_mean=float(p.duration_s.mean()) if len(prior) else np.nan,
            history_mean_speed_kph=float((3600*p.distance_km/p.duration_s.where(p.duration_s>0)).mean()) if len(prior) else np.nan,
            history_ambient_mean=float(p.ambient_mean.mean()) if len(prior) else np.nan,history_pack_min_mean=float(p.pack_min_mean.mean()) if len(prior) else np.nan,
            history_pack_max_mean=float(p.pack_max_mean.mean()) if len(prior) else np.nan,od_history_count=len(same),
            od_history_net_wh_per_km=1000*s.energy_kwh.sum()/s.distance_km.sum() if len(same) and s.distance_km.sum()>0 else np.nan,
            od_history_duration_mean=float(s.duration_s.mean()) if len(same) else np.nan)
        t=thermal[row.vehicle];pos=np.searchsorted(t.timestamp_ns.to_numpy(),row.start_ns-SECOND,side='right')-1
        if pos>=0:
            ob=t.iloc[pos];out.update(past_thermal_min=float(ob.min_temperature),past_thermal_max=float(ob.max_temperature),past_thermal_age_s=(row.cut_ns-int(ob.timestamp_ns)-SECOND)/SECOND)
            tp.append(dict(file=row.file,source=str(ob.file),mode=str(ob['mode']),bucket_ns=int(ob.timestamp_ns),available_ns=int(ob.timestamp_ns)+SECOND,cut_ns=int(row.cut_ns)))
        else:out.update(past_thermal_min=np.nan,past_thermal_max=np.nan,past_thermal_age_s=np.nan)
        for name,v in list(out.items()):
            if name not in ('history_count','history_missing','od_history_count'):out[name+'_missing']=float(not np.isfinite(v))
        features.append(out)
        for j in ix:
            assert c.loc[j,'split']=='train' and j!=i and c.loc[j,'end_ns']+SECOND<=row.cut_ns
            prov.append(dict(file=row.file,history_file=c.loc[j,'file'],history_end_ns=int(c.loc[j,'end_ns']),history_available_ns=int(c.loc[j,'end_ns'])+SECOND,cut_ns=int(row.cut_ns),same_od=bool(route.loc[j,'od_key']==key)))
    return pd.DataFrame(features),prov,tp


def main():
    p=argparse.ArgumentParser()
    for name in ('data','out','todo'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'preflight_checks.json').exists():raise FileExistsError('frozen data already exists')
    mf=json.loads((a.data/'artifact_manifest.json').read_text(encoding='utf-8'));expected={r['name']:r for r in mf['files']};actual=[]
    for name in INPUTS:
        path=a.data/name;digest=sha(path);assert digest==expected[name]['sha256'] and path.stat().st_size==expected[name]['bytes'],name
        actual.append(dict(name=name,sha256=digest,bytes=path.stat().st_size,path=str(path.resolve())))
    source=pd.read_csv(a.data/INPUTS[0]);master=pd.read_csv(a.data/INPUTS[1]);assert len(source)==1272 and source.file.is_unique and len(master)==1396 and master.file.is_unique
    aligned=master.set_index('file').loc[source.file].reset_index();assert np.array_equal(aligned.vehicle,source.vehicle)
    for col in ('temperature_cut_prediction_utc','temperature_cut_remaining_net_candidate_kwh','temperature_cut_dropped_iv_observed_net_kwh','full_source_hybrid_net_kwh','temperature_cut_remaining_route_plausible_km'):
        pd.testing.assert_series_equal(aligned[col],source[col],check_names=False)
    cols=['file','vehicle','start_utc','end_utc','temperature_cut_prediction_utc','temperature_cut_remaining_net_candidate_kwh','temperature_cut_dropped_iv_observed_net_kwh','full_source_hybrid_net_kwh',
        'temperature_cut_remaining_route_plausible_km','gps_path_plausible_km','gap_gt_2s_count','uncertain_charger_signal_or_mode_overlap','main_view_internal_consistency_checked',
        'main_view_5pct_counter_quantization_sufficient','overlap_conflicting_iv_rows','suspected_charger_signal_or_mode_overlap','confirmed_external_charging_from_counter_or_joint_direct','battery_net_quantization_relative_bound']
    c=source[cols].copy().rename(columns={'temperature_cut_prediction_utc':'t_cut','temperature_cut_remaining_net_candidate_kwh':'y_rem','temperature_cut_dropped_iv_observed_net_kwh':'e_prefix',
        'full_source_hybrid_net_kwh':'y_full','temperature_cut_remaining_route_plausible_km':'D'})
    c['start_ns']=ns(c.start_utc);c['end_ns']=ns(c.end_utc);c['cut_ns']=ns(c.t_cut)
    assert np.isfinite(c[['y_rem','e_prefix','y_full','D']]).all().all() and (c.D>0).all()
    identity=float(abs(c.y_full-c.e_prefix-c.y_rem).max());assert identity<=1e-8;c['split'],split_contract=temporal_split(c)
    th=pd.read_csv(a.data/INPUTS[3]);assert th.timestamp_ns.dtype.kind in 'iu';thermal={v:g.sort_values('timestamp_ns').reset_index(drop=True) for v,g in th.groupby('vehicle')}
    routes=[];states=[];oracles=[];ra=[];sp=[];descriptors=[];temperature_conflicts=[];prefix_audit=[]
    with zipfile.ZipFile(a.data/INPUTS[2]) as z:
        assert set(master.file)==set(z.namelist());assert z.testzip() is None
        for i,row in c.iterrows():
            d=pd.read_csv(z.open(row.file));times=ns(d.Timestamp);assert np.all(np.diff(times)>0) and times[0]==row.start_ns and times[-1]==row.end_ns
            np.testing.assert_array_equal(ns(d.signal_bucket_available_utc),times+SECOND)
            power=pd.to_numeric(d.BattVoltage132,errors='coerce').to_numpy(float)*pd.to_numeric(d.RawBattCurrent132,errors='coerce').to_numpy(float)/1000
            ivdt=np.diff(times)/SECOND;ivok=np.isfinite(power[:-1])&np.isfinite(power[1:])&(ivdt>0)&(ivdt<=2)
            prefix_intervals=ivok&(times[1:]<=row.cut_ns)
            prefix_replay=float(np.where(prefix_intervals,(power[:-1]+power[1:])/2*ivdt/3600,0).sum())
            assert abs(prefix_replay-row.e_prefix)<=1e-8,(row.file,'prefix replay')
            used_iv=np.flatnonzero(prefix_intervals)
            prefix_available=int(times[used_iv[-1]+1]+SECOND) if len(used_iv) else int(row.start_ns)
            prefix_audit.append(dict(file=row.file,prefix_replay_kwh=prefix_replay,stored_prefix_kwh=float(row.e_prefix),last_prefix_bucket_available_ns=prefix_available,cut_ns=int(row.cut_ns),prefix_available_at_cut=prefix_available<=row.cut_ns))
            st,prov=latest_state(d,times,int(row.cut_ns),row.file);sp.extend(prov)
            np.testing.assert_allclose([st['pack_min_c'],st['pack_max_c']],[source.temperature_cut_min_temperature.iloc[i],source.temperature_cut_max_temperature.iloc[i]],atol=1e-10)
            # Prefer the current source's valid, ended measurement. Never average mode copies.
            tempbucket=next(p['bucket_ns'] for p in prov if p['feature']=='pack_min_c');other=thermal[row.vehicle];pos=np.searchsorted(other.timestamp_ns.to_numpy(),tempbucket)
            conflict=False
            if pos<len(other) and int(other.timestamp_ns.iloc[pos])==tempbucket:
                ob=other.iloc[pos];conflict=not(np.isclose(ob.min_temperature,st['pack_min_c']) and np.isclose(ob.max_temperature,st['pack_max_c']))
            st['observed_cross_mode_temperature_conflict']=float(conflict)
            if conflict:temperature_conflicts.append(dict(file=row.file,bucket_ns=tempbucket,other_source=str(other.iloc[pos].file),rule='current own-source ended value preferred; no averaging'))
            st['distance_times_ambient_minus20']=row.D*(st['ambient_c']-20);st['distance_times_ambient_minus20_missing']=float(not np.isfinite(st['distance_times_ambient_minus20']));states.append(st)
            lat=pd.to_numeric(d.GPSLatitude04F,errors='coerce').to_numpy(float);lon=pd.to_numeric(d.GPSLongitude04F,errors='coerce').to_numpy(float);ids=np.flatnonzero(valid_gps(lat,lon));gt=times[ids]
            links=haversine(lat[ids],lon[ids]);dt=np.diff(gt)/SECOND;plausible=(dt>0)&(np.divide(links,dt,out=np.full_like(links,np.inf),where=dt>0)*3600<=200)
            replay=float(links[plausible&(gt[1:]>row.cut_ns)].sum());assert abs(replay-row.D)<=1e-8,(row.file,'distance',replay,row.D)
            remaining=(times>=row.cut_ns)&(times<=row.end_ns);geom,ga=geometry_features(lat[remaining],lon[remaining],float(row.D));geom['od_key']=od_key(geom);routes.append(geom)
            crossing=plausible&(gt[:-1]<row.cut_ns)&(gt[1:]>row.cut_ns);crossing_km=float(links[crossing].sum());goodids=np.flatnonzero(valid_gps(lat,lon)&remaining)
            sublinks=haversine(lat[goodids],lon[goodids]);subdt=np.diff(times[goodids])/SECOND;goodsub=(subdt>0)&(np.divide(sublinks,subdt,out=np.full_like(sublinks,np.inf),where=subdt>0)*3600<=200)
            visible=float(sublinks[goodsub].sum());assert abs(row.D-visible-crossing_km)<=1e-8,(row.file,'cross-cut boundary')
            ra.append(dict(file=row.file,cut_ns=int(row.cut_ns),first_remaining_gps_ns=int(times[goodids[0]]),last_remaining_gps_ns=int(times[goodids[-1]]),source_D_km=float(row.D),distance_replay_km=replay,
                visible_remaining_plausible_km=visible,crossing_cut_gps_chord_km=crossing_km,first_gps_offset_s=(times[goodids[0]]-row.cut_ns)/SECOND,last_gps_offset_s=(row.end_ns-times[goodids[-1]])/SECOND,**ga))
            oracles.append(oracle_features(times,lat,lon,int(row.cut_ns),int(row.end_ns)))
            amin=pd.to_numeric(d.VCRIGHT_tempAmbientRaw,errors='coerce').to_numpy(float);fallback=pd.to_numeric(d.VCFRONT_tempAmbient,errors='coerce').to_numpy(float);ambient=np.where(np.isfinite(amin),amin,fallback)
            lo=pd.to_numeric(d.BMSminPackTemperature,errors='coerce').to_numpy(float);hi=pd.to_numeric(d.BMSmaxPackTemperature,errors='coerce').to_numpy(float);tok=np.isfinite(lo)&np.isfinite(hi)&(lo>=-40)&(hi<=100)&(lo<=hi)
            descriptors.append(dict(energy_kwh=float(row.y_full),distance_km=float(row.gps_path_plausible_km),duration_s=(row.end_ns-row.start_ns)/SECOND,
                ambient_mean=float(np.nanmean(ambient)) if np.isfinite(ambient).any() else np.nan,pack_min_mean=float(lo[tok].mean()) if tok.any() else np.nan,pack_max_mean=float(hi[tok].mean()) if tok.any() else np.nan))
            if (i+1)%200==0:print(json.dumps(dict(prepared=i+1,total=len(c))),flush=True)
    route=pd.DataFrame(routes);state=pd.DataFrame(states);oracle=pd.DataFrame(oracles);desc=pd.DataFrame(descriptors)
    c['od_key']=route.od_key;c['distance_bin']=pd.cut(c.D,[-np.inf,2,10,50,np.inf],right=False,labels=['<2','2-10','10-50','>=50']).astype(str)
    eligible=(c.split=='train')&flag(c.main_view_internal_consistency_checked)&~flag(c.uncertain_charger_signal_or_mode_overlap)&~flag(c.confirmed_external_charging_from_counter_or_joint_direct)&(c.overlap_conflicting_iv_rows==0)
    pool=c.loc[eligible];assert len(pool)>0;history,hp,tp=historical_features(c,route,pool,desc,thermal)
    assert all(r['available_ns']<=r['cut_ns'] for r in tp)
    features=pd.concat([route.drop(columns='od_key'),state,history,oracle],axis=1);features.insert(0,'vehicle',c.vehicle);features.insert(0,'file',c.file)
    blocks=dict(R=list(route.drop(columns='od_key').columns),S=list(state.columns),H=list(history.columns),O=list(oracle.columns));contract={}
    for block,columns in blocks.items():
        for col in columns:contract[col]=dict(block=block,oracle=block=='O',formula=col,
            source={'R':'remaining source GPS; frozen budget D','S':'latest valid ended own-source bucket','H':'quality completed training sources; separate thermal telemetry before source start','O':'actual remaining GPS time and duration'}[block],
            available={'R':'given planned-route offline geometry proxy','S':'bucket+1s<=cut','H':'label train ID and end+1s<=cut; telemetry bucket+1s<=source start','O':'after source end; diagnostic only'}[block],missing='NaN and fixed missing flags; native HistGBR routing')
    formulas={'distance_km':'R05 sum plausible GPS chords whose end timestamp > cut; full crossing chord retained',
        'start_lat':'first valid remaining GPS latitude','start_lon':'first valid remaining GPS longitude','end_lat':'last valid remaining GPS latitude','end_lon':'last valid remaining GPS longitude',
        'straight_km':'endpoint haversine, Earth radius6371.0088km','route_to_straight_ratio':'D/straight_km if straight>1e-6km else NaN','ratio_undefined':'straight_km<=1e-6',
        'direction_sin':'sin(endpoint atan2(delta_lon*cos(mean_lat),delta_lat))','direction_cos':'cos(endpoint atan2(delta_lon*cos(mean_lat),delta_lat))',
        'pack_range_c':'pack_max_c-pack_min_c','soc_times_pack_min':'soc*pack_min_c','distance_times_ambient_minus20':'D*(ambient_c-20)',
        'ambient_vcfront_source':'1 if primary ambient has no valid ended bucket and fallback is used',
        'observed_cross_mode_temperature_conflict':'own ended pack temperature differs from deduplicated same completed thermal bucket; own value preferred',
        'history_count':'number of eligible latest<=10 same-vehicle train records completed by cut',
        'history_recent_age_s':'(cut-last_history_end-1s)/1s','history_net_wh_per_km':'1000*sum(full_source_hybrid_net_kwh)/sum(full_source_gps_path_plausible_km)',
        'history_wh_per_km_median':'median(1000*full_source_energy/full_source_distance), distance>0',
        'history_distance_mean':'mean full source distance of eligible latest10','history_duration_mean':'mean completed(source_end-source_start) seconds',
        'history_mean_speed_kph':'mean(3600*full_source_distance/completed_source_duration)',
        'history_ambient_mean':'mean of completed source ambient means, primary/fallback per bucket','history_pack_min_mean':'mean of completed source valid pack-min means','history_pack_max_mean':'mean of completed source valid pack-max means',
        'od_history_count':'same directed250m start/end cells +1km length bin within latest10 history',
        'od_history_net_wh_per_km':'1000*sum(sameOD full-source energy)/sum(sameOD full-source distance)','od_history_duration_mean':'mean completed sameOD source duration',
        'past_thermal_min':'most recent valid telemetry min before source start','past_thermal_max':'most recent valid telemetry max before source start','past_thermal_age_s':'(cut-last_thermal_bucket-1s)/1s',
        'actual_remaining_duration_s':'(source_end-cut) seconds','actual_gps_distance_weighted_v2':'sum(valid_gps_chord_m*gps_speed_mps**2)/sum(valid_gps_chord_m)',
        'actual_observed_stop_fraction':'sum(valid_dt where speed<0.5m/s)/sum(valid_dt)','actual_motion_coverage':'sum(valid_dt)/remaining_duration; dt<=2s and speed<=200km/h',
        'actual_motion_missing':'no valid remaining GPS time intervals','actual_v2_missing':'no positive valid motion distance','actual_stop_missing':'no valid motion intervals',
        'history_missing':'history_count==0'}
    for col in contract:
        if col in formulas:contract[col]['formula']=formulas[col]
        elif col.startswith('turn_abs_'):contract[col]['formula']=col.removeprefix('turn_abs_')+' of absolute unwrapped heading differences on50m equal-spatial route samples (endpoint also included)'
        elif col.endswith('_missing'):contract[col]['formula']='is_not_finite('+col.removesuffix('_missing')+')'
        elif col.endswith('_age_s'):contract[col]['formula']='(cut-last_valid_'+col.removesuffix('_age_s')+'_bucket-1s)/1s'
        elif col in ('soc','ambient_c','pack_min_c','pack_max_c','preconditioning_request'):contract[col]['formula']='latest valid value whose bucket+1s<=cut; source in state_provenance.csv'
    contract['vehicle']=dict(block='R',source='known vehicle',formula='train-only one-hot + unknown',available='at cut',oracle=False)
    forbidden={'y_rem','y_full','e_prefix','duration_s','end_ns','gap_gt_2s_count'};assert not forbidden&set(features.columns)
    used={g:['vehicle']+sum([blocks[k] for k in 'RSHO'[:n]],[]) for g,n in [('R',1),('S',2),('H',3),('O',4)]}
    assert all(not set(blocks['O'])&set(used[g]) for g in ('R','S','H'))
    c['has_history']=history.history_count>0;c['od_seen_train']=c.od_key.isin(set(c.loc[c.split=='train','od_key']));c['ambient_c']=state.ambient_c
    c['ambient_bin']=np.select([state.ambient_c.isna(),state.ambient_c<10,state.ambient_c<=25],['missing','<10','10-25'],default='>25')
    trainrange={col:[float(features.loc[c.split=='train',col].min()),float(features.loc[c.split=='train',col].max())] for col in ('distance_km','soc','ambient_c','pack_min_c') if features.loc[c.split=='train',col].notna().any()}
    outside=np.zeros(len(c),bool)
    for col,(lo,hi) in trainrange.items():outside|=((features[col]<lo)|(features[col]>hi)).fillna(False).to_numpy()
    c['outside_training_range']=outside;summary={}
    for name,g in c.groupby('split'):
        summary[name]=dict(n=len(g),vehicles=g.vehicle.value_counts().to_dict(),cut_min=g.t_cut.min(),cut_max=g.t_cut.max(),ge10km=int((g.D>=10).sum()),
            distance_quantiles={str(q):float(g.D.quantile(q)) for q in (0,.5,.95,1)},history_fraction=float(g.has_history.mean()),outside_training_range=int(g.outside_training_range.sum()),
            ambient_observed=int(g.ambient_c.notna().sum()),internal_consistency=int(flag(g.main_view_internal_consistency_checked).sum()),uncertain_charger=int(flag(g.uncertain_charger_signal_or_mode_overlap).sum()))
    overlaps=[]
    for vehicle,g in c.groupby('vehicle'):
        prior=[]
        for i,row in g.sort_values('start_ns').iterrows():
            prior=[j for j in prior if c.end_ns.iloc[j]>row.start_ns]
            for j in prior:overlaps.append([c.file.iloc[j],row.file])
            prior.append(i)
    assert not overlaps,overlaps
    checks=dict(passed=True,n=1272,unique_files=True,master_zip_bijection=True,label_identity_max_abs_kwh=identity,negative_remaining=int((c.y_rem<0).sum()),
        state_provenance_count=len(sp),all_state_buckets_ended=True,history_provenance_count=len(hp),training_only_completed_history=True,
        telemetry_provenance_count=len(tp),all_telemetry_before_source_start=True,route_budget_replay=True,geometry_50m_no_temporal_density=True,overlapping_driving_targets=overlaps,
        split=split_contract,split_summary=summary,history_pool_n=len(pool),crossing_cut_routes=int(sum(r['crossing_cut_gps_chord_km']>0 for r in ra)),
        route_crossing_cut_max_km=max(r['crossing_cut_gps_chord_km'] for r in ra),current_temperature_conflicts=temperature_conflicts,
        thermal_conflict_limit='R05 historical table deduplicated 24 conflicting mode buckets; discarded values unavailable. Current source conflicts detected where possible; no averaging.',
        preprocessing='None fitted at preparation; vehicle vocabulary and tree binning fit train only',representative_sources=[c.loc[(c.D-target).abs().idxmin(),'file'] for target in (1.,10.,100.)])
    checks['prefix_replay_all1272']=True
    checks['prefix_available_at_cut_n']=sum(r['prefix_available_at_cut'] for r in prefix_audit)
    checks['prefix_boundary_caveat']='R05 trapezoid prefix can use interval-end bucket at cut; that1s mean becomes available cut+1s. Prefix is NOT a learner input. Full-budget readdition is a retrospective accounting output, not certified deployable at exact cut.'
    c.to_csv(a.out/'cohort_manifest.csv',index=False);c[['file','vehicle','t_cut','start_utc','end_utc','split','od_key','D']].to_csv(a.out/'split_manifest.csv',index=False)
    features.to_csv(a.out/'features.csv',index=False);pd.DataFrame(ra).to_csv(a.out/'route_audit.csv',index=False);pd.DataFrame(sp).to_csv(a.out/'state_provenance.csv',index=False)
    pd.DataFrame(prefix_audit).to_csv(a.out/'prefix_audit.csv',index=False)
    pd.DataFrame(hp).to_csv(a.out/'history_provenance.csv',index=False);pd.DataFrame(tp).to_csv(a.out/'thermal_provenance.csv',index=False)
    pool[['file','vehicle','end_utc','end_ns','y_full','gps_path_plausible_km']].to_csv(a.out/'history_pool_manifest.csv',index=False)
    write_json(a.out/'feature_columns.json',dict(blocks=blocks,groups=used));write_json(a.out/'feature_contract.json',contract);write_json(a.out/'preflight_checks.json',checks)
    write_json(a.out/'input_manifest.json',dict(files=actual,artifact_manifest_sha256=sha(a.data/'artifact_manifest.json'),todo_sha256=sha(a.todo),preparation_code_sha256=sha(__file__),input_schema='R05 remaining counter+IV approximation; all1272 source IDs'))
    write_json(a.out/'run_config.json',dict(experiment='EV-B01',prepared_at=pd.Timestamp.now(tz='UTC').isoformat(),fit_started=False,tree=TREE_CONFIG,seed=42,spatial_step_m=50,history_limit=10,threads=4,
        bootstrap_week_blocks=1000,mape_threshold_kwh=.1,validation_tie_simple_relative=.01,gate=dict(validation_mae_improvement=.10,rmse_max_ratio=1.05,each_vehicle_mae_max_ratio=1.10,ge10km_mae_max_ratio=1.10,underestimate_mean_max_ratio=1.05),
        split_summary=summary,adaptations=['Geometry-only; no static map cache','R05 D preserved: crossing-cut GPS chord counted wholly as remaining; separately audited','Native NaN; train-only one-hot; no numeric fill/scale',
        'History label IDs restricted to training IDs in1272 cohort; full-source labels use full-source distance','SameOD summaries restricted to same latest10 history pool; directional250m+1km key',
        'Thermal telemetry takes a bucket already ended before source start; R05 deduplication limitation recorded'],platform=platform.platform(),cloud_project_root=str(Path.cwd()),
        prepared_file_hashes={f:sha(a.out/f) for f in ('features.csv','cohort_manifest.csv','split_manifest.csv','feature_columns.json')}))
    (a.out/'protocol_frozen.md').write_text('# EV-B01 首次拟合前冻结协议\n\n原todo完整快照；实际适配、哈希、日期/ID和因果证据见同目录JSON/CSV，全部在拟合前生成。\n\n'+a.todo.read_text(encoding='utf-8'),encoding='utf-8')
    print(json.dumps(dict(state='prepared_not_fitted',split_summary=summary,checks=True),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
