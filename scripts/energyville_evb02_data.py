"""EV-B02 event-time history adapter. Calibration energy is never loaded/released."""
from __future__ import annotations
import argparse,hashlib,io,json,platform,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from energyville_evb01_data import SECOND,TREE_CONFIG,sha,write_json,flag,haversine,valid_gps,ns

PERMISSIONS={'train':{'train'},'validation':{'train','validation'},'future':{'train','validation','future'}}
META=('file','vehicle','split','start_ns','end_ns','cut_ns','t_cut','start_utc','end_utc','od_key','D','gps_path_plausible_km','distance_bin','ambient_c','ambient_bin','od_seen_train','gap_gt_2s_count','uncertain_charger_signal_or_mode_overlap','suspected_charger_signal_or_mode_overlap','confirmed_external_charging_from_counter_or_joint_direct','main_view_internal_consistency_checked','main_view_5pct_counter_quantization_sufficient','overlap_conflicting_iv_rows')


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.generic):x=x.item()
    return None if isinstance(x,float) and not np.isfinite(x) else x


def digest(x):return hashlib.sha256(json.dumps(clean(x),sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def quality(c):return flag(c.main_view_internal_consistency_checked)&~flag(c.uncertain_charger_signal_or_mode_overlap)&~flag(c.confirmed_external_charging_from_counter_or_joint_direct)&(c.overlap_conflicting_iv_rows==0)


def summary(prior,same,cut):
    p=pd.DataFrame(prior);s=pd.DataFrame(same)
    def mean(column):return float(p[column].mean()) if len(p) else np.nan
    out=dict(history_count=len(p),history_missing=float(not len(p)),history_recent_age_s=(cut-int(p.end_ns.max())-SECOND)/SECOND if len(p) else np.nan,
        history_net_wh_per_km=1000*p.energy_kwh.sum()/p.distance_km.sum() if len(p) and p.distance_km.sum()>0 else np.nan,
        history_wh_per_km_median=float((1000*p.energy_kwh/p.distance_km.where(p.distance_km>0)).median()) if len(p) else np.nan,
        history_distance_mean=mean('distance_km'),history_duration_mean=mean('duration_s'),
        history_mean_speed_kph=float((3600*p.distance_km/p.duration_s.where(p.duration_s>0)).mean()) if len(p) else np.nan,
        history_ambient_mean=mean('ambient_mean'),history_pack_min_mean=mean('pack_min_mean'),history_pack_max_mean=mean('pack_max_mean'),
        od_history_count=len(s),od_history_net_wh_per_km=1000*s.energy_kwh.sum()/s.distance_km.sum() if len(s) and s.distance_km.sum()>0 else np.nan,
        od_history_duration_mean=float(s.duration_s.mean()) if len(s) else np.nan)
    for k,v in list(out.items()):
        if k not in ('history_count','history_missing','od_history_count'):out[k+'_missing']=float(not np.isfinite(v))
    return out


def replay_events(meta,reader,on_query=None):
    """reader called only when eligible source end+1s is reached; not at startup."""
    events=meta.loc[quality(meta)&meta.split.isin(PERMISSIONS)].sort_values(['end_ns','file'])
    queries=meta.loc[meta.split.isin(PERMISSIONS)].sort_values(['cut_ns','file'])
    event_rows=list(events.itertuples());cursor=0;released={};online={};provenance=[];snapshots=[];descriptors={}
    for row in queries.itertuples():
        while cursor<len(event_rows) and int(event_rows[cursor].end_ns)+SECOND<=row.cut_ns:
            e=event_rows[cursor];record=reader(e.Index)
            record.update(file=e.file,vehicle=e.vehicle,split=e.split,end_ns=int(e.end_ns),available_ns=int(e.end_ns)+SECOND,od_key=e.od_key,quality_eligible=True)
            record['history_hash']=digest(record);descriptors[e.Index]=record;released.setdefault(e.vehicle,[]).append(record);cursor+=1
        permission=PERMISSIONS[row.split]
        all_prior=[r for r in released.get(row.vehicle,[]) if r['split'] in permission and r['file']!=row.file]
        prior=all_prior[-10:];same=[r for r in prior if r['od_key']==row.od_key];values=summary(prior,same,int(row.cut_ns));online[row.Index]=values
        for rank,r in enumerate(prior):
            assert r['vehicle']==row.vehicle and r['available_ns']<=row.cut_ns and r['split'] in permission and r['split']!='calibration' and r['file']!=row.file
            provenance.append(dict(query_file=row.file,query_split=row.split,cut_ns=int(row.cut_ns),history_file=r['file'],history_split=r['split'],history_end_ns=r['end_ns'],history_available_ns=r['available_ns'],age_s=(row.cut_ns-r['available_ns'])/SECOND,same_od=r['od_key']==row.od_key,rank_oldest_first=rank,history_hash=r['history_hash'],quality_eligible=True))
        snapshots.append(dict(query_file=row.file,query_split=row.split,cut_ns=int(row.cut_ns),released_event_count=cursor,available_same_vehicle_pool_n=len(all_prior),
            latest10_ids=[r['file'] for r in prior],same_od_latest10_ids=[r['file'] for r in same],history_snapshot_hash=digest(prior),online_feature_hash=digest(values)))
        if on_query:on_query(row,all_prior,prior,same,values)
    return pd.DataFrame.from_dict(online,orient='index').sort_index(),provenance,snapshots,descriptors


def spatial_path(lat,lon,step=50.):
    good=valid_gps(lat,lon);la=np.asarray(lat)[good];lo=np.asarray(lon)[good]
    if not len(la):raise ValueError('missing route proxy')
    keep=np.r_[True,(np.diff(la)!=0)|(np.diff(lo)!=0)];la=la[keep];lo=lo[keep]
    lengths=haversine(la,lo)*1000;cum=np.r_[0,np.cumsum(lengths)]
    pos=np.r_[np.arange(0,cum[-1],step),cum[-1]] if cum[-1]>0 else np.array([0.])
    xy=np.column_stack([np.interp(pos,cum,lo)*111320*np.cos(np.radians(51.)),np.interp(pos,cum,la)*111320])
    return xy,dict(unfiltered_km=float(cum[-1]/1000),space_only_chords_ge250m=int((lengths>=250).sum()),space_only_chords_ge1000m=int((lengths>=1000).sum()),max_spatial_chord_m=float(lengths.max()) if len(lengths) else 0.)


def path_similarity(a,b):
    ab=cKDTree(b).query(a)[0];ba=cKDTree(a).query(b)[0];joined=np.r_[ab,ba]
    return dict(symmetric_nearest_mean_m=float(joined.mean()),symmetric_nearest_p95_m=float(np.quantile(joined,.95)),symmetric_hausdorff_m=float(joined.max()))


def main():
    ap=argparse.ArgumentParser()
    for key in ('prior','data','out','todo'):ap.add_argument('--'+key,type=Path,required=True)
    a=ap.parse_args();assert platform.system()=='Linux' and str(Path.cwd()).startswith('/root/autodl-tmp')
    if (a.out/'preflight_checks.json').exists():raise FileExistsError('EV-B02 already prepared; do not overwrite')
    a.out.mkdir(parents=True,exist_ok=True)
    c=pd.read_csv(a.prior/'cohort_manifest.csv',usecols=list(META));f=pd.read_csv(a.prior/'features.csv');assert len(c)==1272 and list(c.file)==list(f.file) and c.file.is_unique
    targets=pd.concat([pd.read_csv(a.prior/('predictions_'+s+'.csv')) for s in ('train','validation','future')],ignore_index=True).set_index('file')
    assert set(targets.index)==set(c.loc[c.split!='calibration','file']) and targets.index.is_unique
    # Label reads only from three non-calibration prediction artifacts. No cal truth lookup.
    prefix=pd.read_csv(a.prior/'prefix_audit.csv').set_index('file');budget_ns=c.cut_ns+SECOND
    assert (prefix.loc[c.file,'last_prefix_bucket_available_ns'].to_numpy()<=budget_ns).all() and (budget_ns<=c.end_ns).all()
    columns=json.loads((a.prior/'feature_columns.json').read_text());updates=[k for k in columns['blocks']['H'] if not k.startswith('past_thermal')]
    policy=dict(train=['train'],validation=['train','validation'],development=['train','validation','earlier_completed_development'],
        calibration='never query, release, fit, select or score its terminal energy/statistics',release='source_end_ns+1e9<=query_cut_ns, same vehicle, different ID, same quality as EVB01',
        history_limit=10,same_od='inside same latest10 only',past_thermal='all original six thermal columns unchanged',feature_cutoff='t_cut',budget_output='t_cut+1s; suffix still starts at original t_cut',weights='original S/H, rates, vocabulary and order frozen')
    inputs=[]
    for name in ('cohort_manifest.csv','features.csv','split_manifest.csv','feature_columns.json','feature_contract.json','preprocessing_manifest.json','model_S.joblib','model_H.joblib','model_O.joblib','predictions_train.csv','predictions_validation.csv','predictions_future.csv','route_audit.csv','prefix_audit.csv'):
        inputs.append(dict(name=name,path=str((a.prior/name).resolve()),sha256=sha(a.prior/name)))
    original=json.loads((a.prior/'input_manifest.json').read_text())
    for item in original['files']:
        assert sha(a.data/item['name'])==item['sha256'];inputs.append(dict(name=item['name'],path=str((a.data/item['name']).resolve()),sha256=item['sha256']))
    write_json(a.out/'input_manifest.json',dict(inputs=inputs,todo_sha256=sha(a.todo),code_sha256=sha(__file__),prior='EV-B01 frozen artifacts; unchanged'))
    write_json(a.out/'history_policy.json',policy)
    config=dict(experiment='EV-B02',frozen_at=pd.Timestamp.now(tz='UTC').isoformat(),tree=TREE_CONFIG,threads=4,formal_fit_limit=3,calibration_n=125,eval_identity='validation188 + previously seen development193; no new independent test',
        oracle_additions={'O-time':['actual_remaining_duration_s'],'O-motion':['actual_gps_distance_weighted_v2','actual_observed_stop_fraction','actual_v2_missing','actual_stop_missing'],'O-quality':['actual_motion_coverage','actual_motion_missing']},
        refreshed_columns=updates,condition_cells='2 vehicles x4 original distance bins x4 original ambient bins; n<10 descriptive',
        reweight='common nonempty train/val/development cells; additionally report n>=10 common cells; validation means weighted by development cell count',
        repeated_route=dict(window_days=90,all_completed_history_not_latest10=True,spatial_step_m=50,mean_distance_max_m=50,p95_max_m=100,length_ratio=[.8,1.25],diagnostic_only=True),
        geometry=dict(unfiltered_gt_budget_ratio=1.01,space_only_chord_thresholds_m=[250,1000],no_speed_cleaning=True),
        bootstrap=dict(repetitions=1000,seed=42,unit='same ISO week both vehicles together; fixed online errors, no history reshuffle'),
        oracle_concentration=dict(top_fraction=.05,high_concentration_threshold=.5,good_motion_coverage=.95),history_policy=policy)
    write_json(a.out/'run_config.json',config);(a.out/'protocol_frozen.md').write_text('# EV-B02：执行前完整todo快照\n\n'+a.todo.read_text(encoding='utf-8'),encoding='utf-8')
    c.to_csv(a.out/'cohort_reference.csv',index=False);c[['file','vehicle','split','cut_ns','end_ns']].to_csv(a.out/'split_reference.csv',index=False)
    z=zipfile.ZipFile(a.data/'driving_signal_views_1s.zip');read_count=0;path_cache={};route_rows=[];coverage_rows=[];pair_rows=[];feature_provenance=[]
    def reader(i):
        nonlocal read_count
        r=c.loc[i];assert r.split!='calibration';raw=z.read(r.file);d=pd.read_csv(io.BytesIO(raw),usecols=['VCRIGHT_tempAmbientRaw','VCFRONT_tempAmbient','BMSminPackTemperature','BMSmaxPackTemperature'])
        ambient=pd.to_numeric(d.VCRIGHT_tempAmbientRaw,errors='coerce').to_numpy(float);fallback=pd.to_numeric(d.VCFRONT_tempAmbient,errors='coerce').to_numpy(float);ambient=np.where(np.isfinite(ambient),ambient,fallback)
        lo=pd.to_numeric(d.BMSminPackTemperature,errors='coerce').to_numpy(float);hi=pd.to_numeric(d.BMSmaxPackTemperature,errors='coerce').to_numpy(float);good=np.isfinite(lo)&np.isfinite(hi)&(lo>=-40)&(hi<=100)&(lo<=hi)
        read_count+=1
        return dict(energy_kwh=float(targets.loc[r.file,'y_full']),distance_km=float(r.gps_path_plausible_km),duration_s=(r.end_ns-r.start_ns)/SECOND,
            ambient_mean=float(np.nanmean(ambient)) if np.isfinite(ambient).any() else np.nan,pack_min_mean=float(lo[good].mean()) if good.any() else np.nan,pack_max_mean=float(hi[good].mean()) if good.any() else np.nan,raw_source_sha256=hashlib.sha256(raw).hexdigest())
    def route(name):
        if name not in path_cache:
            i=c.index[c.file==name][0];r=c.loc[i];assert r.split!='calibration'
            d=pd.read_csv(z.open(name),usecols=['Timestamp','GPSLatitude04F','GPSLongitude04F']);tn=ns(d.Timestamp);sel=(tn>=r.cut_ns)&(tn<=r.end_ns)
            xy,info=spatial_path(pd.to_numeric(d.GPSLatitude04F,errors='coerce').to_numpy(float)[sel],pd.to_numeric(d.GPSLongitude04F,errors='coerce').to_numpy(float)[sel]);path_cache[name]=(xy,info)
        return path_cache[name]
    def on_query(row,all_prior,prior,same,values):
        for k,v in values.items():
            sources=same if k.startswith('od_history') else prior[-1:] if k.startswith('history_recent_age') else prior
            feature_provenance.append(dict(query_file=row.file,cut_ns=int(row.cut_ns),feature=k,source_ids=json.dumps([p['file'] for p in sources]),history_hash=digest(sources),value=v))
        recent=[r for r in all_prior if r['od_key']==row.od_key and (row.cut_ns-r['available_ns'])/SECOND<=90*86400]
        close=0
        if recent:
            path,_=route(row.file)
            for r in recent:
                other,_=route(r['file']);similarity=path_similarity(path,other);rd=c.loc[c.file==r['file'],'D'].iloc[0];ratio=float(row.D/rd)
                matched=similarity['symmetric_nearest_mean_m']<=50 and similarity['symmetric_nearest_p95_m']<=100 and .8<=ratio<=1.25;close+=int(matched)
                pair_rows.append(dict(query_file=row.file,query_split=row.split,history_file=r['file'],history_split=r['split'],cut_ns=int(row.cut_ns),history_available_ns=r['available_ns'],age_days=(row.cut_ns-r['available_ns'])/SECOND/86400,
                    query_distance_km=float(row.D),history_distance_km=float(rd),distance_ratio=ratio,**similarity,spatial_similarity_rule_passed=matched))
        ages=[(row.cut_ns-r['available_ns'])/SECOND/86400 for r in recent]
        coverage_rows.append(dict(file=row.file,split=row.split,same_od_90d_n=len(recent),at_least1=len(recent)>=1,at_least3=len(recent)>=3,at_least5=len(recent)>=5,spatially_supported_n=close,
            youngest_same_od_age_days=min(ages) if ages else np.nan,oldest_same_od_age_days=max(ages) if ages else np.nan,query_D_km=float(row.D)))
        if len(coverage_rows)%200==0:print(json.dumps(dict(queries=len(coverage_rows),stage='event_release_and_route_audit')),flush=True)
    online,hp,snap,desc=replay_events(c,reader,on_query)
    assert set(online.index)==set(c.index[c.split!='calibration']) and set(online.columns)==set(updates)
    train=c.index[c.split=='train'];old=f.loc[train,updates].to_numpy(float);new=online.loc[train,updates].to_numpy(float)
    np.testing.assert_allclose(new,old,atol=1e-8,rtol=0,equal_nan=True);difference=float(np.nanmax(abs(new-old)))
    # Canonical saved train values avoid meaningless round-trip bin-boundary changes.
    online.loc[train,updates]=f.loc[train,updates].to_numpy();canonicalized=len(train)
    refreshed=f.loc[online.index].copy();refreshed.loc[:,updates]=online.loc[refreshed.index,updates].to_numpy()
    untouched=[k for k in f.columns if k not in updates];pd.testing.assert_frame_equal(refreshed[untouched],f.loc[online.index,untouched])
    refreshed.to_csv(a.out/'features_online.csv',index=False);pd.DataFrame(hp).to_csv(a.out/'history_query_manifest.csv',index=False);pd.DataFrame(feature_provenance).to_csv(a.out/'history_feature_manifest.csv',index=False)
    pd.DataFrame([{**r,'latest10_ids':json.dumps(r['latest10_ids']),'same_od_latest10_ids':json.dumps(r['same_od_latest10_ids'])} for r in snap]).to_csv(a.out/'history_query_snapshots.csv',index=False)
    write_json(a.out/'released_history_records.json',clean(list(desc.values())))
    pd.DataFrame(coverage_rows).to_csv(a.out/'route_history_coverage.csv',index=False);pd.DataFrame(pair_rows).to_csv(a.out/'route_history_pairs.csv',index=False)
    ra=pd.read_csv(a.prior/'route_audit.csv');joined=c[c.split!='calibration'].merge(ra,on='file',validate='one_to_one',suffixes=('','_audit'));geometry=[]
    for r in joined.itertuples():
        flagged=r.unfiltered_geometry_km>1.01*r.D
        info=route(r.file)[1] if flagged else {}
        geometry.append(dict(file=r.file,split=r.split,vehicle=r.vehicle,D=r.D,distance_bin=r.distance_bin,ambient_bin=r.ambient_bin,ambient_c=r.ambient_c,
            unfiltered_geometry_km=r.unfiltered_geometry_km,excess_km=r.unfiltered_geometry_km-r.D,geometry_gt1pct=flagged,ratio=r.unfiltered_geometry_km/r.D,
            main_view_internal_consistency_checked=r.main_view_internal_consistency_checked,uncertain_charger_signal_or_mode_overlap=r.uncertain_charger_signal_or_mode_overlap,gap_gt_2s_count=r.gap_gt_2s_count,**info))
    pd.DataFrame(geometry).to_csv(a.out/'geometry_source_diagnostics.csv',index=False)
    z.close()
    dev=c.index[c.split=='future'];ages=online.loc[dev,'history_recent_age_s'];coverage=pd.DataFrame(coverage_rows)
    checks=dict(passed=True,n_cohort=1272,n_queries=1147,calibration_metadata_only=125,calibration_releases=0,calibration_queries=0,event_reader_calls=read_count,
        released_descriptors=len(desc),queries_all_unique=True,all_history_available_at_cut=True,same_vehicle_different_id=True,permission_by_query_split=True,
        unchanged_S_route_thermal=True,train_formula_replay_max_abs=difference,train_exact_saved_values_canonicalized_n=canonicalized,
        budget_output_delay_s=1,prefix_available_at_budget_all1272=True,suffix_cut_label_unchanged=True,
        development_online_history_available_n=int(ages.notna().sum()),development_online_history_median_age_hours=float(ages.median()/3600),development_online_history_within7d_n=int((ages<=7*86400).sum()),
        development_sameod90d_support_n=int((coverage.loc[coverage.split=='future','same_od_90d_n']>0).sum()),development_geometry_gt1pct_n=int(sum(r['geometry_gt1pct'] and r['split']=='future' for r in geometry)),
        history_label_read='noncal prediction artifacts only; release descriptor when end+1s reached; cal source signals/energy never read',fit_count=0)
    assert checks['development_online_history_available_n']==193 and checks['development_online_history_within7d_n']==190
    assert abs(checks['development_online_history_median_age_hours']-3.6367)<.01
    write_json(a.out/'preflight_checks.json',checks)
    write_json(a.out/'prepared_hashes.json',{p.name:sha(p) for p in a.out.iterdir() if p.is_file() and p.suffix in ('.csv','.json','.md') and p.name!='prepared_hashes.json'})
    print(json.dumps(checks,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
