"""EV-B02: frozen-model history replay followed by exactly three cloud fits."""
from __future__ import annotations
import argparse,itertools,json,os,platform,time
from pathlib import Path
import joblib,numpy as np,pandas as pd,sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits,threadpool_info
from energyville_evb01_data import TREE_CONFIG,write_json,sha,flag,SECOND
from energyville_evb02_data import clean,replay_events
from run_energyville_evb01 import scores,ratio_check

CONTROLS=('S-fixed','H-frozen','H-online','O-combined')
NEW=('O-time','O-motion','O-quality')
GROUPS=CONTROLS+NEW
ADDITIONS={'O-time':['actual_remaining_duration_s'],'O-motion':['actual_gps_distance_weighted_v2','actual_observed_stop_fraction','actual_v2_missing','actual_stop_missing'],'O-quality':['actual_motion_coverage','actual_motion_missing']}


def matrix(f,bundle):
    vocab=bundle['vehicle_vocabulary'];lookup={v:i for i,v in enumerate(vocab)}
    oh=np.eye(len(vocab))[np.array([lookup.get(v,lookup['__unknown__']) for v in f.vehicle])]
    x=np.column_stack([oh,f[bundle['numeric_features']].to_numpy(float)])
    assert not np.isinf(x).any()
    return x


def predict(f,c,bundle):
    b=c.vehicle.map(bundle['baseline']['by_vehicle']).fillna(bundle['baseline']['overall']).to_numpy()*c.D.to_numpy()
    return b+bundle['model'].predict(matrix(f,bundle))


def age_bin(v):
    a=np.asarray(v,float)/86400
    return np.where(~np.isfinite(a),'missing',np.where(a<=1,'0-1d',np.where(a<=7,'1-7d','>7d')))


def evaluate(c,pr,groups=GROUPS):
    result={};y=c.y_rem.to_numpy();v=c.vehicle.to_numpy();prefix=c.e_prefix.to_numpy()
    layers=dict(vehicle=c.vehicle,distance=c.distance_bin,ambient=c.ambient_bin,od_seen_train=c.od_seen_train,
        original_od_history=c.original_od_history,online_od_history=c.online_od_history,original_history_age=c.original_history_age,online_history_age=c.online_history_age,
        uncertain_charger=flag(c.uncertain_charger_signal_or_mode_overlap),suspected_charger=flag(c.suspected_charger_signal_or_mode_overlap),
        confirmed_external_charging=flag(c.confirmed_external_charging_from_counter_or_joint_direct),log_gap_gt2s=c.gap_gt_2s_count>0,
        internally_consistent=flag(c.main_view_internal_consistency_checked),quantization_le5pct=flag(c.main_view_5pct_counter_quantization_sufficient))
    sensitivity=dict(no_uncertain_charger=~flag(c.uncertain_charger_signal_or_mode_overlap),internal_consistency=flag(c.main_view_internal_consistency_checked),
        no_gt2s_gap=c.gap_gt_2s_count==0,quantization_le5pct=flag(c.main_view_5pct_counter_quantization_sufficient),good_motion_coverage=c.actual_motion_coverage>=.95)
    for g in groups:
        p=pr[g];assert np.allclose(p+prefix-c.y_full.to_numpy(),p-y,rtol=0,atol=1e-8)
        s=dict(remaining=scores(y,p,v),full=scores(c.y_full,p+prefix,v),slices={},sensitivity={})
        for name,values in layers.items():
            s['slices'][name]={}
            for layer in sorted(values.unique(),key=str):
                ix=(values==layer).to_numpy();r=scores(y[ix],p[ix],v[ix]);r['descriptive_only']=int(ix.sum())<20;s['slices'][name][str(layer)]=r
        for name,mask in sensitivity.items():
            ix=mask.to_numpy();s['sensitivity'][name]=scores(y[ix],p[ix],v[ix])
        for name,ix in [('ge10km',(c.D>=10).to_numpy()),('ge50km',(c.D>=50).to_numpy())]:
            s[name]=scores(y[ix],p[ix],v[ix]);s[name]['descriptive_only']=int(ix.sum())<20
        result[g]=s
    return result


def gate_a(metrics):
    gates={}
    for partition in ('validation','development'):
        m=metrics[partition];a=m['H-online']['remaining'];h=m['H-frozen']['remaining'];s=m['S-fixed']['remaining']
        r={'vs_H_improve10pct':ratio_check(h['mae_kwh'],a['mae_kwh'],.9),'vs_S_improve5pct':ratio_check(s['mae_kwh'],a['mae_kwh'],.95),
           'vs_S_rmse_guard':ratio_check(s['rmse_kwh'],a['rmse_kwh'],1.05),'vs_S_under_guard':ratio_check(s['underestimate_mean_kwh'],a['underestimate_mean_kwh'],1.05)}
        for v,layer in m['S-fixed']['slices']['vehicle'].items():
            r['vehicle_'+v]=ratio_check(layer['mae_kwh'],m['H-online']['slices']['vehicle'][v]['mae_kwh'],1.1) if layer['n']>=20 else dict(passed=None,n=layer['n'],reason='n<20 descriptive only')
        a10=m['H-online']['ge10km'];s10=m['S-fixed']['ge10km']
        r['ge10km']=ratio_check(s10['mae_kwh'],a10['mae_kwh'],1.1) if s10['n']>=20 else dict(passed=None,n=s10['n'],reason='n<20 descriptive only')
        gates[partition]=r
    return dict(passed=all(v['passed'] is True for r in gates.values() for v in r.values()),partitions=gates,interpretation='predeclared investment gate, not business acceptance or independent statistical confirmation')


def bootstrap(c,pr):
    dt=pd.to_datetime(c.t_cut,utc=True).dt.isocalendar();week=(dt.year.astype(str)+'-W'+dt.week.astype(str)).to_numpy();unique=np.unique(week)
    rng=np.random.default_rng(42);draws=rng.integers(0,len(unique),size=(1000,len(unique)));counts=np.array([(week==w).sum() for w in unique]);rows=[]
    pairs=[('H-frozen','H-online'),('S-fixed','H-online'),('H-frozen','O-combined'),('S-fixed','O-combined')]
    for g in NEW:pairs.extend([('H-frozen',g),('S-fixed',g),(g,'O-combined')])
    for l,r in pairs:
        diff=abs(pr[l]-c.y_rem.to_numpy())-abs(pr[r]-c.y_rem.to_numpy());sums=np.array([diff[week==w].sum() for w in unique]);boot=sums[draws].sum(1)/counts[draws].sum(1)
        rows.append(dict(left=l,right=r,n=len(c),week_blocks=len(unique),mae_improvement_right_kwh=float(diff.mean()),ci95_low_kwh=float(np.quantile(boot,.025)) if len(unique)>=8 else np.nan,
                         ci95_high_kwh=float(np.quantile(boot,.975)) if len(unique)>=8 else np.nan,repetitions=1000,seed=42,unit='fixed online errors; whole ISO week with both cars; no temporal history reshuffle'))
    return pd.DataFrame(rows)


def save_predictions(c,pr,out,partition):
    p=c.copy()
    for g,v in pr.items():p['pred_'+g+'_rem']=v;p['pred_'+g+'_full']=v+c.e_prefix.to_numpy()
    p.to_csv(out/('predictions_'+partition+'.csv'),index=False)


def causal_audit(c,f,online,model,out):
    hp=pd.read_csv(out/'history_query_manifest.csv');lookup=c.set_index('file')
    assert not (hp.query_file==hp.history_file).any()
    assert (hp.history_available_ns<=hp.cut_ns).all()
    assert (hp.history_split!='calibration').all()
    assert (lookup.loc[hp.query_file,'vehicle'].to_numpy()==lookup.loc[hp.history_file,'vehicle'].to_numpy()).all()
    records=json.loads((out/'released_history_records.json').read_text());byfile={r['file']:r for r in records}
    candidates=[r for r in records if r['split']=='future'];target=max(candidates,key=lambda r:r['duration_s'])
    def read(i,changed=False):
        r=dict(byfile[c.loc[i,'file']]);r.pop('history_hash',None)
        if changed and r['file']==target['file']:r['energy_kwh']+=1000.
        return r
    base,_,_,_=replay_events(c,lambda i:read(i));mutated,_,_,_=replay_events(c,lambda i:read(i,True))
    early=c.index[(c.cut_ns<target['available_ns'])&(c.split!='calibration')]
    pd.testing.assert_frame_equal(base.loc[early],mutated.loc[early])
    inputs=f.loc[early].copy();alternate=inputs.copy();updates=list(base.columns)
    inputs.loc[:,updates]=base.loc[early,updates].to_numpy();alternate.loc[:,updates]=mutated.loc[early,updates].to_numpy()
    np.testing.assert_allclose(predict(inputs,c.loc[early],model),predict(alternate,c.loc[early],model),rtol=0,atol=0)
    late=c.index[(c.cut_ns>=target['available_ns'])&(c.split!='calibration')]
    changed=int((base.loc[late,'history_net_wh_per_km'].fillna(-999)!=mutated.loc[late,'history_net_wh_per_km'].fillna(-999)).sum())
    audit=dict(passed=True,history_rows=len(hp),self_in_history=0,wrong_vehicle=0,late_sources=0,calibration_sources=0,
        mutated_file=target['file'],mutated_completed_label_delta_kwh=1000.,source_duration_s=target['duration_s'],source_available_ns=target['available_ns'],
        earlier_queries_identical_features_and_same_saved_H_predictions=len(early),later_queries_with_changed_history=changed,conclusion='unended target mutation cannot affect earlier features/predictions; only subsequent eligible queries can change')
    assert changed>0
    write_json(out/'causal_audit.json',audit)


def diagnostic_tables(c,preds,metrics,out):
    cells=list(itertools.product(sorted(c.vehicle.unique()),['<2','2-10','10-50','>=50'],['<10','10-25','>25','missing']))
    rows=[];values={}
    for partition,s in [('train','train'),('validation','validation'),('development','future')]:
        idx=c.index[c.split==s];cc=c.loc[idx];pr=preds[partition]
        for v,d,a in cells:
            ix=((cc.vehicle==v)&(cc.distance_bin==d)&(cc.ambient_bin==a)).to_numpy();n=int(ix.sum());cell=(v,d,a)
            values[(partition,cell)]=dict(n=n)
            for g in GROUPS:
                err=pr[g][ix]-cc.y_rem.to_numpy()[ix];m=float(abs(err).mean()) if n else np.nan;b=float(err.mean()) if n else np.nan
                values[(partition,cell)][g]=dict(mae=m,bias=b)
                rows.append(dict(partition=partition,vehicle=v,distance_bin=d,ambient_bin=a,n=n,descriptive_only=n<10,group=g,
                    suffix_sum_wh_per_km=1000*cc.loc[ix,'y_rem'].sum()/cc.loc[ix,'D'].sum() if n else np.nan,
                    full_source_sum_wh_per_km=1000*cc.loc[ix,'y_full'].sum()/cc.loc[ix,'gps_path_plausible_km'].sum() if n else np.nan,
                    mae_kwh=m,bias_kwh=b,combined_oracle_gain_from_H_kwh=float((abs(pr['H-frozen'][ix]-cc.y_rem.to_numpy()[ix])-abs(pr['O-combined'][ix]-cc.y_rem.to_numpy()[ix])).mean()) if n else np.nan))
    pd.DataFrame(rows).to_csv(out/'conditional_support.csv',index=False)
    rew=[]
    for minimum in (1,10):
        common=[cell for cell in cells if all(values[(s,cell)]['n']>=minimum for s in ('train','validation','development'))]
        devn=sum(values[('development',cell)]['n'] for cell in common);valn=sum(values[('validation',cell)]['n'] for cell in common)
        for g in GROUPS:
            r=dict(group=g,minimum_cell_n_all3=minimum,common_cell_n=len(common),development_covered_n=devn,development_uncovered_fraction=1-devn/193,validation_covered_n=valn,description_only=True)
            for name in ('mae','bias'):
                r['validation_original_common_'+name]=sum(values[('validation',cell)][g][name]*values[('validation',cell)]['n'] for cell in common)/valn if valn else None
                r['validation_reweighted_'+name]=sum(values[('validation',cell)][g][name]*values[('development',cell)]['n'] for cell in common)/devn if devn else None
                r['development_common_'+name]=sum(values[('development',cell)][g][name]*values[('development',cell)]['n'] for cell in common)/devn if devn else None
            rew.append(r)
    pd.DataFrame(rew).to_csv(out/'reweighted_description.csv',index=False);write_json(out/'reweighted_description.json',clean(rew))
    geometry=pd.read_csv(out/'geometry_source_diagnostics.csv');geo=[]
    concentration=[];gateb={}
    for partition,s in [('validation','validation'),('development','future')]:
        cc=c[c.split==s];pr=preds[partition];y=cc.y_rem.to_numpy()
        gm=geometry[geometry.split==s].set_index('file').loc[cc.file].reset_index()
        for g in GROUPS:gm['error_'+g+'_kwh']=pr[g]-y
        geo.append(gm)
        for g in ('H-online',)+NEW:
            gain=abs(pr['H-frozen']-y)-abs(pr[g]-y);pos=np.maximum(gain,0);top=max(1,int(np.ceil(.05*len(cc))));high=cc.actual_motion_coverage.to_numpy()>=.95
            concentration.append(dict(partition=partition,group=g,mae_improvement_vs_H_kwh=float(gain.mean()),relative_improvement_vs_H=float(gain.mean()/abs(pr['H-frozen']-y).mean()),
                top5pct_n=top,top5pct_positive_gain_share=float(np.sort(pos)[-top:].sum()/pos.sum()) if pos.sum()>0 else None,
                good_coverage_n=int(high.sum()),good_coverage_mae_improvement_vs_H=float(gain[high].mean()) if high.any() else None,
                low_coverage_positive_gain_share=float(pos[~high].sum()/pos.sum()) if pos.sum()>0 else None,
                total_positive_gain_kwh=float(pos.sum()),total_negative_gain_kwh=float(-np.minimum(gain,0).sum())))
    pd.concat(geo).to_csv(out/'geometry_diagnostics.csv',index=False);pd.DataFrame(concentration).to_csv(out/'gain_concentration.csv',index=False)
    for g in NEW:
        checks={}
        for p in ('validation','development'):
            r=next(r for r in concentration if r['partition']==p and r['group']==g)
            checks[p]=dict(vs_H_improve10pct=r['relative_improvement_vs_H']>=.1,top5pct_positive_gain_share=r['top5pct_positive_gain_share'],low_coverage_positive_gain_share=r['low_coverage_positive_gain_share'],
                not_few_or_bad_coverage_dominated=r['top5pct_positive_gain_share'] is not None and r['top5pct_positive_gain_share']<=.5 and r['low_coverage_positive_gain_share']<=.5,
                good_coverage_mae_improvement_vs_H=r['good_coverage_mae_improvement_vs_H'])
        gateb[g]=dict(passed=all(r['vs_H_improve10pct'] and r['not_few_or_bad_coverage_dominated'] for r in checks.values()),checks=checks,
                     caveat='concentration screen is descriptive; correlated proxies and interactions are not isolated causes')
    write_json(out/'stageB_gates.json',clean(gateb))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--prior',type=Path,required=True);a=ap.parse_args();out=a.out
    assert platform.system()=='Linux' and str(Path.cwd()).startswith('/root/autodl-tmp'),'all formal fits on existing authorized cloud'
    if (out/'execution_manifest.json').exists():raise FileExistsError('already started; no silent refit')
    config=json.loads((out/'run_config.json').read_text());assert config['tree']==TREE_CONFIG and config['oracle_additions']==ADDITIONS and config['formal_fit_limit']==3
    assert json.loads((out/'preflight_checks.json').read_text())['passed']
    prepared=json.loads((out/'prepared_hashes.json').read_text())
    for name,digest in prepared.items():assert sha(out/name)==digest,name
    priorhashes=json.loads((out/'input_manifest.json').read_text())['inputs']
    for item in priorhashes:assert sha(item['path'])==item['sha256'],item['name']
    meta=pd.read_csv(out/'cohort_reference.csv');orig=pd.read_csv(a.prior/'features.csv')
    labels=pd.concat([pd.read_csv(a.prior/('predictions_'+s+'.csv'),usecols=['file','y_rem','e_prefix','y_full']) for s in ('train','validation','future')],ignore_index=True).set_index('file')
    c=meta.join(labels,on='file');f=orig;online=pd.read_csv(out/'features_online.csv').set_index('file').loc[c.loc[c.split!='calibration','file']]
    online.index=c.index[c.split!='calibration'];assert len(online)==1147 and c.loc[c.split=='calibration','y_full'].isna().all()
    np.testing.assert_allclose(c.loc[c.split!='calibration','y_full'],c.loc[c.split!='calibration','y_rem']+c.loc[c.split!='calibration','e_prefix'],rtol=0,atol=1e-8)
    c['original_history_age']=age_bin(f.history_recent_age_s);c['original_od_history']=f.od_history_count>0
    c['online_history_age']='calibration_excluded';c['online_od_history']=False
    c.loc[online.index,'online_history_age']=age_bin(online.history_recent_age_s);c.loc[online.index,'online_od_history']=(online.od_history_count>0).to_numpy()
    c['actual_motion_coverage']=f.actual_motion_coverage
    index={name:c.index[c.split==s] for name,s in [('train','train'),('validation','validation'),('development','future')]}
    assert [len(index[p]) for p in index]==[766,188,193]
    started=time.perf_counter();runtime=dict(started_at=pd.Timestamp.now(tz='UTC').isoformat(),platform=platform.platform(),python=platform.python_version(),sklearn=sklearn.__version__,pandas=pd.__version__,numpy=np.__version__,threads=4,cpu_count=os.cpu_count(),groups={})
    execution=dict(planned_new_fits=3,planned_new_groups=NEW,prepared_hashes=prepared,code_hashes={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('energyville_evb02_data.py'),Path(__file__).with_name('energyville_evb02_report.py')]},runtime=runtime,
        control_weights='original S/H/O no refit',fit_features='original frozen training-only H, not online H',calibration_labels='never loaded',development_identity='previous future, already seen, not independent test')
    write_json(out/'execution_manifest.json',execution)
    preds={p:{} for p in index};parity={};stagea={}
    with threadpool_limits(limits=4):
        runtime['threadpools']=threadpool_info();bundles={g:joblib.load(a.prior/('model_'+g+'.joblib')) for g in ('S','H','O')}
        causal_audit(c,f,online,bundles['H'],out)
        for p,idx in index.items():
            saved=pd.read_csv(a.prior/('predictions_'+('future' if p=='development' else p)+'.csv')).set_index('file').loc[c.loc[idx,'file']]
            parity[p]={}
            for g,old in [('S-fixed','S'),('H-frozen','H'),('O-combined','O')]:
                t=time.perf_counter();pr=predict(f.loc[idx],c.loc[idx],bundles[old]);preds[p][g]=pr
                difference=float(np.max(abs(pr-saved['pred_'+old+'_rem'].to_numpy())));assert difference<=1e-8,(p,g,difference)
                parity[p][g]=difference;runtime['groups'].setdefault(g,{})[p+'_predict_s']=time.perf_counter()-t
            t=time.perf_counter();preds[p]['H-online']=predict(online.loc[idx],c.loc[idx],bundles['H']);runtime['groups'].setdefault('H-online',{})[p+'_predict_s']=time.perf_counter()-t
            stagea[p]=evaluate(c.loc[idx],preds[p],CONTROLS)
            save_predictions(c.loc[idx],preds[p],out,'stageA_'+p)
        write_json(out/'control_replay_checks.json',dict(passed=True,tolerance_kwh=1e-8,maximum_absolute_differences=parity))
        write_json(out/'stageA_metrics.json',clean(stagea));write_json(out/'stageA_gate.json',gate_a(stagea))
        print(json.dumps(dict(stage='A_complete_no_fit',mae={p:{g:stagea[p][g]['remaining']['mae_kwh'] for g in CONTROLS} for p in ('validation','development')})),flush=True)
        h=bundles['H'];tr=index['train'];b0=c.loc[tr,'vehicle'].map(h['baseline']['by_vehicle']).fillna(h['baseline']['overall']).to_numpy()*c.loc[tr,'D'].to_numpy()
        fitids=c.loc[tr,'file'].tolist();assert h['train_ids']==fitids;fit_count=0
        # A prior persisted model is never silently re-fitted, including after an interrupted run.
        for g in NEW:
            if (out/('model_'+g+'.joblib')).exists():raise FileExistsError(g+' model already exists')
            bundle={k:h[k] for k in ('vehicle_vocabulary','baseline','tree','train_ids')};bundle['numeric_features']=h['numeric_features']+ADDITIONS[g]
            assert len(bundle['numeric_features'])==len(set(bundle['numeric_features'])) and not any(k in bundle['numeric_features'] for k in ('y_rem','y_full','e_prefix','end_soc','IV'))
            x=matrix(f.loc[tr],bundle);fit_count+=1;assert fit_count<=3
            write_json(out/'fit_progress.json',dict(fit_calls_started=fit_count,current_group=g,train_n=766,train_ids_sha256=__import__('hashlib').sha256(json.dumps(fitids).encode()).hexdigest(),no_calibration=True))
            model=HistGradientBoostingRegressor(**TREE_CONFIG);t=time.perf_counter();model.fit(x,c.loc[tr,'y_rem'].to_numpy()-b0);fit_s=time.perf_counter()-t;assert model.n_iter_==500
            bundle.update(model=model,runner_sha256=sha(__file__),group=g,history_policy='frozen training-only original EVB01 H',oracle=True)
            joblib.dump(bundle,out/('model_'+g+'.joblib'));runtime['groups'][g]=dict(fit_s=fit_s,feature_count=x.shape[1],n_iter=model.n_iter_,new_fit=True)
            for p,idx in index.items():
                t=time.perf_counter();preds[p][g]=predict(f.loc[idx],c.loc[idx],bundle);runtime['groups'][g][p+'_predict_s']=time.perf_counter()-t
            print(json.dumps(dict(group=g,fit_s=fit_s,val_mae=float(abs(preds['validation'][g]-c.loc[index['validation'],'y_rem'].to_numpy()).mean()))),flush=True)
    metrics={}
    for p,idx in index.items():metrics[p]=evaluate(c.loc[idx],preds[p]);save_predictions(c.loc[idx],preds[p],out,p)
    write_json(out/'metrics.json',clean(metrics));pairs=bootstrap(c.loc[index['development']],preds['development']);pairs.to_csv(out/'paired_comparisons.csv',index=False)
    diagnostic_tables(c,preds,metrics,out)
    runtime.update(formal_fit_calls=fit_count,total_s_before_report=time.perf_counter()-started,finished_at=pd.Timestamp.now(tz='UTC').isoformat())
    import resource
    runtime['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(out/'runtime.json',clean(runtime))
    for item in priorhashes:assert sha(item['path'])==item['sha256'],item['name']
    from energyville_evb02_report import report
    report(out)
    write_json(out/'completion.json',dict(state='complete',formal_fits=3,calibration_energy_consumed=0,E5='shelved_not_started',no_next_experiment_started=True,completed_at=pd.Timestamp.now(tz='UTC').isoformat(),
        files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='completion.json'}))
    print(json.dumps(dict(state='complete',stageA_gate=gate_a(metrics)['passed'],mae={p:{g:metrics[p][g]['full']['mae_kwh'] for g in GROUPS} for p in ('validation','development')})),flush=True)


if __name__=='__main__':main()
