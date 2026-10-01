"""Read-only model/data audit plus a separate verification artifact; never refit."""
from __future__ import annotations
import argparse,json,platform,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from energyville_evb01_data import sha,write_json,flag


def independently_score(c,p,g,full=False):
    y=c['y_full' if full else 'y_rem'].to_numpy(float);pred=p['pred_'+g+('_full' if full else '_rem')].to_numpy(float);err=pred-y
    return dict(n=len(y),mae_kwh=float(np.mean(np.abs(err))),rmse_kwh=float(np.sqrt(np.dot(err,err)/len(y))),bias_kwh=float(np.mean(err)),
        wape_pct=float(100*np.sum(np.abs(err))/np.sum(np.abs(y))),underestimate_mean_kwh=float(np.mean(np.clip(-err,0,None))),
        underestimate_p95_kwh=float(np.percentile(np.clip(-err,0,None),95)),mape_n=int(np.sum(y>.1)),mape_coverage=float(np.mean(y>.1)),
        mape_pct=float(np.mean(np.abs(err[y>.1])/y[y>.1])*100) if (y>.1).any() else None,
        vehicle_macro_mae_kwh=float(np.mean([np.abs(err[c.vehicle.to_numpy()==v]).mean() for v in c.vehicle.unique()])))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--models',action='store_true');parser.add_argument('--local-input-base',type=Path);a=parser.parse_args();out=a.out
    def read(name):return json.loads((out/name).read_text(encoding='utf-8'))
    config=read('run_config.json');lock=read('validation_lock.json');completion=read('completion.json');execution=read('execution_manifest.json');decision=read('validation_decision.json')
    inputs=read('input_manifest.json');script_base=Path(__file__).parent
    assert sha(script_base/'energyville_evb01_data.py')==inputs['preparation_code_sha256']
    assert sha(script_base/'run_energyville_evb01.py')==execution['runtime']['runner_sha256']
    if a.local_input_base:
        mapping=[]
        for item in inputs['files']:
            local=a.local_input_base/item['name'];assert local.stat().st_size==item['bytes'] and sha(local)==item['sha256']
            mapping.append(dict(name=item['name'],local_path=str(local.resolve()),cloud_path=item['path'],sha256=item['sha256'],bytes=item['bytes']))
        write_json(out/'input_path_mapping.json',dict(verified=True,files=mapping,cloud_project_root='/root/autodl-tmp',local_project_root=str(script_base.parent.resolve())))
    verified=0;dynamic=[];revised=[]
    for name,digest in completion['files'].items():
        if name=='training.log':
            # The final completion print occurs after this snapshot; intentionally mutable log.
            dynamic.append(name);continue
        if name=='EVB01_实验报告.md' and (out/'report_revision.json').exists():
            revision=read('report_revision.json');assert revision['original_sha256']==digest==sha(out/revision['original_copy']);revised.append(dict(name=name,original_sha256=digest,current_sha256=sha(out/name)))
        else:assert sha(out/name)==digest,(name,'completion snapshot mismatch')
        verified+=1
    for name,digest in execution['frozen_files'].items():assert sha(out/name)==digest,(name,'fit-time freeze mismatch')
    assert sha(out/'validation_decision.json')==lock['decision_sha256']
    assert pd.Timestamp(decision['decided_at'])<=pd.Timestamp(lock['locked_at'])<pd.Timestamp(read('future_evaluation_started.json')['started_at'])
    c=pd.read_csv(out/'cohort_manifest.csv');f=pd.read_csv(out/'features.csv');sp=pd.read_csv(out/'split_manifest.csv');assert len(c)==1272 and c.file.is_unique
    if (out/'todo_input_snapshot.md').exists():assert sha(out/'todo_input_snapshot.md')==inputs['todo_sha256']
    if a.local_input_base:
        examples=[]
        with zipfile.ZipFile(a.local_input_base/'driving_signal_views_1s.zip') as z:
            for name in read('preflight_checks.json')['representative_sources']:
                row=c.loc[c.file==name].iloc[0];d=pd.read_csv(z.open(name));tn=pd.to_datetime(d.Timestamp,utc=True).astype('datetime64[ns, UTC]').astype('int64').to_numpy()
                lo=pd.to_numeric(d.BMSminPackTemperature,errors='coerce').to_numpy(float);hi=pd.to_numeric(d.BMSmaxPackTemperature,errors='coerce').to_numpy(float)
                valid=np.isfinite(lo)&np.isfinite(hi)&(lo>=-40)&(hi<=100)&(lo<=hi);first=int(tn[np.flatnonzero(valid)[0]])
                assert first+1_000_000_000==row.cut_ns
                power=pd.to_numeric(d.BattVoltage132,errors='coerce').to_numpy(float)*pd.to_numeric(d.RawBattCurrent132,errors='coerce').to_numpy(float)/1000
                dt=np.diff(tn)/1e9;ok=(dt>0)&(dt<=2)&np.isfinite(power[:-1])&np.isfinite(power[1:])&(tn[1:]<=row.cut_ns)
                prefix=float(np.where(ok,(power[:-1]+power[1:])/2*dt/3600,0).sum());assert abs(prefix-row.e_prefix)<=1e-8
                times=pd.read_csv(out/'route_audit.csv').set_index('file').loc[name]
                assert times.first_remaining_gps_ns>=row.cut_ns and times.last_remaining_gps_ns<=row.end_ns
                examples.append(dict(file=name,distance_km=float(row.D),t_cut=str(row.t_cut),first_valid_temp_bucket_ns=first,state_first_temp_bucket_completed=True,
                    prefix_recomputed_kwh=prefix,prefix_stored_kwh=float(row.e_prefix),remaining_stored_kwh=float(row.y_rem),full_stored_kwh=float(row.y_full),identity_abs_error_kwh=abs(float(row.y_full-row.y_rem-prefix)),
                    units='voltage V * current A /1000 => kW; trapezoid seconds /3600 => kWh; signed positive discharge',route_first_offset_s=float(times.first_gps_offset_s),route_tail_offset_s=float(times.last_gps_offset_s),
                    prefix_availability_caveat='interval-end bucket at cut becomes available at cut+1s; not learner input'))
        write_json(out/'representative_source_checks.json',dict(passed=True,source='independent raw ZIP reread; nearest1/10/100km selected before fit',examples=examples))
    assert list(c.file)==list(f.file)==list(sp.file);assert np.allclose(c.y_full-c.e_prefix,c.y_rem,atol=1e-8,rtol=0)
    assert c.groupby(pd.to_datetime(c.t_cut,utc=True).dt.strftime('%Y-%m-%d')).split.nunique().max()==1
    for early,later in [('train','validation'),('validation','calibration'),('calibration','future')]:assert c.loc[c.split==early,'end_ns'].max()+1_000_000_000<=c.loc[c.split==later,'cut_ns'].min()
    train=set(c.loc[c.split=='train','file']);pool=pd.read_csv(out/'history_pool_manifest.csv');assert set(pool.file)<=train
    legal_pool=c[c.file.isin(pool.file)];assert flag(legal_pool.main_view_internal_consistency_checked).all() and not flag(legal_pool.uncertain_charger_signal_or_mode_overlap).any()
    assert not flag(legal_pool.confirmed_external_charging_from_counter_or_joint_direct).any() and (legal_pool.overlap_conflicting_iv_rows==0).all()
    hp=pd.read_csv(out/'history_provenance.csv');assert set(hp.history_file)<=train and (hp.file!=hp.history_file).all() and (hp.history_available_ns<=hp.cut_ns).all()
    hpjoin=hp.merge(c[['file','end_ns']],left_on='history_file',right_on='file',suffixes=('','_source'));assert (hpjoin.history_end_ns==hpjoin.end_ns).all()
    state=pd.read_csv(out/'state_provenance.csv');assert (state.available_ns<=state.cut_ns).all() and (state.available_ns==state.bucket_ns+1_000_000_000).all()
    thermal=pd.read_csv(out/'thermal_provenance.csv').merge(c[['file','start_ns']],on='file');assert (thermal.available_ns<=thermal.start_ns).all()
    cols=read('feature_columns.json');contract=read('feature_contract.json')
    for g in ('R','S','H'):assert not(set(cols['groups'][g])&set(cols['blocks']['O'])) and all(not contract[k]['oracle'] for k in cols['groups'][g])
    assert set(f.columns)=={'file','vehicle'}|set(sum(cols['blocks'].values(),[]))
    prep=read('preprocessing_manifest.json');assert set(prep['fitted_ids'])==train
    rates={v:float(g.y_rem.sum()/g.D.sum()) for v,g in c[c.split=='train'].groupby('vehicle')};assert rates==prep['baseline_rates_kwh_per_km']['by_vehicle']
    metrics=read('metrics.json');metric_checks=0;outputs={}
    for part in ('train','validation','future'):
        p=pd.read_csv(out/('predictions_'+part+'.csv'));rows=c[c.split==part].reset_index(drop=True);assert list(rows.file)==list(p.file);outputs[part]=p
        for g in ('B0','R','S','H','O'):
            assert np.allclose(p['pred_'+g+'_full']-p.y_full,p['pred_'+g+'_rem']-p.y_rem,atol=1e-8,rtol=0)
            for full,key in [(False,'remaining'),(True,'full')]:
                for name,v in independently_score(rows,p,g,full).items():
                    expected=metrics[part][g][key][name]
                    if v is not None:assert np.isclose(v,expected,atol=1e-10,rtol=1e-10),(part,g,key,name,v,expected)
                    else:assert expected is None
                    metric_checks+=1
            for layer,mask in dict(no_uncertain_charger=~flag(rows.uncertain_charger_signal_or_mode_overlap),internal_consistency=flag(rows.main_view_internal_consistency_checked),no_gt2s_gap=rows.gap_gt_2s_count==0,quantization_le5pct=flag(rows.main_view_5pct_counter_quantization_sufficient)).items():
                mask=mask.to_numpy();observed=independently_score(rows.loc[mask],p.loc[mask],g);assert np.isclose(observed['mae_kwh'],metrics[part][g]['sensitivity'][layer]['mae_kwh'],atol=1e-10);metric_checks+=1
        baseline=p.vehicle.map(rates)*p.D;assert np.allclose(baseline,p.pred_B0_rem,atol=1e-10,rtol=0)
    assert 'calibration' not in metrics and not list(out.glob('predictions_calibration*'))
    val=outputs['validation'];mae={g:float(abs(val['pred_'+g+'_rem']-val.y_rem).mean()) for g in ('B0','R','S','H','O')}
    def choose(left,right):return left if mae[left]<=mae[right] or mae[left]-mae[right]<=.01*max(min(mae[left],mae[right]),1e-12) else right
    reference=choose('B0','R');candidate=choose('S','H');assert decision['reference']==reference and decision['legal_candidate']==candidate and decision['chosen_legal']==choose(reference,candidate)
    future=outputs['future'];wk=pd.to_datetime(future.t_cut,utc=True).dt.strftime('%G-W%V').str.replace('-W0','-W',regex=False);weeks=sorted(wk.unique());counts=np.array([(wk==w).sum() for w in weeks]);draw=np.random.default_rng(42).integers(0,len(weeks),size=(1000,len(weeks)))
    pairs=pd.read_csv(out/'paired_comparisons.csv')
    for r in pairs.itertuples():
        dif=np.abs(future['pred_'+r.left+'_rem']-future.y_rem)-np.abs(future['pred_'+r.right+'_rem']-future.y_rem);sums=np.array([dif[wk==w].sum() for w in weeks]);bs=sums[draw].sum(axis=1)/counts[draw].sum(axis=1)
        assert np.isclose(dif.mean(),r.mae_improvement_right_kwh,atol=1e-12) and len(weeks)==r.week_blocks
        if len(weeks)>=8:np.testing.assert_allclose(np.quantile(bs,[.025,.975]),[r.ci95_low_kwh,r.ci95_high_kwh],atol=1e-10,rtol=0)
    models_checked=[]
    if a.models:
        import joblib
        from threadpoolctl import threadpool_limits
        for g in ('R','S','H','O'):
            item=joblib.load(out/('model_'+g+'.joblib'));assert item['model'].n_iter_==500 and item['tree']==config['tree'] and set(item['train_ids'])==train
            assert sha(out/('model_'+g+'.joblib'))==lock['model_hashes'][g]
            indices=[c.index[c.file==name][0] for name in outputs['validation'].file.iloc[[0,50,-1]]];vocab=item['vehicle_vocabulary'];xv=np.eye(len(vocab))[[vocab.index(v) if v in vocab else vocab.index('__unknown__') for v in c.vehicle.iloc[indices]]]
            X=np.column_stack([xv,f.loc[indices,item['numeric_features']].to_numpy(float)]);baseline=c.vehicle.iloc[indices].map(rates).to_numpy()*c.D.iloc[indices].to_numpy()
            with threadpool_limits(limits=4):replayed=baseline+item['model'].predict(X)
            pred=outputs['validation'].set_index('file').loc[c.file.iloc[indices],'pred_'+g+'_rem'].to_numpy();np.testing.assert_allclose(replayed,pred,atol=1e-10,rtol=0);models_checked.append(g)
    artifact=dict(passed=True,audited_at=pd.Timestamp.now(tz='UTC').isoformat(),environment=platform.platform(),hash_checked_artifacts=verified,metric_checks=metric_checks,
        state_and_history_causality=True,train_only_preprocessing=True,validation_decision_before_future=True,frozen_future_choice_unchanged=True,executed_code_hashes_match=True,
        calibration_not_scored=True,same1272_unique_ids=True,paired_week_bootstrap_replayed=True,models_replayed=models_checked,
        dynamic_log_hash_exception=dynamic,research_report_revisions=revised,exception_reason='completion print follows initial completion-manifest hash; log is mutable. Narrative report revision keeps original hashed copy. Model/data/decision hashes all match.',
        E5='shelved; not launched by EV-B01',output_artifact_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file() and not p.name.startswith('verification_')})
    write_json(out/('verification_cloud.json' if a.models else 'verification_local.json'),artifact)
    print(json.dumps({k:v for k,v in artifact.items() if k!='output_artifact_hashes'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
