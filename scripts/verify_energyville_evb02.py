"""Independent artifact checks. Re-predict saved models, never fit."""
import argparse,json,platform
from pathlib import Path
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from energyville_evb01_data import sha,write_json,TREE_CONFIG,flag


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--prior',type=Path,required=True);ap.add_argument('--delivery-only',action='store_true');a=ap.parse_args();out=a.out
    assert platform.system()=='Linux'
    load=lambda n:json.loads((out/n).read_text())
    if a.delivery_only:
        delivery=load('delivery_manifest.json')
        for name,digest in delivery['files'].items():assert sha(out/name)==digest,name
        result=dict(passed=True,checked_files=len(delivery['files']),delivery_manifest_sha256=sha(out/'delivery_manifest.json'),mode='final local/cloud delivery byte-identical; no fit or label queries')
        write_json(out/'delivery_verified.json',result);print(json.dumps(result));return
    complete=load('completion.json');assert complete['formal_fits']==3 and complete['calibration_energy_consumed']==0
    for name,d in complete['files'].items():assert sha(out/name)==d,(name,'completion hash mismatch')
    for item in load('input_manifest.json')['inputs']:assert sha(item['path'])==item['sha256'],item['name']
    for name,d in load('execution_manifest.json')['code_hashes'].items():assert sha(Path('scripts')/name)==d,name
    for name in ('preflight_checks.json','causal_audit.json','control_replay_checks.json'):assert load(name)['passed']
    c=pd.read_csv(out/'cohort_reference.csv').set_index('file');f=pd.read_csv(a.prior/'features.csv').set_index('file');online=pd.read_csv(out/'features_online.csv').set_index('file')
    assert len(c)==1272 and c.index.is_unique and c.split.value_counts().to_dict()=={'train':766,'future':193,'validation':188,'calibration':125}
    assert not any(k in c.columns for k in ('y_full','y_rem','e_prefix'))
    assert not set(c.index[c.split=='calibration'])&set(online.index) and len(online)==1147
    updates=load('run_config.json')['refreshed_columns'];unchanged=[k for k in f.columns if k not in updates]
    pd.testing.assert_frame_equal(f.loc[online.index,unchanged],online[unchanged],check_dtype=False)
    np.testing.assert_allclose(f.loc[c.index[c.split=='train'],updates],online.loc[c.index[c.split=='train'],updates],rtol=0,atol=1e-8,equal_nan=True)
    hp=pd.read_csv(out/'history_query_manifest.csv');perms={'train':{'train'},'validation':{'train','validation'},'future':{'train','validation','future'}}
    assert (hp.history_available_ns<=hp.cut_ns).all() and (hp.history_available_ns==hp.history_end_ns+1_000_000_000).all()
    assert (c.loc[hp.query_file,'vehicle'].to_numpy()==c.loc[hp.history_file,'vehicle'].to_numpy()).all()
    assert not (hp.query_file==hp.history_file).any()
    for query,source in zip(hp.query_split,hp.history_split):assert source in perms[query] and source!='calibration'
    assert hp.groupby('query_file').size().max()<=10
    eligible=flag(c.main_view_internal_consistency_checked)&~flag(c.uncertain_charger_signal_or_mode_overlap)&~flag(c.confirmed_external_charging_from_counter_or_joint_direct)&(c.overlap_conflicting_iv_rows==0)
    assert eligible.loc[hp.history_file].all()
    metrics=load('metrics.json');features=load('run_config.json')['oracle_additions'];checked={};replaydiff={};frozenhash={}
    with threadpool_limits(limits=4):
        for partition,split in [('train','train'),('validation','validation'),('development','future')]:
            p=pd.read_csv(out/('predictions_'+partition+'.csv')).set_index('file');expected=c.index[c.split==split];assert p.index.tolist()==expected.tolist()
            assert np.allclose(p.y_full,p.y_rem+p.e_prefix,atol=1e-8,rtol=0)
            replaydiff[partition]={}
            for group in ('S-fixed','H-frozen','H-online','O-combined','O-time','O-motion','O-quality'):
                new=group in features;old={'S-fixed':'S','H-frozen':'H','H-online':'H','O-combined':'O'}.get(group)
                bundle=joblib.load(out/('model_'+group+'.joblib') if new else a.prior/('model_'+old+'.joblib'))
                assert bundle['tree']==TREE_CONFIG and bundle['train_ids']==c.index[c.split=='train'].tolist()
                if new:
                    hb=joblib.load(a.prior/'model_H.joblib');assert bundle['numeric_features']==hb['numeric_features']+features[group]
                    assert bundle['baseline']==hb['baseline'] and bundle['vehicle_vocabulary']==hb['vehicle_vocabulary']
                    assert bundle['model'].n_iter_==500 and not bundle['model'].early_stopping
                source=f.loc[p.index].copy()
                if group=='H-online':
                    source.loc[:,updates]=online.loc[p.index,updates].to_numpy()
                    if partition=='train':source=f.loc[p.index].copy()
                vocab=bundle['vehicle_vocabulary'];mapping={k:i for i,k in enumerate(vocab)};oh=np.eye(len(vocab))[[mapping.get(k,mapping['__unknown__']) for k in source.vehicle]]
                x=np.c_[oh,source[bundle['numeric_features']].to_numpy(float)];base=p.vehicle.map(bundle['baseline']['by_vehicle']).fillna(bundle['baseline']['overall']).to_numpy()*p.D.to_numpy()
                pred=base+bundle['model'].predict(x);d=float(np.max(abs(pred-p['pred_'+group+'_rem'].to_numpy())));assert d<=1e-8,(partition,group,d);replaydiff[partition][group]=d
                er=p['pred_'+group+'_rem'].to_numpy()-p.y_rem.to_numpy();fuller=p['pred_'+group+'_full'].to_numpy()-p.y_full.to_numpy();np.testing.assert_allclose(er,fuller,atol=1e-8,rtol=0)
                s=metrics[partition][group]['full']
                for key,value in [('mae_kwh',abs(er).mean()),('rmse_kwh',np.sqrt(np.mean(er**2))),('bias_kwh',er.mean()),('underestimate_mean_kwh',np.maximum(-er,0).mean()),('underestimate_p95_kwh',np.quantile(np.maximum(-er,0),.95))]:assert abs(s[key]-value)<1e-8,(partition,group,key)
                assert s['n']==len(p) and s['mape_n']==int((p.y_full>.1).sum())
                ok=p.y_full.to_numpy()>.1
                if ok.any():assert abs(s['mape_pct']-100*np.mean(abs(fuller[ok])/p.y_full.to_numpy()[ok]))<1e-8
                assert abs(s['wape_pct']-100*abs(fuller).sum()/abs(p.y_full.to_numpy()).sum())<1e-8
            checked[partition]=len(p)
    cov=pd.read_csv(out/'route_history_coverage.csv');assert cov.file.is_unique and len(cov)==1147
    pairs=pd.read_csv(out/'route_history_pairs.csv');assert (pairs.history_available_ns<=pairs.cut_ns).all() and not (pairs.history_split=='calibration').any()
    cond=pd.read_csv(out/'conditional_support.csv');assert len(cond)==3*32*7 and cond.groupby(['partition','group']).n.sum().isin([766,188,193]).all()
    geo=pd.read_csv(out/'geometry_diagnostics.csv');assert len(geo)==381
    assert len(list(out.glob('model_*.joblib')))==3 and load('runtime.json')['formal_fit_calls']==3
    revision=load('replay_numerical_revision.json');assert revision['no_fit'] and revision['total_original_fits']==3
    assert sha('scripts/repair_energyville_evb02_replay.py')==revision['code_sha256']
    assert revision['training_H_online_now_bitwise_identical_to_H_frozen']
    assert load('fit_progress.json')['fit_calls_started']==3
    result=dict(passed=True,new_fit_count=3,original_EV_B01_inputs_unchanged=True,calibration_target_queries_or_scores=0,checked_predictions=checked,saved_model_replay_max_abs_kwh=replaydiff,
        event_permission_and_quality_valid=True,latest10_preserved=True,unchanged_route_S_and_all6_thermal=True,train_history_unchanged=True,
        energy_conservation_and_independent_metric_recalculation=True,conditional_cells_complete=True,route_pairs_causal=True,E5='shelved_not_started',no_refitting_in_verification=True)
    write_json(out/'verification_cloud.json',result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
