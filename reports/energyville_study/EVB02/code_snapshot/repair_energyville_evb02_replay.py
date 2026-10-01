"""No-fit numerical repair: restore original shared inputs before online replay.

No model/label/source/policy changes. Preserve initial outputs for audit.
"""
import argparse,json,time,zipfile
from pathlib import Path
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from energyville_evb01_data import write_json,sha
from run_energyville_evb02 import CONTROLS,NEW,GROUPS,predict,evaluate,gate_a,bootstrap,diagnostic_tables,save_predictions
from energyville_evb02_report import report


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);a=p.parse_args();out=a.out
    assert not (out/'replay_numerical_revision.json').exists(),'already repaired; no repeated silent replay'
    complete=json.loads((out/'completion.json').read_text());assert complete['formal_fits']==3
    for n,h in complete['files'].items():assert sha(out/n)==h,n
    archive=out/'initial_replay_outputs.zip'
    names=[n for n in complete['files'] if n.startswith(('predictions_','stageA_')) or n in ('metrics.json','paired_comparisons.csv','conditional_support.csv','reweighted_description.csv','reweighted_description.json','geometry_diagnostics.csv','gain_concentration.csv','stageB_gates.json','EVB02_实验报告.md','budget_mae.png','history_error_structure.png')]+['completion.json']
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for n in names:z.write(out/n,arcname=n)
    # Use saved non-cal labels/metadata; never reconstruct or access calibration targets.
    c=pd.concat([pd.read_csv(out/('predictions_'+s+'.csv')) for s in ('train','validation','development')],ignore_index=True)
    c=c.drop(columns=[k for k in c.columns if k.startswith('pred_')])
    f=pd.read_csv(a.prior/'features.csv').set_index('file').loc[c.file].reset_index();c.index=f.index
    stored=pd.read_csv(out/'features_online.csv').set_index('file').loc[c.file].reset_index()
    updates=json.loads((out/'run_config.json').read_text())['refreshed_columns']
    online=f.copy();online.loc[:,updates]=stored[updates].to_numpy()
    tr=c.index[c.split=='train'];online.loc[tr,updates]=f.loc[tr,updates].to_numpy()
    pd.testing.assert_frame_equal(online.loc[tr],f.loc[tr])
    untouched=[k for k in f.columns if k not in updates];pd.testing.assert_frame_equal(online[untouched],f[untouched])
    modelhash={n:sha(out/('model_'+n+'.joblib')) for n in NEW};preds={};metrics={};stagea={};differences={};t=time.perf_counter()
    with threadpool_limits(limits=4):
        bundles={g:joblib.load(a.prior/('model_'+g+'.joblib')) for g in ('S','H','O')}
        bundles.update({g:joblib.load(out/('model_'+g+'.joblib')) for g in NEW})
        for s,split in [('train','train'),('validation','validation'),('development','future')]:
            idx=c.index[c.split==split];cc=c.loc[idx];preds[s]={}
            old=pd.read_csv(out/('predictions_'+s+'.csv')).set_index('file').loc[cc.file]
            for g in GROUPS:
                key={'S-fixed':'S','H-frozen':'H','H-online':'H','O-combined':'O'}.get(g,g)
                preds[s][g]=predict(online.loc[idx] if g=='H-online' else f.loc[idx],cc,bundles[key])
            differences[s]={g:float(np.max(abs(preds[s][g]-old['pred_'+g+'_rem'].to_numpy()))) for g in GROUPS}
            for g in GROUPS:
                if g!='H-online':assert differences[s][g]<=1e-8
            if s=='train':np.testing.assert_array_equal(preds[s]['H-online'],preds[s]['H-frozen'])
            metrics[s]=evaluate(cc,preds[s]);stagea[s]=evaluate(cc,preds[s],CONTROLS)
            save_predictions(cc,preds[s],out,s);save_predictions(cc,{g:preds[s][g] for g in CONTROLS},out,'stageA_'+s)
    write_json(out/'metrics.json',metrics);write_json(out/'stageA_metrics.json',stagea);write_json(out/'stageA_gate.json',gate_a(metrics))
    idx=c.index[c.split=='future'];bootstrap(c.loc[idx],preds['development']).to_csv(out/'paired_comparisons.csv',index=False)
    diagnostic_tables(c,preds,metrics,out)
    for n,h in modelhash.items():assert sha(out/('model_'+n+'.joblib'))==h
    revision=dict(reason='CSV float round-trip can move immutable shared/training inputs across saved tree thresholds; restore shared columns and train H exactly from original input before predicting',
        code_sha256=sha(__file__),initial_outputs_archive_sha256=sha(archive),no_fit=True,new_fits=0,total_original_fits=3,model_hashes_unchanged=modelhash,
        maximum_prediction_change_kwh=differences,training_H_online_now_bitwise_identical_to_H_frozen=True,shared_route_S_all6_thermal_restored_bitwise=True,
        label_policy_splits_unchanged=True,replay_s=time.perf_counter()-t,revised_at=pd.Timestamp.now(tz='UTC').isoformat())
    write_json(out/'replay_numerical_revision.json',revision);report(out)
    write_json(out/'completion.json',dict(state='complete',formal_fits=3,calibration_energy_consumed=0,E5='shelved_not_started',no_next_experiment_started=True,numerical_replay_revision=True,
        completed_at=pd.Timestamp.now(tz='UTC').isoformat(),files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ('completion.json','verification_cloud.json')}))
    print(json.dumps(revision))


if __name__=='__main__':main()
