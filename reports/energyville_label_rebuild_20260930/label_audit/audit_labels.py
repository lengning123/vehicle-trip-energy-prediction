"""Audit all released EnergyVille sessions. Never executes author code.
Multiplexed counters use causal as-of observations in an explicit common
window. Measurement/boundary quality is separate from predictive information.
"""
from pathlib import Path
import io,json,zipfile,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
SRC=ROOT/'reports/public_dataset_audit_20260930/sources'
COUNTERS={'discharge':'TotalDischargeKWh3D2','charge':'TotalChargeKWh3D2','drive':'BMS_kwhDriveDischargeTotal','regen':'BMS_kwhRegenChargeTotal'}
LABELS={'battery_discharge':{'discharge':1},'battery_charge':{'charge':1},'bms_drive_gross':{'drive':1},'bms_regen':{'regen':1},'battery_net':{'discharge':1,'charge':-1},'bms_drive_net':{'drive':1,'regen':-1},'non_drive_residual':{'discharge':1,'drive':-1},'external_charge_balance':{'charge':1,'regen':-1}}
def num(df,c):return pd.to_numeric(df[c],errors='coerce').to_numpy() if c in df else np.full(len(df),np.nan)
def ff(x):return float(x) if np.isfinite(x) else None
def integral(p,s,mask=None):
    dt=np.diff(s);ok=np.isfinite(p[1:])&np.isfinite(p[:-1])&(dt>0)&(dt<=2)
    if mask is not None:ok&=mask[1:]&mask[:-1]
    return (float(np.where(ok,(p[1:]+p[:-1])/2*dt/3600,0).sum()) if ok.any() else np.nan),float(dt[ok].sum())
def audit(name,raw):
    df=pd.read_csv(io.BytesIO(raw));t=pd.to_datetime(df.Timestamp,errors='coerce',utc=True); original_rows=len(df); missing_ts=int(t.isna().sum())
    df=df.loc[t.notna()].reset_index(drop=True);t=pd.to_datetime(df.Timestamp,errors='coerce',utc=True)
    s=(t-t.iloc[0]).dt.total_seconds().to_numpy();dt=np.diff(s);file=name.split('/')[-1];mode=name.split('/')[-2]
    r={'file':file,'zip_path':name,'vehicle':file.split('_')[0],'mode':mode,'rows':original_rows,'start_utc':str(t.iloc[0]),'end_utc':str(t.iloc[-1]),'duration_s':float(s[-1]),'missing_timestamps':missing_ts,'nonpositive_dt':int((dt<=0).sum()),'max_gap_s':float(dt.max()) if len(dt) else 0,'gap_gt_2s_count':int((dt>2).sum()),'gap_gt_60s_count':int((dt>60).sum()),'gap_gt_2s_time_s':float(dt[dt>2].sum()),'sha256':hashlib.sha256(raw).hexdigest()}
    v=num(df,'BattVoltage132');a=num(df,'RawBattCurrent132');p=v*a/1000
    r['voltage_implausible_rows']=int(((v<=0)|(v>655.35)).sum());r['current_implausible_rows']=int(((a<-1138.35)|(a>2138.4)).sum())
    for kind,pi in [('net',p),('discharge',np.maximum(p,0)),('charge',np.maximum(-p,0))]:
        en,obs=integral(pi,s);r['iv_observed_'+kind+'_kwh']=ff(en);r['iv_observed_'+kind+'_coverage']=obs/s[-1] if s[-1]>0 else 0
    cp=num(df,'ChargeLinePower264');ci=num(df,'ChargeLineCurrent264');fc=num(df,'FC_dcCurrent');active=(cp>.2)&(ci>.5)
    r['ac_charger_positive_rows']=int(active.sum());r['dc_charger_positive_rows']=int((fc>.5).sum());r['battery_charge_rows']=int((p<-.2).sum())
    r['ac_charge_observed_kwh'],r['ac_charge_observed_s']=integral(np.where(active,cp,0),s)
    r['source_file_distinct']=int(df.source_file.nunique()) if 'source_file' in df else 0
    arr={}
    for k,c in COUNTERS.items():
        x=num(df,c);valid=np.isfinite(x)&(x>=0)&(x<4294967.295-.0001);ii=np.flatnonzero(valid);arr[k]=(x,ii)
        r.update({k+'_valid_rows':len(ii),k+'_sentinel_or_negative_rows':int((np.isfinite(x)&~valid).sum()),k+'_coverage_ratio':len(ii)/len(df),k+'_negative_steps':int((np.diff(x[ii]) < -1e-7).sum()) if len(ii)>1 else 0,k+'_large_negative_steps':int((np.diff(x[ii])<-.002).sum()) if len(ii)>1 else 0,k+'_first_offset_s':ff(s[ii[0]]) if len(ii) else None,k+'_last_offset_s':ff(s[-1]-s[ii[-1]]) if len(ii) else None,k+'_natural_delta_kwh':float(x[ii[-1]]-x[ii[0]]) if len(ii)>1 else None,k+'_max_observation_gap_s':float(np.diff(s[ii]).max()) if len(ii)>1 else None,k+'_published_not_1wh_grid_fraction':float((np.abs(x[ii]*1000-np.rint(x[ii]*1000))>1e-5).mean()) if len(ii) else None})
    for label,co in LABELS.items():
        r[label+'_tier']='missing';r[label+'_reason']='missing_counter_or_fewer_than_two_observations';r[label+'_kwh']=None
        if not all(len(arr[k][1])>=2 for k in co):continue
        begin=max(s[arr[k][1][0]] for k in co);end=min(s[arr[k][1][-1]] for k in co)
        if begin>=end:r[label+'_reason']='no_common_time_window';continue
        en=0;samples=[];times={};neg=0;large=0
        for k in co:
            x,ii=arr[k];ia=ii[np.searchsorted(s[ii],begin,side='right')-1];ib=ii[np.searchsorted(s[ii],end,side='right')-1]
            r[label+'_component_'+k+'_delta_kwh']=float(x[ib]-x[ia]);en+=co[k]*(x[ib]-x[ia]);samples.append((s[ia],s[ib]));times[k]={'start_utc':str(t.iloc[ia]),'end_utc':str(t.iloc[ib]),'start_age_s':float(begin-s[ia]),'end_age_s':float(end-s[ib])}
            dx=np.diff(x[ii[(ii>=ia)&(ii<=ib)]]);neg+=int((dx < -1e-7).sum());large+=int((dx<-.002).sum())
        st=max(aa for aa,bb in samples);et=s[-1]-min(bb for aa,bb in samples);qb=.002*sum(abs(c) for c in co.values())
        mask=(s>=begin)&(s<=end);iv,iv_s=integral(p,s,mask);pos,_=integral(np.maximum(p,0),s,mask);cha,_=integral(np.maximum(-p,0),s,mask)
        edge,_=integral(np.abs(p),s,(s<=max(aa for aa,bb in samples))|(s>=min(bb for aa,bb in samples)))
        r.update({label+'_kwh':float(en),label+'_window_start_utc':str(t.iloc[np.searchsorted(s,begin)]),label+'_window_end_utc':str(t.iloc[np.searchsorted(s,end)]),label+'_counter_sample_times':json.dumps(times,separators=(',',':')),label+'_start_offset_s':float(st),label+'_end_offset_s':float(et),label+'_max_endpoint_asynchrony_s':max(val for tt in times.values() for nn,val in tt.items() if nn.endswith('_age_s')),label+'_negative_steps':neg,label+'_large_negative_steps':large,label+'_quantization_conservative_kwh':qb,label+'_quantization_relative_bound':qb/abs(en) if abs(en)>0 else None,label+'_quantization_sufficient_5pct':bool(abs(en)>=qb/.05),label+'_visible_edge_abs_iv_kwh':ff(edge),label+'_iv_window_coverage':iv_s/(end-begin),label+'_iv_window_net_kwh':ff(iv),label+'_iv_window_discharge_kwh':ff(pos),label+'_iv_window_charge_kwh':ff(cha)})
        if label=='battery_net':r[label+'_iv_minus_counter_kwh']=ff(iv-en)
        elif label=='battery_discharge':r[label+'_iv_minus_counter_kwh']=ff(pos-en)
        elif label=='battery_charge':r[label+'_iv_minus_counter_kwh']=ff(cha-en)
        reasons=[]
        if r['missing_timestamps'] or r['nonpositive_dt']:reasons.append('invalid_time_grid')
        if large:reasons.append('counter_reset_or_negative_step_gt_2wh')
        elif neg:reasons.append('small_counter_reversal')
        if r['gap_gt_60s_count']:reasons.append('long_gap_breaks_trip_attribution')
        if max(st,et)>5:reasons.append('endpoint_unobserved_gt_5s')
        if label in ['battery_discharge','battery_charge','bms_drive_gross','bms_regen'] and en<-.002-1e-8:reasons.append('negative_nonnegative_component')
        if label=='non_drive_residual' and en<-.004-1e-8:reasons.append('negative_non_drive_residual')
        critical={'invalid_time_grid','counter_reset_or_negative_step_gt_2wh','negative_nonnegative_component','negative_non_drive_residual'}
        r[label+'_tier']='reject' if any(a in critical for a in reasons) else ('candidate' if reasons else 'strong_internal_candidate');r[label+'_reason']=';'.join(reasons) or 'counter_boundary_screen_pass'
    eb=r.get('external_charge_balance_kwh');r['external_charge_counter_evidence']=bool(eb is not None and eb>.02);r['external_charge_direct_evidence']=bool(r['ac_charge_observed_kwh']>=.02 or r['dc_charger_positive_rows']>=5);r['external_charge_any_evidence']=r['external_charge_counter_evidence'] or r['external_charge_direct_evidence']
    if mode=='driving sessions' and r['external_charge_any_evidence']:
        for label in LABELS:
            if r.get(label+'_tier')=='strong_internal_candidate':r[label+'_tier']='candidate';r[label+'_reason']='external_charge_or_stale_charger_evidence_requires_mode_reconstruction'
    d=r.get('battery_discharge_kwh');dr=r.get('bms_drive_gross_kwh');reg=r.get('bms_regen_kwh');ch=r.get('battery_charge_kwh')
    r['drive_greater_than_total_discharge_natural']=bool(d is not None and dr is not None and dr>d+.01);r['regen_greater_than_total_charge_natural']=bool(ch is not None and reg is not None and reg>ch+.01)
    return r
def qs(s):
    s=pd.to_numeric(s,errors='coerce').dropna();return {str(k):float(v) for k,v in s.quantile([0,.05,.5,.95,1]).items()} if len(s) else {}
def main():
    rows=[];errors=[];archive=SRC/'energyville_V2.zip'
    with zipfile.ZipFile(archive) as z:
        names=[n for n in z.namelist() if n.endswith('.csv') and 'sessions/' in n]
        for i,n in enumerate(names):
            try:rows.append(audit(n,z.read(n)))
            except Exception as e:errors.append({'file':n,'error':repr(e)})
            if (i+1)%300==0:print('Audited',i+1,flush=True)
    all_df=pd.DataFrame(rows);all_df.to_csv(OUT/'all_mode_label_audit.csv',index=False,encoding='utf-8-sig');dr=all_df[all_df['mode']=='driving sessions'].copy();dr.to_csv(OUT/'source_label_audit.csv',index=False,encoding='utf-8-sig')
    summary={'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'session_count':len(all_df),'driving_count':len(dr),'errors':errors,'thresholds':{'endpoint_limit_s':5,'long_gap_s':60,'external_balance_limit_kwh':.02},'modes':{},'driving':{}}
    for mode,g in all_df.groupby('mode'):
        summary['modes'][mode]={'files':len(g),'drive_positive_gt_0_01_kwh':int(g.bms_drive_gross_kwh.gt(.01).sum()),'regen_positive_gt_0_01_kwh':int(g.bms_regen_kwh.gt(.01).sum()),'drive_kwh_quantiles':qs(g.bms_drive_gross_kwh),'drive_sum_kwh':float(g.bms_drive_gross_kwh.sum()),'discharge_sum_kwh':float(g.battery_discharge_kwh.sum()),'external_charge_evidence':int(g.external_charge_any_evidence.sum())}
    summary['driving'].update({'external_charge_counter_evidence':int(dr.external_charge_counter_evidence.sum()),'external_charge_direct_evidence':int(dr.external_charge_direct_evidence.sum()),'long_gap_files':int(dr.gap_gt_60s_count.gt(0).sum())})
    for k in COUNTERS:
        summary['driving'][k]={'valid_any_files':int(dr[k+'_valid_rows'].gt(0).sum()),'negative_step_files':int(dr[k+'_negative_steps'].gt(0).sum()),'large_negative_step_files':int(dr[k+'_large_negative_steps'].gt(0).sum()),'start_offset_quantiles':qs(dr[k+'_first_offset_s']),'end_offset_quantiles':qs(dr[k+'_last_offset_s']),'published_non_grid_median_fraction':float(dr[k+'_published_not_1wh_grid_fraction'].median())}
    for label in LABELS:
        aa=dr[label+'_tier'].eq('strong_internal_candidate');x=dr[label+'_kwh']
        summary['driving'][label]={'tier_counts':dr[label+'_tier'].value_counts().to_dict(),'reasons':dr[label+'_reason'].value_counts().to_dict(),'kwh_quantiles_all':qs(x),'zero_or_near_zero_le_0_01_count':int(x.abs().le(.01).sum()),'negative_lt_minus_0_004_count':int(x.lt(-.004).sum()),'A_quant_precision_ge_95pct':int((aa&dr[label+'_quantization_sufficient_5pct'].eq(True)).sum()),'A_visible_edge_kwh_quantiles':qs(dr.loc[aa,label+'_visible_edge_abs_iv_kwh']),'A_endpoint_asynchrony_quantiles':qs(dr.loc[aa,label+'_max_endpoint_asynchrony_s'])}
        diff=label+'_iv_minus_counter_kwh'
        if diff in dr:
            ok=aa&dr[label+'_iv_window_coverage'].ge(.99);summary['driving'][label]['A_iv_coverage_ge_99pct_count']=int(ok.sum());summary['driving'][label]['A_iv_minus_counter_quantiles']=qs(dr.loc[ok,diff])
    base=pd.read_csv(ROOT/'reports/public_dataset_audit_20260930/energyville_candidate_trip_table.csv')[['file','gps_path_km']];joined=dr.merge(base,on='file',validate='one_to_one');summary['driving']['per_vehicle']={}
    for vehicle,g in joined.groupby('vehicle'):
        summary['driving']['per_vehicle'][vehicle]={'files':len(g),'ge_10km_files':int(g.gps_path_km.ge(10).sum()),'labels':{lab:{'A':int(g[lab+'_tier'].eq('strong_internal_candidate').sum()),'A_ge_10km':int((g[lab+'_tier'].eq('strong_internal_candidate')&g.gps_path_km.ge(10)).sum())} for lab in LABELS}}
    ratio=dr.bms_drive_gross_kwh/dr.battery_discharge_kwh.replace(0,np.nan);res=dr.non_drive_residual_kwh
    summary['driving']['drive_over_discharge_quantiles']=qs(ratio);summary['driving']['non_drive_residual_abs_le_0_004_count']=int(res.abs().le(.004).sum());summary['driving']['non_drive_residual_abs_le_0_01_count']=int(res.abs().le(.01).sum());summary['driving']['non_drive_residual_positive_gt_0_02_count']=int(res.gt(.02).sum())
    (OUT/'label_audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps({'driving':len(dr),'all_sessions':len(all_df),'errors':len(errors)}),flush=True)
if __name__=='__main__':main()



