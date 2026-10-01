"""Post-hoc read-only diagnostics. No fitting; calibration labels not scored."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'reports/energyville_study/EVB01'
OUT = Path(__file__).resolve().parent
def load(name): return pd.read_csv(SRC / name)
def flag(s): return s.fillna(False).astype(str).str.lower().isin(['true', '1', '1.0'])
c = load('cohort_manifest.csv')
f = load('features.csv')
a = load('route_audit.csv')
assert c.file.is_unique and f.file.is_unique and len(c) == len(f) == 1272
p = pd.concat([load('predictions_' + s + '.csv') for s in ('train','validation','future')], ignore_index=True)
extra = [x for x in f.columns if x not in p.columns]
p = p.merge(f[['file'] + extra], on='file', validate='one_to_one').merge(a, on='file', validate='one_to_one', suffixes=('', '_audit')).copy()
p['S_error'] = p.pred_S_rem - p.y_rem
p['abs_S_error'] = p.S_error.abs()
p['month'] = pd.to_datetime(p.t_cut, utc=True).dt.strftime('%Y-%m')
def stats(g):
    d = g.D.sum()
    return dict(n=len(g), sum_distance_km=float(d), true_wh_per_km_aggregate=float(1000*g.y_rem.sum()/d) if d>0 else None,
                predicted_S_wh_per_km_aggregate=float(1000*g.pred_S_rem.sum()/d) if d>0 else None,
                S_mae_kwh=float(g.abs_S_error.mean()), S_bias_kwh=float(g.S_error.mean()), S_overprediction_count=int((g.S_error>0).sum()),
                S_underprediction_mean_kwh=float((-g.S_error).clip(lower=0).mean()),
                ambient_median_c=float(g.ambient_c.median()), oracle_mae_kwh=float((g.pred_O_rem-g.y_rem).abs().mean()))
tables = {}
for title, keys in [('split_vehicle_distance',['split','vehicle','distance_bin']),('split_vehicle',['split','vehicle']),
                    ('month_vehicle',['month','vehicle']),('distance_od',['split','distance_bin','od_seen_train'])]:
    rows=[]
    for k,g in p.groupby(keys,dropna=False):
        k=k if isinstance(k,tuple) else (k,)
        rows.append(dict(zip(keys,k),**stats(g)))
    tables[title]=rows
future=p[p.split=='future'].copy()
long=future[future.D>=50]
fco=c[['file','split','vehicle','cut_ns','end_ns','od_key']].merge(f[['file','history_recent_age_s','od_history_count']],on='file',validate='one_to_one')
def eligible(g):
    return flag(g.main_view_internal_consistency_checked)&~flag(g.uncertain_charger_signal_or_mode_overlap)&~flag(g.confirmed_external_charging_from_counter_or_joint_direct)&(g.overlap_conflicting_iv_rows==0)
pool=c[eligible(c)&c.split.isin(['train','validation','future'])]
fresh=[]
for _,r in fco[fco.split=='future'].iterrows():
    past=pool[(pool.vehicle==r.vehicle)&(pool.file!=r.file)&(pool.end_ns+1_000_000_000<=r.cut_ns)]
    age=(r.cut_ns-past.end_ns.max()-1_000_000_000)/1e9 if len(past) else np.nan
    same=past[(past.od_key==r.od_key)&((r.cut_ns-past.end_ns)/1e9<=90*86400)]
    fresh.append(dict(file=r.file,online_latest_age_s=age,online_same_od_90day_n=len(same)))
fresh=pd.DataFrame(fresh)
rfull=ROOT/'reports/energyville_label_rebuild_20260930/data_pipeline/recorded_session_master.csv'
master=pd.read_csv(rfull)
max_end=pd.to_datetime(master.end_utc,utc=True).max()
last_end=pd.to_datetime(future.end_utc,utc=True).max()
def corr(x,y):
    ok=x.notna()&y.notna()
    return float(x[ok].rank().corr(y[ok].rank())) if ok.sum()>=3 else None
summary=dict(
    status='post_hoc_description_no_fit_no_causal_attribution',
    source_hashes={n:hashlib.sha256((SRC/n).read_bytes()).hexdigest() for n in ['features.csv','cohort_manifest.csv','predictions_train.csv','predictions_validation.csv','predictions_future.csv','route_audit.csv']},
    future_S=stats(future), long_future_S=stats(long),
    future_bias_to_mae_ratio=float(abs(future.S_error.mean())/future.abs_S_error.mean()),
    long_abs_error_share=float(long.abs_S_error.sum()/future.abs_S_error.sum()),
    future_same_od_feature_positive_n=int((future.od_history_count>0).sum()),
    future_od_seen_train_n=int(flag(future.od_seen_train).sum()),
    frozen_history_median_age_days=float(future.history_recent_age_s.median()/86400),
    online_history_coverage_simulation=dict(policy='completed eligible train/validation/earlier-future only; exclude all calibration IDs; no fitting or energy-benefit assessment',
        available_n=int(fresh.online_latest_age_s.notna().sum()), median_age_hours=float(fresh.online_latest_age_s.median()/3600),
        within_7days_n=int((fresh.online_latest_age_s<=7*86400).sum()),same_od_90day_available_n=int((fresh.online_same_od_90day_n>0).sum())),
    geometry_future=dict(
        unfiltered_vs_budget_ratio_p50=float((future.unfiltered_geometry_km/future.D).median()),
        unfiltered_vs_budget_ratio_p95=float((future.unfiltered_geometry_km/future.D).quantile(.95)),
        over_1pct_ratio_count=int((future.unfiltered_geometry_km>1.01*future.D).sum()),
        max_excess_km=float((future.unfiltered_geometry_km-future.D).max())),
    observed_future_correlations=dict(note='descriptive, correlated exposures; not independent effects',
        duration_distance=corr(future.actual_remaining_duration_s,future.D),
        motion_v2_distance=corr(future.actual_gps_distance_weighted_v2,future.D),
        error_distance=corr(future.S_error,future.D),
        error_ambient=corr(future.S_error,future.ambient_c)),
    recorded_data_max_end=str(max_end), benchmark_future_max_end=str(last_end),
    source_records_after_benchmark_end=int((pd.to_datetime(master.end_utc,utc=True)>last_end).sum()),
    grouped_descriptions=tables)
def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,list): return [clean(v) for v in x]
    if isinstance(x,(np.integer,np.floating)): x=x.item()
    if isinstance(x,float) and not np.isfinite(x): return None
    if isinstance(x,np.bool_): return bool(x)
    return x
(OUT/'supplemental_diagnostics.json').write_text(json.dumps(clean(summary),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps(clean({k:v for k,v in summary.items() if k not in ('grouped_descriptions','source_hashes')}),ensure_ascii=False,indent=2))
print('SPLIT_VEHICLE_DISTANCE')
for row in tables['split_vehicle_distance']:
    if row['distance_bin']=='>=50': print(json.dumps(clean(row),ensure_ascii=False))
