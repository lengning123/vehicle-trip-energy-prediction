"""EV-B01 fixed four-fit cloud experiment. Validation decision precedes future evaluation."""
from __future__ import annotations
import argparse,json,os,platform,subprocess,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits,threadpool_info
from energyville_evb01_data import TREE_CONFIG,write_json,sha,flag

GROUPS=('B0','R','S','H','O')


def scores(y,p,vehicle):
    y=np.asarray(y,float);p=np.asarray(p,float);v=np.asarray(vehicle);n=len(y)
    if not n:return dict(n=0)
    err=p-y;under=np.maximum(-err,0);ok=y>.1
    per={str(k):float(abs(err[v==k]).mean()) for k in np.unique(v)}
    return dict(n=n,mae_kwh=float(abs(err).mean()),rmse_kwh=float(np.sqrt(np.mean(err**2))),bias_kwh=float(err.mean()),
        wape_pct=float(100*abs(err).sum()/abs(y).sum()) if abs(y).sum()>0 else None,vehicle_macro_mae_kwh=float(np.mean(list(per.values()))),vehicle_mae_kwh=per,
        underestimate_mean_kwh=float(under.mean()),underestimate_p95_kwh=float(np.quantile(under,.95)),
        mape_threshold_kwh=.1,mape_pct=float(100*np.mean(abs(err[ok])/y[ok])) if ok.any() else None,mape_n=int(ok.sum()),mape_coverage=float(ok.mean()))


def basic_rates(c):
    train=c[c.split=='train'];overall=float(train.y_rem.sum()/train.D.sum())
    rates={str(v):float(g.y_rem.sum()/g.D.sum()) for v,g in train.groupby('vehicle')}
    return dict(by_vehicle=rates,overall=overall),c.vehicle.map(rates).fillna(overall).to_numpy()*c.D.to_numpy()


def simple_choice(values,left,right):
    a=values[left]['mae_kwh'];b=values[right]['mae_kwh'];minimum=min(a,b)
    return left if a<=b or a-b<=.01*max(minimum,1e-12) else right


def ratio_check(base,candidate,limit,mode='maximum'):
    if base<1e-10:return dict(passed=None,reason='baseline near zero; unstable ratio',absolute_difference=candidate-base)
    ratio=candidate/base
    return dict(passed=bool(ratio<=limit),ratio=float(ratio),allowed_max_ratio=limit,absolute_difference=float(candidate-base))


def validation_decision(c,preds):
    values={g:scores(c.y_rem,preds[g],c.vehicle) for g in GROUPS}
    reference=simple_choice(values,'B0','R');candidate=simple_choice(values,'S','H');chosen=simple_choice(values,reference,candidate)
    b=values[reference];a=values[candidate];gate={
        'mae_at_least10pct':ratio_check(b['mae_kwh'],a['mae_kwh'],.90),
        'rmse_not_worse5pct':ratio_check(b['rmse_kwh'],a['rmse_kwh'],1.05),
        'underestimate_not_worse5pct':ratio_check(b['underestimate_mean_kwh'],a['underestimate_mean_kwh'],1.05)}
    for v in sorted(c.vehicle.unique()):
        ix=(c.vehicle==v).to_numpy();gate['vehicle_'+str(v)]=ratio_check(float(abs(preds[reference][ix]-c.y_rem.to_numpy()[ix]).mean()),float(abs(preds[candidate][ix]-c.y_rem.to_numpy()[ix]).mean()),1.1)
    ix=(c.D>=10).to_numpy()
    gate['ge10km']=ratio_check(float(abs(preds[reference][ix]-c.y_rem.to_numpy()[ix]).mean()),float(abs(preds[candidate][ix]-c.y_rem.to_numpy()[ix]).mean()),1.1) if ix.any() else dict(passed=None,reason='no validation >=10km observations')
    return dict(reference=reference,legal_candidate=candidate,chosen_legal=chosen,rule='pairwise validation MAE; <=1% difference selects simpler. Oracle excluded.',
        investment_gate_passed=all(x['passed'] is True for x in gate.values()),gates=gate,validation_scores=values,decided_at=pd.Timestamp.now(tz='UTC').isoformat())


def score_partition(c,p):
    out={}
    for g in GROUPS:
        pr=p['pred_'+g+'_rem'].to_numpy();full=p['pred_'+g+'_full'].to_numpy()
        out[g]=dict(remaining=scores(c.y_rem,pr,c.vehicle),full=scores(c.y_full,full,c.vehicle),slices={},sensitivity={})
        assert np.allclose(full-c.y_full,pr-c.y_rem,atol=1e-8,rtol=0)
        definitions=dict(vehicle=c.vehicle,distance=c.distance_bin,od_seen_train=c.od_seen_train,has_history=c.has_history,ambient=c.ambient_bin,
            uncertain_charger=flag(c.uncertain_charger_signal_or_mode_overlap),suspected_charger=flag(c.suspected_charger_signal_or_mode_overlap),
            confirmed_external_charging=flag(c.confirmed_external_charging_from_counter_or_joint_direct),log_gap_gt2s=c.gap_gt_2s_count>0,
            internally_consistent=flag(c.main_view_internal_consistency_checked),quantization_le5pct=flag(c.main_view_5pct_counter_quantization_sufficient))
        for name,values in definitions.items():
            layers={}
            for layer in sorted(values.unique(),key=str):
                ix=(values==layer).to_numpy();result=scores(c.y_rem.to_numpy()[ix],pr[ix],c.vehicle.to_numpy()[ix]);result['descriptive_only']=int(ix.sum())<20;layers[str(layer)]=result
            out[g]['slices'][name]=layers
        for name,mask in dict(no_uncertain_charger=~flag(c.uncertain_charger_signal_or_mode_overlap),internal_consistency=flag(c.main_view_internal_consistency_checked),
            no_gt2s_gap=c.gap_gt_2s_count==0,quantization_le5pct=flag(c.main_view_5pct_counter_quantization_sufficient)).items():
            ix=mask.to_numpy();out[g]['sensitivity'][name]=scores(c.y_rem.to_numpy()[ix],pr[ix],c.vehicle.to_numpy()[ix])
    return out


def paired_bootstrap(c,p,decision):
    dt=pd.to_datetime(c.t_cut,utc=True).dt.isocalendar();week=(dt.year.astype(str)+'-W'+dt.week.astype(str)).to_numpy();unique=np.unique(week);nweeks=len(unique)
    rng=np.random.default_rng(42);draws=rng.integers(0,nweeks,size=(1000,nweeks));counts=np.array([(week==w).sum() for w in unique])
    pairs=[('R','S'),('S','H'),('H','O'),(decision['reference'],decision['legal_candidate']),(decision['chosen_legal'],'O')]+[('B0',g) for g in ('R','S','H','O')]
    records=[];seen=set()
    y=c.y_rem.to_numpy()
    for left,right in pairs:
        if (left,right) in seen:continue
        seen.add((left,right));l=abs(p['pred_'+left+'_rem'].to_numpy()-y);r=abs(p['pred_'+right+'_rem'].to_numpy()-y);diff=l-r
        sums=np.array([diff[week==w].sum() for w in unique]);boot=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
        records.append(dict(left=left,right=right,n=len(c),week_blocks=nweeks,mae_improvement_right_kwh=float(diff.mean()),
            ci95_low_kwh=float(np.quantile(boot,.025)) if nweeks>=8 else None,ci95_high_kwh=float(np.quantile(boot,.975)) if nweeks>=8 else None,
            bootstrap_repetitions=1000,seed=42,unit='whole ISO calendar week; both vehicles sampled together',interpretation='positive means right has lower MAE; <8 weeks no robust interval claim'))
    return pd.DataFrame(records)


def write_predictions(c,pr,path):
    out=c.copy()
    for g in GROUPS:
        out['pred_'+g+'_rem']=pr[g];out['pred_'+g+'_full']=pr[g]+c.e_prefix.to_numpy()
    out.to_csv(path,index=False);return out


def table(metrics,partition):
    rows=['|组别|MAE kWh|RMSE kWh|偏差 kWh|MAPE % (覆盖)|平均低估 kWh|','|---|---:|---:|---:|---:|---:|']
    for g in GROUPS:
        s=metrics[partition][g]['remaining'];m='—' if s['mape_pct'] is None else f"{s['mape_pct']:.2f} ({s['mape_n']}/{s['n']})"
        rows.append(f"|{g}{'（未来Oracle）' if g=='O' else ''}|{s['mae_kwh']:.4f}|{s['rmse_kwh']:.4f}|{s['bias_kwh']:+.4f}|{m}|{s['underestimate_mean_kwh']:.4f}|")
    return '\n'.join(rows)


def report(out,metrics,decision,config,runtime,pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    future=pd.read_csv(out/'predictions_future.csv');groups=list(GROUPS);fv=metrics['future'];chosen=decision['chosen_legal'];reference=decision['reference'];candidate=decision['legal_candidate']
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    ax[0].bar(groups,[fv[g]['remaining']['mae_kwh'] for g in groups],color=['#8796a5','#548cc2','#42a28c','#7269b5','#d5a042']);ax[0].set(ylabel='MAE (kWh)',title='Fixed future holdout; O is oracle')
    vehicles=sorted(future.vehicle.unique());x=np.arange(5)
    for i,v in enumerate(vehicles):ax[1].bar(x+(i-.5)*.35,[fv[g]['remaining']['vehicle_mae_kwh'][v] for g in groups],width=.35,label=v)
    ax[1].set(xticks=x,xticklabels=groups,ylabel='MAE (kWh)',title='Known-vehicle future performance');ax[1].legend();fig.tight_layout();fig.savefig(out/'future_mae.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for g in (chosen,'O'):
        err=(future['pred_'+g+'_rem']-future.y_rem).to_numpy();ax[0].plot(np.sort(abs(err)),np.arange(1,len(err)+1)/len(err),label=g)
        ax[1].scatter(future.D,err,s=15,alpha=.65,label=g)
    ax[0].set(xlabel='Absolute error (kWh)',ylabel='ECDF',title='Legal selection vs oracle');ax[0].legend();ax[1].axhline(0,color='black',lw=.8);ax[1].set(xlabel='Remaining distance (km)',ylabel='Signed error (kWh)',title='Distance/error structure');ax[1].legend();fig.tight_layout();fig.savefig(out/'future_error_structure.png',dpi=160);plt.close(fig)
    ci=pairs[(pairs.left==chosen)&(pairs.right=='O')].iloc[0];lo,hi=ci.ci95_low_kwh,ci.ci95_high_kwh
    oraclegap=fv[chosen]['remaining']['mae_kwh']-fv['O']['remaining']['mae_kwh'];basegap=fv[reference]['remaining']['mae_kwh']-fv[candidate]['remaining']['mae_kwh']
    summary=f"完成云端EV-B01：1272条、两辆BEV，固定四次拟合。验证选择{chosen}，合法信息投入门槛{'通过' if decision['investment_gate_passed'] else '未通过'}。未来MAE为{fv[chosen]['remaining']['mae_kwh']:.3f}kWh，Oracle为{fv['O']['remaining']['mae_kwh']:.3f}kWh。仅两车，标签为计数器+IV近似；未来结果不改变验证选模。E5搁置，未启动。"
    assert len(summary)<=300
    text='# EV-B01：合法状态与历史的信息价值\n\n'+summary+'\n\n## 做了什么\n\n'
    text+='本轮独立于eVED/V4：首个可信包温桶结束时预测剩余电池净量；同一1272源ID、同一划分，全部小/负标签保留，不重筛质量。四组HistGradientBoosting预测B0残差，固定L1、500轮、无随机内部验证、CPU4线程；四次拟合，不调参、不重训train+val。空间几何50m重采样，未接道路/高程地图。完整冻结协议与输入哈希见同目录。\n\n'
    text+='### 实际划分与覆盖\n\n|集合|记录|BEV1/BEV2|截点范围|≥10km|历史可用|超训练合法数值范围|\n|---|---:|---|---|---:|---:|---:|\n'
    for name,s in config['split_summary'].items():text+=f"|{name}|{s['n']}|{s['vehicles']}|{s['cut_min']} 至 {s['cut_max']}|{s['ge10km']}|{s['history_fraction']:.1%}|{s['outside_training_range']}|\n"
    text+='\n日期整块，边界及隔离ID见preflight_checks/split_manifest；校准集未拟合、选模或打分。历史只使用已完成、合格、同车训练ID，至多最近10条；同OD亦限制在这10条中。停车/充电历史温度是另一权限，取source开始前已结束桶，不读取它们的终局能量。预处理词表只fit训练，保留unknown、native NaN。\n\n'
    text+='### 标签/边界预检\n\n所有1272条净量守恒误差≤1e-8kWh；逐条复算IV前缀与剩余预算距离；源CSV/ZIP一一关联。状态桶结束≤截点，历史能量end+1秒≤截点，来源及质量池全部落盘。实际未来、当前标签/IV、文件终点和质量覆盖未进入R/S/H。空间几何函数不接收时间，重复静止采样不改变空间特征。\n\n'
    check=json.loads((out/'preflight_checks.json').read_text());text+=f"特别注意：仅{check['prefix_available_at_cut_n']}/1272条原前缀在截点已完全可得。其余R05梯形前缀可使用截点桶，该桶到cut+1秒才结束。按合同未改标签，前缀从未进入模型；回加后的整程输出只是事后账目，不声称严格cut时可部署。预算距离保留跨截点GPS连线的完整长度，{check['crossing_cut_routes']}条含跨界连线、最大{check['route_crossing_cut_max_km']:.4f}km。无可靠ON/OFF或导航快照。\n\n"
    text+='### 验证选模（先落盘，再查看未来错误）\n\n'+table(metrics,'validation')+'\n\n'
    text+=f"基础参考={reference}，合法候选={candidate}，业务合法选择={chosen}；≤1%差选简单者，O不参与。工程投入门槛{'通过' if decision['investment_gate_passed'] else '未通过'}：\n\n|条件|通过|候选/参考或说明|\n|---|---|---|\n"
    for key,v in decision['gates'].items():text+=f"|{key}|{v['passed']}|{v.get('ratio',v.get('reason'))}|\n"
    text+='\n详见validation_decision.json及validation_lock.json。未来留出不重选、不移动日期、不反向调参。\n\n## 学到了什么\n\n### 固定未来留出\n\n'+table(metrics,'future')+'\n\n'
    text+='剩余与回加整程的MAE/RMSE/偏差相同；完整量MAPE另见metrics.json，阈值真值>0.1kWh，覆盖同时保存。WAPE、车辆宏平均MAE、低估95分位、所有预定义切片及敏感性指标均见JSON。未来各组全覆盖，质量子集不重训。\n\n'
    text+='![未来总体/两车误差](future_mae.png)\n\n![误差与距离](future_error_structure.png)\n\n### 配对变化与不确定性\n\n|左→右|右侧MAE改善kWh|95%周块区间|周块/记录|\n|---|---:|---|---|\n'
    for r in pairs.itertuples():text+=f"|{r.left}→{r.right}|{r.mae_improvement_right_kwh:+.4f}|{r.ci95_low_kwh:+.4f} 至 {r.ci95_high_kwh:+.4f}|{r.week_blocks}/{r.n}|\n" if pd.notna(r.ci95_low_kwh) else f"|{r.left}→{r.right}|{r.mae_improvement_right_kwh:+.4f}|不足8周，仅描述|{r.week_blocks}/{r.n}|\n"
    text+='\n正值表示右侧MAE更低。1000次同日历ISO周整块重采样，两车同周一起，seed42；只刻画两车的时间样本不确定性，不是车队泛化置信区间。区间跨0不等于证明信息无用。\n\n'
    text+='### 分层、训练与反证\n\n|组|训练MAE|验证MAE|未来MAE|≥10km未来MAE|新OD未来MAE|\n|---|---:|---:|---:|---:|---:|\n'
    for g in GROUPS:
        ds=fv[g]['slices']['distance'];ss=fv[g]['slices']['od_seen_train'];long=fv[g]['slices']['vehicle']
        mask=future.D>=10;lm=float(abs(future.loc[mask,'pred_'+g+'_rem']-future.loc[mask,'y_rem']).mean()) if mask.any() else float('nan')
        unseen=ss.get('False',{}).get('mae_kwh',float('nan'))
        text+=f"|{g}|{metrics['train'][g]['remaining']['mae_kwh']:.4f}|{metrics['validation'][g]['remaining']['mae_kwh']:.4f}|{fv[g]['remaining']['mae_kwh']:.4f}|{lm:.4f}|{unseen:.4f}|\n"
    text+='\n|预定义未来质量子集|n|合法选择MAE|Oracle MAE|\n|---|---:|---:|---:|\n'
    for layer,s in fv[chosen]['sensitivity'].items():text+=f"|{layer}|{s['n']}|{s.get('mae_kwh',float('nan')):.4f}|{fv['O']['sensitivity'][layer].get('mae_kwh',float('nan')):.4f}|\n"
    text+='\n切片阈值在拟合前固定，n<20只描述；子集有场景混杂，改善不能归因纯量测质量。新OD是粗空间key而非道路link留出，超训练范围指任一合法数值超训练min/max，不是严格OOD。R05温度表已去重24冲突桶，丢失副本不能恢复；当前source优先值与可辨识冲突独立记录。原始路线位置异常可影响空间转向表示，预算D做了原定义的合理性过滤，几何特征未用未来速度做清洗；因此R弱不能否定完整地图信息。\n\n'
    text+='### H1–H5更新\n\n|假设|本轮更新|支持、反证与缺证|\n|---|---|---|\n'
    text+=f"|H1 合法信息/运行缺口|维持存在，按块评价主导性|未来{reference}→{candidate}改善{basegap:+.4f}kWh；{chosen}→O改善{oraclegap:+.4f}kWh。S混合SOC/外温/包温，不能全归温度；O只诊断有限未来统计，不是理论下界。|\n"
    text+='|H2 标签/边界|内部可用性维持，部署边界风险上调|守恒、前缀与距离逐条复算；prefix可用晚1秒新证据。质量切片不是独立真值，无法识别绝对校准误差。|\n'
    text+='|H3 异构供能/分量语义|BEV混杂下调，组件辨识不变|本轮两BEV，无燃油混合；counter+IV净量仍非认证traction/HVAC。与旧PHEV分数不能比较提升。|\n'
    text+='|H4 任务时点/评价错位|任务目标更对齐，严格截点限制保留|本轮整程净电预算、低估风险与绝对误差统一；剩余预测合法，回加前缀精确实时性未认证，非严格出发前任务。|\n'
    text+='|H5 车辆/路线/时间覆盖|维持高|同日期未来留出、新OD/距离/温度切片；只有两已知车，历史冻结存在陈旧性，不能扩展到陌生车。|\n\n'
    text+='## 因此做什么\n\n'
    if decision['investment_gate_passed'] and basegap>0:
        text+='下一主线：优先完善同条件路线预算与独立区间校准，检查车辆/长程低估，再分边界验证第二公开源；不先扩网络。合法信息通过验证护栏且未来点改善保持，但仍须查看配对区间与小切片，不能直接宣称普适。\n'
    elif oraclegap>0 and np.isfinite(lo) and lo>0:
        text+='下一主线：合法状态/历史未通过完整投入门槛，而Oracle配对区间支持未来运行有价值；优先研究仅训练历史支持的驾驶/交通情景及独立预算区间，不承诺恢复唯一未来轨迹。此轮不能把未观察未来直接塞入部署模型。\n'
    else:
        text+='下一主线：先核查空间路线表示、未来分布偏移、计量/边界及固定树对信息的利用；没有稳定Oracle优势不能推出所有未来信息无价值。优先检查训练/验证差与长程残差，不开启网络或超参网格。\n'
    text+='\n先修正前缀的实时可用边界，再做下一轮实时预算协议：从已结束桶计算causal prefix，重定义相应remaining并记录差异，或明确推迟输出时点；不能悄悄替换本轮冻结标签。本轮未来错误已经查看，若据此改方案，该集合随后称开发基准，不能再称全新最终测试。E5保持搁置，本轮不自动启动任何后续实验。\n\n'
    text+=f"计算：{runtime['platform']}，sklearn {runtime['sklearn']}，numpy {runtime['numpy']}，pandas {runtime['pandas']}，{runtime['threads']}线程。各组耗时见runtime.json，四fit合计{sum(x['fit_s'] for x in runtime['groups'].values()):.2f}秒。参数与缺失/预处理合同见JSON；模型含训练词表、B0率、特征顺序与配置，支持复算。\n\n"
    text+='实现依据：[HistGradientBoostingRegressor官方API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)。运行云端实际版本，不升级已有环境。\n'
    (out/'EVB01_实验报告.md').write_text(text,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();out=a.out
    assert platform.system()=='Linux' and str(Path.cwd()).startswith('/root/autodl-tmp'),'formal fits must run on the authorized cloud'
    if (out/'validation_lock.json').exists() or (out/'execution_manifest.json').exists():raise FileExistsError('formal experiment already started; no silent refit')
    config=json.loads((out/'run_config.json').read_text());checks=json.loads((out/'preflight_checks.json').read_text());assert checks['passed']
    assert config['tree']==TREE_CONFIG
    for name,digest in config['prepared_file_hashes'].items():assert sha(out/name)==digest,(name,'changed after data freeze')
    c=pd.read_csv(out/'cohort_manifest.csv');f=pd.read_csv(out/'features.csv');columns=json.loads((out/'feature_columns.json').read_text());assert list(c.file)==list(f.file) and len(c)==1272
    contract=json.loads((out/'feature_contract.json').read_text())
    for g in ('R','S','H'):assert all(not contract[x]['oracle'] for x in columns['groups'][g])
    tr=np.flatnonzero((c.split=='train').to_numpy());val=np.flatnonzero((c.split=='validation').to_numpy());future=np.flatnonzero((c.split=='future').to_numpy())
    assert len(tr) and len(val) and len(future);vocabulary=sorted(c.vehicle.iloc[tr].unique().tolist());vocab=vocabulary+['__unknown__'];lookup={v:i for i,v in enumerate(vocab)}
    onehot=np.eye(len(vocab))[np.array([lookup.get(v,lookup['__unknown__']) for v in c.vehicle])];rates,b0=basic_rates(c)
    prep=dict(fitted_ids=c.file.iloc[tr].tolist(),vehicle_vocabulary=vocab,unknown_rule='unseen vehicles explicit last channel',numeric='native NaN; no fill/scale',baseline_rates_kwh_per_km=rates,feature_order=columns['groups'])
    write_json(out/'preprocessing_manifest.json',prep)
    hashes={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.suffix in ('.json','.csv','.md')}
    runtime=dict(platform=platform.platform(),python=platform.python_version(),sklearn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__,threads=4,
        cpu_count=os.cpu_count(),started_at=pd.Timestamp.now(tz='UTC').isoformat(),groups={},runner_sha256=sha(__file__))
    write_json(out/'execution_manifest.json',dict(runtime=runtime,frozen_files=hashes,tree=TREE_CONFIG,planned_fits=4,planned_groups=GROUPS,future_evaluation_after_validation_lock=True))
    preds_train={'B0':b0[tr]};preds_val={'B0':b0[val]};models={};started=time.perf_counter()
    with threadpool_limits(limits=4):
        runtime['threadpools']=threadpool_info()
        for g in ('R','S','H','O'):
            num=[x for x in columns['groups'][g] if x!='vehicle'];X=np.column_stack([onehot,f[num].to_numpy(float)])
            assert not np.isinf(X).any();model=HistGradientBoostingRegressor(**TREE_CONFIG);t=time.perf_counter();model.fit(X[tr],c.y_rem.to_numpy()[tr]-b0[tr]);fit_s=time.perf_counter()-t
            assert model.n_iter_==500
            t=time.perf_counter();preds_train[g]=b0[tr]+model.predict(X[tr]);preds_val[g]=b0[val]+model.predict(X[val]);predict_s=time.perf_counter()-t
            models[g]=(model,num);joblib.dump(dict(model=model,numeric_features=num,vehicle_vocabulary=vocab,baseline=rates,tree=TREE_CONFIG,train_ids=prep['fitted_ids'],runner_sha256=sha(__file__)),out/('model_'+g+'.joblib'))
            runtime['groups'][g]=dict(fit_s=fit_s,train_validation_predict_s=predict_s,features=X.shape[1],n_iter=model.n_iter_)
            print(json.dumps(dict(group=g,fit_s=fit_s,val_mae=scores(c.y_rem.iloc[val],preds_val[g],c.vehicle.iloc[val])['mae_kwh'])),flush=True)
        pv=write_predictions(c.iloc[val].reset_index(drop=True),preds_val,out/'predictions_validation.csv');pt=write_predictions(c.iloc[tr].reset_index(drop=True),preds_train,out/'predictions_train.csv')
        decision=validation_decision(c.iloc[val].reset_index(drop=True),preds_val);write_json(out/'validation_decision.json',decision)
        lock=dict(locked_at=pd.Timestamp.now(tz='UTC').isoformat(),decision_sha256=sha(out/'validation_decision.json'),model_hashes={g:sha(out/('model_'+g+'.joblib')) for g in models},frozen_files=hashes,runner_sha256=sha(__file__))
        write_json(out/'validation_lock.json',lock)
        # No future prediction/error is accessed above this committed lock.
        write_json(out/'future_evaluation_started.json',dict(started_at=pd.Timestamp.now(tz='UTC').isoformat(),lock_sha256=sha(out/'validation_lock.json'),protocol='one fixed future evaluation; no tuning/reselection'))
        preds_future={'B0':b0[future]}
        for g,(model,num) in models.items():
            X=np.column_stack([onehot[future],f.loc[future,num].to_numpy(float)]);t=time.perf_counter();preds_future[g]=b0[future]+model.predict(X);runtime['groups'][g]['future_inference_s']=time.perf_counter()-t
        pf=write_predictions(c.iloc[future].reset_index(drop=True),preds_future,out/'predictions_future.csv')
    metrics={name:score_partition(pred,pred) for name,pred in [('train',pt),('validation',pv),('future',pf)]}
    pairs=paired_bootstrap(pf,pf,decision);pairs.to_csv(out/'paired_comparisons.csv',index=False);write_json(out/'metrics.json',metrics)
    runtime['total_s_before_report']=time.perf_counter()-started;runtime['finished_at']=pd.Timestamp.now(tz='UTC').isoformat()
    import resource
    runtime['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    try:runtime['git_head']=subprocess.check_output(['git','rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:runtime['git_head']='cloud copy; no git metadata'
    write_json(out/'runtime.json',runtime);report(out,metrics,decision,config,runtime,pairs)
    write_json(out/'completion.json',dict(state='complete',completed_at=pd.Timestamp.now(tz='UTC').isoformat(),formal_fits=4,E5='shelved_not_started',
        selected_legal=decision['chosen_legal'],files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='completion.json'}))
    print(json.dumps(dict(state='complete',selected_legal=decision['chosen_legal'],future_mae={g:metrics['future'][g]['remaining']['mae_kwh'] for g in GROUPS}),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
