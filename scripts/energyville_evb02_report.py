"""Deterministic EV-B02 report from already completed fits; never train here."""
from pathlib import Path
import json
import numpy as np,pandas as pd

GROUPS=('S-fixed','H-frozen','H-online','O-combined','O-time','O-motion','O-quality')


def report(out):
    out=Path(out);load=lambda n:json.loads((out/n).read_text())
    m=load('metrics.json');a=load('stageA_gate.json');b=load('stageB_gates.json');rt=load('runtime.json');pre=load('preflight_checks.json')
    cov=pd.read_csv(out/'route_history_coverage.csv');geo=pd.read_csv(out/'geometry_diagnostics.csv');rew=pd.read_csv(out/'reweighted_description.csv');conc=pd.read_csv(out/'gain_concentration.csv');pairs=pd.read_csv(out/'paired_comparisons.csv');cells=pd.read_csv(out/'conditional_support.csv')
    dev=pd.read_csv(out/'predictions_development.csv');v=m['validation'];d=m['development'];valonline=v['H-online']['full']['mae_kwh'];devonline=d['H-online']['full']['mae_kwh']
    summary=f"EV-B02已在原云端完成：同1272 ID，不换标签/划分，125条校准终局量未用。相同H权重仅刷新已完成历史，验证/开发MAE={valonline:.4f}/{devonline:.4f}kWh，完整投入门槛{'通过' if a['passed'] else '未通过'}。另固定三次拟合拆耗时/运动/采样Oracle。旧future只作开发参考，E5及新队列未启动。"
    assert len(summary)<=300
    text='# EV-B02：历史新鲜度与运行信息分解实验\n\n'+summary+'\n\n## 1. 目的与预设假设\n\n'
    text+='检验两点：（1）原H无稳定增益，是历史失鲜还是相同映射无法利用近期信息？（2）实际耗时、运动或采样覆盖，哪块更有诊断价值？假设、工程门槛、最大三fit在运行前冻结于protocol_frozen.md/run_config.json，不据结果补拟合。\n\n'
    text+='## 2. 操作、任务与因果合同\n\n'
    text+='本轮不是eVED/V3新增分数，而是EnergyVille两已知BEV记录净电预算。保留EV-B01全部1272 ID：train766（182/584）、validation188（94/94）、calibration125（66/59）、开发参考193（27/166）。开发参考是已查看的旧future，不是新独立测试；无晚于旧future的新记录可用于确认性检验。只两车，不作陌生车辆或车队结论。\n\n'
    text+='输入截止仍为t_cut，预算输出明确推迟至t_cut+1秒；原最新前缀桶此时已结束，1272条全部通过可得时点检查。原y_rem/e_prefix/y_full/D不改，y_full=e_prefix+y_rem。suffix仍从旧cut起算，不称新预算时点后的remaining；预测suffix后加已可得前缀成为整记录预算，剩余与整记录误差相同，但WAPE/MAPE分母不同。该任务不是严格出发前预测，没有独立ON/OFF或导航快照；路线是记录后重建的空间代理。\n\n'
    text+='### 阶段A：零fit、同一保存H权重\n\n'
    text+='S-fixed/H-frozen/O-combined重放原模型及原输入。H-online只刷新25个非thermal历史能量/使用特征；6个past_thermal、当前S、几何、缺失策略、最新10条限制、10条内同OD、词表、B0率及权重保持不变。训练仍仅已完成train；validation允许已完成train及更早validation；开发允许已完成train、validation及更早开发记录。125 calibration从不查询、释放或读其终局能量/源统计。在线开发场景允许先前完成记录的真值成为后续部署观测，与静态不看任何留出标签的实验不同，须分别解释。\n\n'
    text+=f"事件reader只在完整source结束+1秒≤查询cut时释放同车质量合格记录，共{pre['event_reader_calls']}次。目标描述用完整source净量/距离/均温/耗时，不混suffix距离。manifest逐项保存ID、许可、桶结束时点、特征来源和快照hash。训练历史公式复算最大差{pre['train_formula_replay_max_abs']:.2e}，然后使用原保存训练CSV精确值作浮点规范化，避免极小舍入跨树阈值。对一个未结束长记录终局量增加1000kWh，所有更早查询特征和同H预测完全不变；见causal_audit.json。S/H/O控制重放≤1e-8kWh，见control_replay_checks.json。\n\n"
    text+='### 阶段B：恰好三次固定fit\n\n|组|原冻结H追加量|意义与限制|\n|---|---|---|\n|O-time|真实suffix耗时|时间相关运行/负载代理，不认证HVAC|\n|O-motion|真实距离加权v²、观测停车比例及二者missing|运动结构；missing仍可能是量测代理|\n|O-quality|运动覆盖、motion_missing|采样/表示质量代理，不视为交通|\n\n'
    text+='三组训练都用原训练期冻结历史，非H-online；目标B0残差、native NaN、训练词表和ID不变。固定HistGradientBoosting absolute_error、lr0.05、500轮、31叶、minleaf20、l2=1、无内部早停、seed42、CPU4线程。不使用cal，不网格、不多seed、不train+val重训；原H/S/O没有重训。全部实际未来量仅Oracle，不进入H-online，不使用未来IV/终点SOC/能量作为特征，不是可部署模型或理论下界。\n\n'
    text+='## 3. 结果：完整记录预算\n\n'
    for partition,title in [('validation','验证188条'),('development','开发参考193条')]:
        text+=f'### {title}\n\n|组|MAE kWh|RMSE kWh|bias kWh|WAPE %|车辆宏MAE|平均低估|低估p95|MAPE % / 有效数|\n|---|---:|---:|---:|---:|---:|---:|---:|---|\n'
        for g in GROUPS:
            s=m[partition][g]['full'];mp='—' if s['mape_pct'] is None else f"{s['mape_pct']:.2f} / {s['mape_n']}/{s['n']}"
            text+=f"|{g}|{s['mae_kwh']:.4f}|{s['rmse_kwh']:.4f}|{s['bias_kwh']:+.4f}|{s['wape_pct']:.2f}|{s['vehicle_macro_mae_kwh']:.4f}|{s['underestimate_mean_kwh']:.4f}|{s['underestimate_p95_kwh']:.4f}|{mp}|\n"
        text+='\nMAPE仅真值>0.1kWh；其余小/负量仍完整进入MAE/RMSE等主指标。正bias为高估，不裁预测。逐行/分层指标见predictions与metrics.json。\n\n'
    text+='![验证/开发MAE](budget_mae.png)\n\n### 每车与距离反证\n\n|分区/车型/距离|n|S MAE|冻结H MAE|在线H MAE|在线H bias|\n|---|---:|---:|---:|---:|---:|\n'
    for partition in ('validation','development'):
        for kind in ('vehicle','distance'):
            for layer,s in m[partition]['H-online']['slices'][kind].items():
                text+=f"|{partition}/{layer}|{s['n']}|{m[partition]['S-fixed']['slices'][kind][layer]['mae_kwh']:.4f}|{m[partition]['H-frozen']['slices'][kind][layer]['mae_kwh']:.4f}|{s['mae_kwh']:.4f}|{s['bias_kwh']:+.4f}|\n"
    text+='\n原划分BEV1开发只有27条、≥50km仅3条，低样本层只能描述。质量/粗OD/年龄/车/外温完整结果见metrics.json。不能用总体改善遮盖某车退化。\n\n### 阶段A门槛\n\n|分区|条件|通过|比值或原因|\n|---|---|---|---|\n'
    for p,r in a['partitions'].items():
        for k,s in r.items():text+=f"|{p}|{k}|{s['passed']}|{s.get('ratio',s.get('reason'))}|\n"
    text+=f"\n完整门槛{'通过' if a['passed'] else '未通过'}。门槛是研究投入依据，不是业务验收；护栏失败与MAE改善并列报告，负结果不否定其他能利用新鲜输入的映射。\n\n"
    text+='### Oracle单块效果、集中度与护栏\n\n|组/分区|相对H MAE改善|前5%正收益占比|低覆盖正收益占比|高覆盖子集改善kWh|\n|---|---:|---:|---:|---:|\n'
    for r in conc[conc.group.isin(('O-time','O-motion','O-quality'))].itertuples():text+=f"|{r.group}/{r.partition}|{r.relative_improvement_vs_H:.1%}|{r.top5pct_positive_gain_share:.1%}|{r.low_coverage_positive_gain_share:.1%}|{r.good_coverage_mae_improvement_vs_H:+.4f}|\n"
    for g,r in b.items():text+=f"\n- {g}完整双分区10%+非集中门槛：{r['passed']}。\n"
    text+='\n集中度以最大ceil(5%n)条正收益占所有正收益衡量，50%屏幕阈值在fit前冻结；覆盖≥0.95。正收益统计不掩盖负收益（gain_concentration.csv保存二者）。这是描述性筛查，不是因果证明；三个单块与组合O之间仍可能有交互和树表示差异。\n\n'
    text+='### 开发参考周块配对\n\n|左→右|MAE改善kWh|95%周块区间|周数|\n|---|---:|---|---:|\n'
    for r in pairs.itertuples():
        interval=f'[{r.ci95_low_kwh:+.4f}, {r.ci95_high_kwh:+.4f}]' if pd.notna(r.ci95_low_kwh) else '不足8周，仅点估计'
        text+=f"|{r.left}→{r.right}|{r.mae_improvement_right_kwh:+.4f}|{interval}|{r.week_blocks}|\n"
    text+='\n1000次seed42，同ISO周两车一起重采样已固定在线策略误差；不打乱历史重建特征。不是车队泛化区间，也不是严格在线依赖下的覆盖保证，旧开发集非确认性证据。\n\n'
    text+='## 4. 描述诊断与未区分原因\n\n### 新鲜历史与重复路线支持\n\n'
    text+=f"开发193条全有可用历史，中位{pre['development_online_history_median_age_hours']:.3f}小时，190条≤7天，对照冻结约104.9天。及时性提高不等于预算精度必然提高。当前H依然只在最近10条内搜索粗OD。\n\n|分区|n|90天同粗OD≥1/3/5|至少一条空间支持|\n|---|---:|---|---:|\n"
    for p,s in [('train','train'),('validation','validation'),('development','future')]:
        q=cov[cov.split==s];text+=f"|{p}|{len(q)}|{int(q.at_least1.sum())}/{int(q.at_least3.sum())}/{int(q.at_least5.sum())}|{int((q.spatially_supported_n>0).sum())}|\n"
    text+='\n诊断检索全部已完成、合格、同车且90天内同OD历史，不限10条；没有把这个新摘要送入H。用未按未来速度清洗的整条空间路径，50m重采样，双向最近距离均值≤50m/p95≤100m、长度比0.8–1.25作描述性支持。粗OD相同仍不保证同道路；几何异常会影响空间检索。路径误差/年龄/距离逐对见route_history_pairs.csv，不是道路link导航真值。\n\n'
    text+='### 联合条件、共同支持与组成敏感性\n\n'
    text+='车辆×四距离×四外温共32格，三个分区每格列n、suffix/full-source汇总Wh/km、MAE、bias和组合Oracle相对H收益，见conditional_support.csv。n<10只描述，无支持格不能做条件内比较。sum(y_rem)/sum(D)是同suffix量；full-source列另配完整源距离，不混边界。\n\n|共同格每区最少n|格数|开发覆盖/未覆盖|S验证共同格MAE→开发组成重加权|开发共同格S MAE|\n|---|---:|---|---|---:|\n'
    for r in rew[rew.group=='S-fixed'].itertuples():text+=f"|{r.minimum_cell_n_all3}|{r.common_cell_n}|{r.development_covered_n}/193，未覆盖{r.development_uncovered_fraction:.1%}|{r.validation_original_common_mae:.4f}→{r.validation_reweighted_mae:.4f}|{r.development_common_mae:.4f}|\n"
    text+='\n重加权只是验证误差对开发格频率的描述，未改任何预测，不等于季节/温度/坡度因果；不能跨缺格归因。年龄与原同OD分层在metrics.json中全部按同一组样本配对。\n\n### 几何/采样质量\n\n'
    flagged=geo[geo.geometry_gt1pct]
    text+=f"validation/开发geometry比冻结预算D大>1%分别{int((flagged.split=='validation').sum())}/{int((flagged.split=='future').sum())}条；未删源、未重算模型。列车、距离、温度、内部一致性、gap、误差与纯空间≥250/1000m弦长计数，见geometry_diagnostics.csv。弦长只提供位置跳跃线索，没有地图证据时不认证GPS错误。禁止用实际未来速度清洗合法模型特征。预算D自身沿用原标签版本的合理性过滤，路线几何未过滤，两者定义差异可成为表示误差。\n\n"
    text+='![开发误差与距离](history_error_structure.png)\n\n## 5. 解释：H1–H5的支持、反证与缺证\n\n|假设|本轮证据更新|不能推出什么/还缺什么|\n|---|---|---|\n'
    text+=f"|H1 合法信息/历史失鲜|在线H验证/开发相对冻结H改善{(1-valonline/v['H-frozen']['full']['mae_kwh']):.1%}/{(1-devonline/d['H-frozen']['full']['mae_kwh']):.1%}；及时观测已实现，门槛{a['passed']}。单块Oracle显示映射对不同代理的敏感性|固定H反事实不等于最优在线模型；未来统计不是合法输入，也不等同交通/HVAC真实因果|\n"
    text+='|H2 标签/边界|保持原净量与全部ID；1秒延迟使前缀可得；几何质量逐源审计|未提供独立电表真值，不能用某Oracle精度或几何异常证明label错误|\n'
    text+='|H3 分量/供能语义|仍两BEV净电预算，无油电混合；未重新构造组件label|耗时收益不认证HVAC，不能把电池净量拆成准确纯牵引与附件|\n'
    text+='|H4 时点/评价|输出合同更清楚；MAE、偏差、低估及MAPE覆盖共同评估|延迟预算不等于严格出发前；小/负量不删，风险指标改善不能由MAE单独保证|\n'
    text+='|H5 场景覆盖|历史新鲜度、同路线支持、32条件格、开发再加权与长程偏差可直接定位|仅两已知车；共同条件缺格和未测道路/坡度/行为仍混杂；无新独立future|\n\n'
    text+='## 6. 决定：继续什么，暂不做什么\n\n'
    if a['passed']:
        direction='在线H通过预设投入门槛，下一主线可以转向滚动已知车预算，另冻结协议比较全历史/相似路线检索与条件更新；随后才用独立校准做风险区间。'
    else:
        direction='在线H未通过完整投入门槛，不将新鲜度提高宣布为部署成功；继续保留原验证选择S作为合法固定参考。先查看收益车/条件与负收益、共同支持及表示，而非直接扩大网络或推广单一全局历史平均。'
    passing=[g for g,r in b.items() if r['passed']]
    if passing:direction+=' 双分区且非集中门槛通过的Oracle块：'+', '.join(passing)+'；据此优先建设对应合法信息链路，仍须另立协议。'
    else:direction+=' 三个Oracle单块没有通过全部门槛；报告点改善与集中度，不用某一侧最优值选择网络，也不追加第四fit。'
    direction+=' time若只在部分条件有效，合法候选是过去同路线耗时分布而非未来真值；motion若有益且路线支持薄，优先公开道路/限速/坡度；quality若有益，先审采样/几何，不全讲成交通。若点方案仍不稳，再冻结独立条件区间协议，不先画宽区间代替信息辨别。'
    text+=direction+'\n\n本轮不启动E5、区间训练、地图拉取、新公开源混库或新网络网格。125 calibration仅元信息引用，未消费终局量。后续确认性评测需另验公开独立源的计量边界/路线与已知车历史，不能将原cal改名新future。\n\n'
    text+='### 计算成本与复核入口\n\n|组|新增fit秒|特征数（含车型one-hot）|开发推理秒|\n|---|---:|---:|---:|\n'
    for g,r in rt['groups'].items():text+=f"|{g}|{r.get('fit_s',0):.3f}|{r.get('feature_count','旧保存模型')}|{r.get('development_predict_s',0):.4f}|\n"
    text+=f"\n云端{rt['platform']}；Python {rt['python']}、sklearn {rt['sklearn']}、pandas {rt['pandas']}、numpy {rt['numpy']}，4线程，新增fit总数3，总fit{sum(r.get('fit_s',0) for r in rt['groups'].values()):.2f}秒。输入、代码、三保存模型与训练ID、runtime和completion哈希均留存，原EV-B01文件哈希再次核对未变。没有本地训练。\n"
    (out/'EVB02_实验报告.md').write_text(text,encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x=np.arange(len(GROUPS));fig,ax=plt.subplots(figsize=(11,4))
    for i,p in enumerate(('validation','development')):ax.bar(x+(i-.5)*.35,[m[p][g]['full']['mae_kwh'] for g in GROUPS],width=.35,label=p)
    ax.set(xticks=x,xticklabels=GROUPS,ylabel='Full-record budget MAE (kWh)',title='EV-B02: old future is development, Oracle is diagnostic');ax.legend();fig.tight_layout();fig.savefig(out/'budget_mae.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for g in ('S-fixed','H-frozen','H-online'):
        err=(dev['pred_'+g+'_rem']-dev.y_rem).to_numpy();ax[0].scatter(dev.D,err,s=13,alpha=.5,label=g);ax[1].plot(np.sort(abs(err)),np.arange(1,len(err)+1)/len(err),label=g)
    ax[0].axhline(0,color='black',lw=.8);ax[0].set(xlabel='Suffix distance (km)',ylabel='Signed error (kWh)',title='Paired online/frozen errors');ax[1].set(xlabel='Absolute error (kWh)',ylabel='ECDF',title='Same 193 development records')
    for aa in ax:aa.legend()
    fig.tight_layout();fig.savefig(out/'history_error_structure.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);report(p.parse_args().out)
