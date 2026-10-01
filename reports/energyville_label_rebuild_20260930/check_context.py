from pathlib import Path
import shutil,re,json
r=Path('D:/python/homework/POC')
out=r/'reports/energyville_label_rebuild_20260930'
p=r/'电动汽车全程行程能耗预测_近三年文献调研报告.md'
shutil.copy2(p,out/'before_label_rebuild'/p.name)
t=p.read_text(encoding='utf-8')
start=t.index('## 本轮公开数据实物验收')
t=t[:start]+'''## R05 标签重构补充（当前方向D11）

全1396驾驶记录前6秒都有包温，首行温度共同436条件取消。实际交付1359同窗净电量路线候选、1320内部一致性核心、1284完整源计数器+IV补边、1272首温后剩余量标签。主线是已知BEV整程净电量预算与短观测后更新；T/R的多路复用和组件定义未认证，不能把drive/aux称纯牵引/HVAC。相关文献用途与量测区分见[R05管理验收](reports/energyville_label_rebuild_20260930/研究经理验收与方向.md)及[全文依据](reports/energyville_label_rebuild_20260930/literature/整程分量标签的价值与文献依据.md)。

'''+t[start:]
t=t.replace('## 本轮公开数据实物验收（2026-09-30）','## R04历史公开数据实物验收（方向已由D11修订）').replace('资源排序以[公开数据路线与UPECT验收]','R04时资源排序以[公开数据路线与UPECT验收]')
p.write_text(t,encoding='utf-8')
p=r/'decision_log.md';t=p.read_text(encoding='utf-8');t=t.replace('当前决策依据[问题定义]','当前执行D11；历史D00–D10按日期保留。决策依据[问题定义]',1);p.write_text(t,encoding='utf-8')
p=r/'experiments.md';t=p.read_text(encoding='utf-8');t=t.replace('两份Word副本SHA256相同。','R05全量重构2310公开会话并验收净电量/首温截断标签，未训练新模型。两份Word副本SHA256相同。',1);p.write_text(t,encoding='utf-8')
p=out/'update_research_context.py';t=p.read_text(encoding='utf-8-sig');t=t.replace("D=HERE/'data_pipeline'", "D=HERE/'data_pipeline'\nif '\\n## D11 当前决定：' in (ROOT/'decision_log.md').read_text(encoding='utf-8'):\n    raise SystemExit('One-time context update already applied; preserve existing backups.')",1);p.write_text(t,encoding='utf-8')
files=[r/f for f in ['problem.md','hypotheses.md','experiments.md','decision_log.md']]+[out/'研究经理验收与方向.md']
checks={};broken=[]
for p in files:
    t=p.read_text(encoding='utf-8');summary=t.split('## 管理者摘要',1)[1].split('\n## ',1)[0].strip()
    checks[p.name]={'summary_chars':len(summary),'under300':len(summary)<=300}
    for target in re.findall(r'(?<!!)\[[^\]]+\]\(([^)]+)\)',t):
        if target.startswith(('http:','https:','#','codex:')):continue
        target=target.split('#')[0].strip('<>')
        if not (p.parent/target).exists():broken.append({'file':p.name,'target':target})
assert not broken,broken
assert all(v['under300'] for v in checks.values())
assert (r/'decision_log.md').read_text(encoding='utf-8').count('\n## D11 ')==1
assert (r/'experiments.md').read_text(encoding='utf-8').count('\n## R05 ')==1
res={'manager_summaries':checks,'broken_links':broken,'unique_R05_D11':True,'data_acceptance':'manager_acceptance.json'}
(out/'文档核查.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(res,ensure_ascii=False,indent=2))
