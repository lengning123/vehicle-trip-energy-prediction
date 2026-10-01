from pathlib import Path
import json,zipfile,hashlib
import pandas as pd
b=Path('reports/energyville_label_rebuild_20260930/data_pipeline')
m=pd.read_csv(b/'recorded_session_master.csv')
a=pd.read_csv(b.parent/'label_audit/source_label_audit.csv')
j=m[['file','duration_s','main_net_battery_route_candidate']].merge(a[['file','duration_s']],on='file',suffixes=('_pipeline','_audit'),validate='one_to_one')
assert (j.duration_s_pipeline-j.duration_s_audit).abs().max()==0
assert len(m)==1396 and m.file.is_unique
assert (pd.to_datetime(m.net_window_past_temp_available_utc,utc=True)<=pd.to_datetime(m.battery_net_window_start_utc,utc=True)).fillna(True).all()
with zipfile.ZipFile(b/'driving_signal_views_1s.zip') as z:
 assert len(z.namelist())==1396
 bad=z.testzip();assert bad is None
with zipfile.ZipFile(b/'net_battery_route_views_1s.zip') as z:
 assert len(z.namelist())==1396
 bad=z.testzip();assert bad is None
manifest=json.loads((b/'artifact_manifest.json').read_text(encoding='utf-8'))
for r in manifest['files']:
 p=b/r['name'];assert p.stat().st_size==r['bytes'];assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],p.name
print('Accepted: 1396 unique source rows, same durations as independent audit, causal thermal snapshots, ZIP CRC and artifact hashes passed.')

