from pathlib import Path
import sys,io,zipfile,json
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
from audit_labels import COUNTERS,num
base=Path(__file__).resolve().parents[3]
rows=[]
with zipfile.ZipFile(base/'reports/public_dataset_audit_20260930/sources/energyville_V2.zip') as z:
 for n in z.namelist():
  if not n.endswith('.csv') or 'driving sessions/' not in n:continue
  df=pd.read_csv(io.BytesIO(z.read(n)));t=pd.to_datetime(df.Timestamp,utc=True,errors='coerce');s=(t-t.iloc[0]).dt.total_seconds().to_numpy();speed=num(df,'DI_uiSpeed')
  # Blocks with speed observed on >=90% of rows and always exactly zero.
  for start in np.arange(0,s[-1]-59,60):
   ii=np.flatnonzero((s>=start)&(s<start+60))
   if len(ii)<58 or np.isfinite(speed[ii]).mean()<.9 or np.nanmax(np.abs(speed[ii]))>.01:continue
   record={'file':n.split('/')[-1],'start_utc':str(t.iloc[ii[0]]),'end_utc':str(t.iloc[ii[-1]]),'rows':len(ii)}
   ok=True
   for k in ['discharge','drive','regen']:
    x=num(df,COUNTERS[k]);ix=ii[np.isfinite(x[ii])]
    if len(ix)<2:ok=False;break
    record[k+'_delta_kwh']=float(x[ix[-1]]-x[ix[0]])
   if ok:rows.append(record)
pd.DataFrame(rows).to_csv(Path(__file__).parent/'stationary_driving_blocks.csv',index=False,encoding='utf-8-sig')
g=pd.DataFrame(rows)
out={'60s_blocks':len(g),'source_files':g.file.nunique() if len(g) else 0,'drive_positive_ge_5wh_blocks':int(g.drive_delta_kwh.ge(.005-1e-8).sum()) if len(g) else 0,'examples':g[g.drive_delta_kwh.ge(.005-1e-8)].head(12).to_dict('records') if len(g) else []}
(Path(__file__).parent/'stationary_driving_blocks_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))

