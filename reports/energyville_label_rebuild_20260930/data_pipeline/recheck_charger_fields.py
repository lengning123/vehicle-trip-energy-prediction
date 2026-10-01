from pathlib import Path
import pandas as pd,numpy as np,zipfile,io,json
b=Path('reports/energyville_label_rebuild_20260930/data_pipeline')
src=pd.read_csv(b/'driving_source_records.csv')
z=zipfile.ZipFile('reports/public_dataset_audit_20260930/sources/energyville_V2.zip')
rows=[]
for _,r in src[src.charge_active_own_rows.gt(0)].iterrows():
 d=pd.read_csv(io.BytesIO(z.read(r.zip_path)))
 def n(c):return pd.to_numeric(d[c],errors='coerce').to_numpy() if c in d else np.full(len(d),np.nan)
 cp,ci,cv,dc,p=n('ChargeLinePower264'),n('ChargeLineCurrent264'),n('ChargeLineVoltage264'),n('FC_dcCurrent'),n('BattVoltage132')*n('RawBattCurrent132')/1000
 active=((cp>.5)&(ci>.5))|((ci>.5)&(cv>50))|(dc>.5)
 rows.append(dict(file=r.file,cp_alone=int((cp>.5).sum()),ci_cp=int(((cp>.5)&(ci>.5)).sum()),ci_cv=int(((ci>.5)&(cv>50)).sum()),dc=int((dc>.5).sum()),robust_active=int(active.sum()),robust_negative_batt=int((active&(p<-.2)).sum()),powermin=float(np.nanmin(p))))
out=pd.DataFrame(rows);out.to_csv(b/'charger_field_recheck.csv',index=False)
print(out.to_string(index=False))
print('Robust records',int(out.robust_active.gt(0).sum()),'negative batt',int(out.robust_negative_batt.gt(0).sum()))

