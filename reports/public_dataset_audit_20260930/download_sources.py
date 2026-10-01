from pathlib import Path
import urllib.request, json, hashlib, time
BASE=Path('reports/public_dataset_audit_20260930/sources')
BASE.mkdir(parents=True,exist_ok=True)
urls={
'upect_repo.json':'https://api.github.com/repos/RaganrokV/UPECT',
'upect_tree.json':'https://api.github.com/repos/RaganrokV/UPECT/git/trees/main?recursive=1',
'BHD.pkl':'https://raw.githubusercontent.com/RaganrokV/UPECT/main/data/BHD.pkl',
'ChargeCar.pkl':'https://raw.githubusercontent.com/RaganrokV/UPECT/main/data/ChargeCar.pkl',
'SpritMonitor.pkl':'https://raw.githubusercontent.com/RaganrokV/UPECT/main/data/SpritMonitor.pkl',
'VED.pkl':'https://raw.githubusercontent.com/RaganrokV/UPECT/main/data/VED.pkl',
'energyville_metadata.json':'https://rdr.kuleuven.be/api/datasets/:persistentId/?persistentId=doi:10.48804/8KPDTW',
'eve_metadata.json':'https://entrepot.recherche.data.gouv.fr/api/datasets/:persistentId/?persistentId=doi:10.57745/9IJF8G',
'tum_value_overview.csv':'https://raw.githubusercontent.com/tumftm/electric-vehicle-uds-dataset/main/data/value_overview.csv',
}
manifest=[]
for name,url in urls.items():
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'POC-public-data-audit/1.0'})
        with urllib.request.urlopen(req,timeout=60) as r: data=r.read()
        (BASE/name).write_bytes(data)
        info={'name':name,'url':url,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        print(name,len(data),flush=True)
    except Exception as e:
        info={'name':name,'url':url,'error':str(e)}
        print(name,str(e),flush=True)
    manifest.append(info)
(BASE/'download_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
