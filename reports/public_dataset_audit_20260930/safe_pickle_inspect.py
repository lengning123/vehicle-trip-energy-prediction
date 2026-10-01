"""Statically interpret pickle opcodes into inert records. Never invoke serialized functions.
Decode only explicitly recognized numeric byte buffers and pandas tabular metadata.
"""
from pathlib import Path
from dataclasses import dataclass
import pickletools, json, numpy as np, pandas as pd
@dataclass
class Ref:
    module:str
    name:str
@dataclass
class Obj:
    fn:object
    args:object
    state:object=None
MARK=object()
def parse(data):
    st=[]; memo={}; refs=set()
    def items():
        i=len(st)-1
        while st[i] is not MARK:i-=1
        out=st[i+1:]; del st[i:]; return out
    for op,arg,pos in pickletools.genops(data):
        n=op.name
        if n in ('PROTO','FRAME'):continue
        if n=='MARK':st.append(MARK)
        elif n in ('SHORT_BINUNICODE','BINUNICODE','BINUNICODE8','UNICODE','BININT','BININT1','BININT2','LONG1','LONG4','BINFLOAT','FLOAT','INT','SHORT_BINBYTES','BINBYTES','BINBYTES8'):st.append(arg)
        elif n=='NONE':st.append(None)
        elif n in ('NEWTRUE','NEWFALSE'):st.append(n=='NEWTRUE')
        elif n=='EMPTY_TUPLE':st.append(())
        elif n=='EMPTY_LIST':st.append([])
        elif n=='EMPTY_DICT':st.append({})
        elif n=='STACK_GLOBAL':
            name=st.pop(); module=st.pop(); refs.add((module,name)); st.append(Ref(module,name))
        elif n=='GLOBAL':
            module,name=arg.split(' ',1); refs.add((module,name)); st.append(Ref(module,name))
        elif n in ('REDUCE','NEWOBJ'):
            args=st.pop(); fn=st.pop(); st.append(Obj(fn,args))
        elif n=='BUILD':
            state=st.pop(); assert isinstance(st[-1],Obj); st[-1].state=state
        elif n=='MEMOIZE':memo[len(memo)]=st[-1]
        elif n in ('BINPUT','LONG_BINPUT','PUT'):memo[int(arg)]=st[-1]
        elif n in ('BINGET','LONG_BINGET','GET'):st.append(memo[int(arg)])
        elif n=='TUPLE':st.append(tuple(items()))
        elif n in ('TUPLE1','TUPLE2','TUPLE3'):
            k=int(n[-1]); vals=tuple(st[-k:]); del st[-k:]; st.append(vals)
        elif n=='APPENDS':vals=items(); st[-1].extend(vals)
        elif n=='APPEND':v=st.pop(); st[-1].append(v)
        elif n=='SETITEMS':
            vals=items(); assert len(vals)%2==0
            st[-1].update(zip(vals[::2],vals[1::2]))
        elif n=='SETITEM':v=st.pop(); k=st.pop(); st[-1][k]=v
        elif n=='STOP':assert len(st)==1; return st[0],sorted(refs)
        else:raise ValueError((pos,n))
    raise ValueError('No STOP')
def refname(x):return (x.fn.module,x.fn.name) if isinstance(x,Obj) and isinstance(x.fn,Ref) else None
def decode(x):
    if not isinstance(x,Obj):
        if isinstance(x,list):return [decode(v) for v in x]
        if isinstance(x,tuple):return tuple(decode(v) for v in x)
        if isinstance(x,dict):return {k:decode(v) for k,v in x.items()}
        return x
    key=refname(x)
    if key in [('numpy.core.multiarray','_reconstruct'),('numpy._core.multiarray','_reconstruct')]:
        _,shape,dtype,fortran,payload=x.state
        kind=dtype.args[0]
        if kind.startswith('O'):return np.array([decode(v) for v in payload],dtype=object).reshape(shape,order='F' if fortran else 'C')
        assert kind in ('f8','f4','i8','i4','i2','u8','u4','u2','M8','m8','b1'),kind
        endian=dtype.state[1] if dtype.state else '='
        dt=(endian+kind) if kind not in ('M8','m8') else kind+'[ns]'
        return np.frombuffer(payload,dtype=dt).copy().reshape(shape,order='F' if fortran else 'C')
    if key==('pandas._libs.arrays','__pyx_unpickle_NDArrayBacked'):
        return decode(x.state[1])
    if key==('pandas._libs.tslibs.timestamps','_unpickle_timestamp'):
        ns=x.args[0]; return np.datetime64(ns,'ns')
    if key==('pandas.core.indexes.base','_new_Index'):return decode(x.args[1]['data'])
    if key==('pandas.core.frame','DataFrame'):
        mgr=x.state['_mgr']; blocks,axes=mgr.args
        cols=decode(axes[0]).tolist(); data={}
        for block in blocks:
            vals=decode(block.args[0]); placement=block.args[1]
            if isinstance(placement,Obj) and refname(placement)==('builtins','slice'):
                ix=list(range(*placement.args))
            else:ix=decode(placement).tolist()
            if vals.ndim==1:vals=vals.reshape(1,-1)
            assert len(ix)==len(vals),(ix,vals.shape)
            for i,v in zip(ix,vals):data[cols[i]]=v
        assert set(data)==set(cols)
        return pd.DataFrame(data)[cols]
    raise ValueError('Unrecognized inert object '+str(key))
def stats(df):
    out={'rows':len(df),'columns':list(df),'dtypes':df.dtypes.astype(str).to_dict(),'missing':df.isna().sum().astype(int).to_dict(),'duplicates':int(df.duplicated().sum()),'fields':{}}
    for c in df:
        v=df[c]; s={'unique':int(v.nunique(dropna=False))}
        if pd.api.types.is_numeric_dtype(v):
            s.update(min=float(v.min()),median=float(v.median()),max=float(v.max()))
        elif pd.api.types.is_datetime64_any_dtype(v):s.update(min=str(v.min()),max=str(v.max()),midnight=int((v.dt.hour.eq(0)&v.dt.minute.eq(0)).sum()))
        else:s['top']=v.astype(str).value_counts(dropna=False).head(7).to_dict()
        out['fields'][c]=s
    dist=next((c for c in df if c in ('Distance [km]','行程距离')),None)
    if dist:out['distance_ge_10km']=int((pd.to_numeric(df[dist])>=10).sum())
    return out
if __name__=='__main__':
    base=Path(__file__).parent; src=base/'sources'; out=base/'decoded';out.mkdir(exist_ok=True)
    allstats={}
    for path in src.glob('*.pkl'):
        raw,refs=parse(path.read_bytes()); data=decode(raw)
        frames=data if isinstance(data,list) else [data]
        ss=[]
        for i,df in enumerate(frames):
            assert isinstance(df,pd.DataFrame)
            df.to_csv(out/f'{path.stem}_{i}.csv',index=False,encoding='utf-8-sig')
            ss.append(stats(df));print(path.stem,i,df.shape,list(df),flush=True)
        allstats[path.stem]={'serialized_globals':refs,'groups':ss,'total_rows':sum(s['rows'] for s in ss)}
    (base/'upect_actual_data_audit.json').write_text(json.dumps(allstats,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
