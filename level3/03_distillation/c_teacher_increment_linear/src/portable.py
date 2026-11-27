"""Teacher-free NumPy implementation; no sklearn/torch/xgboost imports."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vendor'))
from descriptors import extract_sequence_features
from context_features import context,AA,INDEX

def features(sequences,names):
    desc=[];dimers=[]
    for s in sequences:
        if not s or set(s)-set(AA):raise ValueError('noncanonical sequence')
        row=extract_sequence_features(s);desc.append([row[n] for n in names]);a=np.array([INDEX[v] for v in s])
        dimers.append(np.bincount(a[:-1]*20+a[1:],minlength=400)/max(len(s)-1,1))
    b=np.column_stack([desc,dimers]);x=np.column_stack([b,context(sequences)]).astype(np.float32)
    return b,x

def load(path):
    with np.load(path,allow_pickle=False) as f:a={k:f[k] for k in f.files}
    return json.loads(bytes(a.pop('metadata_utf8')).decode()),a

def predict(meta,a,sequences):
    b,x=features(sequences,meta['descriptor_names']);z=b@a['base_coef']+a['base_intercept'];base=1/(1+np.exp(-np.clip(z,-700,700)))
    if meta['features']=='b1':x=b.astype(np.float32)
    corr=x@a['correction_coef']+a['correction_intercept']
    return np.clip(base+meta['gamma']*corr,1e-6,1-1e-6)
