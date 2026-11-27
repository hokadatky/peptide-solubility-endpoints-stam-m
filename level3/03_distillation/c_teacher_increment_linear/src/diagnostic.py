"""S24 H1 diagnostic; no teacher or prior fitted representation is consumed."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor'))
import json, hashlib, warnings, platform, time
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
import tripeplite as tp
from sequence_features import build_b0_b1


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def write(path, obj):
    with Path(path).open('x') as f: json.dump(obj, f, indent=2, allow_nan=False); f.write('\n')


def check_partition(frame):
    if frame[['record_id','cluster_id','development_fold','label','sequence']].isna().any().any():
        raise ValueError('Missing partition fields')
    if frame.record_id.duplicated().any(): raise ValueError('Duplicate IDs')
    if not set(frame.development_fold).issubset({2,3,4,5,6}): raise ValueError('Forbidden folds')
    if set(frame.development_fold) != {2,3,4,5,6}: raise ValueError('Missing folds')
    if frame.groupby('cluster_id').development_fold.nunique().max() != 1:
        raise ValueError('Cluster leakage')
    if set(frame.label) != {0,1}: raise ValueError('Nonbinary labels')


def load_data():
    cfg=json.loads((ROOT/'protocol/parent_config.json').read_text())
    paths={k:Path(cfg['paths']['project_root'])/cfg['paths'][k]
           for k in ['development','transparent_features','feature_manifest']}
    for k,p in paths.items():
        if sha(p)!=cfg['input_sha256'][k]: raise ValueError('Source hash mismatch: '+k)
    d=pd.read_csv(paths['development'],dtype={'record_id':str,'cluster_id':str})
    f=pd.read_csv(paths['transparent_features'],dtype={'record_id':str})
    if not np.array_equal(d.record_id,f.record_id): raise ValueError('Feature row mismatch')
    mask=d.development_fold.isin([2,3,4,5,6])
    d=d.loc[mask].reset_index(drop=True); f=f.loc[mask].reset_index(drop=True)
    check_partition(d)
    blocks=build_b0_b1(f,d.sequence.tolist(),paths['feature_manifest'])
    return d,blocks,{str(p):sha(p) for p in paths.values()}


def features(fit, seq, desc):
    import torch
    ds=tp.EncodedTripeptideDataset(seq,fit.standardizer.transform(desc))
    values=[]
    def capture(module,inputs,output): values.append(output.detach().cpu().numpy().copy())
    handle=fit.model.combined_norm.register_forward_hook(capture)
    try:
        fit.model.eval()
        with torch.inference_mode():
            for i in range(0,len(ds),256):
                fit.model(*tp.torch_batch(ds,np.arange(i,min(i+256,len(ds))),'cpu')[:7])
    finally: handle.remove()
    return np.concatenate(values)


def logistic(x,y):
    m=make_pipeline(StandardScaler(),LogisticRegression(C=1000,max_iter=20000,tol=1e-4))
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        m.fit(x.astype(float),y)
    if not np.isfinite(m[-1].coef_).all(): raise ValueError('Nonfinite fit')
    return m


def metrics(y,p):
    if not np.isfinite(p).all(): raise ValueError('Nonfinite prediction')
    bins=[]
    for i in range(10):
        mask=(p>=i/10)&((p<(i+1)/10) if i<9 else (p<=1))
        bins.append({'bin':i,'n':int(mask.sum()),'mean_score':float(p[mask].mean()) if mask.any() else None,
                     'observed_rate':float(y[mask].mean()) if mask.any() else None})
    return {'auc':roc_auc_score(y,p),'average_precision':average_precision_score(y,p),
            'brier':brier_score_loss(y,p),'log_loss':log_loss(y,p),'reliability':bins}


def paired_bootstrap(y,a,b,groups,n=10000,seed=20260908):
    y,a,b,groups=map(np.asarray,(y,a,b,groups))
    if not (y.shape==a.shape==b.shape==groups.shape) or y.ndim!=1: raise ValueError('Pair mismatch')
    if not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError('Nonfinite')
    unique,inv=np.unique(groups,return_inverse=True); rng=np.random.default_rng(seed); delta=[]
    for _ in range(n):
        w=np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique))[inv]
        if len(np.unique(y[w>0]))<2: continue
        delta.append(roc_auc_score(y,a,sample_weight=w)-roc_auc_score(y,b,sample_weight=w))
    if len(delta)<n*.95: raise ValueError('Too few valid bootstrap replicates')
    return {'observed_delta':float(roc_auc_score(y,a)-roc_auc_score(y,b)),
            'ci95':np.quantile(delta,[.025,.975]).tolist(),'valid':len(delta),'requested':n,'seed':seed,
            'unit':'30%-identity cluster','conditional_on_fitted_models':True}


def main():
    import torch, sklearn
    out=ROOT/'runs/h1'; out.mkdir(parents=True,exist_ok=False)
    d,blocks,hashes=load_data(); b0,names=blocks['B0']; b1,_=blocks['B1']
    train=np.flatnonzero(d.development_fold.isin([2,3,4,5])); test=np.flatnonzero(d.development_fold==6)
    seq=d.sequence.tolist(); y=d.label.to_numpy()
    write(out/'preflight.json',{'inputs':hashes,'spec_sha256':sha(ROOT/'protocol/SPEC.md'),
          'code_sha256':sha(Path(__file__)),'train_folds':[2,3,4,5],'evaluation_fold':6,
          'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,
          'sklearn':sklearn.__version__,'started_unix':time.time()})
    print('Preflight passed; fitting hard-only TriPepLite on folds 2–5.',flush=True)
    fit=tp.train_fixed_student([seq[i] for i in train],b0[train],y[train],None,alpha=0,epochs=40)
    tp.export_numpy_bundle(fit,out/'hard_student.npz',descriptor_names=names,
                          endpoint='recombinant_soluble_expression_binary_ecoli')
    write(out/'training_history.json',fit.history)
    print('Extracting frozen representation and fitting logistic probes.',flush=True)
    z=features(fit,seq,b0)
    probe=logistic(z[train],y[train]); baseline=logistic(b1[train],y[train])
    preds={'probe':probe.predict_proba(z[test])[:,1],'b1':baseline.predict_proba(b1[test])[:,1],
           'hard':tp.predict_torch(fit,[seq[i] for i in test],b0[test])}
    np.savez_compressed(out/'private_predictions.npz',row_id=d.record_id.to_numpy(dtype=str)[test],
                        cluster=d.cluster_id.to_numpy(dtype=str)[test],label=y[test],**preds)
    for name,m in [('probe',probe),('b1',baseline)]:
        np.savez_compressed(out/(name+'_logistic.npz'),mean=m[0].mean_,scale=m[0].scale_,
                            coef=m[-1].coef_,intercept=m[-1].intercept_)
    report={'scope':'exploratory training-fold diagnostic; not confirmation','promoted':False,
            'train_rows':len(train),'test_rows':len(test),'train_groups':int(d.iloc[train].cluster_id.nunique()),
            'test_groups':int(d.iloc[test].cluster_id.nunique()),'train_label_counts':np.bincount(y[train]).tolist(),
            'test_label_counts':np.bincount(y[test]).tolist(),'dimensions':{'probe':z.shape[1],'b1':b1.shape[1]},
            'metrics':{k:metrics(y[test],p) for k,p in preds.items()},
            'primary_probe_minus_b1':paired_bootstrap(y[test],preds['probe'],preds['b1'],d.cluster_id.to_numpy()[test])}
    lengths=d.sequence.str.len().to_numpy(); lo,hi=int(lengths[train].min()),int(lengths[train].max())
    report['applicability']={'train_length_range':[lo,hi],'test_length_range':[int(lengths[test].min()),int(lengths[test].max())],
         'test_outside_train_length':int(((lengths[test]<lo)|(lengths[test]>hi)).sum()),'canonical_sequences_only':True}
    report['interpretation']='Diagnostic of the fitted representation; no architectural or causal proof.'
    write(out/'result.json',report)
    print(json.dumps({k:v['auc'] for k,v in report['metrics'].items()}),flush=True)
    print(json.dumps(report['primary_probe_minus_b1']),flush=True)

if __name__=='__main__': main()
