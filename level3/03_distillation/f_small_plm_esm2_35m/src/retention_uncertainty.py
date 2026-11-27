"""Paired bootstrap of retention ratios, conditional on frozen OOF predictions."""
import numpy as np,json
from pathlib import Path
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1]

def weighted_auc(y,p,w):
    _,g=np.unique(p,return_inverse=True)
    a=np.bincount(g,weights=w*y);b=np.bincount(g,weights=w*(1-y))
    return float(np.sum(a*(np.cumsum(b)-b/2))/(a.sum()*b.sum()))

def main():
    z=np.load(ROOT/'runs/nested/private_predictions.npz');y=z['label'];keys=['b1','teacher','hard_tuned','distilled'];_,cluster=np.unique(z['cluster'],return_inverse=True);n=cluster.max()+1;rng=np.random.default_rng(20260909)
    prepared={k:np.unique(z[k],return_inverse=True)[1] for k in keys};rows=[]
    for rep in range(10000):
        w=np.bincount(rng.integers(n,size=n),minlength=n)[cluster];auc={}
        for k,g in prepared.items():
            pos=np.bincount(g,weights=w*y);neg=np.bincount(g,weights=w*(1-y));auc[k]=float(np.sum(pos*(np.cumsum(neg)-neg/2))/(pos.sum()*neg.sum()))
            if rep<3:np.testing.assert_allclose(auc[k],roc_auc_score(y,z[k],sample_weight=w),atol=1e-12)
        denom=auc['teacher']-auc['b1'];remaining=auc['teacher']-auc['hard_tuned']
        if denom<=0 or remaining<=0:raise ValueError('Nonpositive bootstrap denominator')
        rows.append([(auc['distilled']-auc['b1'])/denom,(auc['hard_tuned']-auc['b1'])/denom,(auc['distilled']-auc['hard_tuned'])/denom,(auc['distilled']-auc['hard_tuned'])/remaining])
    auc={k:roc_auc_score(y,z[k]) for k in keys};den=auc['teacher']-auc['b1'];points=[(auc['distilled']-auc['b1'])/den,(auc['hard_tuned']-auc['b1'])/den,(auc['distilled']-auc['hard_tuned'])/den,(auc['distilled']-auc['hard_tuned'])/(auc['teacher']-auc['hard_tuned'])];names=['total_retention','hard_pretraining_retention','kd_retention_increment','remaining_gap_recovered']
    report={'scope':'descriptive paired bootstrap ratios conditional on fitted models; no method-selection correction','repeats':10000,'seed':20260909,'unit':'30%-identity cluster','ratios':{k:{'estimate':float(points[i]),'ci95':np.quantile(np.asarray(rows)[:,i],[.025,.975]).tolist()} for i,k in enumerate(names)}}
    (ROOT/'runs/retention_uncertainty.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
