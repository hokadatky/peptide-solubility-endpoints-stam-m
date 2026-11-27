"""Independent metric and weighted-cluster-bootstrap check; no training."""
from pathlib import Path
import json,hashlib
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
import external_runtime as rt
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    result=json.loads((ROOT/'results/external_retrospective.json').read_text());seal=json.loads((ROOT/'manifests/external_development_only_seal.json').read_text());checks=[]
    assert sha(ROOT/'private/external_predictions.npz')==result['private_predictions_sha256']
    assert sha(ROOT/'manifests/external_development_only_seal.json')==result['seal_sha256']
    repair=json.loads((ROOT/'results/SERIALIZATION_REPAIR.json').read_text())
    assert sha(ROOT/'private/external_predictions_safe.npz')==repair['safe_copy_sha256']
    with np.load(ROOT/'private/external_predictions_safe.npz',allow_pickle=False) as z:
        y=z['label'];groups=z['external_cluster_id'];_,g=np.unique(groups,return_inverse=True);ng=int(g.max()+1)
        assert len(y)==3100 and ng==2723
        for arm,record in result['arms'].items():
            assert seal['arms'][arm]['development_seal_diagnostics']['refit']['logistic']['converged']
            assert seal['arms'][arm]['development_seal_diagnostics']['calibration']['converged']
            for kind in ['raw','calibrated']:
                p=z[arm+'__'+kind];assert rt.probability_hash(p)==seal['arms'][arm][kind+'_external_prediction_sha256']
                metrics={'roc_auc':roc_auc_score(y,p),'auprc':average_precision_score(y,p),'brier':brier_score_loss(y,p),'log_loss':log_loss(y,np.clip(p,1e-12,1-1e-12),labels=[0,1])}
                for k,v in metrics.items():np.testing.assert_allclose(v,record[kind]['metrics'][k],atol=1e-12,rtol=0)
                checks.append({'arm':arm,'kind':kind,'metric_match':True,'prediction_seal_match':True})
        p0=z[result['primary_contrast']['reference']+'__calibrated'];p1=z[result['primary_contrast']['candidate']+'__calibrated']
        bins=[np.unique(p,return_inverse=True)[1] for p in [p0,p1]]
        rng=np.random.default_rng(20260909);deltas=[]
        for rep in range(10000):
            w=np.bincount(rng.integers(0,ng,size=ng),minlength=ng)[g];aucs=[]
            for b in bins:
                pos=np.bincount(b,weights=w*y);neg=np.bincount(b,weights=w*(1-y));aucs.append(float(np.sum(pos*(np.cumsum(neg)-neg/2))/(pos.sum()*neg.sum())))
            if rep<5:
                np.testing.assert_allclose(aucs,[roc_auc_score(y,p,sample_weight=w) for p in [p0,p1]],atol=1e-12,rtol=0)
            deltas.append(aucs[1]-aucs[0])
        ci=np.percentile(deltas,[2.5,97.5]);np.testing.assert_allclose(ci,result['primary_contrast']['ci95'],atol=1e-12,rtol=0)
    freeze=json.loads((ROOT/'protocol/PRE_EXECUTION_FREEZE.json').read_text())
    for p,h in freeze['files'].items():assert sha(ROOT/p)==h,p
    preflight=json.loads((ROOT/'protocol/preflight.json').read_text())
    for item in preflight['inputs'].values():assert sha(Path(item['path']))==item['sha256']
    baseline=ROOT.parent/'revision_response_20261016/BASELINE_SHA256.json'
    original=json.loads(baseline.read_text());changes=[x['path'] for x in original if sha(Path(x['path']))!=x['sha256']];assert not changes,changes
    report={'status':'PASS','metrics':checks,'independent_cluster_bootstrap_ci95':ci.tolist(),'replicates':10000,'input_hashes_unchanged':True,'frozen_source_unchanged':True,'original_submission_and_repository_files_unchanged':len(original),'unexpected_changes':changes,'training_rerun':False}
    (ROOT/'results/INDEPENDENT_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
