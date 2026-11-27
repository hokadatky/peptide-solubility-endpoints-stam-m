"""Synthetic checks of transform isolation and clustered uncertainty."""
import numpy as np
from sklearn.metrics import roc_auc_score
import run_models as run

def test_transform_fit_excludes_held_out_rows():
    rng=np.random.default_rng(31);B=rng.normal(size=(330,7));E=rng.normal(size=(330,280));idx=np.arange(300)
    E[300:]+=1000;B[300:]+=1000
    pca=run.prepare_pca(E,idx)
    np.testing.assert_allclose(pca['emb_scaler'].mean_,E[idx].mean(axis=0))
    assert pca['pca'].n_samples_==300
    for mode,k in [('none',0),('full',280),('pca',64)]:
        prep,X=run.representation(B,E,idx,'logit',mode,k,pca)
        np.testing.assert_allclose(X[idx].mean(axis=0),0,atol=1e-12)
        assert prep['joint_scaler'].n_samples_seen_==300
        before=run.transform(B[idx],E[idx],prep)
        run.transform(B[300:]*100,E[300:]*100,prep)
        np.testing.assert_array_equal(before,run.transform(B[idx],E[idx],prep))

def test_cluster_bootstrap_preserves_ties_and_groups():
    rng=np.random.default_rng(52);y=np.tile([0,1],20);groups=np.repeat(np.arange(20),2).astype(str)
    scores={name:rng.integers(0,6,size=len(y))/6 for name,_,_,_ in run.ARMS}
    result=run.bootstrap(y,groups,scores,reps=100)
    unique=np.unique(groups);members=[np.flatnonzero(groups==g) for g in unique];rng=np.random.default_rng(20261016)
    delta=[]
    for _ in range(100):
        rows=np.concatenate([members[g] for g in rng.integers(0,20,size=20)])
        delta.append(roc_auc_score(y[rows],scores['FULL_LOGIT'][rows])-roc_auc_score(y[rows],scores['B1_LOGIT'][rows]))
    np.testing.assert_allclose(result['contrasts'][0]['ci95'],np.percentile(delta,[2.5,97.5]),atol=1e-12)
    np.testing.assert_allclose(result['contrasts'][0]['bonferroni_ci_98_333333'],np.percentile(delta,[100*.05/6,100*(1-.05/6)]),atol=1e-12)
