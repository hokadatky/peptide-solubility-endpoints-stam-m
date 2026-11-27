"""Independent replay, selection/transform audit and ordinary cluster resampling."""
from pathlib import Path
import json,sys,itertools
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
import run_models as run
ROOT=Path(__file__).resolve().parent
def main():
    result=json.loads((ROOT/'results.json').read_text());freeze=json.loads((ROOT/'FREEZE.json').read_text());seal=json.loads((ROOT/'PREDICTION_SEAL.json').read_text())
    assert run.sha(ROOT/'FREEZE.json')==seal['freeze_sha256'];assert run.sha(ROOT/'predictions_sealed.npz')==seal['prediction_archive_sha256'];assert run.sha(ROOT/'PREDICTION_SEAL.json')==result['seal_sha256']
    for p,h in freeze['files'].items():assert run.sha(p)==h,p
    for p,h in freeze['parent_files'].items():assert run.sha(run.PARENT/p)==h,p
    c=json.loads((run.PARENT/'protocol/resolved_config.json').read_text());pre=json.loads((run.PARENT/'protocol/preflight.json').read_text())
    for entry in pre['inputs'].values():assert run.sha(entry['path'])==entry['sha256']
    data=run.rt.load_label_blind_inputs(c);n=len(data['train']);train=data['train'];B=data['b1'];E=data['embeddings']
    refit=np.flatnonzero(train.partition.isin(['fit','model_selection']).to_numpy());cal=np.flatnonzero((train.partition=='calibration').to_numpy());assert len(refit)==9819 and len(cal)==1617
    ext=pd.read_csv(Path(c['paths']['data_root'])/'soluprot_external_test_clustered.csv',dtype={'record_id':str,'external_cluster_id':str});y=ext.label.to_numpy(dtype=int)
    rows=[];scores={}
    with np.load(ROOT/'predictions_sealed.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['record_id'],ext.record_id);np.testing.assert_array_equal(z['external_cluster_id'],ext.external_cluster_id)
        groups=z['external_cluster_id']
        for name,kind,mode,_ in run.ARMS:
            detail=json.loads((ROOT/'selection'/f'{name}.json').read_text());modelpath=ROOT/'models'/f'{name}.joblib'
            assert run.sha(modelpath)==detail['model_sha256']==seal['models'][name]
            eligible=[g for g in detail['grid'] if g['eligible']];selected=max(eligible,key=lambda v:v['selection_auc']);assert selected==detail['selected']
            assert detail['refit_diagnostics']['converged'] and detail['calibration_diagnostics']['converged']
            m=joblib.load(modelpath);prep=m['prep']
            if mode=='pca':
                assert prep['pca'].n_samples_==9819
                assert prep['emb_scaler'].n_samples_seen_==9819
                np.testing.assert_allclose(prep['emb_scaler'].mean_,E[refit].mean(axis=0),atol=1e-12)
            if kind=='logit':
                assert prep['joint_scaler'].n_samples_seen_==9819
                unscaled={**prep,'joint_scaler':None}
                np.testing.assert_allclose(prep['joint_scaler'].mean_,run.transform(B[refit],E[refit],unscaled).mean(axis=0),atol=1e-12)
            X=run.transform(B[n:],E[n:],prep);raw=m['classifier'].predict_proba(X)[:,1];p=m['calibrator'].predict(raw)
            np.testing.assert_allclose(raw,z[name+'__raw'],atol=1e-12,rtol=0);np.testing.assert_allclose(p,z[name+'__calibrated'],atol=1e-12,rtol=0)
            for kindpred,pred in [('raw',raw),('calibrated',p)]:
                independent={'auc':roc_auc_score(y,pred),'average_precision':average_precision_score(y,pred),'brier':brier_score_loss(y,pred),'log_loss':log_loss(y,np.clip(pred,1e-12,1-1e-12),labels=[0,1])}
                for key,val in independent.items():np.testing.assert_allclose(val,result['arms'][name][kindpred][key],atol=1e-12,rtol=0)
                rows.append({'arm':name,'probabilities':kindpred,**independent})
            scores[name]=p
        with np.load(run.PARENT/'private/external_predictions_safe.npz',allow_pickle=False) as old:
            difference=float(np.max(np.abs(z['B1_LOGIT__calibrated']-old['B1_ORDER_LOGIT__calibrated'])))
            np.testing.assert_allclose(z['B1_LOGIT__calibrated'],old['B1_ORDER_LOGIT__calibrated'],atol=1e-10,rtol=0)
    # Independently use ordinary row expansion from sampled clusters, not weighted rank bins.
    unique=np.unique(groups);members=[np.flatnonzero(groups==g) for g in unique];ng=len(unique);rng=np.random.default_rng(20261016)
    values=np.empty((10000,len(scores)));names=list(scores)
    for b in range(10000):
        selection=rng.integers(0,ng,size=ng);indices=np.concatenate([members[g] for g in selection])
        for j,name in enumerate(names):values[b,j]=roc_auc_score(y[indices],scores[name][indices])
    for contrast in result['uncertainty']['contrasts']:
        delta=values[:,names.index(contrast['candidate'])]-values[:,names.index(contrast['baseline'])]
        np.testing.assert_allclose(np.percentile(delta,[2.5,97.5]),contrast['ci95'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(np.percentile(delta,[100*.05/6,100*(1-.05/6)]),contrast['bonferroni_ci_98_333333'],atol=1e-12,rtol=0)
    for j,name in enumerate(names):np.testing.assert_allclose(np.percentile(values[:,j],[2.5,97.5]),result['uncertainty']['arm_auc_ci95'][name],atol=1e-12,rtol=0)
    baseline=json.loads((run.PARENT.parent/'revision_response_20261016/BASELINE_SHA256.json').read_text());changes=[v['path'] for v in baseline if run.sha(v['path'])!=v['sha256']];assert not changes
    pd.DataFrame(rows).to_csv(ROOT/'metrics.tsv',sep='\t',index=False)
    summary={'status':'PASS','model_replay_arms':5,'selection_and_training_local_transform_checks':True,'independent_metrics':True,'independent_row_expansion_bootstrap_replicates':10000,'ci95_and_three_comparison_bonferroni_ci_match':True,'original_B1_max_prediction_difference':difference,'parent_evidence_files_unchanged':len(freeze['parent_files']),'original_submission_and_repository_files_unchanged':len(baseline),'unexpected_changes':changes,'models_refitted_by_verifier':False}
    run.write(ROOT/'VERIFICATION.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
