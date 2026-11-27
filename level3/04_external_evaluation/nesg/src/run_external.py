"""New-output, outcome-delayed execution of the historical V4 external protocol."""
from pathlib import Path
import argparse, hashlib, json, sys, platform, datetime, time
import numpy as np
import pandas as pd
import yaml
import sklearn, scipy
import external_runtime as rt
from analysis_addon.src.clustered_stats import paired_cluster_bootstrap

ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    started=time.time();c=yaml.safe_load((ROOT/'protocol/original_config_v4.yaml').read_text())
    c['run_id']='stam_nesg_external_20261016';c['paths']['run_root']=str(ROOT)
    paths={'train_partitions':'soluprot_train_partitions.csv','external_clustered':'soluprot_external_test_clustered.csv','development_features':'solubility_transparent_features_development.csv','external_features':'solubility_transparent_features_external.csv','feature_manifest':'solubility_transparent_features.json','embedding_index':'embeddings/soluprot_embedding_index.csv','prott5_array':'embeddings/soluprot_prott5.npy'}
    d=Path(c['paths']['data_root']);checks={}
    for key,rel in paths.items():
        checks[key]={'path':str(d/rel),'sha256':sha(d/rel)}
        if checks[key]['sha256']!=c['input_hashes'][key]:raise ValueError('Input digest mismatch: '+key)
    historical=Path(c['paths']['scope_root'])/'analysis_addon_run_20260810_v4/manifests/internal-primary.json'
    assert json.loads(historical.read_text())['status']=='completed'
    packages={'numpy':np.__version__,'pandas':pd.__version__,'scikit-learn':sklearn.__version__,'scipy':scipy.__version__}
    previous=json.loads((Path(c['paths']['scope_root'])/'analysis_addon_run_20260810_v4/environment/sc8_environment.json').read_text())['packages']
    assert all(previous[k]==v for k,v in packages.items()),packages
    data=rt.load_label_blind_inputs(c);train=data['train'];ext=data['external_without_outcome']
    assert 'label' not in ext.columns
    overlap=set(train.sequence)&set(ext.sequence)
    assert not overlap,'Exact train/external sequence overlap'
    assert set(train.record_id).isdisjoint(set(ext.record_id))
    assert train.groupby('cluster_id')['partition'].nunique().max()==1
    role_summary=train.groupby('partition').agg(rows=('record_id','size'),groups=('cluster_id','nunique'),positive=('label','sum')).to_dict('index')
    preflight={'at':now(),'status':'PASS','inputs':checks,'environment':{'python':sys.version,'platform':platform.platform(),'packages':packages},'development_roles':role_summary,'external_rows':len(ext),'external_groups':int(ext.external_cluster_id.nunique()),'external_labels_parsed':False,'exact_sequence_overlap':0,'historical_internal_manifest':{'path':str(historical),'sha256':sha(historical)},'feature_hashes':data['preseal_hashes']}
    write(ROOT/'protocol/preflight.json',preflight)
    if args.preflight_only:
        print(json.dumps({k:preflight[k] for k in ['status','development_roles','external_rows','external_groups','external_labels_parsed','exact_sequence_overlap']},indent=2));return
    if (ROOT/'results/external_retrospective.json').exists():raise RuntimeError('Completed result exists; use verification, do not overwrite.')
    write(ROOT/'protocol/resolved_config.json',c)
    code_files=sorted((ROOT/'src').glob('*.py'))+sorted((ROOT/'vendor').rglob('*.py'))+[ROOT/'protocol/ANALYSIS_PLAN.md',ROOT/'protocol/resolved_config.json']
    freeze={str(p.relative_to(ROOT)):sha(p) for p in code_files}
    freeze_path=ROOT/'protocol/PRE_EXECUTION_FREEZE.json'
    if freeze_path.exists():
        assert json.loads(freeze_path.read_text())['files']==freeze,'Source changed after freeze'
    else:write(freeze_path,{'at':now(),'files':freeze,'external_outcomes_historically_viewed':True,'new_execution_outcomes_not_parsed':True})
    print('PREFLIGHT PASS; fitting on development labels only',flush=True)
    results,seal=rt.build_development_only_predictions(c,ROOT,data)
    seal['sealed_at_utc']=now();seal['new_run_freeze_sha256']=sha(freeze_path)
    seal_path=rt.write_development_only_seal(ROOT,seal)
    print('PREDICTIONS SEALED; loading external outcomes',flush=True)
    y=rt.load_external_outcomes_after_seal(c,ext,seal_path);groups=ext.external_cluster_id.astype(str).to_numpy()
    pred={'record_id':ext.record_id.astype(str).to_numpy(),'external_cluster_id':groups,'label':y}
    for arm,v in results.items():
        for kind in ['raw','calibrated']:pred[arm+'__'+kind]=v[kind]
    (ROOT/'private').mkdir(exist_ok=True)
    np.savez_compressed(ROOT/'private/external_predictions.npz',**pred)
    reference='B1_ORDER_LOGIT';candidate='B1_ORDER_PLUS_PROTT5_PLS_LOGIT'
    contrast=paired_cluster_bootstrap(y,results[reference]['calibrated'],results[candidate]['calibrated'],groups,'roc_auc',10000,20260909)
    assert contrast['n_replicates_valid']==10000,contrast
    report={'run_id':c['run_id'],'completed_at_utc':now(),'analysis_scope':'retrospective_external_evaluation_with_historical_outcome_exposure','endpoint':'recombinant soluble expression in Escherichia coli, not direct aqueous peptide solubility','n_rows':len(y),'n_groups':len(np.unique(groups)),'positive':int(y.sum()),'negative':int((1-y).sum()),'largest_group':int(pd.Series(groups).value_counts().max()),'primary_contrast':{'reference':reference,'candidate':candidate,'probability_type':'development-calibrated',**contrast},'arms':{a:{'raw':rt.frozen._arm_metrics(y,v['raw'],groups),'calibrated':rt.frozen._arm_metrics(y,v['calibrated'],groups),'selected_k_C':v['selected'],'calibration_n':v['calibration_n']} for a,v in results.items()},'ci_scope':'Conditional on fixed predictions; no refitting, historical method-selection or between-study uncertainty','external_tuning':False,'elapsed_seconds':time.time()-started,'seal_sha256':sha(seal_path),'private_predictions_sha256':sha(ROOT/'private/external_predictions.npz'),'manuscript_modified':False}
    for rel,h in freeze.items():assert sha(ROOT/rel)==h,'Frozen source changed'
    write(ROOT/'results/external_retrospective.json',report)
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
