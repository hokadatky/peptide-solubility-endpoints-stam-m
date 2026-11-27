"""Bounded, development-selected exploratory head comparison. See PLAN.md."""
from pathlib import Path
import sys,json,hashlib,datetime,time,warnings,itertools
import numpy as np
import pandas as pd
import joblib,sklearn,scipy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent
sys.path.insert(0,str(PARENT/'src'))
import external_runtime as rt
from analysis_addon.src.nested_models import PlattCalibrator
C_GRID=[1e-4,1e-3,.01,.1,1.,10.,100.,1000.,10000.]
ARMS=[('B1_LOGIT','logit','none',[0]),('FULL_LOGIT','logit','full',[1024]),('PCA_LOGIT','logit','pca',[64,128,256]),('B1_HGB','hgb','none',[0]),('PCA_HGB','hgb','pca',[64,128])]
CONTRASTS=[('FULL_LOGIT','B1_LOGIT'),('PCA_LOGIT','B1_LOGIT'),('PCA_HGB','B1_HGB')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def prepare_pca(E,idx):
    scaler=StandardScaler().fit(E[idx]);pca=PCA(n_components=256,svd_solver='full').fit(scaler.transform(E[idx]))
    return {'emb_scaler':scaler,'pca':pca}
def transform(B,E,prep):
    mode=prep['mode'];k=prep['k']
    if mode=='none':X=B
    elif mode=='full':X=np.hstack([B,E])
    else:X=np.hstack([B,prep['pca'].transform(prep['emb_scaler'].transform(E))[:,:k]])
    if prep.get('joint_scaler') is not None:X=prep['joint_scaler'].transform(X)
    return X
def representation(B,E,idx,kind,mode,k,pca):
    prep={'mode':mode,'k':k,'joint_scaler':None}
    if mode=='pca':prep.update(pca)
    X=transform(B,E,prep)
    if kind=='logit':
        prep['joint_scaler']=StandardScaler().fit(X[idx]);X=prep['joint_scaler'].transform(X)
    return prep,X
def params(kind):
    if kind=='logit':return [{'C':c} for c in C_GRID]
    return [dict(max_leaf_nodes=leaf,min_samples_leaf=sample,l2_regularization=l2) for leaf,sample,l2 in itertools.product([15,31],[100,50,20],[10.,1.])]
def fit_model(kind,X,y,param):
    if kind=='logit':model=LogisticRegression(**param,solver='lbfgs',max_iter=20000,tol=1e-4,penalty='l2',class_weight=None)
    else:model=HistGradientBoostingClassifier(**param,learning_rate=.05,max_iter=300,early_stopping=False,random_state=20261016)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always');model.fit(X,y)
    conv=any(issubclass(w.category,ConvergenceWarning) for w in captured)
    if kind=='logit':conv=conv or int(model.n_iter_.max())>=20000
    diag={'converged':not conv,'iterations':np.asarray(model.n_iter_).tolist(),'warnings':[str(w.message) for w in captured]}
    return model,diag
def metrics(y,p):
    return {'auc':float(roc_auc_score(y,p)),'average_precision':float(average_precision_score(y,p)),'brier':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,np.clip(p,1e-12,1-1e-12),labels=[0,1]))}
def bootstrap(y,groups,scores,reps=10000):
    _,g=np.unique(groups,return_inverse=True);ng=int(g.max()+1);names=list(scores)
    bins=[np.unique(scores[name],return_inverse=True)[1] for name in names]
    rng=np.random.default_rng(20261016);values=np.empty((reps,len(names)))
    for r in range(reps):
        w=np.bincount(rng.integers(0,ng,size=ng),minlength=ng)[g]
        for j,b in enumerate(bins):
            pos=np.bincount(b,weights=w*y);neg=np.bincount(b,weights=w*(1-y))
            values[r,j]=np.sum(pos*(np.cumsum(neg)-neg/2))/(pos.sum()*neg.sum())
            if r<5:np.testing.assert_allclose(values[r,j],roc_auc_score(y,scores[names[j]],sample_weight=w),atol=1e-12,rtol=0)
    contrasts=[]
    for cand,base in CONTRASTS:
        delta=values[:,names.index(cand)]-values[:,names.index(base)]
        contrasts.append({'candidate':cand,'baseline':base,'delta_auc':float(roc_auc_score(y,scores[cand])-roc_auc_score(y,scores[base])),'ci95':np.percentile(delta,[2.5,97.5]).tolist(),'bonferroni_ci_98_333333':np.percentile(delta,[100*.05/6,100*(1-.05/6)]).tolist()})
    return {'replicates':reps,'seed':20261016,'arm_auc_ci95':{name:np.percentile(values[:,j],[2.5,97.5]).tolist() for j,name in enumerate(names)},'contrasts':contrasts}
def main():
    started=time.time()
    if (ROOT/'results.json').exists():raise RuntimeError('Completed exploratory results exist; do not overwrite or add models.')
    assert json.loads((ROOT/'INPUT_AUDIT.json').read_text())['status']=='PASS'
    preflight=json.loads((PARENT/'protocol/preflight.json').read_text())
    for item in preflight['inputs'].values():assert sha(item['path'])==item['sha256']
    packages={'numpy':np.__version__,'pandas':pd.__version__,'scikit-learn':sklearn.__version__,'scipy':scipy.__version__}
    assert packages==preflight['environment']['packages']
    dependencies=[ROOT/'PLAN.md',ROOT/'run_models.py',ROOT/'audit_inputs.py',ROOT/'INPUT_AUDIT.json',PARENT/'src/external_runtime.py']+sorted((PARENT/'vendor').rglob('*.py'))
    parent_files={str(p.relative_to(PARENT)):sha(p) for p in PARENT.rglob('*') if p.is_file() and ROOT not in p.parents and '__pycache__' not in p.parts}
    freeze={'at_utc':now(),'files':{str(p):sha(p) for p in dependencies},'parent_files':parent_files,'packages':packages,'new_model_external_outcomes_not_evaluated':True,'previous_NESG_results_seen':True}
    if (ROOT/'FREEZE.json').exists():
        old=json.loads((ROOT/'FREEZE.json').read_text());assert old['files']==freeze['files'];assert old['parent_files']==parent_files
    else:write(ROOT/'FREEZE.json',freeze)
    c=json.loads((PARENT/'protocol/resolved_config.json').read_text());data=rt.load_label_blind_inputs(c)
    train=data['train'];n=len(train);y=train.label.to_numpy(dtype=int);part=train.partition.to_numpy();B=data['b1'];E=data['embeddings'];ext=data['external_without_outcome']
    idx={key:np.flatnonzero(part==key) for key in ['fit','model_selection','calibration']}
    for a,b in itertools.combinations(idx,2):assert set(train.cluster_id.iloc[idx[a]]).isdisjoint(train.cluster_id.iloc[idx[b]])
    fit=idx['fit'];sel=idx['model_selection'];cal=idx['calibration'];refit=np.concatenate([fit,sel]);external=np.arange(n,len(B))
    print('Input audit PASS; development-only model selection begins',flush=True)
    pca_fit=prepare_pca(E,fit);pca_refit=prepare_pca(E,refit)
    report={};predictions={'record_id':ext.record_id.to_numpy(dtype=str),'external_cluster_id':ext.external_cluster_id.to_numpy(dtype=str)};models={}
    for name,kind,mode,dimensions in ARMS:
        checkpoint=ROOT/'checkpoints'/f'{name}.npz';modelpath=ROOT/'models'/f'{name}.joblib';detailpath=ROOT/'selection'/f'{name}.json'
        if checkpoint.exists():
            detail=json.loads(detailpath.read_text());assert sha(modelpath)==detail['model_sha256'];assert sha(checkpoint)==detail['checkpoint_sha256']
            with np.load(checkpoint,allow_pickle=False) as z:
                for key in ['raw','calibrated']:predictions[name+'__'+key]=z[key]
            report[name]=detail;models[name]=sha(modelpath);print('Resume '+name,flush=True);continue
        records=[];best=None;bestkey=None
        for k in dimensions:
            prep,X=representation(B,E,fit,kind,mode,k,pca_fit)
            for param in params(kind):
                model,diag=fit_model(kind,X[fit],y[fit],param)
                p=model.predict_proba(X[sel])[:,1];auc=float(roc_auc_score(y[sel],p));eligible=diag['converged'] and np.isfinite(p).all()
                row={'k':k,**param,'selection_auc':auc,'eligible':bool(eligible),'diagnostics':diag};records.append(row)
                # Enumerations prefer simplicity; replace only on a strictly larger AUC.
                if eligible and (bestkey is None or auc>bestkey):best=row;bestkey=auc
            print(name+' k='+str(k)+' development grid complete',flush=True)
        if best is None:raise RuntimeError('No converged candidates: '+name)
        selected={k:best[k] for k in params(kind)[0]};prep,X=representation(B,E,refit,kind,mode,best['k'],pca_refit)
        model,diag=fit_model(kind,X[refit],y[refit],selected);assert diag['converged'],(name,diag)
        raw_cal=model.predict_proba(X[cal])[:,1];calibrator=PlattCalibrator().fit(raw_cal,y[cal])
        assert calibrator.model.coef_[0,0]>0,'Unexpected development-calibration sign inversion'
        raw=model.predict_proba(X[external])[:,1];calibrated=calibrator.predict(raw)
        assert np.isfinite(raw).all() and np.isfinite(calibrated).all()
        modelpath.parent.mkdir(exist_ok=True);joblib.dump({'prep':prep,'classifier':model,'calibrator':calibrator,'kind':kind,'name':name},modelpath,compress=3)
        checkpoint.parent.mkdir(exist_ok=True);np.savez_compressed(checkpoint,raw=raw,calibrated=calibrated,calibration_raw=raw_cal,calibration_y=y[cal])
        detail={'kind':kind,'mode':mode,'selected':best,'grid':records,'refit_diagnostics':diag,'calibration_diagnostics':calibrator.diagnostics,'calibration_slope':float(calibrator.model.coef_[0,0]),'calibration_intercept':float(calibrator.model.intercept_[0]),'calibration_partition_raw_metrics':metrics(y[cal],raw_cal),'model_sha256':sha(modelpath),'checkpoint_sha256':sha(checkpoint)}
        write(detailpath,detail);report[name]=detail;models[name]=sha(modelpath)
        predictions[name+'__raw']=raw;predictions[name+'__calibrated']=calibrated
        print(name+' selected '+str({k:v for k,v in best.items() if k!='diagnostics'})+'; predictions saved, external metrics not computed',flush=True)
    predpath=ROOT/'predictions_sealed.npz';np.savez_compressed(predpath,**predictions)
    seal={'at_utc':now(),'prediction_archive_sha256':sha(predpath),'models':models,'freeze_sha256':sha(ROOT/'FREEZE.json'),'all_arms_fitted_before_external_evaluation':True,'external_outcomes_used_for_current_hyperparameter_selection':False,'previous_external_outcomes_seen':True}
    write(ROOT/'PREDICTION_SEAL.json',seal)
    print('All five arms sealed; evaluating external outcomes',flush=True)
    outcomes=pd.read_csv(Path(c['paths']['data_root'])/'soluprot_external_test_clustered.csv',usecols=['record_id','label'],dtype={'record_id':str})
    assert np.array_equal(outcomes.record_id.to_numpy(dtype=str),predictions['record_id']);yext=outcomes.label.to_numpy(dtype=int)
    scores={name:predictions[name+'__calibrated'] for name,_,_,_ in ARMS}
    res={'scope':'post-NESG-result exploratory model development; no confirmatory claim','completed_at_utc':now(),'n_external':len(yext),'groups_external':len(np.unique(predictions['external_cluster_id'])),'external_positive':int(yext.sum()),'arms':{name:{'raw':metrics(yext,predictions[name+'__raw']),'calibrated':metrics(yext,predictions[name+'__calibrated']),'selected':{k:v for k,v in report[name]['selected'].items() if k!='diagnostics'},'calibration_partition_raw_metrics':report[name]['calibration_partition_raw_metrics']} for name,_,_,_ in ARMS},'uncertainty':bootstrap(yext,predictions['external_cluster_id'],scores),'elapsed_seconds':time.time()-started,'seal_sha256':sha(ROOT/'PREDICTION_SEAL.json'),'input_preflight_sha256':sha(PARENT/'protocol/preflight.json')}
    for p,h in freeze['files'].items():assert sha(p)==h
    for p,h in parent_files.items():assert sha(PARENT/p)==h,p
    write(ROOT/'results.json',res);print(json.dumps(res,indent=2),flush=True)
if __name__=='__main__':main()
