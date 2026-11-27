"""Nested grouped B1-residual distillation; all outcomes exploratory."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import sys
sys.dont_write_bytecode=True
from diagnostic import *
import torch, copy
from sklearn.cross_decomposition import PLSRegression
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
SEED=20260908


def split_indices(folds,groups,allowed,heldout):
    if heldout not in allowed or not set(allowed).issubset({2,3,4,5,6}):raise ValueError('Forbidden split')
    tr=np.flatnonzero(np.isin(folds,[f for f in allowed if f!=heldout]));va=np.flatnonzero(folds==heldout)
    if set(groups[tr])&set(groups[va]):raise ValueError('Group leakage')
    if not len(tr) or not len(va):raise ValueError('Empty split')
    return tr,va


def teacher(b1,emb,y,tr,te):
    sc=StandardScaler().fit(emb[tr]);xx=sc.transform(emb[tr])
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        pls=PLSRegression(n_components=8,max_iter=500,tol=1e-6).fit(xx,y[tr])
    m=logistic(np.column_stack([b1[tr],pls.transform(xx)]),y[tr])
    return np.clip(m.predict_proba(np.column_stack([b1[te],pls.transform(sc.transform(emb[te]))]))[:,1],1e-6,1-1e-6)


def targets(b1,emb,y,folds,groups,allowed,ledger):
    soft=np.full(len(y),np.nan)
    for fold in allowed:
        tr,te=split_indices(folds,groups,allowed,fold)
        soft[te]=teacher(b1,emb,y,tr,te)
        ledger.append({'teacher_train_folds':[int(x) for x in allowed if x!=fold],'teacher_target_fold':int(fold),
              'train_group_hash':hashlib.sha256('\n'.join(sorted(set(groups[tr]))).encode()).hexdigest(),
              'target_group_hash':hashlib.sha256('\n'.join(sorted(set(groups[te]))).encode()).hexdigest()})
    return soft


def initial_model():
    tp.set_deterministic_seed(SEED)
    m=tp.make_model()
    torch.nn.init.zeros_(m.head[3].weight);torch.nn.init.zeros_(m.head[3].bias)
    return m


def fit_student(seq,desc,y,soft,offset,tr,va,alpha,epochs=40,early=True):
    sc=tp.fit_descriptor_standardizer(desc[tr])
    ds=tp.EncodedTripeptideDataset([seq[i] for i in tr],sc.transform(desc[tr]),y[tr],
          None if alpha==0 else np.clip(soft[tr],1e-6,1-1e-6))
    vd=tp.EncodedTripeptideDataset([seq[i] for i in va],sc.transform(desc[va]))
    model=initial_model();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    history=[];best_epoch=0;best_loss=float(log_loss(y[va],expit(offset[va]))) if early else None
    stale=0
    def predict(ds,off):
        model.eval();p=[]
        with torch.inference_mode():
            for s in range(0,len(ds),256):
                ii=np.arange(s,min(s+256,len(ds)))
                z=model(*tp.torch_batch(ds,ii,'cpu')[:7]).numpy().astype(float)+off[ii]
                p.extend(expit(z))
        return np.asarray(p)
    for ep in range(1,epochs+1):
        model.train();order=np.random.default_rng(SEED+ep-1).permutation(len(ds));loss_sum=0.
        for s in range(0,len(ds),128):
            ii=order[s:s+128];batch=tp.torch_batch(ds,ii,'cpu')
            z=model(*batch[:7])+torch.tensor(offset[tr[ii]],dtype=torch.float32)
            loss,_,_=tp.distillation_loss(z,batch[7],batch[8],alpha=alpha,temperature=2.)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            loss_sum+=float(loss.detach())*len(ii)
        row={'epoch':ep,'train_loss':loss_sum/len(ds)}
        if early:
            val=float(log_loss(y[va],predict(vd,offset[va])));row['validation_log_loss']=val
            if val<best_loss-.0001:best_loss=val;best_epoch=ep;stale=0
            else:stale+=1
        history.append(row)
        if early and stale>=5:break
    # Early runs select only budget/loss, never return a mismatched last model as best.
    fit=tp.FitResult(model.cpu().eval(),sc,history,SEED,alpha,2.)
    return fit,{'best_epoch':best_epoch if early else epochs,'best_log_loss':best_loss,'history':history}


def export(fit,base,path,names):
    tp.export_numpy_bundle(fit,path,descriptor_names=names,endpoint='recombinant_soluble_expression_binary_ecoli',
           extra_metadata={'residual_on_frozen_B1':True,'promoted':False})
    with np.load(path,allow_pickle=False) as f:a={k:f[k] for k in f.files}
    a.update(base_mean=base[0].mean_,base_scale=base[0].scale_,base_coef=base[-1].coef_,base_intercept=base[-1].intercept_)
    with path.open('wb') as f:np.savez_compressed(f,**a)


def numpy_predict(path,seq,desc,b1):
    meta,a=tp.load_numpy_bundle(path)
    p=tp.predict_numpy_state(meta,a,seq,desc)
    z=((b1-a['base_mean'])/a['base_scale'])@a['base_coef'].T+a['base_intercept']
    return expit(z.ravel()+logit(np.clip(p,1e-15,1-1e-15)))


def main():
    import sklearn
    out=ROOT/'runs/nested';out.mkdir(parents=True,exist_ok=False)
    d,blocks,hashes=load_data();b0,names=blocks['B0'];b1,_=blocks['B1']
    cfg=json.loads((ROOT/'protocol/parent_config.json').read_text());path=Path(cfg['paths']['project_root'])/cfg['paths']['prott5']
    if sha(path)!=cfg['input_sha256']['prott5']:raise ValueError('Embedding hash')
    raw=pd.read_csv(Path(cfg['paths']['project_root'])/cfg['paths']['development'],usecols=['development_fold'])
    emb=np.asarray(np.load(path,mmap_mode='r')[:len(raw)][raw.development_fold.isin([2,3,4,5,6]).to_numpy()],dtype=float)
    folds=d.development_fold.to_numpy();groups=d.cluster_id.to_numpy();y=d.label.to_numpy();seq=d.sequence.tolist()
    write(out/'preflight.json',{'input_hashes':{**hashes,str(path):sha(path)},
        'source_hashes':{str(p.relative_to(ROOT)):sha(p) for folder in ['src','vendor','protocol'] for p in (ROOT/folder).glob('*') if p.is_file()},
        'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'sklearn':sklearn.__version__,'started_unix':time.time(),
        'scope':'exploratory nested CV on previously used folds2–6; not confirmation'})
    pred={k:np.full(len(y),np.nan) for k in ['b1','teacher','distilled','hard_tuned','hard_matched']};details=[]
    with threadpool_limits(limits=1):
        for outer in [2,3,4,5,6]:
            outer_dir=out/f'outer_{outer}';outer_dir.mkdir()
            allowed=[x for x in [2,3,4,5,6] if x!=outer];selection={a:[] for a in [0.,.25,.5]};ledger=[]
            print(f'outer {outer}: inner selection starts',flush=True)
            for inner in allowed:
                tr,va=split_indices(folds,groups,allowed,inner);inner_folds=[x for x in allowed if x!=inner]
                soft=targets(b1,emb,y,folds,groups,inner_folds,ledger)
                baseline=logistic(b1[tr],y[tr]);offset=baseline.decision_function(b1)
                for alpha in [0.,.25,.5]:
                    _,rec=fit_student(seq,b0,y,soft,offset,tr,va,alpha)
                    selection[alpha].append({'inner_fold':inner,**rec})
                    print(f'outer {outer} inner {inner} alpha {alpha}: epoch {rec["best_epoch"]}, loss {rec["best_log_loss"]:.4f}',flush=True)
            write(outer_dir/'inner_selection.json',{str(k):v for k,v in selection.items()})
            selected=min([.25,.5],key=lambda a:(np.mean([x['best_log_loss'] for x in selection[a]]),a))
            budget=lambda a:int(np.floor(np.median([x['best_epoch'] for x in selection[a]])+.5))
            dep,h_ep=budget(selected),budget(0.)
            tr,te=split_indices(folds,groups,[2,3,4,5,6],outer)
            baseline=logistic(b1[tr],y[tr]);offset=baseline.decision_function(b1)
            soft=targets(b1,emb,y,folds,groups,allowed,ledger)
            pred['b1'][te]=expit(offset[te]);pred['teacher'][te]=teacher(b1,emb,y,tr,te)
            parity={}
            for arm,alpha,ep in [('distilled',selected,dep),('hard_tuned',0.,h_ep),('hard_matched',0.,dep)]:
                fit,rec=fit_student(seq,b0,y,soft,offset,tr,te,alpha,epochs=ep,early=False)
                p=expit(offset[te]+logit(np.clip(tp.predict_torch(fit,[seq[i] for i in te],b0[te]),1e-15,1-1e-15)))
                pred[arm][te]=p
                target=outer_dir/(arm+'.npz');export(fit,baseline,target,names)
                pn=numpy_predict(target,[seq[i] for i in te],b0[te],b1[te])
                parity[arm]=float(np.max(np.abs(p-pn)))
                if parity[arm]>1e-5:raise ValueError('NumPy parity failure')
                write(outer_dir/(arm+'_training.json'),rec)
            write(outer_dir/'teacher_memberships.json',ledger)
            record={'outer_fold':outer,'alpha':selected,'distilled_epochs':dep,'hard_tuned_epochs':h_ep,
                    'n_train':len(tr),'n_test':len(te),'parity_max_abs':parity,
                    'metrics':{k:metrics(y[te],v[te]) for k,v in pred.items()}}
            write(outer_dir/'result.json',record);details.append(record)
            print('outer result '+json.dumps({k:v['auc'] for k,v in record['metrics'].items()}),flush=True)
    np.savez_compressed(out/'private_predictions.npz',row_id=d.record_id.to_numpy(dtype=str),cluster=groups.astype(str),fold=folds,label=y,**pred)
    report={'scope':'exploratory nested within-development; not confirmation','promoted':False,
        'n_rows':len(y),'n_groups':len(np.unique(groups)),'folds':[2,3,4,5,6],
        'metrics':{k:metrics(y,v) for k,v in pred.items()},
        'primary_distilled_minus_matched_hard':paired_bootstrap(y,pred['distilled'],pred['hard_matched'],groups),
        'secondary_distilled_minus_b1':paired_bootstrap(y,pred['distilled'],pred['b1'],groups),
        'outer_results':details,'historical_gap_resolved':False}
    write(out/'result.json',report);print('DONE '+json.dumps({k:v['auc'] for k,v in report['metrics'].items()}),flush=True)

if __name__=='__main__':main()
