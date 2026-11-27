import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
from teacher_reference import *
from context_features import context
from sklearn.linear_model import Ridge
import xgboost as xgb
from itertools import combinations
SEED=20260909

class Data:
    def __init__(self):
        self.d,self.blocks,hashes=load_data();self.b,self.names=self.blocks['B1'];self.y=self.d.label.to_numpy();self.f=self.d.development_fold.to_numpy();self.g=self.d.cluster_id.to_numpy();self.seq=self.d.sequence.tolist()
        cfg=json.loads((ROOT/'protocol/parent_config.json').read_text());base=Path(cfg['paths']['project_root'])
        embp=base/cfg['paths']['prott5'];idxp=base/cfg['paths']['embedding_index'];assert sha(embp)==cfg['input_sha256']['prott5'];assert sha(idxp)==cfg['input_sha256']['embedding_index']
        raw=pd.read_csv(base/cfg['paths']['development']);index=pd.read_csv(idxp);assert np.array_equal(index.record_id[:len(raw)],raw.record_id)
        self.emb=np.asarray(np.load(embp,mmap_mode='r')[:len(raw)][raw.development_fold.isin([2,3,4,5,6])],float)
        self.x={'b1':self.b.astype(np.float32),'context':np.column_stack([self.b,context(self.seq)]).astype(np.float32)}
        self.cache=ROOT/'runs/target_cache';self.cache.mkdir(exist_ok=True)
        self.input_hashes={**hashes,str(embp):sha(embp),str(idxp):sha(idxp)}
    def indices(self,allowed):return np.flatnonzero(np.isin(self.f,allowed))
    def pair(self,train,test):
        assert set(train)<=set(range(2,7)) and set(test)<=set(range(2,7)) and not set(train)&set(test)
        tr=self.indices(train);te=self.indices(test);assert not set(self.g[tr])&set(self.g[te])
        key='tr'+''.join(map(str,sorted(train)))+'_te'+''.join(map(str,sorted(test)))
        path=self.cache/(key+'.npz')
        if path.exists():
            with np.load(path) as z:return z['b'],z['t']
        baseline=logistic(self.b[tr],self.y[tr]);b=baseline.predict_proba(self.b[te])[:,1];t=teacher(self.b,self.emb,self.y,tr,te)
        np.savez_compressed(path,b=b,t=t)
        write(self.cache/(key+'.json'),{'train_folds':train,'target_folds':test,'n_train':len(tr),'n_target':len(te),'training_groups_sha256':hashlib.sha256('\n'.join(sorted(set(self.g[tr]))).encode()).hexdigest(),'target_groups_sha256':hashlib.sha256('\n'.join(sorted(set(self.g[te]))).encode()).hexdigest()})
        return b,t
    def targets(self,allowed):
        b=np.full(len(self.y),np.nan);t=b.copy()
        for f in allowed:
            te=self.indices([f]);b[te],t[te]=self.pair([x for x in allowed if x!=f],[f])
        return b,t

def regressor(family,param):
    if family=='ridge':return make_pipeline(StandardScaler(),Ridge(alpha=param,solver='lsqr',tol=1e-6,max_iter=10000))
    return xgb.XGBRegressor(n_estimators=300,max_depth=int(param),learning_rate=.03,min_child_weight=20,reg_lambda=10,subsample=1,colsample_bytree=1,random_state=SEED,n_jobs=1,tree_method='hist',objective='reg:squarederror')

def score(base,correction,gamma):return np.clip(base+gamma*correction,1e-6,1-1e-6)

def preflight(out,data):
    import sklearn
    write(out/'preflight.json',{'started_unix':time.time(),'input_hashes':data.input_hashes,'source_hashes':{str(p.relative_to(ROOT)):sha(p) for folder in ['src','vendor','protocol'] for p in (ROOT/folder).glob('*') if p.is_file()},'versions':{'numpy':np.__version__,'sklearn':sklearn.__version__,'xgboost':xgb.__version__},'feature_dimensions':{k:v.shape[1] for k,v in data.x.items()},'scope':'exploratory; no folds0/1 or external labels'})

def screen():
    out=ROOT/'runs/screen';out.mkdir(exist_ok=False);d=Data();preflight(out,d)
    tr=d.indices([2,3,4]);va=d.indices([5]);bo,to=d.targets([2,3,4]);bv,tv=d.pair([2,3,4],[5]);records=[]
    with threadpool_limits(limits=1):
        for feature in ['b1','context']:
            for family,params in [('ridge',[10,100,1000]),('xgb',[2,4])]:
                for param in params:
                    for arm,target in [('distilled',to-bo),('hard',d.y-bo)]:
                        m=regressor(family,param).fit(d.x[feature][tr],target[tr]);corr=m.predict(d.x[feature][va]);row={'features':feature,'family':family,'param':param,'arm':arm,'increment_correlation':float(np.corrcoef(corr,tv-bv)[0,1]),'increment_mse':float(np.mean((corr-(tv-bv))**2)),'scores':{str(g):metrics(d.y[va],score(bv,corr,g)) for g in [0,.25,.5,1]}}
                        records.append(row);write(out/f'{feature}_{family}_{param}_{arm}.json',row);print(feature,family,param,arm,'corr',round(row['increment_correlation'],3),'AUC', {k:round(v['auc'],4) for k,v in row['scores'].items()},flush=True)
    choices=[(v['auc'],row['features']=='b1',row['family']=='ridge',-float(g),row) for row in records if row['arm']=='distilled' for g,v in row['scores'].items()]
    winner=max(choices,key=lambda z:z[:4])[-1]
    result={'baseline':metrics(d.y[va],bv),'teacher':metrics(d.y[va],tv),'selected_features':winner['features'],'selected_family':winner['family'],'all_records':records,'scope':'mechanism screen only; fold5 selection, not evaluation evidence'}
    write(out/'result.json',result);print('SELECTED',winner['features'],winner['family'],flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):screen()
