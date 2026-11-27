from increment import *
from scipy.special import logit,expit
from scipy.spatial.distance import cdist
from strong_control import fit as fit_logistic
import joblib

class Map:
    def fit(self,x,tr,width):
        self.columns=np.r_[np.arange(25),np.arange(425,545),np.arange(x.shape[1]-353,x.shape[1])]
        self.scaler=StandardScaler().fit(x[tr][:,self.columns]);z=np.clip(self.scaler.transform(x[:,self.columns]),-5,5)
        ii=np.random.default_rng(SEED).choice(tr,min(512,len(tr)),replace=False);self.anchors=z[ii].copy();self.width=width;return self
    def transform(self,x):
        z=np.clip(self.scaler.transform(x[:,self.columns]),-5,5);k=np.exp(-cdist(z,self.anchors,'sqeuclidean')/(2*len(self.columns)*self.width**2))
        return np.column_stack([x,k]).astype(np.float32)

def lg(p):return logit(np.clip(p,1e-6,1-1e-6))
def response(b,c,g):return np.clip(expit(lg(b)+g*c),1e-6,1-1e-6)
def blend(b,p,w):return np.clip(expit((1-w)*lg(b)+w*lg(p)),1e-6,1-1e-6)
def choose(rows,arm):
    keys=set((r['width'],r['param'],r['gamma']) for r in rows if arm in r);means={k:float(np.mean([r[arm] for r in rows if arm in r and (r['width'],r['param'],r['gamma'])==k])) for k in keys};best=max(means.values());k=min([k for k,v in means.items() if v>=best-1e-6],key=lambda k:(k[2],-k[1],-k[0]))
    return {'width':k[0],'param':k[1],'gamma':k[2],'inner_auc':means[k]}

def choose_log(rows):
    keys=set((r['width'],r['C'],r['w']) for r in rows);means={k:float(np.mean([r['auc'] for r in rows if (r['width'],r['C'],r['w'])==k])) for k in keys};best=max(means.values());k=min([k for k,v in means.items() if v>=best-1e-6],key=lambda k:(k[2],k[1],-k[0]));return {'width':k[0],'C':k[1],'w':k[2],'inner_auc':means[k]}

def main():
    out=ROOT/'runs/nested';out.mkdir(exist_ok=False);d=Data();preflight(out,d);x=d.x['context'];preds={k:np.full(len(d.y),np.nan) for k in ['b1','teacher','distilled','hard_matched','hard_tuned','shuffled','nonlinear_logistic']};details=[]
    for outer in range(2,7):
        od=out/f'outer_{outer}';od.mkdir();allowed=[f for f in range(2,7) if f!=outer];rows=[];lr=[]
        for inner in allowed:
            train=[f for f in allowed if f!=inner];tr=d.indices(train);va=d.indices([inner]);bo,to=d.targets(train);bv,tv=d.pair(train,[inner]);kd=lg(to)-lg(bo);hard=logit(.05+.9*d.y)-lg(bo)
            for width in [.5,1,2]:
                mapper=Map().fit(x,tr,width);z=mapper.transform(x)
                for param in [1000,10000,30000]:
                    fits={arm:regressor('ridge',param).fit(z[tr],target[tr]).predict(z[va]) for arm,target in [('distilled',kd),('hard',hard)]}
                    for gamma in [0,.5,1,2,4]:rows.append({'inner':inner,'width':width,'param':param,'gamma':gamma,**{arm:float(roc_auc_score(d.y[va],response(bv,v,gamma))) for arm,v in fits.items()}})
                for c in [.0001,.001,.01,.1]:
                    p=fit_logistic(z[tr],d.y[tr],c).predict_proba(z[va])[:,1]
                    for w in [0,.25,.5,.75,1]:lr.append({'inner':inner,'width':width,'C':c,'w':w,'auc':float(roc_auc_score(d.y[va],blend(bv,p,w)))})
            print('nonlinear outer',outer,'inner',inner,'done',flush=True)
        sel=choose(rows,'distilled');hs=choose(rows,'hard');ls=choose_log(lr);write(od/'selection.json',{'distilled':sel,'hard':hs,'logistic':ls,'inner_results':rows,'logistic_inner':lr});tr=d.indices(allowed);te=d.indices([outer]);bo,to=d.targets(allowed);bv,tv=d.pair(allowed,[outer]);preds['b1'][te]=bv;preds['teacher'][te]=tv;baseline=logistic(d.b[tr],d.y[tr])
        maps={width:Map().fit(x,tr,width) for width in set([sel['width'],hs['width'],ls['width']])}
        for arm,cfg in [('distilled',sel),('hard_matched',sel),('hard_tuned',hs),('shuffled',sel)]:
            target=(logit(.05+.9*d.y)-lg(bo))[tr] if arm.startswith('hard') else (lg(to)-lg(bo))[tr]
            if arm=='shuffled':target=np.random.default_rng(SEED+outer).permutation(target)
            mapper=maps[cfg['width']];z=mapper.transform(x);fit=regressor('ridge',cfg['param']).fit(z[tr],target);preds[arm][te]=response(bv,fit.predict(z[te]),cfg['gamma'])
            joblib.dump({'mapper':mapper,'gamma':cfg['gamma'],'baseline':baseline,'regressor':fit,'descriptor_names':d.blocks['B0'][1],'promoted':False},od/(arm+'.joblib'),compress=3)
        z=maps[ls['width']].transform(x);p=fit_logistic(z[tr],d.y[tr],ls['C']).predict_proba(z[te])[:,1];preds['nonlinear_logistic'][te]=blend(bv,p,ls['w'])
        r={'fold':outer,'selections':{'distilled':sel,'hard':hs,'logistic':ls},'metrics':{k:metrics(d.y[te],v[te]) for k,v in preds.items()}};write(od/'result.json',r);details.append(r);print('NONLINEAR OUTER',outer,{k:v['auc'] for k,v in r['metrics'].items()},sel,flush=True)
    old=np.load(ROOT/'runs/prior_baseline_private_predictions.npz');assert np.array_equal(old['row_id'],d.d.record_id.to_numpy(dtype=str));preds['prior_transparent']=old['selected'];np.savez_compressed(out/'private_predictions.npz',row_id=d.d.record_id.to_numpy(dtype=str),cluster=d.g.astype(str),fold=d.f,label=d.y,**preds)
    m={k:metrics(d.y,v) for k,v in preds.items()};contrasts={k:paired_bootstrap(d.y,preds['distilled'],preds[k],d.g,seed=SEED) for k in ['b1','hard_tuned','nonlinear_logistic','prior_transparent']};ret=(m['distilled']['auc']-m['b1']['auc'])/(m['teacher']['auc']-m['b1']['auc']);passed=m['distilled']['auc']-m['b1']['auc']>=.01 and ret>=.2 and all(contrasts[k]['ci95'][0]>0 for k in ['nonlinear_logistic','prior_transparent'])
    write(out/'result.json',{'scope':'exploratory nonlinear cycle; no independent confirmation','metrics':m,'distilled_minus':contrasts,'teacher_gain_retention':ret,'research_gate_passed':bool(passed),'promoted':False,'outer_results':details});print('NONLINEAR DONE',{k:v['auc'] for k,v in m.items()},ret,passed,flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
