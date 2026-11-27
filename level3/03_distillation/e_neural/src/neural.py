from increment import *
import torch
from scipy.special import expit,logit
import joblib
torch.set_num_threads(1)

def lg(p):return logit(np.clip(p,1e-6,1-1e-6))
def model(seed):
    torch.manual_seed(seed);m=torch.nn.Sequential(torch.nn.Linear(2498,64),torch.nn.SiLU(),torch.nn.Dropout(.3),torch.nn.Linear(64,1))
    torch.nn.init.zeros_(m[-1].weight);torch.nn.init.zeros_(m[-1].bias);return m

def fit(x,y,t,bo,bv,tr,va,alpha,epochs=40,early=True,seed=SEED):
    sc=StandardScaler().fit(x[tr]);tx=torch.from_numpy(np.clip(sc.transform(x),-5,5).astype(np.float32));target=torch.from_numpy(((1-alpha)*y+alpha*t)[tr].astype(np.float32));offset=torch.from_numpy(lg(bo[tr]).astype(np.float32));m=model(seed);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.1)
    best=roc_auc_score(y[va],bv);best_epoch=0;stale=0;history=[]
    for ep in range(1,epochs+1):
        m.train();order=np.random.default_rng(seed+ep-1).permutation(len(tr));total=0.
        for start in range(0,len(tr),256):
            ii=order[start:start+256];z=m(tx[tr[ii]]).ravel()+offset[ii];loss=torch.nn.functional.binary_cross_entropy_with_logits(z,target[ii]);assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();total+=float(loss.detach())*len(ii)
        r={'epoch':ep,'training_loss':total/len(tr)}
        if early:
            m.eval()
            with torch.inference_mode():p=expit(lg(bv)+m(tx[va]).ravel().numpy().astype(float))
            auc=roc_auc_score(y[va],p);r['validation_auc']=auc
            if auc>best+.0001:best=auc;best_epoch=ep;stale=0
            else:stale+=1
        history.append(r)
        if early and stale>=5:break
    return m.eval(),sc,{'best_epoch':best_epoch if early else epochs,'best_auc':float(best) if early else None,'history':history}

def main():
    out=ROOT/'runs/nested';out.mkdir(exist_ok=False);d=Data();preflight(out,d);write(out/'torch_environment.json',{'version':torch.__version__,'threads':torch.get_num_threads()});x=d.x['context'];preds={k:np.full(len(d.y),np.nan) for k in ['b1','teacher','distilled','hard_tuned','hard_matched']};seedpreds={f'{arm}_{s}':np.full(len(d.y),np.nan) for arm in ['distilled','hard_tuned','hard_matched'] for s in [SEED,SEED+1]};details=[]
    for outer in range(2,7):
        od=out/f'outer_{outer}';od.mkdir();allowed=[f for f in range(2,7) if f!=outer];select={a:[] for a in [0,.5,1]}
        for inner in allowed:
            train=[f for f in allowed if f!=inner];tr=d.indices(train);va=d.indices([inner]);bo,to=d.targets(train);bv,tv=d.pair(train,[inner])
            for alpha in [0,.5,1]:
                _,_,r=fit(x,d.y,to,bo,bv,tr,va,alpha);select[alpha].append({'inner':inner,**r})
                print('neural outer',outer,'inner',inner,'alpha',alpha,'epoch',r['best_epoch'],'auc',r['best_auc'],flush=True)
        alpha=min([.5,1],key=lambda a:(-np.mean([r['best_auc'] for r in select[a]]),a));budget=lambda a:int(np.floor(np.median([r['best_epoch'] for r in select[a]])+.5));dep=budget(alpha);hep=budget(0)
        write(od/'selection.json',{'alpha':alpha,'distilled_epochs':dep,'hard_epochs':hep,'inner':{str(k):v for k,v in select.items()}})
        tr=d.indices(allowed);te=d.indices([outer]);bo,to=d.targets(allowed);bv,tv=d.pair(allowed,[outer]);preds['b1'][te]=bv;preds['teacher'][te]=tv;baseline=logistic(d.b[tr],d.y[tr])
        for arm,a,ep in [('distilled',alpha,dep),('hard_tuned',0,hep),('hard_matched',0,dep)]:
            pp=[]
            for seed in [SEED,SEED+1]:
                m,sc,r=fit(x,d.y,to,bo,bv,tr,te,a,epochs=ep,early=False,seed=seed)
                with torch.inference_mode():p=expit(lg(bv)+m(torch.from_numpy(np.clip(sc.transform(x[te]),-5,5).astype(np.float32))).ravel().numpy().astype(float))
                pp.append(p);seedpreds[f'{arm}_{seed}'][te]=p
                joblib.dump({'state':{k:v.detach().numpy() for k,v in m.state_dict().items()},'scaler':sc,'baseline':baseline,'descriptor_names':d.blocks['B0'][1],'seed':seed,'alpha':a,'epochs':ep,'promoted':False},od/f'{arm}_{seed}.joblib',compress=3)
                write(od/f'{arm}_{seed}_history.json',r)
            preds[arm][te]=np.mean(pp,axis=0)
        r={'fold':outer,'alpha':alpha,'distilled_epochs':dep,'hard_epochs':hep,'metrics':{k:metrics(d.y[te],v[te]) for k,v in preds.items()}};write(od/'result.json',r);details.append(r);print('NEURAL OUTER',outer,{k:v['auc'] for k,v in r['metrics'].items()},flush=True)
    old=np.load(ROOT/'runs/prior_baseline.npz');assert np.array_equal(old['row_id'],d.d.record_id.to_numpy(dtype=str));preds['transparent_blend']=old['selected'];np.savez_compressed(out/'private_predictions.npz',row_id=d.d.record_id.to_numpy(dtype=str),cluster=d.g.astype(str),fold=d.f,label=d.y,**preds,**seedpreds)
    m={k:metrics(d.y,v) for k,v in preds.items()};cs={k:paired_bootstrap(d.y,preds['distilled'],preds[k],d.g,seed=SEED) for k in ['b1','hard_tuned','transparent_blend']};ret=(m['distilled']['auc']-m['b1']['auc'])/(m['teacher']['auc']-m['b1']['auc']);write(out/'result.json',{'scope':'exploratory neural cycle, not confirmation','metrics':m,'seed_metrics':{k:metrics(d.y,v) for k,v in seedpreds.items()},'distilled_minus':cs,'teacher_gain_retention':ret,'promoted':False,'outer_results':details});print('NEURAL DONE',{k:v['auc'] for k,v in m.items()},ret,flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
