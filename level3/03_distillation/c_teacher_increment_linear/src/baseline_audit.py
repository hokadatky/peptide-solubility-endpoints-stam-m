from strong_control import fit
from increment import *
from scipy.special import logit,expit

def blend(b,p,w):return np.clip(expit((1-w)*logit(np.clip(b,1e-6,1-1e-6))+w*logit(np.clip(p,1e-6,1-1e-6))),1e-6,1-1e-6)

def pick(rows,family=None):
    rows=[r for r in rows if family is None or r['family']==family];keys=set((r['family'],r['C'],r['w']) for r in rows)
    av={k:float(np.mean([r['auc'] for r in rows if (r['family'],r['C'],r['w'])==k])) for k in keys};best=max(av.values());k=min([k for k,v in av.items() if v>=best-1e-6],key=lambda k:(k[1],k[2],k[0]))
    return {'family':k[0],'C':k[1],'w':k[2],'inner_auc':av[k]}

def main():
    out=ROOT/'runs/baseline_audit';out.mkdir(exist_ok=False);d=Data();preflight(out,d);pred={k:np.full(len(d.y),np.nan) for k in ['b1_tuned','context_blend','selected']};details=[]
    for outer in range(2,7):
        allowed=[f for f in range(2,7) if f!=outer];rows=[]
        for inner in allowed:
            train=[f for f in allowed if f!=inner];tr=d.indices(train);va=d.indices([inner]);bv,_=d.pair(train,[inner])
            for c in [.0001,.001,.01,.1,1,10,1000]:
                p=fit(d.b[tr],d.y[tr],c).predict_proba(d.b[va])[:,1];rows.append({'inner':inner,'family':'b1','C':c,'w':0,'auc':float(roc_auc_score(d.y[va],p))})
            for c in [.0001,.001,.01,.1,1,10]:
                p=fit(d.x['context'][tr],d.y[tr],c).predict_proba(d.x['context'][va])[:,1]
                for w in [0,.25,.5,.75,1]:rows.append({'inner':inner,'family':'blend','C':c,'w':w,'auc':float(roc_auc_score(d.y[va],blend(bv,p,w)))})
        choices={'b1_tuned':pick(rows,'b1'),'context_blend':pick(rows,'blend'),'selected':pick(rows)};tr=d.indices(allowed);te=d.indices([outer]);bv,_=d.pair(allowed,[outer])
        for arm,cfg in choices.items():
            x=d.b if cfg['family']=='b1' else d.x['context'];p=fit(x[tr],d.y[tr],cfg['C']).predict_proba(x[te])[:,1];pred[arm][te]=p if cfg['family']=='b1' else blend(bv,p,cfg['w'])
        r={'fold':outer,'selection':choices,'inner':rows,'metrics':{k:metrics(d.y,v[te]) if False else metrics(d.y[te],v[te]) for k,v in pred.items()}};write(out/f'outer_{outer}.json',r);details.append(r);print('BASELINE',outer,{k:v['auc'] for k,v in r['metrics'].items()},choices['selected'],flush=True)
    np.savez_compressed(out/'private_predictions.npz',row_id=d.d.record_id.to_numpy(dtype=str),label=d.y,**pred);raw=np.load(ROOT/'runs/final_scale/private_predictions.npz');assert np.array_equal(raw['row_id'],d.d.record_id.to_numpy(dtype=str))
    contrast=paired_bootstrap(d.y,raw['distilled'],pred['selected'],d.g,seed=SEED)
    write(out/'result.json',{'scope':'exploratory sensitivity to stronger transparent baselines; student fixed','metrics':{k:metrics(d.y,v) for k,v in pred.items()},'distilled_minus_selected_transparent':contrast,'outer_results':details});print('BASELINE DONE',contrast,flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
