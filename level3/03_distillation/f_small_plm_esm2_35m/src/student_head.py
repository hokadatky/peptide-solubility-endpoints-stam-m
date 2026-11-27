from increment import *
import joblib

def soft_fit(x,y,q,c):
    sc=StandardScaler().fit(x);z=sc.transform(x).astype(float);m=LogisticRegression(C=c,solver='lbfgs',max_iter=20000,tol=1e-4)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        if np.array_equal(q,y):m.fit(z,y)
        else:m.fit(np.vstack([z,z]),np.r_[np.ones(len(y)),np.zeros(len(y))],sample_weight=np.r_[q,1-q])
    return make_pipeline(sc,m)

def choose(rows,soft):
    rows=[r for r in rows if (r['alpha']>0)==soft];keys=set((r['alpha'],r['C']) for r in rows);avg={k:float(np.mean([r['auc'] for r in rows if (r['alpha'],r['C'])==k])) for k in keys};best=max(avg.values());a,c=min([k for k,v in avg.items() if v>=best-1e-6],key=lambda k:(k[1],k[0]));return {'alpha':a,'C':c,'inner_auc':avg[(a,c)]}

def main():
    assert (ROOT/'runs/ENCODER_IMPORT_QA.json').exists(),'Encoder import must be verified first'
    out=ROOT/'runs/nested';out.mkdir(exist_ok=False);d=Data();preflight(out,d)
    with np.load(ROOT/'runs/esm35_development.npz') as f:assert np.array_equal(f['row_id'],d.d.record_id.to_numpy(dtype=str));x=np.column_stack([d.b,f['embeddings']])
    preds={k:np.full(len(d.y),np.nan) for k in ['b1','teacher','distilled','hard_tuned','hard_matched']};details=[]
    for outer in range(2,7):
        od=out/f'outer_{outer}';od.mkdir();allowed=[f for f in range(2,7) if f!=outer];rows=[]
        for inner in allowed:
            train=[f for f in allowed if f!=inner];tr=d.indices(train);va=d.indices([inner]);_,t=d.targets(train)
            for alpha in [0,.25,.5,.75,1]:
                q=(1-alpha)*d.y[tr]+alpha*t[tr]
                for c in [.0001,.001,.01,.1,1,10]:
                    fit=soft_fit(x[tr],d.y[tr],q,c);p=fit.predict_proba(x[va])[:,1];rows.append({'inner':inner,'alpha':alpha,'C':c,'auc':float(roc_auc_score(d.y[va],p))})
            print('semantic outer',outer,'inner',inner,'done',flush=True)
        sel=choose(rows,True);hs=choose(rows,False);write(od/'selection.json',{'distilled':sel,'hard':hs,'inner':rows})
        tr=d.indices(allowed);te=d.indices([outer]);_,t=d.targets(allowed);bv,tv=d.pair(allowed,[outer]);preds['b1'][te]=bv;preds['teacher'][te]=tv
        for arm,cfg in [('distilled',sel),('hard_tuned',hs),('hard_matched',{'alpha':0,'C':sel['C']})]:
            q=(1-cfg['alpha'])*d.y[tr]+cfg['alpha']*t[tr];fit=soft_fit(x[tr],d.y[tr],q,cfg['C']);preds[arm][te]=fit.predict_proba(x[te])[:,1];joblib.dump(fit,od/(arm+'.joblib'),compress=3)
        r={'fold':outer,'selection':sel,'hard_selection':hs,'metrics':{k:metrics(d.y[te],v[te]) for k,v in preds.items()}};write(od/'result.json',r);details.append(r);print('SEMANTIC OUTER',outer,{k:v['auc'] for k,v in r['metrics'].items()},sel,hs,flush=True)
    np.savez_compressed(out/'private_predictions.npz',row_id=d.d.record_id.to_numpy(dtype=str),cluster=d.g.astype(str),fold=d.f,label=d.y,**preds)
    m={k:metrics(d.y,v) for k,v in preds.items()};cs={k:paired_bootstrap(d.y,preds['distilled'],preds[k],d.g,seed=SEED) for k in ['b1','hard_tuned','hard_matched']};ret=(m['distilled']['auc']-m['b1']['auc'])/(m['teacher']['auc']-m['b1']['auc'])
    write(out/'result.json',{'scope':'exploratory frozen semantic student; not independent confirmation','metrics':m,'distilled_minus':cs,'teacher_gain_retention':ret,'research_gate_passed':bool(ret>=.8 and cs['hard_tuned']['ci95'][0]>0),'original_5MiB_gate_passed':False,'promoted':False,'outer_results':details});print('SEMANTIC DONE',{k:v['auc'] for k,v in m.items()},ret,cs['hard_tuned'],flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
