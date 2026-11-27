from increment import *

def fit(x,y,c):
    m=make_pipeline(StandardScaler(),LogisticRegression(C=c,solver='lbfgs',max_iter=20000,tol=1e-4))
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning);m.fit(x,y)
    return m

def main():
    out=ROOT/'runs/strong_control';out.mkdir(exist_ok=False);d=Data();preflight(out,d);x=d.x['context'];p=np.full(len(d.y),np.nan);details=[]
    for outer in range(2,7):
        allowed=[f for f in range(2,7) if f!=outer];rows=[]
        for inner in allowed:
            tr=d.indices([f for f in allowed if f!=inner]);va=d.indices([inner])
            for c in [.0001,.001,.01,.1,1,10]:rows.append({'inner':inner,'C':c,'auc':float(roc_auc_score(d.y[va],fit(x[tr],d.y[tr],c).predict_proba(x[va])[:,1]))})
        avg={c:float(np.mean([r['auc'] for r in rows if r['C']==c])) for c in [.0001,.001,.01,.1,1,10]};best=max(avg.values());chosen=min(c for c,v in avg.items() if v>=best-1e-6)
        tr=d.indices(allowed);te=d.indices([outer]);m=fit(x[tr],d.y[tr],chosen);p[te]=m.predict_proba(x[te])[:,1]
        row={'fold':outer,'selected_C':chosen,'inner':rows,'metrics':metrics(d.y[te],p[te])};details.append(row);write(out/f'outer_{outer}.json',row);print('STRONG',outer,chosen,row['metrics']['auc'],flush=True)
    np.savez_compressed(out/'private_predictions.npz',row_id=d.d.record_id.to_numpy(dtype=str),fold=d.f,cluster=d.g.astype(str),label=d.y,prediction=p)
    raw=np.load(ROOT/'runs/extended/private_predictions.npz');assert np.array_equal(raw['row_id'],d.d.record_id.to_numpy(dtype=str))
    contrast=paired_bootstrap(d.y,raw['distilled'],p,d.g,seed=SEED)
    write(out/'result.json',{'scope':'exploratory stronger hard-label control; student unchanged','metrics':metrics(d.y,p),'distilled_minus_context_logistic':contrast,'outer_results':details});print('STRONG DONE',metrics(d.y,p)['auc'],contrast,flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
