from increment import *
import joblib
from scipy.special import logit,expit

def target_vectors(y,b,t,space):
    if space=='probability':return t-b,y-b
    lb=logit(np.clip(b,1e-6,1-1e-6))
    return logit(np.clip(t,1e-6,1-1e-6))-lb,logit(.05+.9*y)-lb

def predict_score(b,corr,gamma,space):
    return score(b,corr,gamma) if space=='probability' else np.clip(expit(logit(np.clip(b,1e-6,1-1e-6))+gamma*corr),1e-6,1-1e-6)

def choose(rows,arm):
    keys=sorted(set((r['space'],r['param'],r['gamma']) for r in rows));means={k:float(np.mean([r[arm] for r in rows if (r['space'],r['param'],r['gamma'])==k])) for k in keys}
    best=max(means.values());k=min([k for k,v in means.items() if v>=best-1e-6],key=lambda k:(k[2],-k[1],k[0]!='probability'))
    return {'space':k[0],'param':k[1],'gamma':k[2],'inner_mean_auc':means[k]}

def main():
    assert (ROOT/'runs/nested/private_predictions.npz').exists(),'Complete Stage B outer fits first'
    out=ROOT/'runs/extended';out.mkdir(exist_ok=False);d=Data();preflight(out,d);x=d.x['context'];preds={k:np.full(len(d.y),np.nan) for k in ['b1','teacher','distilled','hard_matched','hard_tuned','shuffled']};details=[]
    for outer in range(2,7):
        od=out/f'outer_{outer}';od.mkdir();allowed=[f for f in range(2,7) if f!=outer];rows=[]
        for inner in allowed:
            train=[f for f in allowed if f!=inner];tr=d.indices(train);va=d.indices([inner]);bo,to=d.targets(train);bv,tv=d.pair(train,[inner])
            for space in ['probability','logit']:
                kd,hard=target_vectors(d.y,bo,to,space)
                for param in [1000,10000,100000]:
                    fits={arm:regressor('ridge',param).fit(x[tr],target[tr]).predict(x[va]) for arm,target in [('distilled',kd),('hard',hard)]}
                    for gamma in [0,.25,.5,1]:rows.append({'inner':inner,'space':space,'param':param,'gamma':gamma,**{arm:float(roc_auc_score(d.y[va],predict_score(bv,fits[arm],gamma,space))) for arm in fits}})
            print('extended outer',outer,'inner',inner,'done',flush=True)
        sel=choose(rows,'distilled');hs=choose(rows,'hard');write(od/'selection.json',{'distilled':sel,'hard':hs,'inner_results':rows})
        tr=d.indices(allowed);te=d.indices([outer]);bo,to=d.targets(allowed);bv,tv=d.pair(allowed,[outer]);preds['b1'][te]=bv;preds['teacher'][te]=tv;baseline=logistic(d.b[tr],d.y[tr])
        for arm,cfg in [('distilled',sel),('hard_matched',sel),('hard_tuned',hs),('shuffled',sel)]:
            kd,hard=target_vectors(d.y,bo,to,cfg['space']);target=hard[tr] if arm.startswith('hard') else kd[tr]
            if arm=='shuffled':target=np.random.default_rng(SEED+outer).permutation(target)
            fit=regressor('ridge',cfg['param']).fit(x[tr],target);preds[arm][te]=predict_score(bv,fit.predict(x[te]),cfg['gamma'],cfg['space'])
            joblib.dump({'feature':'context','family':'ridge','space':cfg['space'],'gamma':cfg['gamma'],'baseline':baseline,'regressor':fit,'descriptor_names':d.blocks['B0'][1],'promoted':False},od/(arm+'.joblib'),compress=3)
        r={'fold':outer,'n':len(te),'distilled_selection':sel,'hard_selection':hs,'metrics':{k:metrics(d.y[te],v[te]) for k,v in preds.items()}}
        write(od/'result.json',r);details.append(r);print('EXTENDED OUTER',outer,{k:round(v['auc'],5) for k,v in r['metrics'].items()},sel,flush=True)
    np.savez_compressed(out/'private_predictions.npz',row_id=d.d.record_id.to_numpy(dtype=str),fold=d.f,cluster=d.g.astype(str),label=d.y,**preds)
    m={k:metrics(d.y,v) for k,v in preds.items()};contrasts={k:paired_bootstrap(d.y,preds['distilled'],preds[k],d.g,seed=SEED) for k in ['b1','hard_matched','hard_tuned','shuffled']}
    gain=m['distilled']['auc']-m['b1']['auc'];retention=gain/(m['teacher']['auc']-m['b1']['auc']);passed=gain>=.01 and retention>=.2 and contrasts['b1']['ci95'][0]>0 and contrasts['hard_tuned']['ci95'][0]>0
    write(out/'result.json',{'scope':'exploratory amended nested development; not confirmation','n_rows':len(d.y),'n_groups':len(set(d.g)),'metrics':m,'distilled_minus':contrasts,'teacher_gain_retention':retention,'research_progression_gate_passed':bool(passed),'historical_80_percent_retention_met':bool(retention>=.8),'promoted':False,'outer_results':details})
    print('EXTENDED DONE',{k:v['auc'] for k,v in m.items()},'retention',retention,'gate',passed,flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
