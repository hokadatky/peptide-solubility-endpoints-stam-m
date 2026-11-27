from increment import *
import joblib

def choose(rows,arm):
    keys=sorted(set((r['param'],r['gamma']) for r in rows))
    avg={key:float(np.mean([r[arm] for r in rows if (r['param'],r['gamma'])==key])) for key in keys}
    best=max(avg.values());eligible=[k for k,v in avg.items() if v>=best-1e-6]
    # gamma first, then stronger ridge regularization or shallower XGBoost.
    winner=min(eligible,key=lambda k:(k[1],-k[0] if rows[0]['family']=='ridge' else k[0]))
    return {'param':winner[0],'gamma':winner[1],'inner_mean_auc':avg[winner]}

def main():
    out=ROOT/'runs/nested';out.mkdir(exist_ok=False);d=Data();preflight(out,d)
    screen=json.loads((ROOT/'runs/screen/result.json').read_text());family=screen['selected_family'];feature=screen['selected_features'];params=[10,100,1000] if family=='ridge' else [2,4];x=d.x[feature]
    preds={k:np.full(len(d.y),np.nan) for k in ['b1','teacher','distilled','hard_matched','hard_tuned','shuffled']};details=[]
    for outer in range(2,7):
        od=out/f'outer_{outer}';od.mkdir();allowed=[f for f in range(2,7) if f!=outer];rows=[]
        for inner in allowed:
            train=[f for f in allowed if f!=inner];tr=d.indices(train);va=d.indices([inner]);bo,to=d.targets(train);bv,tv=d.pair(train,[inner])
            fits={}
            for param in params:
                for arm,target in [('distilled',to-bo),('hard',d.y-bo)]:fits[(param,arm)]=regressor(family,param).fit(x[tr],target[tr]).predict(x[va])
                for gamma in [0,.25,.5,1]:rows.append({'family':family,'inner':inner,'param':param,'gamma':gamma,**{arm:float(roc_auc_score(d.y[va],score(bv,fits[(param,arm)],gamma))) for arm in ['distilled','hard']}})
            print('outer',outer,'inner',inner,'done',flush=True)
        sel=choose(rows,'distilled');hard=choose(rows,'hard');write(od/'selection.json',{'family':family,'feature':feature,'distilled':sel,'hard':hard,'inner_results':rows})
        tr=d.indices(allowed);te=d.indices([outer]);bo,to=d.targets(allowed);bv,tv=d.pair(allowed,[outer]);preds['b1'][te]=bv;preds['teacher'][te]=tv
        baseline=logistic(d.b[tr],d.y[tr]);np.testing.assert_allclose(baseline.predict_proba(d.b[te])[:,1],bv,rtol=1e-10)
        delta=to[tr]-bo[tr];shuffled=np.random.default_rng(SEED+outer).permutation(delta)
        for arm,target,config in [('distilled',delta,sel),('hard_matched',d.y[tr]-bo[tr],sel),('hard_tuned',d.y[tr]-bo[tr],hard),('shuffled',shuffled,sel)]:
            fit=regressor(family,config['param']).fit(x[tr],target);corr=fit.predict(x[te]);preds[arm][te]=score(bv,corr,config['gamma'])
            joblib.dump({'feature':feature,'family':family,'gamma':config['gamma'],'baseline':baseline,'regressor':fit,'descriptor_names':d.blocks['B0'][1],'promoted':False},od/(arm+'.joblib'),compress=3)
        r={'fold':outer,'n':len(te),'distilled_selection':sel,'hard_selection':hard,'metrics':{k:metrics(d.y[te],v[te]) for k,v in preds.items()},'increment_correlation':float(np.corrcoef((preds['distilled'][te]-bv),tv-bv)[0,1]) if sel['gamma'] else None}
        write(od/'result.json',r);details.append(r);print('OUTER',outer,{k:round(v['auc'],5) for k,v in r['metrics'].items()},flush=True)
    np.savez_compressed(out/'private_predictions.npz',row_id=d.d.record_id.to_numpy(dtype=str),fold=d.f,cluster=d.g.astype(str),label=d.y,**preds)
    m={k:metrics(d.y,v) for k,v in preds.items()};contrasts={k:paired_bootstrap(d.y,preds['distilled'],preds[k],d.g,seed=SEED) for k in ['b1','hard_matched','hard_tuned','shuffled']}
    gain=m['distilled']['auc']-m['b1']['auc'];teacher_gain=m['teacher']['auc']-m['b1']['auc'];retention=gain/teacher_gain
    passed=gain>=.01 and retention>=.2 and contrasts['b1']['ci95'][0]>0 and contrasts['hard_tuned']['ci95'][0]>0
    r={'scope':'exploratory data-informed nested evaluation; not confirmation','n_rows':len(d.y),'n_groups':len(set(d.g)),'family':family,'features':feature,'metrics':m,'distilled_minus':contrasts,'teacher_gain_retention':retention,'research_progression_gate_passed':bool(passed),'historical_80_percent_retention_met':bool(retention>=.8),'promoted':False,'outer_results':details}
    write(out/'result.json',r);print('DONE', {k:v['auc'] for k,v in m.items()},'retention',retention,'gate',passed,flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
