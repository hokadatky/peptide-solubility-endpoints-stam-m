from increment import *
import joblib
import portable

def export(bundle,path):
    assert bundle['family']=='ridge'
    bs,bm=bundle['baseline'];rs,rm=bundle['regressor']
    bc=bm.coef_.ravel()/bs.scale_;bi=float(bm.intercept_[0]-bs.mean_@bc)
    rc=rm.coef_.astype(float)/rs.scale_;ri=float(rm.intercept_-rs.mean_@rc)
    meta={'schema':'crossfit-increment-ridge-v1','features':bundle['feature'],'gamma':bundle['gamma'],'descriptor_names':bundle['descriptor_names'],'promoted':False,'dtype':'float64 folded coefficients; context inputs float32','endpoint':'recombinant soluble expression in E. coli'}
    np.savez_compressed(path,metadata_utf8=np.frombuffer(json.dumps(meta).encode(),dtype=np.uint8),base_coef=bc,base_intercept=bi,correction_coef=rc,correction_intercept=ri)

def main():
    out=ROOT/'runs/engineering';out.mkdir(exist_ok=False);d=Data();raw=np.load(ROOT/'runs/nested/private_predictions.npz');parity={}
    for fold in range(2,7):
        p=ROOT/f'runs/nested/outer_{fold}/distilled.joblib';dst=out/f'distilled_fold{fold}.npz';bundle=joblib.load(p);export(bundle,dst);m,a=portable.load(dst)
        idx=d.indices([fold]);got=portable.predict(m,a,[d.seq[i] for i in idx]);delta=float(np.max(np.abs(got-raw['distilled'][idx])));assert delta<1e-5,delta;parity[str(fold)]=delta
    # Predeclared representative fold6, independent of best outer result.
    path=out/'distilled_fold6.npz';m,a=portable.load(path)
    report={'source_sha256':sha(ROOT/'runs/nested/outer_6/distilled.joblib'),'portable_sha256':sha(path),'source_npz_bytes':path.stat().st_size,'coefficient_count':sum(v.size for v in a.values()),'array_bytes':sum(v.nbytes for v in a.values()),'parity_all_outer_rows':parity,'batch':1,'warmups':20,'repeats':100,'platform':platform.platform(),'numpy':np.__version__,'timings':{},'scope':'one BLAS thread; complete sequence-to-score; synthetic benchmark; no teacher timing'}
    with threadpool_limits(limits=1):
        for length in [100,300,500,1000,2825]:
            seq=(portable.AA*((length+19)//20))[:length]
            for _ in range(20):portable.predict(m,a,[seq])
            ms=[]
            for _ in range(100):
                t=time.perf_counter_ns();portable.predict(m,a,[seq]);ms.append((time.perf_counter_ns()-t)/1e6)
            report['timings'][str(length)]={'median_ms':float(np.median(ms)),'p95_ms':float(np.quantile(ms,.95))}
    write(out/'result.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
