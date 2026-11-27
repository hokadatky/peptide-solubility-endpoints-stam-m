from increment import *
import joblib
import portable_logit as pt

def main():
    out=ROOT/'runs/engineering_final';out.mkdir(exist_ok=False);d=Data();raw=np.load(ROOT/'runs/final_scale/private_predictions.npz');qa={};qa32={}
    for fold in range(2,7):
        bundle=joblib.load(ROOT/f'runs/final_scale/outer_{fold}/distilled.joblib');assert bundle['space']=='logit'
        bs,bm=bundle['baseline'];rs,rm=bundle['regressor'];bc=bm.coef_.ravel()/bs.scale_;bi=float(bm.intercept_[0]-bs.mean_@bc)
        rc=rm.coef_.astype(float)/rs.scale_;ri=float(rm.intercept_-rs.mean_@rc);coef=bundle['gamma']*rc;coef[:425]+=bc;inter=bi+bundle['gamma']*ri
        meta={'schema':'fused-context-logit-increment-v1','descriptor_names':bundle['descriptor_names'],'promoted':False,'endpoint':'recombinant soluble expression in E. coli'}
        mbytes=np.frombuffer(json.dumps(meta).encode(),dtype=np.uint8)
        for dtype in ['float64','float32']:
            dst=out/f'fold{fold}_{dtype}.npz';np.savez_compressed(dst,metadata_utf8=mbytes,coef=coef.astype(dtype),intercept=np.asarray(inter,dtype=dtype))
            m,a=pt.load(dst);idx=d.indices([fold]);p=pt.predict(m,a,[d.seq[i] for i in idx]);err=float(np.max(np.abs(p-raw['distilled'][idx])));assert dtype=='float32' or err<1e-5; (qa if dtype=='float64' else qa32)[str(fold)]=err
    path=out/'fold6_float64.npz';m,a=pt.load(path);report={'source_npz_bytes':path.stat().st_size,'float32_npz_bytes':(out/'fold6_float32.npz').stat().st_size,'portable_sha256':sha(path),'inference_parameters':sum(v.size for v in a.values()),'array_bytes':sum(v.nbytes for v in a.values()),'parity_float64':qa,'parity_float32_all_oof_rows':qa32,'timings':{},'batch':1,'warmups':20,'repeats':100,'platform':platform.platform(),'numpy':np.__version__,'scope':'primary float64 coefficient storage; NumPy mixed precision input features, float64 accumulation; one BLAS thread; complete sequence-to-score, synthetic inputs'}
    with threadpool_limits(limits=1):
        for length in [100,300,500,1000,2825]:
            seq=(pt.features.__globals__['AA']*((length+19)//20))[:length]
            for _ in range(20):pt.predict(m,a,[seq])
            ms=[]
            for _ in range(100):
                t=time.perf_counter_ns();pt.predict(m,a,[seq]);ms.append((time.perf_counter_ns()-t)/1e6)
            report['timings'][str(length)]={'median_ms':float(np.median(ms)),'p95_ms':float(np.quantile(ms,.95))}
    write(out/'result.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
