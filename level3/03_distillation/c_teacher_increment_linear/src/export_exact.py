from increment import *
import joblib
import portable_exact as pt

def main():
    out=ROOT/'runs/engineering_final_exact';out.mkdir(exist_ok=False);d=Data();raw=np.load(ROOT/'runs/final_scale/private_predictions.npz');qa={};auc={}
    for fold in range(2,7):
        b=joblib.load(ROOT/f'runs/final_scale/outer_{fold}/distilled.joblib');bs,bm=b['baseline'];rs,rm=b['regressor']
        m={'schema':'context-increment-exact-scalers-v1','descriptor_names':b['descriptor_names'],'space':b['space'],'gamma':b['gamma'],'promoted':False}
        a={'base_mean':bs.mean_,'base_scale':bs.scale_,'base_coef':bm.coef_.ravel(),'base_intercept':bm.intercept_[0],'correction_mean':rs.mean_,'correction_scale':rs.scale_,'correction_coef':rm.coef_,'correction_intercept':rm.intercept_}
        dst=out/f'fold{fold}.npz';np.savez_compressed(dst,metadata_utf8=np.frombuffer(json.dumps(m).encode(),dtype=np.uint8),**a)
        m,a=pt.load(dst);idx=d.indices([fold]);p=pt.predict(m,a,[d.seq[i] for i in idx]);err=float(np.max(np.abs(p-raw['distilled'][idx])));assert err<1e-5,err;qa[str(fold)]=err;auc[str(fold)]=float(roc_auc_score(d.y[idx],p)-roc_auc_score(d.y[idx],raw['distilled'][idx]))
    path=out/'fold6.npz';m,a=pt.load(path);r={'scope':'original fitted scaler arithmetic, float32 correction/float64 baseline; full sequence-to-score; one BLAS thread; synthetic timing inputs; no teacher latency measured','portable_sha256':sha(path),'source_npz_bytes':path.stat().st_size,'inference_coefficients':sum(v.size for k,v in a.items() if k.endswith(('coef','intercept'))),'normalization_constants':sum(v.size for k,v in a.items() if k.endswith(('mean','scale'))),'array_bytes':sum(v.nbytes for v in a.values()),'parity_all_oof_rows':qa,'auc_delta_export_per_fold':auc,'timings':{},'batch':1,'warmups':20,'repeats':100,'platform':platform.platform(),'numpy':np.__version__}
    for length in [100,300,500,1000,2825]:
        seq=('ACDEFGHIKLMNPQRSTVWY'*((length+19)//20))[:length]
        for _ in range(20):pt.predict(m,a,[seq])
        times=[]
        for _ in range(100):
            t=time.perf_counter_ns();pt.predict(m,a,[seq]);times.append((time.perf_counter_ns()-t)/1e6)
        r['timings'][str(length)]={'median_ms':float(np.median(times)),'p95_ms':float(np.quantile(times,.95))}
    write(out/'result.json',r);print(json.dumps(r,indent=2))
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
