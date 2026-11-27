import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
from runtime import *
import time,hashlib,platform,pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parents[1]
SNAP=Path.home()/'.cache/huggingface/hub/models--facebook--esm2_t12_35M_UR50D/snapshots/6fbf070e65b0b7291e7bbcd451118c216cff79d8'

def main():
    torch.set_num_threads(1);out=ROOT/'runs/engineering';out.mkdir(exist_ok=False);s=Student(ROOT/'runs/encoder_export',SNAP,ROOT/'runs/head_export/fold6.npz');raw=pd.read_csv(BASE/'data/solubility/soluprot_train_partitions.csv');raw=raw[raw.development_fold.isin([2,3,4,5,6])].reset_index(drop=True);pr=np.load(ROOT/'runs/nested/private_predictions.npz');assert np.array_equal(pr['row_id'],raw.record_id.to_numpy(dtype=str))
    ids=np.flatnonzero(raw.development_fold.to_numpy()==6);checks=[int(ids[0]),int(ids[len(ids)//2]),int(ids[-1]),int(ids[np.argmax(raw.iloc[ids].sequence.str.len().to_numpy())])];errs=[]
    for i in checks:errs.append(float(abs(s.predict([raw.iloc[i].sequence])[0]-pr['distilled'][i])))
    assert max(errs)<1e-4,errs
    enc=json.loads((ROOT/'runs/encoder_export/result.json').read_text());head=json.loads((ROOT/'runs/head_export/result.json').read_text())
    r={'scope':'one CPU thread; complete sequence-to-score with frozen ESM35; no teacher in runtime; synthetic repeated-alphabet inputs','torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),'full_parameters':enc['active_encoder_parameters']+906,'encoder_fp32_bytes':enc['artifacts']['float32']['bytes'],'head_npz_bytes':head['fold6_npz_bytes'],'full_weight_artifact_bytes':enc['artifacts']['float32']['bytes']+head['fold6_npz_bytes'],'prediction_array_parity_head':head['all_oof_head_parity'],'fresh_raw_sequence_probe_probability_errors':errs,'fresh_probe_count':len(errs),'warmups':3,'repeats':10,'batch':1,'timings':{}}
    for length in [100,300,500,1022]:
        seq=(AA*((length+19)//20))[:length]
        for _ in range(3):s.predict([seq])
        ms=[]
        for _ in range(10):
            t=time.perf_counter_ns();s.predict([seq]);ms.append((time.perf_counter_ns()-t)/1e6)
        b=s.b1([seq]);e=s.embed([seq]);hh=[]
        for _ in range(100):
            t=time.perf_counter_ns();s.cached_score(b,e);hh.append((time.perf_counter_ns()-t)/1e6)
        r['timings'][str(length)]={'sequence_to_score_median_ms':float(np.median(ms)),'sequence_to_score_p95_ms':float(np.quantile(ms,.95)),'cached_head_only_median_ms':float(np.median(hh))}
        print(length,r['timings'][str(length)],flush=True)
    (out/'result.json').write_text(json.dumps(r,indent=2));print('BENCHMARK DONE',r['full_parameters'],r['full_weight_artifact_bytes'],flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
