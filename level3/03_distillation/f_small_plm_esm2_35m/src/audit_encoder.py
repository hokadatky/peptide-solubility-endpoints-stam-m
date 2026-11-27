import os
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib,time
import numpy as np,pandas as pd,torch
from transformers import AutoModel,AutoTokenizer
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parents[1]
CACHE=BASE/'paper/general_peptide_solubility_scoping/analysis_distillation_rescue_run_20260813_v2/correct_residue_pool_esm35_design.npz'
SNAP=Path.home()/'.cache/huggingface/hub/models--facebook--esm2_t12_35M_UR50D/snapshots/6fbf070e65b0b7291e7bbcd451118c216cff79d8'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    torch.set_num_threads(1);raw=pd.read_csv(BASE/'data/solubility/soluprot_train_partitions.csv');cfg=json.loads((ROOT/'protocol/parent_config.json').read_text());assert sha(BASE/cfg['paths']['development'])==cfg['input_sha256']['development']
    with np.load(CACHE,allow_pickle=False) as z:idx=z['row_indices'];emb=z['embeddings'];meta=json.loads(bytes(z['metadata_json_utf8']).decode())
    assert len(np.unique(idx))==len(idx) and meta['model']=='facebook/esm2_t12_35M_UR50D'
    selected=np.flatnonzero(raw.development_fold.isin([2,3,4,5,6]));lookup={int(v):i for i,v in enumerate(idx)};assert all(int(i) in lookup for i in selected)
    subset=emb[[lookup[int(i)] for i in selected]];assert subset.shape==(8198,480)
    tok=AutoTokenizer.from_pretrained(str(SNAP),local_files_only=True);m=AutoModel.from_pretrained(str(SNAP),local_files_only=True).eval()
    probes=[int(selected[0]),int(selected[len(selected)//2]),int(selected[-1]),int(selected[np.argmax(raw.iloc[selected].sequence.str.len().to_numpy())])];errs=[]
    for i in probes:
        s=raw.iloc[i].sequence;s=s if len(s)<=1022 else s[:511]+s[-511:];a=tok([s],return_tensors='pt',padding=True,truncation=True,max_length=1024,return_special_tokens_mask=True)
        with torch.inference_mode():
            h=m(input_ids=a['input_ids'],attention_mask=a['attention_mask']).last_hidden_state;mask=a['attention_mask']*(1-a['special_tokens_mask']);p=(h*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True);err=float(np.max(np.abs(p.numpy()[0]-emb[lookup[i]])));errs.append(err)
    assert max(errs)<.001,errs
    np.savez_compressed(ROOT/'runs/esm35_development.npz',row_id=raw.iloc[selected].record_id.to_numpy(dtype=str),embeddings=subset)
    report={'import_source':str(CACHE),'source_sha256':sha(CACHE),'metadata':meta,'retained_folds':[2,3,4,5,6],'retained_rows':len(selected),'checkpoint_revision':SNAP.name,'checkpoint_files':{p.name:sha(p) for p in SNAP.iterdir() if p.is_file()},'torch':torch.__version__,'device':'cpu','fresh_probe_max_abs_embedding_errors':errs,'probe_rows':probes,'model_parameters_all_loaded':sum(p.numel() for p in m.parameters()),'created_unix':time.time(),'scope':'frozen pretrained embedding import only; no previous fitted head or predictions'}
    (ROOT/'runs/ENCODER_IMPORT_QA.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
