from pathlib import Path
import json,hashlib
import torch
from transformers import AutoModel
from safetensors.torch import save_file
ROOT=Path(__file__).resolve().parents[1]
SNAP=Path.home()/'.cache/huggingface/hub/models--facebook--esm2_t12_35M_UR50D/snapshots/6fbf070e65b0b7291e7bbcd451118c216cff79d8'

def main():
    out=ROOT/'runs/encoder_export';out.mkdir(exist_ok=False);m=AutoModel.from_pretrained(str(SNAP),local_files_only=True,add_pooling_layer=False)
    state={k:v.detach().contiguous() for k,v in m.state_dict().items() if not k.startswith('contact_head.')};count=sum(v.numel() for k,v in m.named_parameters() if not k.startswith('contact_head.'));m.config.to_json_file(out/'config.json')
    paths={}
    for dtype in ['float32','float16']:
        path=out/f'encoder_{dtype}.safetensors';save_file({k:v.to(getattr(torch,dtype)).contiguous() for k,v in state.items()},str(path));paths[dtype]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (out/'result.json').write_text(json.dumps({'active_encoder_parameters':count,'stored_scalar_count':sum(v.numel() for v in state.values()),'full_predictor_learned_parameters':count+906,'removed_unused_pooler':True,'removed_unused_contact_head':True,'artifacts':paths,'FP16_scope':'storage only, upcast FP32; predictive validation pending; not primary inference'},indent=2));print(count,paths)
if __name__=='__main__':main()
