"""Standalone student runtime: frozen ESM35 encoder + B1 + distilled head."""
from pathlib import Path
import sys,json
import numpy as np,torch
from transformers import EsmConfig,EsmModel,AutoTokenizer
from safetensors.torch import load_file
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vendor'))
from descriptors import extract_sequence_features
AA='ACDEFGHIKLMNPQRSTVWY';INDEX={x:i for i,x in enumerate(AA)}

class Student:
    def __init__(self,encoder_dir,tokenizer_dir,head_path):
        enc=Path(encoder_dir);cfg=EsmConfig.from_json_file(str(enc/'config.json'));self.encoder=EsmModel(cfg,add_pooling_layer=False);self.encoder.contact_head=None
        self.encoder.load_state_dict(load_file(str(enc/'encoder_float32.safetensors')),strict=True);self.encoder.eval();self.tokenizer=AutoTokenizer.from_pretrained(str(tokenizer_dir),local_files_only=True)
        with np.load(head_path,allow_pickle=False) as z:self.meta=json.loads(bytes(z['metadata_utf8']).decode());self.head={k:z[k] for k in ['mean','scale','coef','intercept']}
    def b1(self,sequences):
        rows=[]
        for s in sequences:
            if not s or set(s)-set(AA):raise ValueError('Noncanonical sequence')
            d=extract_sequence_features(s);a=np.asarray([INDEX[v] for v in s]);pairs=np.bincount(a[:-1]*20+a[1:],minlength=400)/max(len(s)-1,1);rows.append(np.r_[[d[n] for n in self.meta['descriptor_names']],pairs])
        return np.asarray(rows)
    def embed(self,sequences):
        ss=[s if len(s)<=1022 else s[:511]+s[-511:] for s in sequences];t=self.tokenizer(ss,return_tensors='pt',padding=True,truncation=True,max_length=1024,return_special_tokens_mask=True)
        with torch.inference_mode():
            h=self.encoder(input_ids=t['input_ids'],attention_mask=t['attention_mask']).last_hidden_state;mask=t['attention_mask']*(1-t['special_tokens_mask']);return ((h*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True)).numpy()
    def cached_score(self,b,e):
        x=np.column_stack([b,e]);h=self.head;z=((x-h['mean'])/h['scale'])@h['coef']+h['intercept'];return 1/(1+np.exp(-np.clip(z,-700,700)))
    def predict(self,sequences):return self.cached_score(self.b1(sequences),self.embed(sequences))
