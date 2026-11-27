"""Read-only source/feature audit; outputs confined to this exploratory root."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[2]
sys.path.insert(0,str(WORK/'src'))
from prepare_solubility_benchmark import join_official_split
from build_solubility_features import extract_features
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    data=WORK/'data/solubility';source=WORK/'data/external/soluprot/soluprot_data'
    names=json.loads((data/'solubility_transparent_features.json').read_text())['feature_groups']['primary_aac_plus_physicochemical']
    out={'status':'PASS','splits':{},'files':{}}
    for split,derived,features in [('train','soluprot_train_partitions.csv','development'),('external_test','soluprot_external_test_clustered.csv','external')]:
        stem='training_set' if split=='train' else 'test_set'
        original=join_official_split(source/(stem+'.csv'),source/(stem+'.fasta'),split)
        df=pd.read_csv(data/derived,dtype={'record_id':str,'source_id':str})
        pd.testing.assert_frame_equal(original,df[original.columns],check_dtype=False)
        saved=pd.read_csv(data/f'solubility_transparent_features_{features}.csv').set_index('record_id').loc[df.record_id,names].to_numpy()
        recomputed=pd.DataFrame([extract_features(s) for s in df.sequence])[names].to_numpy()
        np.testing.assert_allclose(saved,recomputed,atol=1e-10,rtol=1e-10)
        summary={'rows':len(df),'labels_match_original':True,'sequences_match_original':True,'all_primary_features_recomputed':True,'max_feature_abs_error':float(np.max(np.abs(saved-recomputed))),'length_by_label':{},'terminal_polyhistidine':{}}
        for label in [0,1]:
            lengths=df.loc[df.label==label,'length']
            summary['length_by_label'][str(label)]={'n':len(lengths),'median':float(lengths.median()),'mean':float(lengths.mean()),'q10':float(lengths.quantile(.1)),'q90':float(lengths.quantile(.9))}
        summary['length_auc_descriptive']=float(roc_auc_score(df.label,df.length))
        # Fixed descriptive tag indicator, not a trimming/exclusion rule.
        tag=df.sequence.map(lambda s:'HHHHH' in s[:30] or 'HHHHH' in s[-30:])
        for value in [False,True]:
            sub=df[tag==value]
            summary['terminal_polyhistidine'][str(value)]={'n':len(sub),'positive_fraction':float(sub.label.mean()) if len(sub) else None}
        out['splits'][split]=summary
        for p in [source/(stem+'.csv'),source/(stem+'.fasta'),data/derived]:out['files'][str(p)]=sha(p)
    meta=json.loads((data/'embeddings/soluprot_prott5_metadata.json').read_text())
    assert sha(data/'embeddings/soluprot_prott5.npy')==meta['output_sha256']
    for k,f in [('development','soluprot_train_partitions.csv'),('external_test','soluprot_external_test_clustered.csv'),('index','embeddings/soluprot_embedding_index.csv')]:assert sha(data/f)==meta['input_sha256'][k]
    out['embedding_provenance']={'status':'PASS','pooling':meta['pooling'],'special_tokens_excluded':meta['special_tokens_excluded'],'checkpoint':meta['configuration']['checkpoint'],'dimension':meta['dimension'],'input_hashes_match_extraction_metadata':True,'embedding_values_not_reextracted':True}
    (ROOT/'INPUT_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='files'},indent=2))
if __name__=='__main__':main()
