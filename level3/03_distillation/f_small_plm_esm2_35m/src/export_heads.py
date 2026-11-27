from increment import *
import joblib

def main():
    out=ROOT/'runs/head_export';out.mkdir(exist_ok=False);d=Data();e=np.load(ROOT/'runs/esm35_development.npz');x=np.column_stack([d.b,e['embeddings']]);raw=np.load(ROOT/'runs/nested/private_predictions.npz');errs={}
    for fold in range(2,7):
        m=joblib.load(ROOT/f'runs/nested/outer_{fold}/distilled.joblib');sc,lr=m;idx=d.indices([fold]);p=expit(((x[idx]-sc.mean_)/sc.scale_)@lr.coef_.ravel()+lr.intercept_[0]);err=float(np.max(np.abs(p-raw['distilled'][idx])));assert err<1e-10;errs[str(fold)]=err
        meta={'descriptor_names':d.blocks['B0'][1],'encoder_revision':'6fbf070e65b0b7291e7bbcd451118c216cff79d8','fold':fold,'endpoint':'recombinant soluble expression in E. coli','promoted':False}
        np.savez_compressed(out/f'fold{fold}.npz',mean=sc.mean_,scale=sc.scale_,coef=lr.coef_.ravel(),intercept=lr.intercept_[0],metadata_utf8=np.frombuffer(json.dumps(meta).encode(),dtype=np.uint8))
    write(out/'result.json',{'all_oof_head_parity':errs,'head_coefficients':906,'normalization_constants':1810,'fold6_npz_bytes':(out/'fold6.npz').stat().st_size,'sha256':sha(out/'fold6.npz')});print(errs)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
