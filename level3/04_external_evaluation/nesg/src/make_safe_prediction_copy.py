"""Non-model serialization repair: create a pickle-free copy, retaining original."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import external_runtime as rt
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((ROOT/'results/external_retrospective.json').read_text());seal=json.loads((ROOT/'manifests/external_development_only_seal.json').read_text());c=json.loads((ROOT/'protocol/resolved_config.json').read_text())
a=ROOT/'private/external_predictions.npz';b=ROOT/'private/external_predictions_safe.npz';assert sha(a)==r['private_predictions_sha256']
# Read numeric arrays without pickle. Reconstruct IDs from the already verified source order.
with np.load(a,allow_pickle=False) as z: payload={k:z[k] for k in z.files if k not in ['record_id','external_cluster_id']}
d=pd.read_csv(Path(c['paths']['data_root'])/'soluprot_external_test_clustered.csv',usecols=['record_id','external_cluster_id','label'],dtype={'record_id':str,'external_cluster_id':str})
assert np.array_equal(d.label.to_numpy(),payload['label'])
ids=d.record_id.to_numpy(dtype=str);groups=d.external_cluster_id.to_numpy(dtype=str)
assert rt.string_array_hash(ids)==seal['preseal_input_hashes']['external_record_order']
payload.update(record_id=ids,external_cluster_id=groups);np.savez_compressed(b,**payload)
with np.load(b,allow_pickle=False) as z:
 for k,v in payload.items():assert np.array_equal(z[k],v)
log={'reason':'Original local NPZ stored pandas identifier arrays as object dtype. Numeric predictions/labels remain unchanged; source-aligned Unicode IDs allow pickle-free loading. Original artifact retained unchanged.','original_sha256':sha(a),'safe_copy_sha256':sha(b),'numeric_predictions_unchanged':True,'record_order_verified_against_preseal_hash':True}
(ROOT/'results/SERIALIZATION_REPAIR.json').write_text(json.dumps(log,indent=2)+'\n');print(json.dumps(log,indent=2))
