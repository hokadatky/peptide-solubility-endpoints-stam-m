import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from neural import model,fit

def test_zero_output_is_baseline_correction_zero():
    with torch.inference_mode():assert torch.equal(model(1)(torch.ones(3,2498)),torch.zeros(3,1))

def test_fixed_budget_weights_do_not_depend_on_outer_labels():
    rng=np.random.default_rng(3);x=rng.normal(size=(12,2498)).astype(np.float32);y=np.arange(12)%2;t=np.ones(12)*.6;b=np.ones(12)*.5;tr=np.arange(8);te=np.arange(8,12)
    a,_,_=fit(x,y,t,b,b[te],tr,te,.5,epochs=1,early=False,seed=1)
    y2=y.copy();y2[te]=1-y2[te]
    z,_,_=fit(x,y2,t,b,b[te],tr,te,.5,epochs=1,early=False,seed=1)
    for k,v in a.state_dict().items():assert torch.equal(v,z.state_dict()[k])
