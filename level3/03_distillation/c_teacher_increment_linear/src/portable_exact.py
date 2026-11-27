"""Portable teacher-free student preserving fitted scaler arithmetic."""
from portable import features,load
import numpy as np

def predict(meta,a,sequences):
    b,x=features(sequences,meta['descriptor_names'])
    bz=(b-a['base_mean'])/a['base_scale'];bz=bz@a['base_coef']+a['base_intercept']
    x-=a['correction_mean'];x/=a['correction_scale']
    correction=x@a['correction_coef']+a['correction_intercept']
    if meta['space']=='logit':
        # Match training/evaluation: baseline probabilities clipped before logit.
        bz=np.clip(bz,np.log(1e-6/(1-1e-6)),np.log((1-1e-6)/1e-6))
        z=bz+meta['gamma']*correction
    else:
        base=1/(1+np.exp(-np.clip(bz,-700,700)))
        return np.clip(base+meta['gamma']*correction,1e-6,1-1e-6)
    return np.clip(1/(1+np.exp(-np.clip(z,-700,700))),1e-6,1-1e-6)
