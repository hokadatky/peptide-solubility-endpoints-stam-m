"""Fused linear logit-increment student: NumPy + deterministic descriptors."""
from portable import features,load
import numpy as np

def predict(meta,a,sequences):
    b,x=features(sequences,meta['descriptor_names'])
    # Original B1 columns are float64; all other context features are float32.
    x=x.astype(np.float64);x[:,:425]=b
    z=x@a['coef']+a['intercept']
    return np.clip(1/(1+np.exp(-np.clip(z,-700,700))),1e-6,1-1e-6)
