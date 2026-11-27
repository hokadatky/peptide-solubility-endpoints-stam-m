"""Label-free context features; fixed AA alphabet and no fitted vocabulary."""
import numpy as np
AA='ACDEFGHIKLMNPQRSTVWY'
INDEX={a:i for i,a in enumerate(AA)}
GROUPS=('AGV','ILFP','YMTS','HNQW','RK','DE','C')
GROUP={a:i for i,g in enumerate(GROUPS) for a in g}

def context(sequences):
    rows=[]
    for s in sequences:
        if not s or set(s)-set(AA):raise ValueError('noncanonical sequence')
        a=np.array([INDEX[x] for x in s]);parts=[]
        for seg in [a[:32],a[-32:],*np.array_split(a,4)]:
            parts.extend(np.bincount(seg,minlength=20)/max(len(seg),1))
        for lag in [2,3,4,8]:parts.extend(np.bincount(a[:-lag]*20+a[lag:],minlength=400)/max(len(a)-lag,1))
        g=np.array([GROUP[x] for x in s]);parts.extend(np.bincount(g[:-2]*49+g[1:-1]*7+g[2:],minlength=343)/max(len(g)-2,1))
        for prop in [np.array([float(x in 'AVILMFWY') for x in s]),np.array([int(x in 'KR')-int(x in 'DE') for x in s])]:
            for lag in [1,2,4,8,16]:parts.append(float(np.mean(prop[:-lag]*prop[lag:])) if len(prop)>lag else 0.)
        rows.append(parts)
    return np.asarray(rows,dtype=np.float32)
