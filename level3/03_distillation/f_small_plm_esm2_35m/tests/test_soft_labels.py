import sys
from pathlib import Path
import numpy as np
from scipy.special import expit
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from student_head import soft_fit

def test_weighted_duplicate_loss_matches_fractional_bce():
    z=np.array([-2.,.1,3.]);q=np.array([.1,.4,.9]);fractional=np.logaddexp(0,z)-q*z
    duplicate=q*(np.logaddexp(0,z)-z)+(1-q)*np.logaddexp(0,z)
    np.testing.assert_allclose(fractional,duplicate,atol=1e-14)

def test_soft_fit_matches_gradient_optimum():
    rng=np.random.default_rng(1);x=rng.normal(size=(80,5));q=expit(x[:,0]-.2*x[:,1]);y=(q>.5).astype(int);c=.1;m=soft_fit(x,y,q,c);s,head=m;z=s.transform(x);p=m.predict_proba(x)[:,1]
    gradient=z.T@(p-q)+head.coef_.ravel()/c
    assert np.max(np.abs(gradient))<.005
    assert abs(np.sum(p-q))<.005
