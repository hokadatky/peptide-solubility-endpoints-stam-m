import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from nonlinear import Map,lg,response

def test_mapper_fits_training_rows_only_and_is_reproducible():
    x=np.random.default_rng(1).normal(size=(600,2498)).astype(np.float32);tr=np.arange(550)
    a=Map().fit(x,tr,.5);xx=x.copy();xx[550:]=10000;b=Map().fit(xx,tr,.5)
    np.testing.assert_array_equal(a.anchors,b.anchors)
    np.testing.assert_array_equal(a.scaler.mean_,b.scaler.mean_)
    assert a.anchors.shape==(512,498)
    assert a.transform(x[:2]).shape==(2,3010)

def test_exact_teacher_correction_identity():
    b=np.array([.1,.5,.9]);t=np.array([.3,.4,.95])
    np.testing.assert_allclose(response(b,lg(t)-lg(b),1),t,atol=1e-12)
