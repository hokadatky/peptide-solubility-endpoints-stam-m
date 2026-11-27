import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from context_features import context
from increment import score,Data

def test_context_is_finite_length_stable_and_order_sensitive():
    a=context(['ACDEFGHIKLMNPQRSTVWY','YWVTSRQPNMLKIHGFEDCA','A'])
    assert a.shape==(3,2073) and np.isfinite(a).all()
    assert not np.array_equal(a[0],a[1])
    with pytest.raises(ValueError):context(['AX'])

def test_zero_correction_and_zero_gamma_retain_baseline():
    b=np.array([.1,.3,.5,.9]);np.testing.assert_array_equal(score(b,np.zeros(4),1),b)
    np.testing.assert_array_equal(score(b,np.ones(4),0),b)

def test_forbidden_target_splits_fail_before_data_access():
    d=object.__new__(Data)
    with pytest.raises(AssertionError):d.pair([2,3],[3])
    with pytest.raises(AssertionError):d.pair([0,2],[3])

def test_selection_uses_arm_specific_inner_scores_and_conservative_ties():
    from nested import choose
    rows=[{'family':'ridge','param':p,'gamma':g,'distilled':.7,'hard':.8 if g==1 else .7} for p in [10,1000] for g in [0,1]]
    assert choose(rows,'distilled')['gamma']==0
    assert choose(rows,'distilled')['param']==1000
    assert choose(rows,'hard')['gamma']==1

def test_teacher_and_baseline_targets_exclude_each_target_fold():
    d=object.__new__(Data);d.y=np.zeros(6);d.f=np.array([2,2,3,3,4,4]);calls=[]
    def pair(train,test):
        calls.append((train,test));n=len(d.indices(test));return np.ones(n)*.2,np.ones(n)*.4
    d.pair=pair
    b,t=d.targets([2,3,4]);np.testing.assert_allclose(t-b,.2)
    assert len(calls)==3
    assert all(set(tr).isdisjoint(te) and len(tr)==2 for tr,te in calls)

def test_logit_increment_recovers_teacher_with_identity_regressor():
    from nested_extended import target_vectors,predict_score
    b=np.array([.2,.4,.8]);t=np.array([.3,.7,.65]);y=np.array([0,1,1]);kd,_=target_vectors(y,b,t,'logit')
    np.testing.assert_allclose(predict_score(b,kd,1,'logit'),t,rtol=1e-12)
    np.testing.assert_allclose(predict_score(b,kd,0,'logit'),b,rtol=1e-12)
