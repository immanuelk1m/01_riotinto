import json
import pickle
import hashlib
import numpy as np
import pandas as pd
from analysis.causal_study import contiguous, parents, contributions, cached, paired_ci, zero_before_start


def test_definite_parents_only():
    g=np.full((4,4,3),'',dtype='<U3')
    g[1,0,2]='-->'; g[2,0,0]='o->';g[3,0,0]='<->'
    assert parents(g,['price','a','b','c']) == {('a',2)}


def test_contiguous_does_not_stitch():
    X=pd.DataFrame({'a':[1,2,np.nan,3,4,5]},index=pd.date_range('2020-01-05',periods=6,freq='W-SUN'))
    assert contiguous(X).index.equals(X.index[3:])


def test_zero_before_start_fills_only_leading_driver_gaps():
    X=pd.DataFrame({'price':[np.nan,1,2,3],'a':[np.nan,np.nan,5,6],'b':[1,np.nan,3,4]},index=pd.date_range('2020-01-05',periods=4,freq='W-SUN'))
    Z=zero_before_start(X)
    assert list(Z.a)==[0,0,5,6]
    assert np.isnan(Z.b.iloc[1]) and np.isnan(Z.price.iloc[0])
    assert len(contiguous(Z))==2


def test_contributions_parent_only_and_training_frozen():
    rng=np.random.default_rng(0)
    X=pd.DataFrame(rng.normal(size=(100,3)),columns=['price','a','b'])
    X.price=2*X.a
    fit=np.arange(100)<70
    for spline in (False,True):
        c=contributions(X,{('a',0)},fit,spline)
        changed=X.copy();changed.loc[70:,'price']=9999
        pd.testing.assert_frame_equal(c,contributions(changed,{('a',0)},fit,spline))
        assert (c.b==0).all()
    assert not contributions(X,set(),fit).to_numpy().any()


def test_integrity(tmp_path):
    p=pickle.dumps({'a':1});(tmp_path/'result.pkl').write_bytes(p)
    (tmp_path/'sha256').write_text(hashlib.sha256(p).hexdigest())
    files = ['result.pkl','sha256','contributions.parquet','hits.parquet','ranking.parquet','parents.json','spec.json','predictions.parquet','model.pkl']
    for name in files[2:]: (tmp_path/name).write_bytes(b'export')
    (tmp_path/'manifest.json').write_text(json.dumps({name:hashlib.sha256((tmp_path/name).read_bytes()).hexdigest() for name in files}))
    assert cached(tmp_path)=={'a':1}
    (tmp_path/'result.pkl').write_bytes(b'broken')
    assert cached(tmp_path) is None


def test_paired_same_weeks():
    a=pd.Series([True]*12,index=pd.date_range('2020-01-05',periods=12,freq='W-SUN'))
    assert paired_ci({'x':a},{'x':a},20)==[0.,0.,0.]


def test_study_scope_changes_identity():
    from analysis.causal_study import fingerprint
    from analysis import causal_common as cc
    assert fingerprint(['coffee'],['pcmci']) != fingerprint(None,None)
    assert fingerprint() == fingerprint(list(cc.SERIES))


def test_saved_model_reproduces_predictions():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(80, 2)), columns=['price', 'a'])
    X.price = X.a * 2 + 3
    for spline in (False, True):
        _, prediction, fitted = contributions(X, {('a', 0)}, np.arange(80) < 60, spline, details=True)
        fitted = pickle.loads(pickle.dumps(fitted))
        if spline:
            actual = fitted['estimator'].predict(fitted['transformer'].transform(pd.DataFrame({'0': X.a})))
        else:
            actual = fitted['intercept'] + X.a * fitted['coefficients'][0]
        np.testing.assert_allclose(prediction, actual)


def test_missing_artifact_rejects_cache(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps({'missing.parquet':'abc'}))
    assert cached(tmp_path) is None
