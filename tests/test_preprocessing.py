import numpy as np
import pandas as pd
import pytest

from phi4finance import Scaler, lag_embed


@pytest.mark.parametrize("method", ["minmax", "absmax", "none"])
def test_roundtrip(method):
    X = np.random.default_rng(0).normal(0, 0.02, (50, 3))
    s = Scaler(method).fit(X)
    assert np.allclose(s.inverse_transform(s.transform(X)), X)


def test_minmax_range_and_absmax_keeps_zero():
    X = np.random.default_rng(0).normal(0, 0.02, (50, 3))
    Z = Scaler("minmax").fit_transform(X)
    assert np.allclose(Z.min(0), -1) and np.allclose(Z.max(0), 1)
    assert np.allclose(Scaler("absmax").fit(X).transform(np.zeros(3)), 0)


def test_fit_on_train_only():
    train, test = np.array([-0.01, 0.0, 0.02]), np.array([0.05])
    s = Scaler("absmax").fit(train)
    assert np.isclose(s.transform(test)[0], 2.5)   # test extremes are not peeked at


def test_column_selection_and_dataframe():
    df = pd.DataFrame({"A": [-1.0, 1.0], "B": [-2.0, 4.0]})
    s = Scaler("minmax").fit(df)
    assert isinstance(s.transform(df), pd.DataFrame)
    assert np.isclose(s.transform(4.0, cols=1), 1.0)
    assert np.isclose(s.inverse_transform(-1.0, cols=1), -2.0)
    assert np.isclose(s.scale_factor(cols=1), 3.0)
    s2 = Scaler.from_dict(s.to_dict())
    assert np.allclose(s2.transform(df.values), s.transform(df.values))


def test_unfitted_and_constant():
    with pytest.raises(RuntimeError):
        Scaler().transform([1.0])
    with pytest.raises(ValueError):
        Scaler().fit(np.ones(5))


def test_lag_embed_order():
    X = lag_embed(np.arange(6), window=3)
    assert X.shape == (4, 3)
    assert X[0].tolist() == [0, 1, 2] and X[-1].tolist() == [3, 4, 5]
