"""Unit tests for the walk-forward vol-forecasting harness (purge + ΔR² bootstrap)."""

import numpy as np

from experiments.volatility_forecasting import delta_r2_pvalue, oos_r2, walk_forward_predict


class _RecordingModel:
    """Stub regressor that records the training-set size it was fitted on."""

    train_sizes = []

    def fit(self, X, y):
        _RecordingModel.train_sizes.append(len(y))
        self._mean = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(X.shape[0], self._mean)


def test_purge_excludes_boundary_labels():
    """With purge=p, no training row within p of the test block is used."""
    T, first, step, purge = 1500, 1000, 63, 20
    rng = np.random.default_rng(0)
    Xf = rng.normal(size=(T, 3))
    y = rng.normal(size=T)

    _RecordingModel.train_sizes = []
    walk_forward_predict(Xf, y, _RecordingModel, first, step=step, purge=purge)

    starts = list(range(first, T, step))
    assert len(_RecordingModel.train_sizes) == len(starts)
    # train slice is [0 : i - purge] on fully-finite synthetic data
    assert _RecordingModel.train_sizes == [i - purge for i in starts]


def test_purge_zero_matches_unpurged_boundary():
    T, first, step = 1400, 1000, 63
    rng = np.random.default_rng(1)
    Xf = rng.normal(size=(T, 2))
    y = rng.normal(size=T)

    _RecordingModel.train_sizes = []
    walk_forward_predict(Xf, y, _RecordingModel, first, step=step, purge=0)
    assert _RecordingModel.train_sizes[0] == first


def test_delta_r2_pvalue_detects_better_model():
    rng = np.random.default_rng(2)
    T = 2000
    y = rng.normal(size=T)
    yhat_good = y + 0.3 * rng.normal(size=T)
    yhat_bad = np.zeros(T)

    dr2, p = delta_r2_pvalue(y, yhat_bad, yhat_good, n_boot=500, seed=0)
    assert dr2 > 0
    assert p < 0.05
    # and symmetric: the worse model shows no improvement
    dr2_rev, p_rev = delta_r2_pvalue(y, yhat_good, yhat_bad, n_boot=500, seed=0)
    assert dr2_rev < 0
    assert p_rev > 0.5


def test_oos_r2_perfect_prediction():
    y = np.arange(10.0)
    assert oos_r2(y, y) == 1.0
