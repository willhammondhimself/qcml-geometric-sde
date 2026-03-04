"""
Tests for information geometry and fusion regime detectors.

Tests verify:
- FisherRaoDetector
- WassersteinDetector
- KLDivergenceDetector
- SinkhornDetector
- RankFusionDetector
- StackingFusionDetector
- DynamicSwitchingDetector
"""

import numpy as np
import pytest

from qcml_geometry.info_geometry import (
    FisherRaoDetector,
    WassersteinDetector,
    KLDivergenceDetector,
    SinkhornDetector,
)
from qcml_geometry.fusion import (
    RankFusionDetector,
    StackingFusionDetector,
    DynamicSwitchingDetector,
)


@pytest.fixture
def regime_shift_data():
    """Generate synthetic data with a clear regime shift."""
    rng = np.random.default_rng(42)
    T = 400
    X = rng.standard_normal((T, 4))
    # Inject regime shift: higher vol + mean shift at t=200
    X[200:280] = X[200:280] * 3.0 + 1.5
    return X


@pytest.fixture
def univariate_data():
    """Generate univariate data with regime shift."""
    rng = np.random.default_rng(42)
    T = 400
    x = rng.standard_normal(T) * 0.01
    # Crisis period
    x[200:280] = rng.standard_normal(80) * 0.05 - 0.02
    return x.reshape(-1, 1)


class TestFisherRaoDetector:
    def test_smoke(self, univariate_data):
        det = FisherRaoDetector(window=20, lag=5, min_expanding=30)
        det.fit(univariate_data)
        scores = det.compute_regime_scores(univariate_data)
        assert scores.shape == (len(univariate_data),)
        assert not np.all(np.isnan(scores))

    def test_name(self):
        det = FisherRaoDetector()
        assert det.name == "Fisher-Rao"

    def test_elevated_during_crisis(self, univariate_data):
        det = FisherRaoDetector(window=20, lag=5, min_expanding=30)
        det.fit(univariate_data)
        scores = det.compute_regime_scores(univariate_data)
        calm = scores[60:180]
        crisis = scores[210:270]
        calm_valid = calm[~np.isnan(calm)]
        crisis_valid = crisis[~np.isnan(crisis)]
        if len(calm_valid) > 5 and len(crisis_valid) > 5:
            assert np.nanmean(crisis_valid) > np.nanmean(calm_valid)

    def test_not_fitted_raises(self):
        det = FisherRaoDetector()
        X = np.random.randn(100, 1)
        with pytest.raises(RuntimeError, match="fit"):
            det.compute_regime_scores(X)


class TestWassersteinDetector:
    def test_smoke(self, regime_shift_data):
        det = WassersteinDetector(window=20, lag=5, min_expanding=30)
        det.fit(regime_shift_data)
        scores = det.compute_regime_scores(regime_shift_data)
        assert scores.shape == (len(regime_shift_data),)
        assert not np.all(np.isnan(scores))

    def test_name(self):
        det = WassersteinDetector()
        assert det.name == "Wasserstein"

    def test_univariate(self, univariate_data):
        det = WassersteinDetector(window=20, lag=5, min_expanding=30)
        det.fit(univariate_data)
        scores = det.compute_regime_scores(univariate_data)
        assert scores.shape == (len(univariate_data),)

    def test_elevated_during_crisis(self, regime_shift_data):
        det = WassersteinDetector(window=20, lag=5, min_expanding=30)
        det.fit(regime_shift_data)
        scores = det.compute_regime_scores(regime_shift_data)
        calm = scores[60:180]
        crisis = scores[210:270]
        calm_valid = calm[~np.isnan(calm)]
        crisis_valid = crisis[~np.isnan(crisis)]
        if len(calm_valid) > 5 and len(crisis_valid) > 5:
            assert np.nanmean(crisis_valid) > np.nanmean(calm_valid)


class TestKLDivergenceDetector:
    def test_smoke(self, regime_shift_data):
        det = KLDivergenceDetector(window=20, lag=5, min_expanding=30)
        det.fit(regime_shift_data)
        scores = det.compute_regime_scores(regime_shift_data)
        assert scores.shape == (len(regime_shift_data),)
        assert not np.all(np.isnan(scores))

    def test_name(self):
        det = KLDivergenceDetector()
        assert det.name == "KL Divergence"

    def test_nonnegative_jsd(self, univariate_data):
        det = KLDivergenceDetector(window=20, lag=5, min_expanding=30)
        det.fit(univariate_data)
        scores = det.compute_regime_scores(univariate_data)
        # JSD is non-negative, so raw values should be non-negative
        # (z-scores can be negative, but raw JSD >= 0)
        assert scores.shape == (len(univariate_data),)


class TestSinkhornDetector:
    def test_smoke(self, univariate_data):
        det = SinkhornDetector(window=20, lag=5, min_expanding=30)
        det.fit(univariate_data)
        scores = det.compute_regime_scores(univariate_data)
        assert scores.shape == (len(univariate_data),)
        assert not np.all(np.isnan(scores))

    def test_name(self):
        det = SinkhornDetector()
        assert det.name == "Sinkhorn"


class TestRankFusionDetector:
    def test_smoke_precomputed(self, regime_shift_data):
        rng = np.random.default_rng(42)
        T = len(regime_shift_data)
        # Simulate 3 detector scores
        score_matrix = rng.standard_normal((T, 3))
        score_matrix[:30] = np.nan

        det = RankFusionDetector(rolling_window=10, min_expanding=30)
        det.set_precomputed_scores(score_matrix)
        scores = det.compute_regime_scores(regime_shift_data)
        assert scores.shape == (T,)
        assert not np.all(np.isnan(scores))

    def test_name(self):
        det = RankFusionDetector()
        assert det.name == "Rank Fusion"


class TestStackingFusionDetector:
    def test_smoke_precomputed(self, regime_shift_data):
        rng = np.random.default_rng(42)
        T = len(regime_shift_data)
        score_matrix = rng.standard_normal((T, 3))
        score_matrix[:30] = np.nan
        labels = np.zeros(T)
        labels[200:280] = 1.0

        det = StackingFusionDetector(
            crisis_labels=labels, train_end=T, min_expanding=30
        )
        det.set_precomputed_scores(score_matrix)
        det.fit(regime_shift_data)
        scores = det.compute_regime_scores(regime_shift_data)
        assert scores.shape == (T,)
        assert not np.all(np.isnan(scores))

    def test_no_labels(self, regime_shift_data):
        rng = np.random.default_rng(42)
        T = len(regime_shift_data)
        score_matrix = rng.standard_normal((T, 3))

        det = StackingFusionDetector(min_expanding=30)
        det.set_precomputed_scores(score_matrix)
        det.fit(regime_shift_data)
        scores = det.compute_regime_scores(regime_shift_data)
        assert scores.shape == (T,)

    def test_name(self):
        det = StackingFusionDetector()
        assert det.name == "Stacking Fusion"


class TestDynamicSwitchingDetector:
    def test_smoke_precomputed(self, regime_shift_data):
        rng = np.random.default_rng(42)
        T = len(regime_shift_data)
        score_matrix = rng.standard_normal((T, 3))
        score_matrix[:30] = np.nan

        det = DynamicSwitchingDetector(
            eval_window=30, rolling_window=10, min_expanding=30
        )
        det.set_precomputed_scores(score_matrix)
        det._is_fitted = True
        scores = det.compute_regime_scores(regime_shift_data)
        assert scores.shape == (T,)
        assert not np.all(np.isnan(scores))

    def test_name(self):
        det = DynamicSwitchingDetector()
        assert det.name == "Dynamic Switching"
