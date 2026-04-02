"""
Unit tests for classical baseline regime detectors
(qcml.regime.classical_baselines).

Tests use REAL market data from Polygon API (SPY around the 2008 crisis)
following the same pattern as tests/test_quantum_indicators.py.

Tested detectors:
- QCMLChernDetector: Wraps existing Chern pipeline
- RollingVolatilityDetector: Rolling vol z-score
- CUSUMDetector: Cumulative sum change-point detector
- HMMRegimeDetector: 2-state Gaussian HMM
- RandomForestRegimeDetector: Supervised RF baseline
"""

import numpy as np
import pandas as pd
import pytest
import sys

sys.path.insert(0, "..")

from dotenv import load_dotenv

load_dotenv()

from qcml_geometry import BaseRegimeDetector
from experiments.baselines import (
    RollingVolatilityDetector,
    CUSUMDetector,
    HMMRegimeDetector,
    RandomForestRegimeDetector,
    RollingWindowRFDetector,
    VIXThresholdDetector,
)
from experiments.additional_detectors import (
    QCMLChernDetector,
    GeometricConsensusDetector,
)


# ---------------------------------------------------------------------------
# Fixtures: Real market data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_market_data():
    """
    Fetch real SPY data around the 2008 crisis from Polygon.

    Uses a ~6 month window (2008-06 to 2008-12) that spans the
    Lehman Brothers collapse, providing a genuine regime transition.
    Cached at module scope so the API is only called once per test run.
    """
    from experiments.data_loader import PolygonDataSource, MinimalFeatureEngine

    ds = PolygonDataSource()
    df = ds.fetch_equities(["SPY"], start_date="2008-06-01", end_date="2008-12-31")
    assert len(df) > 50, f"Expected >50 rows, got {len(df)}"

    prices_df = df["close"].unstack(level="symbol")
    fe = MinimalFeatureEngine()
    features_df = fe.create_feature_matrix(prices_df)
    features_df = features_df.dropna()

    return features_df


@pytest.fixture(scope="module")
def feature_array(real_market_data):
    """Numpy feature array from real market data."""
    return real_market_data.values.astype(np.float64)


# ---------------------------------------------------------------------------
# Base interface contract tests
# ---------------------------------------------------------------------------


class TestBaseRegimeDetector:
    """Test that all detectors satisfy the BaseRegimeDetector interface."""

    @pytest.fixture(
        params=[
            RollingVolatilityDetector,
            CUSUMDetector,
        ]
    )
    def stateless_detector_cls(self, request):
        return request.param

    def test_fit_returns_self(self, stateless_detector_cls, feature_array):
        detector = stateless_detector_cls()
        result = detector.fit(feature_array)
        assert result is detector

    def test_scores_1d_correct_length(self, stateless_detector_cls, feature_array):
        detector = stateless_detector_cls()
        detector.fit(feature_array)
        scores = detector.compute_regime_scores(feature_array)
        assert scores.ndim == 1
        assert len(scores) == len(feature_array)

    def test_has_name(self, stateless_detector_cls):
        detector = stateless_detector_cls()
        assert isinstance(detector.name, str)
        assert len(detector.name) > 0


# ---------------------------------------------------------------------------
# Rolling Volatility
# ---------------------------------------------------------------------------


class TestRollingVolatilityDetector:

    def test_output_has_nans_at_start(self, feature_array):
        det = RollingVolatilityDetector(vol_window=20, min_expanding=60)
        det.fit(feature_array)
        scores = det.compute_regime_scores(feature_array)
        # First min_expanding entries should be NaN
        assert np.isnan(scores[0])

    def test_non_nan_scores_are_finite(self, feature_array):
        det = RollingVolatilityDetector(vol_window=20, min_expanding=60)
        det.fit(feature_array)
        scores = det.compute_regime_scores(feature_array)
        valid = scores[~np.isnan(scores)]
        assert len(valid) > 0
        assert np.all(np.isfinite(valid))


# ---------------------------------------------------------------------------
# CUSUM
# ---------------------------------------------------------------------------


class TestCUSUMDetector:

    def test_starts_at_zero(self, feature_array):
        det = CUSUMDetector(burn_in=60)
        det.fit(feature_array)
        scores = det.compute_regime_scores(feature_array)
        assert scores[0] == 0.0

    def test_non_negative(self, feature_array):
        det = CUSUMDetector(burn_in=60)
        det.fit(feature_array)
        scores = det.compute_regime_scores(feature_array)
        assert np.all(scores >= 0.0)

    def test_detects_mean_shift(self):
        """CUSUM should accumulate when abs(returns) consistently exceed k."""
        np.random.seed(42)
        # Calm period then volatile period
        calm = np.cumsum(np.random.randn(100) * 0.01)
        volatile = np.cumsum(np.random.randn(100) * 0.05)
        X = np.concatenate([calm, volatile]).reshape(-1, 1)

        det = CUSUMDetector(burn_in=60)
        det.fit(X)
        scores = det.compute_regime_scores(X)

        # Last 50 scores should be larger than first 50 scores on average
        first_50 = np.mean(scores[:50])
        last_50 = np.mean(scores[-50:])
        assert last_50 > first_50


# ---------------------------------------------------------------------------
# HMM
# ---------------------------------------------------------------------------


class TestHMMRegimeDetector:

    def test_fit_returns_self(self, feature_array):
        det = HMMRegimeDetector(n_iter=20, seed=42)
        result = det.fit(feature_array)
        assert result is det

    def test_scores_in_0_1(self, feature_array):
        det = HMMRegimeDetector(n_iter=20, seed=42)
        det.fit(feature_array)
        scores = det.compute_regime_scores(feature_array)
        valid = scores[~np.isnan(scores)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 1.0)

    def test_two_states(self, feature_array):
        det = HMMRegimeDetector(n_iter=20, seed=42)
        det.fit(feature_array)
        assert det._model.n_components == 2

    def test_scores_correct_length(self, feature_array):
        det = HMMRegimeDetector(n_iter=20, seed=42)
        det.fit(feature_array)
        scores = det.compute_regime_scores(feature_array)
        assert len(scores) == len(feature_array)


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------


class TestRandomForestRegimeDetector:

    def test_fit_with_labels(self, feature_array):
        det = RandomForestRegimeDetector(n_estimators=10, seed=42, lookback=10)
        T = len(feature_array)
        y = np.zeros(T)
        y[T // 2 :] = 1  # Second half is "crisis"
        det.fit_with_labels(feature_array, y)
        assert det._model is not None

    def test_scores_in_0_1(self, feature_array):
        det = RandomForestRegimeDetector(n_estimators=10, seed=42, lookback=10)
        T = len(feature_array)
        y = np.zeros(T)
        y[T // 2 :] = 1
        det.fit_with_labels(feature_array, y)
        scores = det.compute_regime_scores(feature_array)
        valid = scores[~np.isnan(scores)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 1.0)

    def test_scores_correct_length(self, feature_array):
        det = RandomForestRegimeDetector(n_estimators=10, seed=42, lookback=10)
        T = len(feature_array)
        y = np.zeros(T)
        y[T // 2 :] = 1
        det.fit_with_labels(feature_array, y)
        scores = det.compute_regime_scores(feature_array)
        assert len(scores) == T

    def test_raises_without_labels(self, feature_array):
        det = RandomForestRegimeDetector(n_estimators=10, seed=42, lookback=10)
        det.fit(feature_array)
        with pytest.raises(RuntimeError, match="fit_with_labels"):
            det.compute_regime_scores(feature_array)


# ---------------------------------------------------------------------------
# QCML Chern wrapper
# ---------------------------------------------------------------------------


class TestQCMLChernDetector:

    def test_fit_returns_self(self, feature_array):
        det = QCMLChernDetector(hilbert_dim=4, window_size=10, n_pca_components=5, seed=42)
        result = det.fit(feature_array)
        assert result is det

    def test_scores_correct_length(self, feature_array):
        det = QCMLChernDetector(hilbert_dim=4, window_size=10, n_pca_components=5, seed=42)
        det.fit(feature_array)
        scores = det.compute_regime_scores(feature_array)
        assert len(scores) == len(feature_array)

    def test_scores_have_valid_values(self, feature_array):
        det = QCMLChernDetector(hilbert_dim=4, window_size=10, n_pca_components=5, seed=42)
        det.fit(feature_array)
        scores = det.compute_regime_scores(feature_array)
        valid = scores[~np.isnan(scores)]
        assert len(valid) > 0
        assert np.all(np.isfinite(valid))

    def test_raises_without_fit(self):
        det = QCMLChernDetector()
        with pytest.raises(RuntimeError):
            det.compute_regime_scores(np.random.randn(100, 5))


# ---------------------------------------------------------------------------
# Geometric Consensus Detector
# ---------------------------------------------------------------------------


class TestGeometricConsensusDetector:

    def test_fit_returns_self(self, feature_array):
        det = GeometricConsensusDetector(
            hilbert_dim=4,
            n_pca_components=3,
            n_curvature_components=3,
            min_expanding=20,
            seed=42,
        )
        result = det.fit(feature_array)
        assert result is det

    def test_scores_correct_length(self, feature_array):
        det = GeometricConsensusDetector(
            hilbert_dim=4,
            n_pca_components=3,
            n_curvature_components=3,
            min_expanding=20,
            seed=42,
        )
        det.fit(feature_array)
        scores = det.compute_regime_scores(feature_array)
        assert len(scores) == len(feature_array)

    def test_scores_finite_or_nan(self, feature_array):
        det = GeometricConsensusDetector(
            hilbert_dim=4,
            n_pca_components=3,
            n_curvature_components=3,
            min_expanding=20,
            seed=42,
        )
        det.fit(feature_array)
        scores = det.compute_regime_scores(feature_array)
        # All values should be finite (no Inf); NaN is acceptable at start
        assert not np.any(np.isinf(scores))

    def test_name_property(self):
        det = GeometricConsensusDetector()
        assert det.name == "Geometric Consensus"

    def test_raises_without_fit(self):
        det = GeometricConsensusDetector()
        with pytest.raises(RuntimeError):
            det.compute_regime_scores(np.random.randn(100, 5))

    def test_consensus_fewer_detections_than_chern(self, feature_array):
        """Consensus should be more selective than standalone Chern."""
        # Fit standalone Chern
        chern = QCMLChernDetector(
            hilbert_dim=4,
            window_size=10,
            n_pca_components=3,
            seed=42,
        )
        chern.fit(feature_array)
        chern_scores = chern.compute_regime_scores(feature_array)

        # Fit consensus
        consensus = GeometricConsensusDetector(
            hilbert_dim=4,
            n_pca_components=3,
            n_curvature_components=3,
            min_expanding=20,
            seed=42,
        )
        consensus.fit(feature_array)
        consensus_scores = consensus.compute_regime_scores(feature_array)

        # Count threshold exceedances for Chern (using fixed threshold)
        valid_chern = chern_scores[~np.isnan(chern_scores)]
        if len(valid_chern) > 0:
            chern_threshold = np.nanmean(valid_chern) + 1.5 * np.nanstd(valid_chern)
            chern_detections = np.sum(chern_scores > chern_threshold)
        else:
            chern_detections = 0

        # Consensus detections = nonzero scores
        consensus_detections = np.sum(consensus_scores > 0)

        # Consensus should have fewer or equal detections
        assert consensus_detections <= chern_detections, (
            f"Consensus ({consensus_detections}) should not exceed "
            f"standalone Chern ({chern_detections})"
        )


# ---------------------------------------------------------------------------
# Evaluation helpers (from regime_comparison.py)
# ---------------------------------------------------------------------------


class TestEvaluationHelpers:
    """Test the adaptive threshold and persistence filter helpers."""

    def test_persistence_filter_removes_single_spike(self):
        """Single True surrounded by False should be removed."""
        # Import from the experiments module
        sys.path.insert(0, str(pytest.importorskip("pathlib").Path(__file__).parent.parent))
        from experiments.regime_comparison import apply_persistence_filter

        mask = np.array([False, False, True, False, False, False])
        result = apply_persistence_filter(mask, min_persistence=3)
        assert not np.any(result)

    def test_persistence_filter_keeps_sustained_run(self):
        """Run of >= min_persistence should be kept."""
        sys.path.insert(0, str(pytest.importorskip("pathlib").Path(__file__).parent.parent))
        from experiments.regime_comparison import apply_persistence_filter

        mask = np.array([False, True, True, True, True, False])
        result = apply_persistence_filter(mask, min_persistence=3)
        # The run [1:5] has 4 True values, should be kept
        assert np.sum(result) == 4

    def test_adaptive_threshold_shape(self):
        """Output length should match input."""
        sys.path.insert(0, str(pytest.importorskip("pathlib").Path(__file__).parent.parent))
        from experiments.regime_comparison import compute_adaptive_threshold

        np.random.seed(42)
        scores = np.random.randn(200)
        thresholds = compute_adaptive_threshold(scores, min_expanding=20, quantile=0.95)
        assert len(thresholds) == len(scores)


# ---------------------------------------------------------------------------
# Rolling Window RF Detector
# ---------------------------------------------------------------------------


class TestRollingWindowRFDetector:

    def test_fit_rolling_creates_model(self):
        """fit_rolling with synthetic VIX should create a trained model."""
        np.random.seed(42)
        X = np.random.randn(300, 8)
        vix = np.random.uniform(15, 35, size=300)
        det = RollingWindowRFDetector(n_estimators=10, seed=42)
        det.fit_rolling(X, vix)
        assert det._model is not None

    def test_scores_in_0_1(self):
        """All valid scores should be in [0, 1]."""
        np.random.seed(42)
        X = np.random.randn(300, 8)
        vix = np.random.uniform(15, 35, size=300)
        det = RollingWindowRFDetector(n_estimators=10, seed=42)
        det.fit_rolling(X, vix)
        scores = det.compute_regime_scores(X)
        valid = scores[~np.isnan(scores)]
        assert len(valid) > 0
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 1.0)

    def test_scores_correct_length(self):
        """Output length should match input length."""
        np.random.seed(42)
        X = np.random.randn(300, 8)
        vix = np.random.uniform(15, 35, size=300)
        det = RollingWindowRFDetector(n_estimators=10, seed=42)
        det.fit_rolling(X, vix)
        scores = det.compute_regime_scores(X)
        assert len(scores) == len(X)

    def test_raises_without_fit(self):
        """Should raise RuntimeError if fit_rolling not called."""
        det = RollingWindowRFDetector()
        with pytest.raises(RuntimeError, match="fit_rolling"):
            det.compute_regime_scores(np.random.randn(100, 8))

    def test_name_property(self):
        det = RollingWindowRFDetector()
        assert det.name == "Rolling RF (VIX)"


# ---------------------------------------------------------------------------
# VIX Threshold Detector
# ---------------------------------------------------------------------------


class TestVIXThresholdDetector:

    def test_set_vix_and_score(self):
        """set_vix + compute_regime_scores should return correct shape."""
        np.random.seed(42)
        vix = np.random.uniform(12, 40, size=200)
        det = VIXThresholdDetector(min_expanding=60)
        det.set_vix(vix)
        scores = det.compute_regime_scores(np.empty((200, 4)))
        assert len(scores) == 200
        # First min_expanding entries should be NaN
        assert np.isnan(scores[0])
        # Should have valid scores after burn-in
        valid = scores[~np.isnan(scores)]
        assert len(valid) > 0

    def test_raises_without_vix(self):
        """Should raise RuntimeError if set_vix not called."""
        det = VIXThresholdDetector()
        with pytest.raises(RuntimeError, match="set_vix"):
            det.compute_regime_scores(np.random.randn(100, 4))

    def test_z_scores_finite(self):
        """All non-NaN scores should be finite (no inf)."""
        np.random.seed(42)
        vix = np.random.uniform(10, 50, size=300)
        det = VIXThresholdDetector(min_expanding=30)
        det.set_vix(vix)
        scores = det.compute_regime_scores(np.empty((300, 4)))
        valid = scores[~np.isnan(scores)]
        assert np.all(np.isfinite(valid))

    def test_name_property(self):
        det = VIXThresholdDetector()
        assert det.name == "VIX Level"
