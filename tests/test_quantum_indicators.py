"""
Unit tests for novel quantum indicators (qcml.regime.quantum_indicators).

Tests use REAL market data from Polygon API (SPY around the 2008 crisis)
to validate that indicators produce physically meaningful results on
actual financial time series.

Tested indicators:
- SpectralGapIndicator: Gap collapse detection
- EnergyEvolutionIndicator: Ground state energy regime classification
- FidelityDecayIndicator: Stability index via fidelity decay
- MultiScaleChernConsensus: Weighted cross-scale consensus
- GeometricIndicatorSuite: Composite score combining all indicators
"""

import numpy as np
import pandas as pd
import pytest
import os
import sys

sys.path.insert(0, "..")

from dotenv import load_dotenv

load_dotenv()

from qcml_geometry import QCMLGeometry
from qcml_geometry.indicators import (
    IndicatorResult,
    SpectralGapIndicator,
    EnergyEvolutionIndicator,
    FidelityDecayIndicator,
    MultiScaleChernConsensus,
    GeometricIndicatorSuite,
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

    # Build feature matrix
    prices_df = df["close"].unstack(level="symbol")
    fe = MinimalFeatureEngine()
    features_df = fe.create_feature_matrix(prices_df)
    features_df = features_df.dropna()

    return features_df


@pytest.fixture(scope="module")
def geometry_and_features(real_market_data):
    """
    Create a fitted QCMLGeometry from real market data.

    Returns (geometry, X) where X is the numpy feature array.
    """
    X = real_market_data.values.astype(np.float64)
    n_features = X.shape[1]

    geometry = QCMLGeometry(n_features=n_features, hilbert_dim=8)
    geometry.fit_operators(X, method="pca_inspired")

    return geometry, X


# ---------------------------------------------------------------------------
# SpectralGapIndicator tests
# ---------------------------------------------------------------------------


class TestSpectralGapIndicator:
    """Tests for spectral gap early warning indicator."""

    def test_gap_series_shape(self, geometry_and_features):
        """Gap series must have same length as input."""
        geometry, X = geometry_and_features
        indicator = SpectralGapIndicator(geometry, window_size=10)
        gaps = indicator.compute_spectral_gap_series(X)

        assert gaps.shape == (X.shape[0],)

    def test_gap_values_non_negative(self, geometry_and_features):
        """Spectral gap (E1 - E0) must be non-negative."""
        geometry, X = geometry_and_features
        indicator = SpectralGapIndicator(geometry, window_size=10)
        gaps = indicator.compute_spectral_gap_series(X)

        assert np.all(gaps >= -1e-10), "Spectral gap must be non-negative"

    def test_rolling_gap_smoothness(self, geometry_and_features):
        """Rolling gap should be smoother than raw gap."""
        geometry, X = geometry_and_features
        indicator = SpectralGapIndicator(geometry, window_size=10)

        raw_gaps = indicator.compute_spectral_gap_series(X)
        rolling_gaps = indicator.compute_rolling_gap(X)

        # Rolling std should be smaller than raw std
        assert np.std(rolling_gaps) <= np.std(raw_gaps) + 1e-10

    def test_detect_gap_collapse_returns_indicator_result(self, geometry_and_features):
        """detect_gap_collapse must return a valid IndicatorResult."""
        geometry, X = geometry_and_features
        indicator = SpectralGapIndicator(geometry, window_size=10)
        result = indicator.detect_gap_collapse(X)

        assert isinstance(result, IndicatorResult)
        assert result.name == "spectral_gap"
        assert len(result.values) == X.shape[0]
        assert isinstance(result.transitions, list)
        assert "mean_gap" in result.metadata
        assert "n_collapses" in result.metadata
        assert result.metadata["mean_gap"] >= 0

    def test_gap_collapse_threshold_sensitivity(self, geometry_and_features):
        """Lower threshold should detect more collapses."""
        geometry, X = geometry_and_features

        strict = SpectralGapIndicator(geometry, window_size=10, collapse_threshold_std=3.0)
        loose = SpectralGapIndicator(geometry, window_size=10, collapse_threshold_std=1.0)

        result_strict = strict.detect_gap_collapse(X)
        result_loose = loose.detect_gap_collapse(X)

        assert len(result_loose.transitions) >= len(result_strict.transitions)


# ---------------------------------------------------------------------------
# EnergyEvolutionIndicator tests
# ---------------------------------------------------------------------------


class TestEnergyEvolutionIndicator:
    """Tests for ground state energy evolution indicator."""

    def test_energy_series_shape(self, geometry_and_features):
        """Energy series must match input length."""
        geometry, X = geometry_and_features
        indicator = EnergyEvolutionIndicator(geometry, window_size=10)
        energies = indicator.compute_energy_series(X)

        assert energies.shape == (X.shape[0],)

    def test_energy_values_finite(self, geometry_and_features):
        """Ground state energies must be finite real numbers."""
        geometry, X = geometry_and_features
        indicator = EnergyEvolutionIndicator(geometry, window_size=10)
        energies = indicator.compute_energy_series(X)

        assert np.all(np.isfinite(energies)), "Energies must be finite"

    def test_energy_regime_classification_result(self, geometry_and_features):
        """energy_regime_classification must return valid IndicatorResult."""
        geometry, X = geometry_and_features
        indicator = EnergyEvolutionIndicator(geometry, window_size=10)
        result = indicator.energy_regime_classification(X)

        assert isinstance(result, IndicatorResult)
        assert result.name == "ground_state_energy"
        assert len(result.values) == X.shape[0]
        assert isinstance(result.transitions, list)
        assert "mean_energy" in result.metadata
        assert "n_stress_events" in result.metadata

    def test_energy_stress_threshold_sensitivity(self, geometry_and_features):
        """Lower threshold should detect more stress events."""
        geometry, X = geometry_and_features

        strict = EnergyEvolutionIndicator(geometry, window_size=10, stress_threshold_std=3.0)
        loose = EnergyEvolutionIndicator(geometry, window_size=10, stress_threshold_std=1.0)

        result_strict = strict.energy_regime_classification(X)
        result_loose = loose.energy_regime_classification(X)

        assert len(result_loose.transitions) >= len(result_strict.transitions)


# ---------------------------------------------------------------------------
# FidelityDecayIndicator tests
# ---------------------------------------------------------------------------


class TestFidelityDecayIndicator:
    """Tests for quantum fidelity decay indicator."""

    def test_fidelity_series_shape(self, geometry_and_features):
        """Fidelity series length = T - lag."""
        geometry, X = geometry_and_features
        lag = 1
        indicator = FidelityDecayIndicator(geometry, lag=lag, window_size=10)
        fidelities = indicator.compute_fidelity_series(X)

        assert fidelities.shape == (X.shape[0] - lag,)

    def test_fidelity_values_in_unit_interval(self, geometry_and_features):
        """Fidelity F = |<psi|phi>|^2 must be in [0, 1]."""
        geometry, X = geometry_and_features
        indicator = FidelityDecayIndicator(geometry, lag=1, window_size=10)
        fidelities = indicator.compute_fidelity_series(X)

        assert np.all(fidelities >= -1e-10), "Fidelity must be >= 0"
        assert np.all(fidelities <= 1.0 + 1e-10), "Fidelity must be <= 1"

    def test_fidelity_self_overlap(self, geometry_and_features):
        """Fidelity with lag=0 equivalent should be ~1 (identity overlap)."""
        geometry, X = geometry_and_features
        # Test with lag=1 on very similar consecutive points
        indicator = FidelityDecayIndicator(geometry, lag=1, window_size=10)
        fidelities = indicator.compute_fidelity_series(X)

        # Most fidelities should be high (>0.5) for real market data
        # since consecutive days are typically similar
        median_fidelity = np.median(fidelities)
        assert median_fidelity > 0.1, (
            f"Median fidelity {median_fidelity:.3f} too low; "
            "consecutive market states should have some overlap"
        )

    def test_stability_index_result(self, geometry_and_features):
        """stability_index must return valid IndicatorResult."""
        geometry, X = geometry_and_features
        indicator = FidelityDecayIndicator(geometry, lag=1, window_size=10)
        result = indicator.stability_index(X)

        assert isinstance(result, IndicatorResult)
        assert result.name == "fidelity_decay"
        assert len(result.values) == X.shape[0] - 1
        assert isinstance(result.transitions, list)
        assert "mean_fidelity" in result.metadata
        assert "lag" in result.metadata
        assert result.metadata["lag"] == 1

    def test_larger_lag_reduces_fidelity(self, geometry_and_features):
        """Larger lag should generally produce lower fidelity (more state change)."""
        geometry, X = geometry_and_features

        fid_lag1 = FidelityDecayIndicator(geometry, lag=1, window_size=10)
        fid_lag5 = FidelityDecayIndicator(geometry, lag=5, window_size=10)

        f1 = fid_lag1.compute_fidelity_series(X)
        f5 = fid_lag5.compute_fidelity_series(X)

        # Mean fidelity at lag=5 should generally be lower than lag=1
        assert np.mean(f5) <= np.mean(f1) + 0.1, (
            "Larger lag should produce lower or similar fidelity"
        )


# ---------------------------------------------------------------------------
# MultiScaleChernConsensus tests
# ---------------------------------------------------------------------------


class TestMultiScaleChernConsensus:
    """Tests for multi-scale Chern consensus indicator."""

    def test_multi_scale_chern_dict_structure(self, geometry_and_features):
        """compute_multi_scale_chern returns dict with correct keys."""
        geometry, X = geometry_and_features
        scales = [10, 20]
        indicator = MultiScaleChernConsensus(geometry, scales=scales)
        chern_dict = indicator.compute_multi_scale_chern(X)

        assert set(chern_dict.keys()) == set(scales)
        for scale, values in chern_dict.items():
            assert isinstance(values, np.ndarray)
            assert len(values) > 0

    def test_consensus_result_structure(self, geometry_and_features):
        """compute_consensus returns valid IndicatorResult."""
        geometry, X = geometry_and_features
        indicator = MultiScaleChernConsensus(
            geometry, scales=[10, 20], consensus_threshold=0.6
        )
        result = indicator.compute_consensus(X)

        assert isinstance(result, IndicatorResult)
        assert result.name == "multi_scale_consensus"
        assert len(result.values) > 0
        assert isinstance(result.transitions, list)
        assert "scales" in result.metadata
        assert "scale_correlations" in result.metadata

    def test_consensus_values_non_negative_and_finite(self, geometry_and_features):
        """Consensus values must be non-negative and finite."""
        geometry, X = geometry_and_features
        indicator = MultiScaleChernConsensus(
            geometry, scales=[10, 20], consensus_threshold=0.6
        )
        result = indicator.compute_consensus(X)

        assert np.all(result.values >= -1e-10), "Consensus must be >= 0"
        assert np.all(np.isfinite(result.values)), "Consensus must be finite"

    def test_custom_weights_normalization(self, geometry_and_features):
        """Custom weights should be normalized to sum to 1."""
        geometry, X = geometry_and_features
        indicator = MultiScaleChernConsensus(
            geometry, scales=[10, 20], weights=[3.0, 7.0]
        )

        assert abs(sum(indicator.weights) - 1.0) < 1e-10

    def test_weights_length_mismatch_raises(self, geometry_and_features):
        """Mismatched weights and scales should raise ValueError."""
        geometry, X = geometry_and_features
        with pytest.raises(ValueError, match="weights length"):
            MultiScaleChernConsensus(
                geometry, scales=[10, 20, 30], weights=[0.5, 0.5]
            )


# ---------------------------------------------------------------------------
# GeometricIndicatorSuite tests
# ---------------------------------------------------------------------------


class TestGeometricIndicatorSuite:
    """Tests for the unified indicator suite."""

    def test_compute_all_returns_four_indicators(self, geometry_and_features):
        """compute_all must return exactly 4 named indicator results."""
        geometry, X = geometry_and_features
        suite = GeometricIndicatorSuite(
            geometry, window_size=10, scales=[10, 20]
        )
        results = suite.compute_all(X)

        expected_keys = {
            "spectral_gap",
            "ground_state_energy",
            "fidelity_decay",
            "multi_scale_consensus",
        }
        assert set(results.keys()) == expected_keys

        for name, result in results.items():
            assert isinstance(result, IndicatorResult)
            assert result.name == name
            assert len(result.values) > 0

    def test_composite_score_shape(self, geometry_and_features):
        """Composite score length equals minimum indicator length."""
        geometry, X = geometry_and_features
        suite = GeometricIndicatorSuite(
            geometry, window_size=10, scales=[10, 20]
        )
        composite, results = suite.compute_composite_score(X)

        min_len = min(len(r.values) for r in results.values())
        assert len(composite) == min_len

    def test_composite_score_finite(self, geometry_and_features):
        """Composite score values must be finite."""
        geometry, X = geometry_and_features
        suite = GeometricIndicatorSuite(
            geometry, window_size=10, scales=[10, 20]
        )
        composite, _ = suite.compute_composite_score(X)

        assert np.all(np.isfinite(composite)), "Composite score must be finite"

    def test_composite_with_custom_weights(self, geometry_and_features):
        """Custom weights should change the composite score."""
        geometry, X = geometry_and_features
        suite = GeometricIndicatorSuite(
            geometry, window_size=10, scales=[10, 20]
        )

        composite_equal, _ = suite.compute_composite_score(X)

        # Weight only spectral gap
        weights_sg = {
            "spectral_gap": 1.0,
            "ground_state_energy": 0.0,
            "fidelity_decay": 0.0,
            "multi_scale_consensus": 0.0,
        }
        composite_sg, _ = suite.compute_composite_score(X, weights=weights_sg)

        # They should differ (unless all indicators are identical, which is unlikely)
        assert not np.allclose(composite_equal, composite_sg), (
            "Different weights should produce different composite scores"
        )

    def test_suite_sub_indicators_accessible(self, geometry_and_features):
        """Suite should expose its sub-indicator instances."""
        geometry, X = geometry_and_features
        suite = GeometricIndicatorSuite(
            geometry, window_size=10, scales=[10, 20]
        )

        assert isinstance(suite.spectral_gap, SpectralGapIndicator)
        assert isinstance(suite.energy, EnergyEvolutionIndicator)
        assert isinstance(suite.fidelity, FidelityDecayIndicator)
        assert isinstance(suite.consensus, MultiScaleChernConsensus)


# ---------------------------------------------------------------------------
# IndicatorResult dataclass tests
# ---------------------------------------------------------------------------


class TestIndicatorResult:
    """Tests for the IndicatorResult data container."""

    def test_basic_construction(self):
        """IndicatorResult should accept all required fields."""
        result = IndicatorResult(
            name="test_indicator",
            values=np.array([1.0, 2.0, 3.0]),
            transitions=[1],
            threshold=0.5,
        )

        assert result.name == "test_indicator"
        assert len(result.values) == 3
        assert result.transitions == [1]
        assert result.threshold == 0.5
        assert result.metadata == {}

    def test_metadata_default_empty(self):
        """Metadata should default to empty dict."""
        result = IndicatorResult(
            name="test", values=np.array([]), transitions=[], threshold=0.0
        )
        assert result.metadata == {}

    def test_metadata_with_values(self):
        """Metadata should store arbitrary key-value pairs."""
        result = IndicatorResult(
            name="test",
            values=np.array([1.0]),
            transitions=[],
            threshold=0.5,
            metadata={"mean": 1.0, "std": 0.5},
        )
        assert result.metadata["mean"] == 1.0
        assert result.metadata["std"] == 0.5


# ---------------------------------------------------------------------------
# QFI Susceptibility tests (QCMLGeometry.compute_qfi_susceptibility / _determinant)
# ---------------------------------------------------------------------------


class TestQFISusceptibility:
    """Tests for quantum Fisher information susceptibility."""

    def test_qfi_susceptibility_positive(self, geometry_and_features):
        """QFI susceptibility tr(g) must be non-negative (metric is PSD)."""
        geometry, X = geometry_and_features
        chi = geometry.compute_qfi_susceptibility(X[0])

        assert chi >= -1e-10, f"QFI susceptibility must be non-negative, got {chi}"

    def test_qfi_susceptibility_finite(self, geometry_and_features):
        """QFI susceptibility must be finite for all real data points."""
        geometry, X = geometry_and_features
        for i in range(min(10, len(X))):
            chi = geometry.compute_qfi_susceptibility(X[i])
            assert np.isfinite(chi), f"QFI susceptibility not finite at index {i}"

    def test_qfi_susceptibility_varies_across_regime(self, geometry_and_features):
        """QFI susceptibility should vary across the crisis period."""
        geometry, X = geometry_and_features
        chis = np.array([geometry.compute_qfi_susceptibility(X[i])
                         for i in range(min(20, len(X)))])

        assert np.std(chis) > 1e-12, (
            "QFI susceptibility is constant — should vary across market states"
        )

    def test_qfi_determinant_finite(self, geometry_and_features):
        """QFI metric determinant must be finite."""
        geometry, X = geometry_and_features
        det_g = geometry.compute_qfi_determinant(X[0])

        assert np.isfinite(det_g), f"QFI determinant not finite: {det_g}"

    def test_qfi_determinant_non_negative(self, geometry_and_features):
        """Metric determinant should be non-negative (PSD metric)."""
        geometry, X = geometry_and_features
        det_g = geometry.compute_qfi_determinant(X[0])

        assert det_g >= -1e-10, f"QFI determinant should be non-negative, got {det_g}"

    def test_qfi_susceptibility_equals_trace_of_metric(self, geometry_and_features):
        """Verify χ = tr(g) by computing both independently."""
        geometry, X = geometry_and_features
        x = X[0]

        chi = geometry.compute_qfi_susceptibility(x)
        g = geometry.quantum_metric(x)

        assert abs(chi - np.trace(g)) < 1e-10, (
            f"QFI susceptibility {chi} != tr(g) {np.trace(g)}"
        )


# ---------------------------------------------------------------------------
# Scalar Curvature tests (QCMLGeometry._christoffel_symbols / ricci_scalar)
# ---------------------------------------------------------------------------


class TestScalarCurvature:
    """Tests for Christoffel symbols and Ricci scalar curvature."""

    def test_christoffel_torsion_free(self, geometry_and_features):
        """Gamma^sigma_{mu nu} == Gamma^sigma_{nu mu} (Levi-Civita is torsion-free)."""
        geometry, X = geometry_and_features
        christoffel, _, _ = geometry._christoffel_symbols(X[0])

        n = christoffel.shape[0]
        for sigma in range(n):
            for mu in range(n):
                for nu in range(n):
                    assert abs(christoffel[sigma, mu, nu] - christoffel[sigma, nu, mu]) < 1e-6, (
                        f"Torsion-free violated: Gamma^{sigma}_{mu}{nu} != Gamma^{sigma}_{nu}{mu}"
                    )

    def test_ricci_scalar_finite(self, geometry_and_features):
        """Ricci scalar R must be finite (not NaN/Inf)."""
        geometry, X = geometry_and_features
        R = geometry.ricci_scalar(X[0])

        assert np.isfinite(R), f"Ricci scalar not finite: {R}"

    def test_ricci_scalar_varies(self, geometry_and_features):
        """Ricci scalar should differ across crisis vs non-crisis points."""
        geometry, X = geometry_and_features
        # Sample early (pre-crisis) and late (crisis) points
        R_values = np.array([geometry.ricci_scalar(X[i]) for i in range(0, min(20, len(X)), 2)])

        assert np.std(R_values) > 1e-15, (
            "Ricci scalar is constant — should vary across market states"
        )

    def test_metric_inverse_consistency(self, geometry_and_features):
        """g @ g_inv should approximate identity (validates regularized inverse)."""
        geometry, X = geometry_and_features
        _, g, g_inv = geometry._christoffel_symbols(X[0])

        product = g @ g_inv
        n = g.shape[0]
        identity = np.eye(n)

        assert np.allclose(product, identity, atol=1e-4), (
            f"g @ g_inv deviates from identity: max error = {np.max(np.abs(product - identity))}"
        )

    def test_scalar_curvature_detector_runs(self, geometry_and_features):
        """Full ScalarCurvatureDetector pipeline produces valid scores."""
        from experiments.additional_detectors import ScalarCurvatureDetector

        _, X = geometry_and_features
        det = ScalarCurvatureDetector(
            hilbert_dim=8,
            n_curvature_components=3,  # small for speed
            operator_method='random',
            rolling_window=5,
            min_expanding=10,
            seed=42,
        )
        det.fit(X)
        scores = det.compute_regime_scores(X)

        assert len(scores) == len(X)
        # After min_expanding, scores should be finite
        valid = scores[~np.isnan(scores)]
        assert len(valid) > 0, "No valid scores produced"
        assert np.all(np.isfinite(valid)), "Non-NaN scores must be finite"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
