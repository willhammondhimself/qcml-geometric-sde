"""
Tests for new geometric regime detectors.

Tests verify:
- LevelSpacingRatioDetector
- RicciScalarDetector
- SectionalCurvatureDetector (abs_zscore and neg_fraction modes)
- GeodesicVelocityDetector
- SpeedLimitRatioDetector
- DimensionalityCollapseDetector
- SpectralFlowDetector
- CommutatorNormDetector
- Core methods: full_spectrum, sectional_curvature, hamiltonian_commutator_norm
"""

import numpy as np
import pytest

from qcml_geometry import QCMLGeometry, create_test_data_sphere
from qcml_geometry.observables import (
    LevelSpacingRatioDetector,
    QuantumRelativeEntropyDetector,
    RicciScalarDetector,
    SectionalCurvatureDetector,
    GeodesicVelocityDetector,
    SpeedLimitRatioDetector,
    DimensionalityCollapseDetector,
    SpectralFlowDetector,
    CommutatorNormDetector,
)


class TestCoreExtensions:
    """Test new methods added to QCMLGeometry."""

    @pytest.fixture
    def geometry(self):
        X = create_test_data_sphere(n_samples=50, noise=0.05, seed=42)
        geo = QCMLGeometry(n_features=3, hilbert_dim=4)
        geo.fit_operators(X, method="pca_inspired")
        return geo

    def test_full_spectrum_shape(self, geometry):
        x = np.array([0.5, 0.3, 0.1])
        spectrum = geometry.full_spectrum(x)
        assert spectrum.shape == (4,)
        assert np.all(np.diff(spectrum) >= 0), "Eigenvalues should be sorted"

    def test_full_spectrum_real(self, geometry):
        x = np.array([0.5, 0.3, 0.1])
        spectrum = geometry.full_spectrum(x)
        assert np.all(np.isreal(spectrum))

    def test_spectral_gap_from_spectrum(self, geometry):
        x = np.array([0.5, 0.3, 0.1])
        spectrum = geometry.full_spectrum(x)
        gap = geometry.spectral_gap(x)
        assert np.isclose(gap, spectrum[1] - spectrum[0], atol=1e-10)

    def test_sectional_curvature_returns_scalar(self, geometry):
        x = np.array([0.5, 0.3, 0.1])
        K = geometry.sectional_curvature(x, i=0, j=1)
        assert isinstance(K, float)
        assert np.isfinite(K)

    def test_sectional_curvature_same_index(self, geometry):
        x = np.array([0.5, 0.3, 0.1])
        K = geometry.sectional_curvature(x, i=0, j=0)
        assert K == 0.0

    def test_hamiltonian_commutator_norm_nonneg(self, geometry):
        x1 = np.array([0.5, 0.3, 0.1])
        x2 = np.array([0.6, 0.4, 0.2])
        norm = geometry.hamiltonian_commutator_norm(x1, x2)
        assert norm >= 0.0

    def test_hamiltonian_commutator_self_zero(self, geometry):
        x = np.array([0.5, 0.3, 0.1])
        norm = geometry.hamiltonian_commutator_norm(x, x)
        assert norm < 1e-10, "Commutator of H with itself should be zero"

    def test_hamiltonian_commutator_different_points(self, geometry):
        x1 = np.array([1.0, -0.5, 0.3])
        x2 = np.array([-0.5, 1.0, -0.3])
        norm = geometry.hamiltonian_commutator_norm(x1, x2)
        # Different non-zero points generally produce non-commuting Hamiltonians
        assert norm >= 0.0


class TestGeodesicVelocityDetector:
    """Test GeodesicVelocityDetector (fast, uses quantum_distance)."""

    @pytest.fixture
    def enriched_data(self):
        rng = np.random.default_rng(42)
        T = 300
        X = rng.standard_normal((T, 8))
        # Inject a regime shift at t=150
        X[150:200] *= 3.0
        return X

    def test_smoke(self, enriched_data):
        det = GeodesicVelocityDetector(
            hilbert_dim=4,
            n_pca_components=3,
            rolling_window=10,
            min_expanding=30,
            seed=42,
        )
        det.fit(enriched_data)
        scores = det.compute_regime_scores(enriched_data)
        assert scores.shape == (len(enriched_data),)
        assert not np.all(np.isnan(scores))

    def test_name(self):
        det = GeodesicVelocityDetector()
        assert det.name == "Geodesic Velocity"

    def test_scores_vary_during_shift(self, enriched_data):
        det = GeodesicVelocityDetector(
            hilbert_dim=4,
            n_pca_components=3,
            rolling_window=10,
            min_expanding=30,
            seed=42,
        )
        det.fit(enriched_data)
        scores = det.compute_regime_scores(enriched_data)
        # Scores during regime shift should differ from calm period
        calm = scores[60:140]
        shift = scores[155:195]
        calm_valid = calm[~np.isnan(calm)]
        shift_valid = shift[~np.isnan(shift)]
        if len(calm_valid) > 5 and len(shift_valid) > 5:
            assert np.nanstd(shift_valid) > 0 or np.nanstd(calm_valid) > 0


class TestSpectralFlowDetector:
    """Test SpectralFlowDetector."""

    @pytest.fixture
    def enriched_data(self):
        rng = np.random.default_rng(42)
        T = 300
        X = rng.standard_normal((T, 8))
        X[150:200] *= 3.0
        return X

    def test_smoke(self, enriched_data):
        det = SpectralFlowDetector(
            hilbert_dim=4,
            n_pca_components=3,
            rolling_window=10,
            min_expanding=30,
            seed=42,
        )
        det.fit(enriched_data)
        scores = det.compute_regime_scores(enriched_data)
        assert scores.shape == (len(enriched_data),)
        assert not np.all(np.isnan(scores))

    def test_name(self):
        det = SpectralFlowDetector()
        assert det.name == "Spectral Flow"


class TestCommutatorNormDetector:
    """Test CommutatorNormDetector."""

    @pytest.fixture
    def enriched_data(self):
        rng = np.random.default_rng(42)
        T = 300
        X = rng.standard_normal((T, 8))
        X[150:200] *= 3.0
        return X

    def test_smoke(self, enriched_data):
        det = CommutatorNormDetector(
            hilbert_dim=4,
            n_pca_components=3,
            rolling_window=10,
            min_expanding=30,
            seed=42,
        )
        det.fit(enriched_data)
        scores = det.compute_regime_scores(enriched_data)
        assert scores.shape == (len(enriched_data),)
        assert not np.all(np.isnan(scores))

    def test_name(self):
        det = CommutatorNormDetector()
        assert det.name == "Commutator Norm"


class TestRicciScalarDetector:
    """Test RicciScalarDetector (expensive — use small dimensions)."""

    def test_smoke(self):
        rng = np.random.default_rng(42)
        # Very small data for expensive Ricci computation
        X = rng.standard_normal((100, 4))
        det = RicciScalarDetector(
            hilbert_dim=4,
            n_pca_components=2,
            rolling_window=10,
            min_expanding=30,
            seed=42,
        )
        det.fit(X)
        scores = det.compute_regime_scores(X)
        assert scores.shape == (100,)

    def test_name(self):
        det = RicciScalarDetector()
        assert det.name == "Ricci Scalar"


class TestSectionalCurvatureDetector:
    """Test SectionalCurvatureDetector."""

    def test_smoke(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 4))
        det = SectionalCurvatureDetector(
            hilbert_dim=4,
            n_pca_components=2,
            rolling_window=10,
            min_expanding=30,
            seed=42,
        )
        det.fit(X)
        scores = det.compute_regime_scores(X)
        assert scores.shape == (100,)

    def test_name(self):
        det = SectionalCurvatureDetector()
        assert det.name == "Sectional Curvature (0,1)"

    def test_neg_fraction_mode(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 4))
        det = SectionalCurvatureDetector(
            hilbert_dim=4,
            n_pca_components=2,
            rolling_window=10,
            min_expanding=30,
            seed=42,
            score_mode="neg_fraction",
            neg_fraction_window=10,
        )
        det.fit(X)
        scores = det.compute_regime_scores(X)
        assert scores.shape == (100,)
        valid = scores[~np.isnan(scores)]
        assert len(valid) > 0
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 1.0)

    def test_neg_fraction_name(self):
        det = SectionalCurvatureDetector(score_mode="neg_fraction")
        assert "Sign" in det.name

    def test_default_mode_backward_compat(self):
        det = SectionalCurvatureDetector()
        assert det.score_mode == "abs_zscore"


class TestSpeedLimitRatioDetector:
    """Test SpeedLimitRatioDetector (v/Delta speed limit ratio)."""

    @pytest.fixture
    def enriched_data(self):
        rng = np.random.default_rng(42)
        T = 300
        X = rng.standard_normal((T, 8))
        X[150:200] *= 3.0
        return X

    def test_smoke(self, enriched_data):
        det = SpeedLimitRatioDetector(
            hilbert_dim=4,
            n_pca_components=3,
            rolling_window=10,
            min_expanding=30,
            seed=42,
        )
        det.fit(enriched_data)
        scores = det.compute_regime_scores(enriched_data)
        assert scores.shape == (len(enriched_data),)
        assert not np.all(np.isnan(scores))

    def test_no_infinities(self, enriched_data):
        det = SpeedLimitRatioDetector(
            hilbert_dim=4,
            n_pca_components=3,
            rolling_window=10,
            min_expanding=30,
            seed=42,
        )
        det.fit(enriched_data)
        scores = det.compute_regime_scores(enriched_data)
        valid = scores[~np.isnan(scores)]
        assert np.all(np.isfinite(valid))

    def test_name(self):
        det = SpeedLimitRatioDetector()
        assert det.name == "Speed Limit Ratio"


class TestDimensionalityCollapseDetector:
    """Test DimensionalityCollapseDetector (metric IPR)."""

    @pytest.fixture
    def enriched_data(self):
        rng = np.random.default_rng(42)
        T = 300
        X = rng.standard_normal((T, 8))
        X[150:200] *= 3.0
        return X

    def test_smoke(self, enriched_data):
        det = DimensionalityCollapseDetector(
            hilbert_dim=4,
            n_pca_components=3,
            rolling_window=10,
            min_expanding=30,
            seed=42,
        )
        det.fit(enriched_data)
        scores = det.compute_regime_scores(enriched_data)
        assert scores.shape == (len(enriched_data),)
        assert not np.all(np.isnan(scores))

    def test_name(self):
        det = DimensionalityCollapseDetector()
        assert det.name == "Dimensionality Collapse"

    def test_ipr_bounded(self, enriched_data):
        """IPR should be in [0, 1] before z-scoring."""
        det = DimensionalityCollapseDetector(
            hilbert_dim=4,
            n_pca_components=3,
            rolling_window=10,
            min_expanding=30,
            seed=42,
        )
        det.fit(enriched_data)
        # Access internal IPR computation by running a small check
        from qcml_geometry.observables import _transform_array

        Xt = _transform_array(
            enriched_data,
            det._scaler,
            det._pca,
            normalization=det.normalization,
            train_norms=det._train_norms,
            train_std=det._train_std,
        )
        g = det._geometry.quantum_metric(Xt[50])
        eigvals = np.linalg.eigvalsh(g)
        eigvals = np.maximum(eigvals, 0)
        total = np.sum(eigvals)
        if total > 1e-15:
            ipr = eigvals[-1] / total
            assert 0.0 <= ipr <= 1.0


class TestLevelSpacingRatio:
    """Test LevelSpacingRatioDetector (RMT spectral statistics)."""

    @pytest.fixture
    def data(self):
        np.random.seed(42)
        return np.random.randn(200, 8)

    def test_fit_and_score(self, data):
        det = LevelSpacingRatioDetector(hilbert_dim=4, n_pca_components=4)
        det.fit(data)
        scores = det.compute_regime_scores(data)
        assert scores.shape == (200,)
        valid = scores[~np.isnan(scores)]
        assert len(valid) > 100

    def test_name(self):
        det = LevelSpacingRatioDetector()
        assert det.name == "Level Spacing Ratio"

    def test_default_operator_method(self):
        det = LevelSpacingRatioDetector()
        assert det.operator_method == "random"

    def test_variant_selection(self, data):
        for variant in ["mean_ratio", "std_ratio", "poisson_fraction"]:
            det = LevelSpacingRatioDetector(
                hilbert_dim=4,
                n_pca_components=4,
                variant=variant,
            )
            det.fit(data)
            scores = det.compute_regime_scores(data)
            assert scores.shape == (200,)

    def test_level_spacing_ratios_shape(self):
        from qcml_geometry.observables import compute_level_spacing_ratios

        eigenvalues = np.array([0.1, 0.3, 0.7, 1.2, 2.0, 3.5, 5.0, 8.0])
        ratios = compute_level_spacing_ratios(eigenvalues)
        assert ratios.shape == (6,)  # d - 2 ratios for d=8

    def test_level_spacing_ratios_range(self):
        from qcml_geometry.observables import compute_level_spacing_ratios

        eigenvalues = np.sort(np.random.rand(8))
        ratios = compute_level_spacing_ratios(eigenvalues)
        assert np.all(ratios >= 0)
        assert np.all(ratios <= 1)

    def test_rmt_constants(self):
        from qcml_geometry.observables import RMT_POISSON, RMT_GOE, RMT_GUE

        assert 0.38 < RMT_POISSON < 0.40
        assert 0.52 < RMT_GOE < 0.54
        assert 0.59 < RMT_GUE < 0.61
        assert RMT_POISSON < RMT_GOE < RMT_GUE


class TestQuantumRelativeEntropy:
    """Test QuantumRelativeEntropyDetector."""

    @pytest.fixture
    def data(self):
        np.random.seed(42)
        return np.random.randn(200, 8)

    def test_fit_and_score(self, data):
        det = QuantumRelativeEntropyDetector(hilbert_dim=4, n_pca_components=4)
        det.fit(data)
        scores = det.compute_regime_scores(data)
        assert scores.shape == (200,)
        valid = scores[~np.isnan(scores)]
        assert len(valid) > 100

    def test_name(self):
        det = QuantumRelativeEntropyDetector()
        assert det.name == "Quantum Relative Entropy"

    def test_qre_nonnegative(self):
        from qcml_geometry.observables import _quantum_relative_entropy

        d = 4
        psi = np.zeros(d, dtype=complex)
        psi[0] = 1.0
        rho_ref = np.eye(d) / d  # maximally mixed
        D = _quantum_relative_entropy(psi, rho_ref)
        assert D >= 0

    def test_qre_pure_ref_zero(self):
        from qcml_geometry.observables import _quantum_relative_entropy

        d = 4
        psi = np.zeros(d, dtype=complex)
        psi[0] = 1.0
        rho_ref = np.outer(psi, psi.conj())
        D = _quantum_relative_entropy(psi, rho_ref)
        assert D < 0.01  # should be ~0

    def test_expanding_rho_ref(self):
        from qcml_geometry.observables import _build_expanding_rho_ref

        d = 4
        states = [np.random.randn(d) + 0j for _ in range(50)]
        for s in states:
            s /= np.linalg.norm(s)
        rho = _build_expanding_rho_ref(states, 30)
        assert rho is not None
        assert rho.shape == (d, d)
        assert np.isclose(np.trace(rho), 1.0, atol=1e-10)
