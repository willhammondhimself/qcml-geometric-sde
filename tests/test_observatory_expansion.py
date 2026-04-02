"""
Smoke tests for the 10 new observatory expansion observables (Mar 2026).

Tests verify:
- Core methods: spectral_entropy, geometric_phase_rate, hamiltonian_sensitivity,
  geodesic_curvature, effective_state_dimension, qgt_phase_rigidity,
  reduced_state_purity, spectral_complexity, berry_velocity_coupling,
  ricci_scalar_rate
- Detector classes: SpectralEntropyDetector, GeometricPhaseRateDetector,
  HamiltonianSensitivityDetector, GeodesicCurvatureDetector,
  EffectiveStateDimensionDetector, QGTPhaseRigidityDetector,
  ReducedPurityDetector, SpectralComplexityDetector,
  BerryVelocityCouplingDetector, CurvatureRateDetector
"""

import numpy as np
import pytest

from qcml_geometry import QCMLGeometry, create_test_data_sphere
from qcml_geometry.observables import (
    BerryVelocityCouplingDetector,
    CurvatureRateDetector,
    EffectiveStateDimensionDetector,
    GeodesicCurvatureDetector,
    GeometricPhaseRateDetector,
    HamiltonianSensitivityDetector,
    QGTPhaseRigidityDetector,
    ReducedPurityDetector,
    SpectralComplexityDetector,
    SpectralEntropyDetector,
)


class TestNewCoreMethods:
    """Test new methods added to QCMLGeometry."""

    @pytest.fixture
    def geometry(self):
        X = create_test_data_sphere(n_samples=50, noise=0.05, seed=42)
        geo = QCMLGeometry(n_features=3, hilbert_dim=4)
        geo.fit_operators(X, method="pca_inspired")
        return geo

    @pytest.fixture
    def geometry_large(self):
        """Larger Hilbert space for reduced purity bipartition."""
        X = create_test_data_sphere(n_samples=50, noise=0.05, seed=42)
        geo = QCMLGeometry(n_features=3, hilbert_dim=8)
        geo.fit_operators(X, method="random")
        return geo

    def test_spectral_entropy_nonneg(self, geometry):
        x = np.array([0.5, 0.3, 0.1])
        S = geometry.spectral_entropy(x, c=1.0)
        assert isinstance(S, float)
        assert S >= 0.0

    def test_spectral_entropy_bounded(self, geometry):
        x = np.array([0.5, 0.3, 0.1])
        S = geometry.spectral_entropy(x)
        assert S <= np.log(geometry.hilbert_dim) + 1e-6

    def test_geometric_phase_rate_finite(self, geometry):
        x1 = np.array([0.5, 0.3, 0.1])
        x2 = np.array([0.6, 0.4, 0.2])
        gpr = geometry.geometric_phase_rate(x1, x2)
        assert isinstance(gpr, float)
        assert np.isfinite(gpr)

    def test_geometric_phase_rate_finite_values(self, geometry):
        x = np.array([0.5, 0.3, 0.1])
        gpr = geometry.geometric_phase_rate(x, x)
        assert np.isfinite(gpr)

    def test_hamiltonian_sensitivity_nonneg(self, geometry):
        x1 = np.array([0.5, 0.3, 0.1])
        x2 = np.array([0.6, 0.4, 0.2])
        hs = geometry.hamiltonian_sensitivity(x1, x2)
        assert isinstance(hs, float)
        assert hs >= -1e-12  # variance is non-negative

    def test_hamiltonian_sensitivity_same_point(self, geometry):
        x = np.array([0.5, 0.3, 0.1])
        hs = geometry.hamiltonian_sensitivity(x, x)
        assert hs < 1e-10

    def test_geodesic_curvature_finite(self, geometry):
        x_prev = np.array([0.3, 0.2, 0.1])
        x_curr = np.array([0.5, 0.3, 0.1])
        x_next = np.array([0.7, 0.4, 0.2])
        gc = geometry.geodesic_curvature(x_prev, x_curr, x_next)
        assert isinstance(gc, float)
        assert np.isfinite(gc)
        assert gc >= 0.0

    def test_effective_state_dimension_bounded(self, geometry):
        x_points = np.array(
            [
                [0.5, 0.3, 0.1],
                [0.6, 0.4, 0.2],
                [0.7, 0.5, 0.3],
                [0.4, 0.2, 0.0],
            ]
        )
        states = []
        for x in x_points:
            psi = geometry.quasi_coherent_state(x)
            states.append(psi)
        states = np.array(states)
        esd = geometry.effective_state_dimension(states)
        assert isinstance(esd, float)
        assert esd >= 1.0 - 1e-6
        assert esd <= len(states) + 1e-6

    def test_qgt_phase_rigidity_nonneg(self, geometry):
        x = np.array([0.5, 0.3, 0.1])
        rho = geometry.qgt_phase_rigidity(x)
        assert isinstance(rho, float)
        assert rho >= 0.0

    def test_reduced_state_purity_bounded(self, geometry_large):
        x = np.array([0.5, 0.3, 0.1])
        purity = geometry_large.reduced_state_purity(x, partition=(2, 4))
        assert isinstance(purity, float)
        assert 0.0 <= purity <= 1.0 + 1e-6

    def test_reduced_state_purity_dimension_check(self, geometry_large):
        """Partition must multiply to hilbert_dim."""
        x = np.array([0.5, 0.3, 0.1])
        purity = geometry_large.reduced_state_purity(x, partition=(2, 4))
        assert np.isfinite(purity)

    def test_spectral_complexity_nonneg(self, geometry):
        x = np.array([0.5, 0.3, 0.1])
        sc = geometry.spectral_complexity(x, c=1.0)
        assert isinstance(sc, float)
        assert sc >= 0.0

    def test_spectral_complexity_bounded(self, geometry):
        x = np.array([0.5, 0.3, 0.1])
        sc = geometry.spectral_complexity(x)
        assert sc <= np.log(geometry.hilbert_dim) + 1e-6

    def test_berry_velocity_coupling_nonneg(self, geometry):
        x_curr = np.array([0.5, 0.3, 0.1])
        x_prev = np.array([0.4, 0.2, 0.0])
        bvc = geometry.berry_velocity_coupling(x_curr, x_prev)
        assert isinstance(bvc, float)
        assert bvc >= 0.0

    def test_ricci_scalar_rate_nonneg(self, geometry):
        x_curr = np.array([0.5, 0.3, 0.1])
        x_prev = np.array([0.4, 0.2, 0.0])
        rsr = geometry.ricci_scalar_rate(x_curr, x_prev)
        assert isinstance(rsr, float)
        assert rsr >= 0.0


class TestNewDetectors:
    """Smoke tests for the 10 new detector classes."""

    @pytest.fixture
    def test_data(self):
        """Generate synthetic test data (200 points, 5 features)."""
        rng = np.random.default_rng(42)
        T = 200
        X = rng.standard_normal((T, 5)) * 0.01
        # Add a volatility spike to simulate crisis
        X[150:170, :] *= 3.0
        return X

    @pytest.fixture
    def test_data_large(self):
        """Larger dataset for detectors that need more data."""
        rng = np.random.default_rng(42)
        T = 300
        X = rng.standard_normal((T, 5)) * 0.01
        X[200:220, :] *= 3.0
        return X

    def _smoke_test_detector(self, detector_cls, X, **kwargs):
        """Common smoke test: fit + compute_regime_scores returns array of correct length."""
        det = detector_cls(
            hilbert_dim=4,
            n_pca_components=3,
            rolling_window=10,
            operator_method="random",
            seed=42,
            normalization="sphere",
            **kwargs,
        )
        det.fit(X)
        scores = det.compute_regime_scores(X)
        assert scores.shape == (X.shape[0],), f"Expected ({X.shape[0]},), got {scores.shape}"
        # At least some non-NaN scores
        assert np.any(np.isfinite(scores)), "All scores are NaN"
        return scores

    def test_spectral_entropy_detector(self, test_data):
        self._smoke_test_detector(SpectralEntropyDetector, test_data)

    def test_geometric_phase_rate_detector(self, test_data):
        self._smoke_test_detector(GeometricPhaseRateDetector, test_data)

    def test_hamiltonian_sensitivity_detector(self, test_data):
        self._smoke_test_detector(HamiltonianSensitivityDetector, test_data)

    def test_geodesic_curvature_detector(self, test_data_large):
        self._smoke_test_detector(
            GeodesicCurvatureDetector,
            test_data_large,
            subsample=3,
        )

    def test_effective_state_dimension_detector(self, test_data):
        self._smoke_test_detector(EffectiveStateDimensionDetector, test_data)

    def test_qgt_phase_rigidity_detector(self, test_data):
        self._smoke_test_detector(QGTPhaseRigidityDetector, test_data)

    def test_reduced_purity_detector(self, test_data):
        det = ReducedPurityDetector(
            hilbert_dim=8,
            n_pca_components=3,
            rolling_window=10,
            operator_method="random",
            seed=42,
            normalization="sphere",
            partition=(2, 4),
        )
        det.fit(test_data)
        scores = det.compute_regime_scores(test_data)
        assert scores.shape == (test_data.shape[0],)
        assert np.any(np.isfinite(scores))

    def test_spectral_complexity_detector(self, test_data):
        self._smoke_test_detector(SpectralComplexityDetector, test_data)

    def test_berry_velocity_coupling_detector(self, test_data):
        self._smoke_test_detector(BerryVelocityCouplingDetector, test_data)

    def test_curvature_rate_detector(self, test_data_large):
        self._smoke_test_detector(
            CurvatureRateDetector,
            test_data_large,
            subsample=3,
        )

    def test_detector_names_unique(self):
        """All new detectors have unique names."""
        detectors = [
            SpectralEntropyDetector,
            GeometricPhaseRateDetector,
            HamiltonianSensitivityDetector,
            GeodesicCurvatureDetector,
            EffectiveStateDimensionDetector,
            QGTPhaseRigidityDetector,
            ReducedPurityDetector,
            SpectralComplexityDetector,
            BerryVelocityCouplingDetector,
            CurvatureRateDetector,
        ]
        names = []
        for cls in detectors:
            det = cls(hilbert_dim=4, n_pca_components=3, seed=42)
            names.append(det.name)
        assert len(names) == len(set(names)), f"Duplicate names: {names}"


class TestNewBaselines:
    """Smoke tests for the 4 new baseline detectors."""

    @pytest.fixture
    def test_data(self):
        rng = np.random.default_rng(42)
        T = 200
        X = rng.standard_normal((T, 5)) * 0.01
        X[150:170, :] *= 3.0
        return X

    def test_ewma_detector(self, test_data):
        from experiments.baselines import EWMADetector

        det = EWMADetector(decay=0.94, min_expanding=30)
        det.fit(test_data)
        scores = det.compute_regime_scores(test_data)
        assert scores.shape == (test_data.shape[0],)
        assert np.any(np.isfinite(scores))

    def test_mahalanobis_detector(self, test_data):
        from experiments.baselines import MahalanobisDetector

        det = MahalanobisDetector(min_expanding=30)
        det.fit(test_data)
        scores = det.compute_regime_scores(test_data)
        assert scores.shape == (test_data.shape[0],)
        assert np.any(np.isfinite(scores))

    def test_structural_break_detector(self, test_data):
        from experiments.baselines import StructuralBreakDetector

        det = StructuralBreakDetector(min_expanding=30)
        det.fit(test_data)
        scores = det.compute_regime_scores(test_data)
        assert scores.shape == (test_data.shape[0],)

    def test_transfer_entropy_detector(self, test_data):
        from experiments.baselines import TransferEntropyDetector

        det = TransferEntropyDetector(te_window=30, min_expanding=30)
        det.fit(test_data)
        scores = det.compute_regime_scores(test_data)
        assert scores.shape == (test_data.shape[0],)
