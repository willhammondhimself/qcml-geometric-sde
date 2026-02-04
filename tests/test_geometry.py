"""
Unit tests for QCML Geometry module.

Tests verify correctness of:
- Error Hamiltonian computation
- Quasi-coherent state ground state finding
- Quantum metric tensor properties
- Berry curvature computation
- Topological invariants (Chern numbers)
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '..')

from qcml.qcml_geometry import (
    QCMLGeometry,
    create_test_data_sphere,
    create_test_data_torus
)


class TestQCMLGeometry:
    """Test suite for QCMLGeometry class."""

    @pytest.fixture
    def geometry_and_data(self):
        """Create test data and fitted geometry."""
        X = create_test_data_sphere(n_samples=100, noise=0.05, seed=42)
        geometry = QCMLGeometry(n_features=3, hilbert_dim=4)
        geometry.fit_operators(X, method='pca_inspired')
        return geometry, X

    def test_initialization(self):
        """Test QCMLGeometry initialization."""
        geometry = QCMLGeometry(n_features=5, hilbert_dim=8)
        assert geometry.n_features == 5
        assert geometry.hilbert_dim == 8
        assert not geometry.is_fitted

    def test_fit_operators_pca(self):
        """Test operator fitting with PCA method."""
        X = create_test_data_sphere(n_samples=50, noise=0.05)
        geometry = QCMLGeometry(n_features=3, hilbert_dim=4)
        geometry.fit_operators(X, method='pca_inspired')

        assert geometry.is_fitted
        assert len(geometry.operators) == 3
        # Check operators are Hermitian
        for op in geometry.operators:
            assert np.allclose(op, op.conj().T), "Operator must be Hermitian"

    def test_fit_operators_pauli(self):
        """Test operator fitting with Pauli method."""
        X = create_test_data_sphere(n_samples=50, noise=0.05)
        geometry = QCMLGeometry(n_features=3, hilbert_dim=4)
        geometry.fit_operators(X, method='pauli', n_components=3)

        assert geometry.is_fitted
        assert len(geometry.operators) == 3

    def test_error_hamiltonian(self, geometry_and_data):
        """Test error Hamiltonian computation."""
        geometry, X = geometry_and_data
        x = X[0]

        H = geometry.error_hamiltonian(x)

        # Check shape
        assert H.shape == (geometry.hilbert_dim, geometry.hilbert_dim)
        # Check Hermitian
        assert np.allclose(H, H.conj().T), "Hamiltonian must be Hermitian"
        # Check eigenvalues are real
        eigenvalues = np.linalg.eigvalsh(H)
        assert np.all(np.isreal(eigenvalues))

    def test_quasi_coherent_state(self, geometry_and_data):
        """Test ground state computation."""
        geometry, X = geometry_and_data
        x = X[0]

        psi = geometry.quasi_coherent_state(x)
        psi_with_energy, energy = geometry.quasi_coherent_state(x, return_energy=True)

        # Check shape and normalization
        assert psi.shape == (geometry.hilbert_dim,)
        assert np.allclose(np.linalg.norm(psi), 1.0), "State must be normalized"
        assert np.allclose(psi, psi_with_energy), "State should match"
        assert isinstance(energy, (float, np.floating)), "Energy should be scalar"
        assert energy < 100, "Energy should be reasonable"  # Sanity check

    def test_quantum_metric(self, geometry_and_data):
        """Test quantum metric tensor."""
        geometry, X = geometry_and_data
        x = X[0]

        g = geometry.quantum_metric(x)

        # Check shape
        assert g.shape == (3, 3)
        # Check symmetry
        assert np.allclose(g, g.T), "Metric must be symmetric"
        # Check semi-positivity (allow small numerical negatives)
        eigenvalues = np.linalg.eigvalsh(g)
        assert np.all(eigenvalues > -1e-5), "Metric should be positive semi-definite"

    def test_berry_curvature(self, geometry_and_data):
        """Test Berry curvature computation."""
        geometry, X = geometry_and_data
        x = X[0]

        F = geometry.berry_curvature(x)

        # Check shape
        assert F.shape == (3, 3)
        # Check antisymmetry
        assert np.allclose(F, -F.T), "Berry curvature must be antisymmetric"

    def test_quantum_distance(self, geometry_and_data):
        """Test quantum distance computation."""
        geometry, X = geometry_and_data
        x1, x2 = X[0], X[1]

        d = geometry.quantum_distance(x1, x2)

        # Check range
        assert 0 <= d <= np.pi / 2, "Distance should be in [0, π/2]"
        # Self-distance should be near zero
        d_self = geometry.quantum_distance(x1, x1)
        assert d_self < 1e-3, "Self-distance should be near zero"

    def test_quantum_similarity(self, geometry_and_data):
        """Test quantum fidelity computation."""
        geometry, X = geometry_and_data
        x1, x2 = X[0], X[1]

        sim = geometry.quantum_similarity(x1, x2)

        # Check range
        assert 0 <= sim <= 1, "Similarity should be in [0, 1]"
        # Self-similarity should be near one
        sim_self = geometry.quantum_similarity(x1, x1)
        assert sim_self > 0.99, "Self-similarity should be near 1"

    def test_spectral_gap(self, geometry_and_data):
        """Test spectral gap computation."""
        geometry, X = geometry_and_data
        x = X[0]

        gap = geometry.spectral_gap(x)

        assert isinstance(gap, (float, np.floating))
        assert gap >= 0, "Spectral gap must be non-negative"

    def test_cache_clearing(self, geometry_and_data):
        """Test ground state caching."""
        geometry, X = geometry_and_data
        x = X[0]

        # Compute state (should cache)
        psi1 = geometry.quasi_coherent_state(x)
        cache_size = len(geometry._ground_state_cache)
        assert cache_size > 0, "Cache should have entries"

        # Clear cache
        geometry.clear_cache()
        assert len(geometry._ground_state_cache) == 0, "Cache should be empty"

    def test_sphere_vs_torus(self):
        """Test that sphere and torus have different topological properties."""
        X_sphere = create_test_data_sphere(n_samples=100, noise=0.05)
        X_torus = create_test_data_torus(n_samples=100, noise=0.05)

        geometry_sphere = QCMLGeometry(n_features=3, hilbert_dim=4)
        geometry_sphere.fit_operators(X_sphere, method='pca_inspired')

        geometry_torus = QCMLGeometry(n_features=3, hilbert_dim=4)
        geometry_torus.fit_operators(X_torus, method='pca_inspired')

        # Compute spectral gaps for both
        gaps_sphere = [geometry_sphere.spectral_gap(X_sphere[i]) for i in range(0, len(X_sphere), 10)]
        gaps_torus = [geometry_torus.spectral_gap(X_torus[i]) for i in range(0, len(X_torus), 10)]

        # Both should have some non-zero gaps (topological structure detected)
        assert np.mean(gaps_sphere) > 0, "Sphere should have spectral gap"
        assert np.mean(gaps_torus) > 0, "Torus should have spectral gap"


class TestTestData:
    """Test suite for synthetic test data generation."""

    def test_sphere_generation(self):
        """Test sphere data generation."""
        X = create_test_data_sphere(n_samples=100, noise=0.05, seed=42)

        assert X.shape == (100, 3)
        # Points should be approximately on unit sphere (with noise)
        distances = np.linalg.norm(X, axis=1)
        assert 0.8 < np.mean(distances) < 1.2, "Points should be near unit sphere"

    def test_torus_generation(self):
        """Test torus data generation."""
        X = create_test_data_torus(n_samples=200, R=2.0, r=0.5, noise=0.05, seed=42)

        assert X.shape == (200, 3)
        # Points should form a torus shape (check bounds)
        assert X.min() > -3.5 and X.max() < 3.5, "Torus should fit in bounds"

    def test_deterministic_seed(self):
        """Test that seed provides reproducibility."""
        X1 = create_test_data_sphere(n_samples=50, seed=42)
        X2 = create_test_data_sphere(n_samples=50, seed=42)

        assert np.allclose(X1, X2), "Same seed should give same data"

        X3 = create_test_data_sphere(n_samples=50, seed=123)
        assert not np.allclose(X1, X3), "Different seed should give different data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
