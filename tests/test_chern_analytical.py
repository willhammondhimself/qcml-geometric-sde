"""
Analytical tests for Berry curvature and Chern number computation.

For the QCML error Hamiltonian H(x) = 1/2 * sum_k (A_k - x_k*I)^2,
the effective Hamiltonian (modulo identity) is -sum_k x_k A_k, which is
linear in x.

Key mathematical facts tested here:
  1. For hilbert_dim=2 with pure Pauli operators, the d-vector is linear
     and Berry curvature vanishes identically (away from degeneracies).
  2. Non-commuting operators in hilbert_dim >= 4 produce non-trivial Berry
     curvature through level repulsion in the ground-state manifold.
  3. The plaquette method converges with grid refinement.
  4. A level crossing (degeneracy) at the origin contributes a topological
     charge of 1/2 in the plaquette sum for 2-level systems.
"""

import numpy as np
import pytest

from qcml_geometry.core import QCMLGeometry


class TestChernAnalytical:
    """Analytical tests for Berry curvature and Chern number."""

    def _make_2level_geometry(self) -> QCMLGeometry:
        """Create a 2-level system with Pauli-Z and Pauli-X operators."""
        sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)

        geom = QCMLGeometry(n_features=2, hilbert_dim=2)
        geom.set_operators([sigma_z, sigma_x])
        return geom

    def _make_4level_geometry(self) -> QCMLGeometry:
        """Create a 4-level system with random non-commuting Hermitian operators.

        Non-commuting operators in higher dimensions produce non-trivial
        Berry curvature through the level repulsion mechanism.
        """
        rng = np.random.RandomState(42)
        def rand_hermitian(n):
            A = rng.randn(n, n) + 1j * rng.randn(n, n)
            return (A + A.conj().T) / 2

        A0 = rand_hermitian(4)
        A1 = rand_hermitian(4)

        geom = QCMLGeometry(n_features=2, hilbert_dim=4)
        geom.set_operators([A0, A1])
        return geom

    def _make_grid(self, n_grid: int, extent: float, n_features: int = 2,
                   offset: float = 0.0) -> np.ndarray:
        """Create a 2D grid of points in parameter space."""
        vals = np.linspace(-extent + offset, extent + offset, n_grid)
        X_grid = np.zeros((n_grid, n_grid, n_features))
        for i, xi in enumerate(vals):
            for j, xj in enumerate(vals):
                X_grid[i, j, 0] = xi
                X_grid[i, j, 1] = xj
        return X_grid

    def test_chern_vanishes_away_from_degeneracy_2level(self):
        """For 2-level system, Chern number is 0 when grid avoids the origin.

        The Berry curvature vanishes for the linear d-vector of the error
        Hamiltonian. When the grid does not contain the level crossing at
        the origin, the plaquette sum should give C ~ 0.
        """
        geom = self._make_2level_geometry()

        # Grid entirely in positive quadrant — no degeneracy
        X_grid = self._make_grid(n_grid=20, extent=1.5, offset=2.0)

        chern = geom.chern_number(X_grid, indices=(0, 1), method='plaquette')
        assert abs(chern) < 0.05, (
            f"Chern number |C| = {abs(chern):.4f} for 2-level system away from "
            f"degeneracy, expected ~0"
        )

    def test_degeneracy_contributes_half_charge_2level(self):
        """Level crossing at origin contributes topological charge of 1/2.

        When the grid contains the degeneracy point (origin), the plaquette
        method accumulates a Berry phase of pi (= Chern 1/2) from the
        gauge singularity, consistent with a Dirac monopole of charge 1/2.
        """
        geom = self._make_2level_geometry()

        # Grid centered on origin — contains the degeneracy
        X_grid = self._make_grid(n_grid=30, extent=2.0)

        chern = geom.chern_number(X_grid, indices=(0, 1), method='plaquette')
        assert abs(abs(chern) - 0.5) < 0.05, (
            f"Chern number |C| = {abs(chern):.4f} with degeneracy, "
            f"expected ~0.5 (half-monopole charge)"
        )

    def test_berry_curvature_vanishes_away_from_degeneracy(self):
        """Berry curvature should be ~0 away from level crossing for 2-level system."""
        geom = self._make_2level_geometry()

        x_far = np.array([2.0, 1.0])
        F = geom.berry_curvature(x_far)

        assert abs(F[0, 1]) < 0.01, (
            f"Berry curvature F_01 = {F[0, 1]:.6f} far from degeneracy, "
            f"expected ~0 for 2-level error Hamiltonian"
        )

    def test_berry_curvature_nontrivial_4level(self):
        """4-level system with non-commuting operators has non-trivial Berry curvature."""
        geom = self._make_4level_geometry()

        test_points = [
            np.array([0.5, 0.3]),
            np.array([1.0, 0.5]),
            np.array([0.3, 0.7]),
        ]

        max_curvature = 0.0
        for x in test_points:
            F = geom.berry_curvature(x)
            max_curvature = max(max_curvature, abs(F[0, 1]))

        assert max_curvature > 0.01, (
            f"Max Berry curvature F_01 = {max_curvature:.6f} across test points, "
            f"expected non-zero for 4-level system with non-commuting operators"
        )

    def test_chern_convergence_with_grid_refinement(self):
        """Chern number should converge as grid is refined."""
        geom = self._make_4level_geometry()

        chern_values = []
        for n_grid in [10, 20, 30]:
            X_grid = self._make_grid(n_grid, extent=2.0)
            chern = geom.chern_number(X_grid, indices=(0, 1), method='plaquette')
            chern_values.append(chern)

        # Successive differences should decrease (convergence)
        diff1 = abs(chern_values[1] - chern_values[0])
        diff2 = abs(chern_values[2] - chern_values[1])

        assert diff2 < diff1 + 0.01, (
            f"Chern values {[f'{c:.4f}' for c in chern_values]} should converge "
            f"(diff1={diff1:.4f}, diff2={diff2:.4f})"
        )

    def test_spectral_gap_minimum_at_origin(self):
        """Spectral gap should be smallest near the origin."""
        geom = self._make_2level_geometry()

        gap_origin = geom.spectral_gap(np.array([0.0, 0.0]))
        gap_far = geom.spectral_gap(np.array([3.0, 3.0]))

        assert gap_origin < gap_far, (
            f"Gap at origin ({gap_origin:.4f}) should be < gap far away ({gap_far:.4f})"
        )

    def test_spectral_gap_positive_away_from_origin(self):
        """Spectral gap should be strictly positive away from degeneracy."""
        geom = self._make_2level_geometry()

        test_points = [
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([1.0, 1.0]),
            np.array([2.0, 2.0]),
        ]

        for x in test_points:
            gap = geom.spectral_gap(x)
            assert gap > 0.1, (
                f"Spectral gap at {x} is {gap:.4f}, expected > 0.1"
            )

    def test_chern_nontrivial_4level(self):
        """4-level system should have non-zero Chern number over a large grid."""
        geom = self._make_4level_geometry()

        X_grid = self._make_grid(n_grid=25, extent=2.0)
        chern = geom.chern_number(X_grid, indices=(0, 1), method='plaquette')

        assert abs(chern) > 0.05, (
            f"Chern number |C| = {abs(chern):.4f} for 4-level system, "
            f"expected non-zero"
        )
