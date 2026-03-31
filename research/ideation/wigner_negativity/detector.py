"""
Wigner Negativity Detector for Financial Regime Detection

Computes the discrete Wigner function negativity of the QCML density matrix
rho = |psi(x)><psi(x)| as a non-classicality indicator.

For a d-dimensional system, the discrete Wigner function is defined on a
d x d phase-space grid (Gibbons et al. 2004, Gross 2006):

    W(alpha) = (1/d) Tr(A_alpha * rho)

where A_alpha are the d^2 phase-point operators forming a complete,
informationally-complete set. For general d (not necessarily prime),
we use the Wootters construction via displacement operators built from
generalized Pauli (clock and shift) matrices.

The Wigner negativity is:

    N(rho) = (sum_alpha |W(alpha)| - 1) / 2

This equals zero for "classical" states (those with non-negative Wigner
representation) and is positive for non-classical states. In quantum
information, Wigner negativity is a resource for quantum computation
(Veitch et al. 2012).

Hypothesis: During financial crises, the QCML ground state enters a
"non-classical" regime with elevated Wigner negativity, reflecting
the breakdown of Gaussian/classical descriptions of market dynamics.

References:
    - Gibbons, Hoffman, Wootters (2004). Discrete phase space based on
      finite fields. Phys. Rev. A 70, 062101.
    - Gross (2006). Hudson's theorem for finite-dimensional quantum systems.
      J. Math. Phys. 47, 122107.
    - Veitch, Ferrie, Gross, Emerson (2012). Negative quasi-probability as
      a resource for quantum computation. New J. Phys. 14, 113011.
    - Wootters (1987). A Wigner-function formulation of finite-state quantum
      mechanics. Ann. Phys. 176, 1-21.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, List

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qcml_geometry.core import QCMLGeometry

logger = logging.getLogger(__name__)


def _build_clock_shift(d: int):
    """Build generalized Pauli clock (Z) and shift (X) operators for dimension d.

    Z|j> = omega^j |j>   where omega = exp(2*pi*i/d)
    X|j> = |j+1 mod d>

    Args:
        d: Hilbert space dimension.

    Returns:
        (Z, X): Clock and shift matrices, each (d, d) complex.
    """
    omega = np.exp(2j * np.pi / d)

    # Clock operator: diagonal with omega^j
    Z = np.diag([omega ** j for j in range(d)])

    # Shift operator: cyclic permutation
    X = np.zeros((d, d), dtype=np.complex128)
    for j in range(d):
        X[(j + 1) % d, j] = 1.0

    return Z, X


def _build_displacement_operators(d: int):
    """Build the d^2 displacement operators D(p, q) = tau^(p*q) * X^p * Z^q.

    where tau = exp(i*pi*(d+1)/d) for even d, or tau = exp(i*pi/d) for odd d.
    These form the displacement operator basis for the discrete Wigner function.

    Args:
        d: Hilbert space dimension.

    Returns:
        displacements: Array of shape (d, d, d, d) where displacements[p, q]
            is the displacement operator D(p, q).
    """
    Z, X = _build_clock_shift(d)

    # Phase factor: for the Wootters construction, we need
    # D(p,q) = tau^(p*q) X^p Z^q where tau handles the non-commutativity
    # For general d, tau = exp(i*pi/d) works (Gross 2006 Eq. 3)
    if d % 2 == 0:
        # For even d, use tau = exp(i*pi*(d+1)/d) (Gross convention)
        tau = np.exp(1j * np.pi * (d + 1) / d)
    else:
        tau = np.exp(1j * np.pi / d)

    # Precompute X^p and Z^q
    X_powers = [np.eye(d, dtype=np.complex128)]
    Z_powers = [np.eye(d, dtype=np.complex128)]
    for _ in range(d - 1):
        X_powers.append(X_powers[-1] @ X)
        Z_powers.append(Z_powers[-1] @ Z)

    displacements = np.zeros((d, d, d, d), dtype=np.complex128)
    for p in range(d):
        for q in range(d):
            phase = tau ** (p * q)
            displacements[p, q] = phase * (X_powers[p] @ Z_powers[q])

    return displacements


def _build_phase_point_operators(d: int):
    """Build the d^2 phase-point operators A(p, q) for the discrete Wigner function.

    A(p, q) = (1/d) * sum_{p', q'} omega^(p*q' - q*p') * D(p', q')

    where omega = exp(2*pi*i/d) and D are displacement operators.
    Each A(p, q) is Hermitian with Tr(A(p, q)) = 1.

    Args:
        d: Hilbert space dimension.

    Returns:
        phase_point_ops: Array of shape (d, d, d, d) where phase_point_ops[p, q]
            is the phase-point operator A(p, q).
    """
    omega = np.exp(2j * np.pi / d)
    displacements = _build_displacement_operators(d)

    phase_point_ops = np.zeros((d, d, d, d), dtype=np.complex128)

    for p in range(d):
        for q in range(d):
            A_pq = np.zeros((d, d), dtype=np.complex128)
            for pp in range(d):
                for qp in range(d):
                    phase = omega ** (p * qp - q * pp)
                    A_pq += phase * displacements[pp, qp]
            phase_point_ops[p, q] = A_pq / d

    return phase_point_ops


def compute_wigner_function(rho: np.ndarray, phase_point_ops: np.ndarray) -> np.ndarray:
    """Compute the discrete Wigner function W(p, q) for density matrix rho.

    W(p, q) = (1/d) * Tr(A(p, q) * rho)

    The Wigner function is real-valued and sums to 1 for any valid
    density matrix.

    Args:
        rho: Density matrix of shape (d, d).
        phase_point_ops: Phase-point operators of shape (d, d, d, d).

    Returns:
        W: Wigner function values on d x d grid, shape (d, d).
    """
    d = rho.shape[0]
    W = np.zeros((d, d), dtype=np.float64)

    for p in range(d):
        for q in range(d):
            # W(p, q) = (1/d) Tr(A(p,q) * rho)
            W[p, q] = np.real(np.trace(phase_point_ops[p, q] @ rho)) / d

    return W


def compute_wigner_negativity(rho: np.ndarray, phase_point_ops: np.ndarray) -> float:
    """Compute the Wigner negativity of a density matrix.

    N(rho) = (sum_{p,q} |W(p,q)| - 1) / 2

    This is zero for states with non-negative Wigner representation
    and positive for "non-classical" states.

    Args:
        rho: Density matrix of shape (d, d).
        phase_point_ops: Phase-point operators of shape (d, d, d, d).

    Returns:
        negativity: Non-negative scalar. Zero means classically representable.
    """
    W = compute_wigner_function(rho, phase_point_ops)
    return float((np.sum(np.abs(W)) - 1.0) / 2.0)


class WignerNegativityDetector:
    """Regime detection via Wigner function negativity of QCML ground states.

    The detector:
    1. Fits a QCML geometry (PCA + Hermitian operators) on the data
    2. For each time step, computes the ground state |psi(x_t)>
    3. Forms rho_t = |psi><psi| (pure state density matrix)
    4. Computes the discrete Wigner function on a d x d phase-space grid
    5. Measures Wigner negativity N(rho) = (sum|W| - 1) / 2
    6. Returns expanding-window z-scored negativity as the anomaly score

    Higher scores indicate more "non-classical" behavior, hypothesized
    to spike before/during financial crises.

    Args:
        hilbert_dim: Dimension of the QCML Hilbert space (default 8).
        n_pca_components: Number of PCA components for feature reduction.
        operator_method: Method for constructing QCML operators.
        rolling_window: Window for rolling mean smoothing.
        min_expanding: Minimum observations before z-scoring begins.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        hilbert_dim: int = 8,
        n_pca_components: int = 8,
        operator_method: str = 'random',
        scale_exponent: Optional[float] = None,
        rolling_window: int = 20,
        min_expanding: int = 60,
        seed: int = 42,
        normalization: str = 'soft',
    ):
        self.hilbert_dim = hilbert_dim
        self.n_pca_components = n_pca_components
        self.operator_method = operator_method
        self.scale_exponent = scale_exponent
        self.rolling_window = rolling_window
        self.min_expanding = min_expanding
        self.seed = seed
        self.normalization = normalization

        self._geometry = None
        self._scaler = None
        self._pca = None
        self._train_norms = None
        self._train_std = None
        self._phase_point_ops = None

    @property
    def name(self) -> str:
        return "Wigner Negativity"

    def fit(self, X: np.ndarray) -> 'WignerNegativityDetector':
        """Fit the detector on feature matrix X.

        Args:
            X: Feature matrix of shape (T, n_features).

        Returns:
            self
        """
        np.random.seed(self.seed)
        n_components = min(self.n_pca_components, X.shape[1])

        self._scaler = StandardScaler()
        self._scaler.fit(X)

        self._pca = PCA(n_components=n_components)
        X_scaled = self._scaler.transform(X)
        self._pca.fit(X_scaled)

        X_pca_raw = self._pca.transform(X_scaled)
        self._train_norms = np.linalg.norm(X_pca_raw, axis=1)
        self._train_std = np.std(X_pca_raw, axis=0)

        # Apply normalization
        if self.normalization == 'sphere':
            norms = np.linalg.norm(X_pca_raw, axis=1, keepdims=True)
            X_pca = X_pca_raw / (norms + 1e-8)
        elif self.normalization == 'soft':
            median_norm = np.median(self._train_norms)
            norms = np.linalg.norm(X_pca_raw, axis=1, keepdims=True)
            X_pca = X_pca_raw / (norms + median_norm)
        else:
            X_pca = X_pca_raw

        self._geometry = QCMLGeometry(
            n_features=X_pca.shape[1], hilbert_dim=self.hilbert_dim
        )
        self._geometry.fit_operators(
            X_pca, method=self.operator_method,
            scale_exponent=self.scale_exponent,
        )

        # Pre-build phase-point operators (expensive but done once)
        logger.info(f"Building phase-point operators for d={self.hilbert_dim}...")
        self._phase_point_ops = _build_phase_point_operators(self.hilbert_dim)
        logger.info("Phase-point operators built.")

        return self

    def _transform_point(self, x_raw: np.ndarray) -> np.ndarray:
        """Transform a single raw data point through scaler + PCA + normalization."""
        x_scaled = self._scaler.transform(x_raw.reshape(1, -1))
        x_pca = self._pca.transform(x_scaled).ravel()

        if self.normalization == 'sphere':
            norm = np.linalg.norm(x_pca)
            return x_pca / (norm + 1e-8)
        elif self.normalization == 'soft':
            median_norm = np.median(self._train_norms)
            norm = np.linalg.norm(x_pca)
            return x_pca / (norm + median_norm)
        return x_pca

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute Wigner negativity z-scores for each time step.

        Uses expanding-window z-score normalization to avoid future data leakage.

        Args:
            X: Feature matrix of shape (T, n_features).

        Returns:
            z_scores: Array of shape (T,) with NaN for warmup period.
        """
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        import pandas as pd

        T = X.shape[0]
        negativity_vals = np.empty(T)

        for t in range(T):
            x_t = self._transform_point(X[t])
            psi = self._geometry.quasi_coherent_state(x_t)

            # Pure state density matrix: rho = |psi><psi|
            rho = np.outer(psi, psi.conj())

            # Compute Wigner negativity
            negativity_vals[t] = compute_wigner_negativity(rho, self._phase_point_ops)

        # Expanding-window z-score (causal: only uses past data)
        rolling_vals = (
            pd.Series(negativity_vals)
            .rolling(window=self.rolling_window, min_periods=1)
            .mean()
            .values
        )

        z_scores = np.full(T, np.nan)
        start = max(self.min_expanding, 0)
        for t in range(start, T):
            past = rolling_vals[:t]
            past_valid = past[~np.isnan(past)]
            if len(past_valid) < 10:
                continue
            mu = np.mean(past_valid)
            sigma = np.std(past_valid, ddof=1)
            if sigma > 1e-12:
                z_scores[t] = (rolling_vals[t] - mu) / sigma
            else:
                z_scores[t] = 0.0

        return z_scores

    def compute_raw_negativity(self, X: np.ndarray) -> np.ndarray:
        """Compute raw Wigner negativity values (without z-scoring).

        Useful for analysis and visualization.

        Args:
            X: Feature matrix of shape (T, n_features).

        Returns:
            negativity: Array of shape (T,) with raw negativity values.
        """
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_raw_negativity().")

        T = X.shape[0]
        negativity_vals = np.empty(T)

        for t in range(T):
            x_t = self._transform_point(X[t])
            psi = self._geometry.quasi_coherent_state(x_t)
            rho = np.outer(psi, psi.conj())
            negativity_vals[t] = compute_wigner_negativity(rho, self._phase_point_ops)

        return negativity_vals
