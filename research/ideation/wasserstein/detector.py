"""
Wasserstein/Trace-Distance Detector for Financial Regime Detection

Implements multiple quantum transport distances between consecutive quantum
states as regime detection signals. For pure states |psi_t>, the density
matrix is rho_t = |psi_t><psi_t|.

Three distances are available (selected via `distance_mode`):

1. 'trace' (default)
   Trace distance: T(rho_t, rho_{t-1}) = Tr(|rho_t - rho_{t-1}|) / 2
   For pure states: T = sqrt(1 - |<psi_t|psi_{t-1}>|^2)
   Equals the Wasserstein-1 distance on the Bloch sphere under the geodesic
   metric. Maximum sensitivity to orthogonal state changes.

2. 'bures'
   Bures distance: d_B = sqrt(2 * (1 - sqrt(F)))
   where F = |<psi_t|psi_{t-1}>|^2 is fidelity.
   Geodesic distance on the space of density matrices under the Bures metric.
   Better metric properties than raw fidelity; emphasises near-orthogonal
   transitions more than fidelity does.

3. 'eigenvalue_wasserstein'
   Classical Wasserstein-1 on eigenvalue spectra:
   W_1(p_t, p_{t-1}) = sum_k |lambda_t^(k) - lambda_{t-1}^(k)|
   where eigenvalues are sorted in ascending order.
   For pure states the non-trivial eigenvalue is always 1 (one eigenvalue=1,
   rest=0), so this collapses to 0. Used only with mixed states; included
   for completeness.

Mathematical relationship between metrics (pure states, fidelity F):
   fidelity overlap:  F = |<psi_t|psi_{t-1}>|^2  in [0, 1]
   trace distance:    T = sqrt(1 - F)              in [0, 1]
   Bures distance:    B = sqrt(2(1 - sqrt(F)))     in [0, sqrt(2)]
   geodesic angle:    theta = arccos(sqrt(F))      in [0, pi/2]

Scoring:
   Raw distances are smoothed with a rolling mean, then z-scored with an
   expanding window (same protocol as all other observatory detectors).

References:
   Nielsen & Chuang (2000), "Quantum Computation and Quantum Information".
   Bures (1969), "An extension of Kakutani's theorem on infinite product
       measures to the tensor product of semifinite w*-algebras".
   Wasserstein (1969), "Markov processes over denumerable products".
"""

import logging
from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import sys
import os

# Add project root to path for imports
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from qcml_geometry.core import QCMLGeometry
from qcml_geometry.observables import (
    BaseRegimeDetector,
    ExpandingWindowMixin,
    _apply_normalization,
    _transform_array,
    _transform_point,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Core distance functions
# =============================================================================

def trace_distance_pure(psi_t: np.ndarray, psi_prev: np.ndarray) -> float:
    """Compute trace distance between two pure states via fidelity.

    For pure states |psi_t> and |psi_{t-1}|:
        T(rho_t, rho_{t-1}) = sqrt(1 - |<psi_t|psi_{t-1}>|^2)

    This is exact (no approximation) and equals the operator-norm formula
    Tr(|rho_t - rho_{t-1}|) / 2 for rank-1 density matrices.

    Args:
        psi_t: Quantum state vector at time t, shape (hilbert_dim,).
        psi_prev: Quantum state vector at time t-1, shape (hilbert_dim,).

    Returns:
        Trace distance in [0, 1]. 0 = identical states, 1 = orthogonal.
    """
    fidelity = abs(np.vdot(psi_prev, psi_t)) ** 2
    # Clip for numerical safety before sqrt
    fidelity = np.clip(fidelity, 0.0, 1.0)
    return float(np.sqrt(1.0 - fidelity))


def bures_distance_pure(psi_t: np.ndarray, psi_prev: np.ndarray) -> float:
    """Compute Bures distance between two pure states.

    Bures distance:  d_B = sqrt(2 * (1 - sqrt(F)))
    where F = |<psi_t|psi_{t-1}>|^2.

    The Bures metric is the quantum generalization of the Fisher-Rao metric
    on probability distributions. It is the geodesic distance on the space
    of density matrices under the quantum Fisher information metric.

    Args:
        psi_t: Quantum state vector at time t, shape (hilbert_dim,).
        psi_prev: Quantum state vector at time t-1, shape (hilbert_dim,).

    Returns:
        Bures distance in [0, sqrt(2)].
    """
    fidelity = abs(np.vdot(psi_prev, psi_t)) ** 2
    fidelity = np.clip(fidelity, 0.0, 1.0)
    return float(np.sqrt(2.0 * (1.0 - np.sqrt(fidelity))))


def eigenvalue_wasserstein(
    psi_t: np.ndarray,
    psi_prev: np.ndarray,
) -> float:
    """Compute classical W_1 Wasserstein distance between eigenvalue spectra.

    For pure states, rho = |psi><psi| has eigenvalues {1, 0, 0, ...}.
    All pure states have identical spectra so this returns 0. This mode
    is provided for correctness/documentation; it is meaningful only for
    mixed states (would require computing partial traces or thermal states).

    For pure states the trace distance and Bures distance are the correct
    quantum analogs of Wasserstein distance.

    Args:
        psi_t: Quantum state vector at time t, shape (hilbert_dim,).
        psi_prev: Quantum state vector at time t-1, shape (hilbert_dim,).

    Returns:
        W_1 distance between sorted eigenvalue spectra (0.0 for pure states).
    """
    # Both are rank-1 density matrices: eigenvalues are [1, 0, ..., 0]
    # sorted ascending = [0, ..., 0, 1] for both -> difference = 0
    # Include implementation for completeness
    d = len(psi_t)
    eigs_t = np.zeros(d)
    eigs_t[-1] = 1.0
    eigs_prev = np.zeros(d)
    eigs_prev[-1] = 1.0
    return float(np.sum(np.abs(eigs_t - eigs_prev)))


def _compute_quantum_distance(
    psi_t: np.ndarray,
    psi_prev: np.ndarray,
    mode: str,
) -> float:
    """Dispatch to the requested distance function.

    Args:
        psi_t: State vector at time t.
        psi_prev: State vector at time t-1.
        mode: One of 'trace', 'bures', 'eigenvalue_wasserstein'.

    Returns:
        Distance value (float >= 0).
    """
    if mode == 'trace':
        return trace_distance_pure(psi_t, psi_prev)
    elif mode == 'bures':
        return bures_distance_pure(psi_t, psi_prev)
    elif mode == 'eigenvalue_wasserstein':
        return eigenvalue_wasserstein(psi_t, psi_prev)
    else:
        raise ValueError(
            f"Unknown distance_mode '{mode}'. "
            "Choose from: 'trace', 'bures', 'eigenvalue_wasserstein'."
        )


# =============================================================================
# Shared helper functions (mirror loschmidt_echo pattern)
# =============================================================================

def _expanding_zscore(
    raw_values: np.ndarray,
    rolling_window: int,
    min_expanding: int,
    T: int,
    skip_nan_start: int = 0,
) -> np.ndarray:
    """Compute expanding-window z-score of rolling-mean values.

    Matches the standard observatory detector pattern.

    Args:
        raw_values: 1-D array of raw distance values, length T.
        rolling_window: Window size for rolling mean smoothing.
        min_expanding: Minimum history before computing z-score.
        T: Total time series length.
        skip_nan_start: Number of leading NaN indices to skip.

    Returns:
        z_scores: 1-D array of length T, NaN where insufficient history.
    """
    rolling_vals = (
        pd.Series(raw_values)
        .rolling(window=rolling_window, min_periods=1)
        .mean()
        .values
    )
    z_scores = np.full(T, np.nan)
    start = max(min_expanding, skip_nan_start)
    for t in range(start, T):
        past = rolling_vals[skip_nan_start:t]
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


def _standard_init(detector, **kwargs):
    """Shared __init__ attribute assignment for QCML detectors."""
    for k, v in kwargs.items():
        setattr(detector, k, v)
    detector._geometry = None
    detector._scaler = None
    detector._pca = None
    detector._snapshots = None
    detector._train_norms = None
    detector._train_std = None
    detector._epsilon = 1e-5


def _standard_qcml_fit(detector, X: np.ndarray):
    """Shared fit logic for ExpandingWindowMixin + BaseRegimeDetector subclasses.

    Args:
        detector: Detector instance with standard QCML attributes.
        X: Raw feature matrix of shape (T, n_features).

    Returns:
        detector (fitted).
    """
    if detector.expanding_refit_interval is not None:
        return detector._fit_expanding(X)

    np.random.seed(detector.seed)
    n_components = min(detector.n_pca_components, X.shape[1])
    fit_end = detector.causal_fit_length or X.shape[0]

    detector._scaler = StandardScaler()
    detector._scaler.fit(X[:fit_end])

    detector._pca = PCA(n_components=n_components)
    X_scaled_fit = detector._scaler.transform(X[:fit_end])
    detector._pca.fit(X_scaled_fit)

    X_pca_raw = detector._pca.transform(X_scaled_fit)
    detector._train_norms = np.linalg.norm(X_pca_raw, axis=1)
    detector._train_std = np.std(X_pca_raw, axis=0)
    X_pca_fit = _apply_normalization(
        X_pca_raw, detector.normalization, detector._train_norms, detector._train_std,
    )

    if detector.adaptive_epsilon:
        detector._epsilon = 1e-3 * np.median(np.abs(X_pca_fit))
    else:
        detector._epsilon = 1e-5

    detector._geometry = QCMLGeometry(
        n_features=X_pca_fit.shape[1], hilbert_dim=detector.hilbert_dim,
    )
    custom_ops = getattr(detector, 'custom_operators', None)
    if custom_ops is not None:
        detector._geometry.set_operators(custom_ops)
    else:
        detector._geometry.fit_operators(
            X_pca_fit,
            method=detector.operator_method,
            scale_exponent=detector.scale_exponent,
        )

    return detector


# =============================================================================
# Detector class
# =============================================================================

class WassersteinStateDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via quantum transport distance between consecutive states.

    For each time step t, computes a distance between the quasi-coherent
    ground states |psi_t> and |psi_{t-1}| of the error Hamiltonian:

        H(x_t) = 1/2 * sum_k (A_k - x_t^k * I)^2

    Available distances (`distance_mode`):
        'trace'                  — T = sqrt(1 - |<psi_t|psi_{t-1}>|^2)
        'bures'                  — B = sqrt(2(1 - sqrt(|<psi_t|psi_{t-1}>|^2)))
        'eigenvalue_wasserstein' — Classical W_1 on sorted eigenvalue spectra
                                   (always 0 for pure states; for diagnostics)

    For pure states, trace distance and Bures distance are both exact quantum
    analogs of the Wasserstein-1 distance on the Bloch sphere. Trace distance
    is the canonical choice; Bures emphasises near-orthogonal transitions more.

    Score = expanding-window z-score of rolling-mean distances.

    Attributes:
        distance_mode: Which distance formula to use.
        hilbert_dim: Dimension of Hilbert space.
        n_pca_components: Number of PCA components for feature reduction.
        operator_method: QCML operator construction method.
        rolling_window: Rolling mean window (trading days).
        min_expanding: Minimum history for expanding z-score.
        seed: Random seed.
        normalization: Post-PCA normalization mode.
        adaptive_epsilon: Scale epsilon to data.
    """

    def __init__(
        self,
        distance_mode: str = 'trace',
        hilbert_dim: int = 4,
        n_pca_components: int = 8,
        operator_method: str = 'random',
        scale_exponent: Optional[float] = None,
        rolling_window: int = 20,
        min_expanding: int = 60,
        seed: int = 42,
        causal_fit_length: Optional[int] = None,
        expanding_refit_interval: Optional[int] = None,
        normalization: str = 'soft',
        adaptive_epsilon: bool = True,
        custom_operators: Optional[List[np.ndarray]] = None,
    ):
        """Initialize WassersteinStateDetector.

        Args:
            distance_mode: Distance metric. One of:
                'trace' (default) — trace distance T = sqrt(1 - F)
                'bures'           — Bures distance B = sqrt(2(1-sqrt(F)))
                'eigenvalue_wasserstein' — W_1 on eigenvalue spectra (always 0
                    for pure states; included for completeness)
            hilbert_dim: Hilbert space dimension. Use 4 or 6 to avoid
                Kramers degeneracy (avoid powers of 2 with pca_inspired method
                or use operator_method='random').
            n_pca_components: Number of PCA components.
            operator_method: Operator construction method ('random' recommended).
            scale_exponent: PCA eigenvalue scaling exponent.
            rolling_window: Rolling mean window size (trading days).
            min_expanding: Minimum samples before computing z-score.
            seed: Random seed for reproducibility.
            causal_fit_length: Fit only on first N samples (None = all).
            expanding_refit_interval: Refit interval for expanding windows.
            normalization: Post-PCA normalization: 'sphere', 'soft', 'clip', 'none'.
            adaptive_epsilon: If True, scale epsilon to data magnitude.
            custom_operators: Override learned operators with provided list.

        Example:
            >>> det = WassersteinStateDetector(distance_mode='trace', hilbert_dim=4)
            >>> det.fit(X_train)
            >>> scores = det.compute_regime_scores(X_full)
        """
        valid_modes = {'trace', 'bures', 'eigenvalue_wasserstein'}
        if distance_mode not in valid_modes:
            raise ValueError(
                f"distance_mode must be one of {valid_modes}, got '{distance_mode}'"
            )

        _standard_init(
            self,
            distance_mode=distance_mode,
            hilbert_dim=hilbert_dim,
            n_pca_components=n_pca_components,
            operator_method=operator_method,
            scale_exponent=scale_exponent,
            rolling_window=rolling_window,
            min_expanding=min_expanding,
            seed=seed,
            causal_fit_length=causal_fit_length,
            expanding_refit_interval=expanding_refit_interval,
            normalization=normalization,
            adaptive_epsilon=adaptive_epsilon,
            custom_operators=custom_operators,
        )

    @property
    def name(self) -> str:
        return f"Wasserstein ({self.distance_mode})"

    def fit(self, X: np.ndarray, **kwargs) -> 'WassersteinStateDetector':
        """Fit scaler, PCA, and QCML geometry.

        Args:
            X: Feature matrix of shape (T, n_features).

        Returns:
            self
        """
        return _standard_qcml_fit(self, X)

    def _compute_ground_states(
        self,
        X_raw: np.ndarray,
        X_transformed: np.ndarray,
    ) -> np.ndarray:
        """Compute ground state vectors for all time steps.

        Args:
            X_raw: Raw feature matrix (T, n_features), used when snapshots
                are active.
            X_transformed: PCA-transformed feature matrix (T, n_pca_components).

        Returns:
            psi_array: Complex array of shape (T, hilbert_dim). Each row is
                a normalized ground state |psi_t>.
        """
        T = len(X_transformed)
        psi_array = np.empty((T, self.hilbert_dim), dtype=np.complex128)

        for t in range(T):
            if self._snapshots is not None:
                geo, xt = self._transform_point_at(X_raw[t], t)
            else:
                geo = self._geometry
                xt = X_transformed[t]

            psi_array[t] = geo.quasi_coherent_state(xt)

        return psi_array

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute regime scores for all time steps.

        For each t >= 1: distance(psi_t, psi_{t-1}).
        For t=0: NaN (no predecessor).

        Scores are rolling-mean smoothed then z-scored with an expanding
        window starting at min_expanding.

        Args:
            X: Feature matrix of shape (T, n_features).

        Returns:
            z_scores: 1-D array of length T. NaN where insufficient history.
        """
        if self._geometry is None and self._snapshots is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        X_transformed = _transform_array(
            X, self._scaler, self._pca,
            normalization=self.normalization,
            train_norms=self._train_norms,
            train_std=self._train_std,
        )
        T = len(X_transformed)

        # --- Compute ground states for all t ---
        psi_array = self._compute_ground_states(X, X_transformed)

        # --- Compute consecutive distances ---
        raw_distances = np.full(T, np.nan)
        for t in range(1, T):
            raw_distances[t] = _compute_quantum_distance(
                psi_array[t], psi_array[t - 1], self.distance_mode,
            )

        # --- Rolling mean + expanding z-score ---
        z_scores = _expanding_zscore(
            raw_values=raw_distances,
            rolling_window=self.rolling_window,
            min_expanding=self.min_expanding,
            T=T,
            skip_nan_start=1,  # skip t=0 which is always NaN
        )

        return z_scores
