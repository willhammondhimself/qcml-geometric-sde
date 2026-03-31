"""
Quantum Relative Entropy Detector for Financial Regime Detection

Computes the quantum relative entropy D(rho_t || rho_ref) between the current
pure state density matrix and a rolling-window average reference state.

For a pure state rho_t = |psi_t><psi_t| and a mixed reference state rho_ref:

    D(rho_t || rho_ref) = Tr(rho_t * (log rho_t - log rho_ref))
                        = -<psi_t| log(rho_ref) |psi_t>

since S(rho_t) = 0 for a pure state and Tr(rho_t) = 1.

Physical interpretation:
  - D measures the "surprise" of seeing rho_t given the reference distribution rho_ref.
  - When the current state differs strongly from the recent average (crisis onset),
    D(rho_t || rho_ref) spikes because rho_t lives in a region of low support of rho_ref.
  - When rho_ref has near-zero eigenvalues along the direction of psi_t,
    D diverges (regularized via epsilon floor on eigenvalues).

This generalizes fidelity-based measures by using the full eigenvalue spectrum
of rho_ref rather than just the leading overlap. It is more sensitive to
low-probability directions.

Variants implemented:
  1. D with W=20 rolling window (short-term reference)
  2. D with W=60 rolling window (medium-term reference)
  3. Simple infidelity 1 - <psi_t|rho_ref|psi_t> (fidelity complement)
  4. D with expanding window (all past states as reference)

Reference:
  Umegaki (1962), "Conditional expectation in an operator algebra."
  Vedral (2002), "The role of relative entropy in quantum information theory."
"""

import logging
import os
import sys
from typing import Optional, List

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add project root to path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from qcml_geometry.core import QCMLGeometry
from qcml_geometry.observables import (
    BaseRegimeDetector,
    ExpandingWindowMixin,
    _apply_normalization,
    _transform_array,
)

logger = logging.getLogger(__name__)

EPSILON = 1e-10  # Regularization floor for log(rho_ref) eigenvalues


# ---------------------------------------------------------------------------
# Core computation: quantum relative entropy for pure vs mixed state
# ---------------------------------------------------------------------------

def quantum_relative_entropy(
    psi: np.ndarray,
    rho_ref: np.ndarray,
    epsilon: float = EPSILON,
) -> float:
    """Compute D(|psi><psi| || rho_ref) = -<psi| log(rho_ref) |psi>.

    For a pure state rho = |psi><psi|, the von Neumann entropy S(rho) = 0,
    so D(rho || sigma) = -S(rho) - Tr(rho log sigma) = -<psi|log(sigma)|psi>.

    Uses eigendecomposition of rho_ref for numerically stable matrix log:
        log(rho_ref) = V @ diag(log(max(lambda_i, epsilon))) @ V^T

    Args:
        psi: Pure state vector of shape (d,), normalized.
        rho_ref: Reference density matrix of shape (d, d), Hermitian PSD.
        epsilon: Floor for eigenvalues to avoid log(0).

    Returns:
        D: Quantum relative entropy in nats (non-negative for valid inputs).
    """
    # Eigendecompose rho_ref
    eigenvalues, eigenvectors = np.linalg.eigh(rho_ref)

    # Regularize: floor eigenvalues at epsilon
    eigenvalues_reg = np.maximum(eigenvalues, epsilon)

    # Compute log(rho_ref) = V @ diag(log(lambda)) @ V^T
    log_eigenvalues = np.log(eigenvalues_reg)

    # D = -<psi| log(rho_ref) |psi>
    # = -<psi| V diag(log(lambda)) V^T |psi>
    # = -sum_i log(lambda_i) |<v_i|psi>|^2
    coeffs = eigenvectors.conj().T @ psi  # (d,) projections of psi onto eigenbasis
    probs = np.abs(coeffs) ** 2  # |<v_i|psi>|^2

    D = -np.sum(log_eigenvalues * probs)

    return float(max(D, 0.0))  # Should be non-negative, clamp for numerics


def simple_infidelity(
    psi: np.ndarray,
    rho_ref: np.ndarray,
) -> float:
    """Compute 1 - <psi|rho_ref|psi> (infidelity with mixed reference).

    This is the simplest fidelity-based divergence measure. Unlike quantum
    relative entropy, it does not use the full eigenvalue spectrum --
    it only measures how much probability weight rho_ref assigns to |psi>.

    Args:
        psi: Pure state vector of shape (d,), normalized.
        rho_ref: Reference density matrix of shape (d, d).

    Returns:
        infidelity: 1 - F(psi, rho_ref) in [0, 1].
    """
    fidelity = np.real(np.vdot(psi, rho_ref @ psi))
    return float(max(1.0 - fidelity, 0.0))


def build_rolling_rho_ref(
    states: List[np.ndarray],
    t: int,
    window: int,
) -> Optional[np.ndarray]:
    """Build rolling-window average density matrix.

    rho_ref = (1/W) * sum_{s=t-W}^{t-1} |psi_s><psi_s|

    Args:
        states: List of all state vectors.
        t: Current time index.
        window: Lookback window size W.

    Returns:
        rho_ref: Density matrix of shape (d, d), or None if insufficient history.
    """
    if t < window:
        return None

    d = states[0].shape[0]
    rho = np.zeros((d, d), dtype=complex)
    for s in range(t - window, t):
        psi_s = states[s]
        rho += np.outer(psi_s, psi_s.conj())
    rho /= window

    return rho


def build_expanding_rho_ref(
    states: List[np.ndarray],
    t: int,
    min_window: int = 20,
) -> Optional[np.ndarray]:
    """Build expanding-window average density matrix using all past states.

    rho_ref = (1/t) * sum_{s=0}^{t-1} |psi_s><psi_s|

    Args:
        states: List of all state vectors.
        t: Current time index.
        min_window: Minimum number of past states required.

    Returns:
        rho_ref: Density matrix of shape (d, d), or None if insufficient history.
    """
    if t < min_window:
        return None

    d = states[0].shape[0]
    rho = np.zeros((d, d), dtype=complex)
    for s in range(t):
        psi_s = states[s]
        rho += np.outer(psi_s, psi_s.conj())
    rho /= t

    return rho


# ---------------------------------------------------------------------------
# Expanding z-score helper
# ---------------------------------------------------------------------------

def _expanding_zscore(
    raw_values: np.ndarray,
    rolling_window: int,
    min_expanding: int,
    T: int,
    skip_nan_start: int = 0,
) -> np.ndarray:
    """Compute expanding-window z-score of rolling-mean values.

    Causal: z-score at time t uses only data up to t-1.

    Args:
        raw_values: Raw signal of shape (T,).
        rolling_window: Window for initial rolling mean smoothing.
        min_expanding: Minimum samples before z-score is computed.
        T: Total length.
        skip_nan_start: Skip first N entries when computing expanding stats.

    Returns:
        z_scores: Z-scored signal of shape (T,), NaN before min_expanding.
    """
    import pandas as pd

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


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class QuantumRelativeEntropyDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via quantum relative entropy D(rho_t || rho_ref).

    Computes D(|psi_t><psi_t| || rho_ref) where rho_ref is the rolling-window
    average density matrix of recent ground states. D measures how "surprising"
    the current state is relative to the recent trajectory.

    Score = expanding z-score of rolling-mean relative entropy.

    Supports four variants via the `variant` parameter:
        'D_W20': Relative entropy with W=20 rolling window
        'D_W60': Relative entropy with W=60 rolling window
        'infidelity': Simple infidelity 1 - <psi_t|rho_ref|psi_t>
        'D_expanding': Relative entropy with expanding window (all past states)

    Attributes:
        state_window: Number of past ground states for rho_ref (W).
        variant: Which measure to compute.
    """

    def __init__(
        self,
        hilbert_dim: int = 8,
        state_window: int = 20,
        variant: str = 'D_W20',
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
        """Initialize Quantum Relative Entropy Detector.

        Args:
            hilbert_dim: Hilbert space dimension (default 8 for 3-qubit system).
            state_window: Rolling window size for reference density matrix (W).
            variant: Measure variant ('D_W20', 'D_W60', 'infidelity', 'D_expanding').
            n_pca_components: Number of PCA components for feature reduction.
            operator_method: Operator construction method. 'random' recommended.
            scale_exponent: PCA eigenvalue scaling exponent.
            rolling_window: Window for rolling mean smoothing.
            min_expanding: Minimum expanding window size for z-score.
            seed: Random seed for reproducibility.
            causal_fit_length: Fit on first N samples only (None = all).
            expanding_refit_interval: Refit interval for expanding windows.
            normalization: Post-PCA normalization mode.
            adaptive_epsilon: Adapt numerical epsilon to data scale.
            custom_operators: Override learned operators.
        """
        self.hilbert_dim = hilbert_dim
        self.state_window = state_window
        self.variant = variant
        self.n_pca_components = n_pca_components
        self.operator_method = operator_method
        self.scale_exponent = scale_exponent
        self.rolling_window = rolling_window
        self.min_expanding = min_expanding
        self.seed = seed
        self.causal_fit_length = causal_fit_length
        self.expanding_refit_interval = expanding_refit_interval
        self.normalization = normalization
        self.adaptive_epsilon = adaptive_epsilon
        self.custom_operators = custom_operators

        # Internal state (set during fit)
        self._geometry: Optional[QCMLGeometry] = None
        self._scaler: Optional[StandardScaler] = None
        self._pca: Optional[PCA] = None
        self._snapshots = None
        self._train_norms: Optional[np.ndarray] = None
        self._train_std: Optional[np.ndarray] = None
        self._epsilon = 1e-5

    @property
    def name(self) -> str:
        return f"Quantum Relative Entropy ({self.variant})"

    def fit(self, X: np.ndarray, **kwargs) -> 'QuantumRelativeEntropyDetector':
        """Fit scaler, PCA, and QCML geometry from feature matrix.

        Args:
            X: Feature matrix of shape (T, n_features).

        Returns:
            self
        """
        if self.expanding_refit_interval is not None:
            return self._fit_expanding(X)

        np.random.seed(self.seed)
        n_components = min(self.n_pca_components, X.shape[1])
        fit_end = self.causal_fit_length or X.shape[0]

        self._scaler = StandardScaler()
        self._scaler.fit(X[:fit_end])

        self._pca = PCA(n_components=n_components)
        X_scaled_fit = self._scaler.transform(X[:fit_end])
        self._pca.fit(X_scaled_fit)

        X_pca_raw = self._pca.transform(X_scaled_fit)
        self._train_norms = np.linalg.norm(X_pca_raw, axis=1)
        self._train_std = np.std(X_pca_raw, axis=0)
        X_pca_fit = _apply_normalization(
            X_pca_raw, self.normalization,
            self._train_norms, self._train_std,
        )

        if self.adaptive_epsilon:
            self._epsilon = 1e-3 * np.median(np.abs(X_pca_fit))
        else:
            self._epsilon = 1e-5

        self._geometry = QCMLGeometry(
            n_features=X_pca_fit.shape[1],
            hilbert_dim=self.hilbert_dim,
        )

        if self.custom_operators is not None:
            self._geometry.set_operators(self.custom_operators)
        else:
            self._geometry.fit_operators(
                X_pca_fit,
                method=self.operator_method,
                scale_exponent=self.scale_exponent,
            )

        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute regime scores as z-scored quantum relative entropy.

        Args:
            X: Feature matrix of shape (T, n_features).

        Returns:
            scores: 1-D array of length T. Higher z-score = more anomalous.
        """
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        Xt = _transform_array(
            X, self._scaler, self._pca,
            normalization=self.normalization,
            train_norms=self._train_norms,
            train_std=self._train_std,
        )
        T = len(Xt)

        # Collect all ground states
        states = []
        for t in range(T):
            if self._snapshots:
                geo, xt = self._transform_point_at(X[t], t)
            else:
                geo, xt = self._geometry, Xt[t]
            states.append(geo.quasi_coherent_state(xt))

        # Compute the chosen variant for each time step
        W = self.state_window
        vals = np.full(T, np.nan)

        for t in range(T):
            try:
                if self.variant == 'D_expanding':
                    rho_ref = build_expanding_rho_ref(states, t, min_window=20)
                else:
                    rho_ref = build_rolling_rho_ref(states, t, W)

                if rho_ref is None:
                    continue

                psi_t = states[t]

                if self.variant == 'infidelity':
                    vals[t] = simple_infidelity(psi_t, rho_ref)
                else:
                    vals[t] = quantum_relative_entropy(psi_t, rho_ref)

            except Exception as e:
                logger.warning(f"QRE failed at t={t}: {e}")
                vals[t] = np.nan

        skip_start = W if self.variant != 'D_expanding' else 20
        return _expanding_zscore(
            vals, self.rolling_window, self.min_expanding, T,
            skip_nan_start=skip_start,
        )
