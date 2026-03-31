"""
Channel Capacity (Holevo Information) Detector for Financial Regime Detection

Computes the Holevo information chi = S(rho_avg) for an ensemble of pure ground
states within a rolling window, where:

  rho_avg = (1/W) * sum_{s=1}^{W} |psi(t-W+s)><psi(t-W+s)|
  chi = -Tr(rho_avg * log(rho_avg))  (von Neumann entropy)

Since each ensemble member is a pure state with S(|psi><psi|) = 0, the Holevo
bound chi = S(rho_avg) - sum p_i S(rho_i) = S(rho_avg).

Physical interpretation:
  - chi measures the DIVERSITY of ground states within the window.
  - If states converge (crisis convergence): rho_avg becomes more pure -> chi drops.
  - If states fluctuate wildly (crisis divergence): rho_avg becomes more mixed -> chi rises.
  - Maximum chi = log(d) when states span all d dimensions uniformly.

This is complementary to Effective State Dimension, which measures D_eff via the
inverse participation ratio of the Gram matrix. Channel capacity uses von Neumann
entropy of the average density matrix, which weights eigenvalues logarithmically
rather than quadratically.

Relationship: D_eff = exp(S_2(rho_avg)) where S_2 is Renyi-2 entropy.
Channel capacity uses S_1 (von Neumann). These differ when eigenvalue spectrum
is non-uniform.

Reference:
  Holevo (1973), "Bounds for the quantity of information transmitted by a
  quantum communication channel."
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


# ---------------------------------------------------------------------------
# Core computation: Holevo information for pure state ensemble
# ---------------------------------------------------------------------------

def holevo_information(ground_states: List[np.ndarray]) -> float:
    """Compute Holevo information for a uniform ensemble of pure states.

    chi = S(rho_avg) = -Tr(rho_avg * log(rho_avg))

    where rho_avg = (1/W) * sum_s |psi_s><psi_s|.

    For pure states, S(|psi><psi|) = 0, so the Holevo bound equals
    the von Neumann entropy of the mixture.

    Args:
        ground_states: List of W state vectors, each of shape (d,), normalized.

    Returns:
        chi: Holevo information in nats. In [0, log(d)].
    """
    W = len(ground_states)
    if W < 2:
        return 0.0

    d = ground_states[0].shape[0]

    # Build time-averaged density matrix
    rho_avg = np.zeros((d, d), dtype=complex)
    for psi in ground_states:
        rho_avg += np.outer(psi, psi.conj())
    rho_avg /= W

    # Von Neumann entropy
    eigenvalues = np.linalg.eigvalsh(rho_avg)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]

    if len(eigenvalues) == 0:
        return 0.0

    return float(-np.sum(eigenvalues * np.log(eigenvalues)))


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

class ChannelCapacityDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via Holevo information of rolling ground state ensemble.

    Collects QCML ground states |psi(t)> in a rolling window of size W, then
    computes chi = S(rho_avg) where rho_avg = (1/W) sum |psi><psi|.

    chi measures the channel capacity (diversity) of the quantum state
    trajectory. During regime transitions, chi changes as states either
    converge (collapse) or diverge (explore new directions).

    Score = expanding z-score of rolling-mean Holevo information.

    Attributes:
        state_window: Number of ground states in the rolling window (W).
    """

    def __init__(
        self,
        hilbert_dim: int = 8,
        state_window: int = 60,
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
        """Initialize Channel Capacity Detector.

        Args:
            hilbert_dim: Hilbert space dimension (default 8 for 3-qubit system).
            state_window: Rolling window size for ground state collection (W).
            n_pca_components: Number of PCA components for feature reduction.
            operator_method: Operator construction method. 'random' recommended.
            scale_exponent: PCA eigenvalue scaling exponent (pca_* methods only).
            rolling_window: Window for rolling mean smoothing of chi values.
            min_expanding: Minimum expanding window size for z-score computation.
            seed: Random seed for reproducibility.
            causal_fit_length: Fit on first N samples only (None = all).
            expanding_refit_interval: Refit interval for expanding windows.
            normalization: Post-PCA normalization mode ('soft', 'sphere', etc.).
            adaptive_epsilon: Adapt numerical epsilon to data scale.
            custom_operators: Override learned operators with custom list.
        """
        self.hilbert_dim = hilbert_dim
        self.state_window = state_window
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
        return f"Channel Capacity (W={self.state_window})"

    def fit(self, X: np.ndarray, **kwargs) -> 'ChannelCapacityDetector':
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
        """Compute regime scores as z-scored Holevo information.

        Args:
            X: Feature matrix of shape (T, n_features).

        Returns:
            scores: 1-D array of length T. Higher absolute z-score = more anomalous.
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
        W = self.state_window

        # Collect all ground states first
        states = []
        for t in range(T):
            if self._snapshots:
                geo, xt = self._transform_point_at(X[t], t)
            else:
                geo, xt = self._geometry, Xt[t]
            states.append(geo.quasi_coherent_state(xt))

        # Compute rolling Holevo information
        vals = np.full(T, np.nan)
        for t in range(W, T):
            window_states = states[t - W:t]
            try:
                vals[t] = holevo_information(window_states)
            except Exception as e:
                logger.warning(f"Holevo info failed at t={t}: {e}")
                vals[t] = np.nan

        return _expanding_zscore(vals, self.rolling_window, self.min_expanding, T, skip_nan_start=W)
