"""
Partition Function / Free Energy Detector for Financial Regime Detection

Applies statistical mechanics to the QCML error Hamiltonian spectrum to detect
market phase transitions.  The canonical partition function

    Z(beta) = Tr(e^{-beta * H}) = sum_k exp(-beta * E_k)

and its derived thermodynamic quantities exhibit singularities (or sharp peaks)
at phase transitions:

    Free energy   : F = -ln(Z) / beta
    Internal energy: U = <E>_beta = sum_k E_k * p_k
    Specific heat  : C = beta^2 * Var_beta(E) = beta^2 * (<E^2> - <E>^2)

where  p_k = exp(-beta * E_k) / Z  is the Boltzmann weight.

The specific heat C peaks sharply at phase transitions, making it the
primary detection score.  We z-score C over an expanding window to
produce a stationary, comparable signal.

Reference:
    Sachdev (1999) "Quantum Phase Transitions", Cambridge.
    Goldenfeld (1992) "Lectures on Phase Transitions", Addison-Wesley.
"""

import logging
import os
import sys
from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Project root path
# ---------------------------------------------------------------------------
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
# Core thermodynamic computation
# ---------------------------------------------------------------------------

def compute_partition_thermodynamics(
    eigenvalues: np.ndarray,
    beta: float,
) -> dict:
    """Compute partition function and derived thermodynamic observables.

    Given the sorted spectrum {E_k} of H(x), computes:
        - Z(beta): partition function = sum_k exp(-beta * E_k)
        - F(beta): free energy = -ln(Z) / beta
        - U(beta): internal energy = <E>_beta = sum_k E_k * p_k
        - C(beta): specific heat = beta^2 * Var_beta(E)

    The computation is numerically stabilized by shifting energies:
        Z = exp(-beta * E_0) * sum_k exp(-beta * (E_k - E_0))
    which cancels in ratios and only adds beta*E_0 to ln Z.

    Args:
        eigenvalues: Sorted real eigenvalue spectrum of H(x),
                     shape (hilbert_dim,).
        beta: Inverse temperature parameter (> 0).

    Returns:
        dict with keys:
            'Z': partition function (float > 0)
            'ln_Z': log partition function (float)
            'F': free energy (float)
            'U': internal energy = <E>_beta (float)
            'C': specific heat = beta^2 * Var(E)_beta (float >= 0)
            'E_var': energy variance under Boltzmann distribution (float >= 0)
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    E0 = eigenvalues[0]  # Ground state energy for numerical stability

    # Boltzmann weights (shifted for stability)
    shifted = eigenvalues - E0
    log_weights = -beta * shifted
    # Subtract max for further numerical safety (max is 0 here since all >=0)
    weights = np.exp(log_weights)
    Z_shifted = np.sum(weights)

    # ln Z = -beta * E0 + ln(Z_shifted)
    ln_Z = -beta * E0 + np.log(Z_shifted + 1e-300)
    Z = np.exp(ln_Z)

    # Boltzmann probabilities
    probs = weights / (Z_shifted + 1e-300)

    # Internal energy: <E>_beta
    U = float(np.sum(eigenvalues * probs))

    # Energy variance: <E^2>_beta - <E>_beta^2
    E_sq_mean = float(np.sum(eigenvalues**2 * probs))
    E_var = max(E_sq_mean - U**2, 0.0)  # numerical clamp to non-negative

    # Specific heat: C = beta^2 * Var(E)
    C = beta**2 * E_var

    # Free energy: F = -ln(Z) / beta
    F = -ln_Z / beta

    return {
        'Z': float(Z),
        'ln_Z': float(ln_Z),
        'F': float(F),
        'U': float(U),
        'C': float(C),
        'E_var': float(E_var),
    }


# ---------------------------------------------------------------------------
# Helper: expanding-window z-score (matches other observatory detectors)
# ---------------------------------------------------------------------------

def _expanding_zscore(
    raw_values: np.ndarray,
    rolling_window: int,
    min_expanding: int,
    T: int,
    skip_nan_start: int = 0,
) -> np.ndarray:
    """Compute expanding-window z-score of rolling-mean values.

    Args:
        raw_values: 1-D array of raw signal values, length T.
        rolling_window: Rolling mean window size.
        min_expanding: Minimum number of past observations before z-scoring.
        T: Total time-series length.
        skip_nan_start: Number of leading NaN entries to skip.

    Returns:
        z_scores: 1-D array of length T (NaN where insufficient history).
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


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class PartitionFunctionDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via the specific heat of the QCML Hamiltonian spectrum.

    Computes the canonical partition function Z(beta) = Tr(e^{-beta*H}) from
    the full eigenspectrum of H(x) at each time step, then derives the specific
    heat C = beta^2 * Var_beta(E).  C peaks sharply at statistical-mechanical
    phase transitions; empirically this correlates with financial regime onset.

    The raw C series is smoothed with a rolling mean and z-scored over an
    expanding window (no look-ahead bias).

    Attributes:
        beta: Inverse temperature parameter controlling sensitivity.
            Small beta (~0.1): broad, high-temperature limit — all eigenvalues
            contribute equally, C is low.
            Large beta (~10): low-temperature limit — only ground/first-excited
            states contribute, C peaks at spectral-gap closings.
            Recommended range: 0.5–5.0.  Default 1.0.
    """

    def __init__(
        self,
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
        beta: float = 1.0,
    ):
        """Initialize PartitionFunctionDetector.

        Args:
            hilbert_dim: Hilbert space dimension (default 4 = 2-qubit system).
            n_pca_components: Number of PCA components for feature reduction.
            operator_method: Operator construction method.
                'random' avoids Kramers degeneracy (recommended).
            scale_exponent: PCA eigenvalue scaling exponent for pca_* methods.
            rolling_window: Window size for rolling mean smoothing of raw C.
            min_expanding: Minimum expanding history before z-scoring begins.
            seed: Random seed for reproducibility.
            causal_fit_length: Fit only on first N samples (None = all).
            expanding_refit_interval: Refit geometry every N steps (None = static).
            normalization: Post-PCA normalization ('soft', 'sphere', 'none', 'clip').
            adaptive_epsilon: Scale epsilon to data magnitude.
            custom_operators: Override fitted operators with a fixed list.
            beta: Inverse temperature for partition function Z = Tr(e^{-beta*H}).
                Controls how sharply the spectrum is weighted toward the ground
                state.  Default 1.0.
        """
        # Initialize all shared attributes via setattr (mirrors other detectors)
        self.hilbert_dim = hilbert_dim
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
        self.beta = beta

        # Internal state (set by fit)
        self._geometry: Optional[QCMLGeometry] = None
        self._scaler: Optional[StandardScaler] = None
        self._pca: Optional[PCA] = None
        self._snapshots = None
        self._train_norms: Optional[np.ndarray] = None
        self._train_std: Optional[np.ndarray] = None
        self._epsilon: float = 1e-5

    @property
    def name(self) -> str:
        return f"Partition Function (beta={self.beta})"

    def fit(self, X: np.ndarray, **kwargs) -> 'PartitionFunctionDetector':
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
            X_pca_raw, self.normalization, self._train_norms, self._train_std,
        )

        if self.adaptive_epsilon:
            self._epsilon = 1e-3 * np.median(np.abs(X_pca_fit))
        else:
            self._epsilon = 1e-5

        self._geometry = QCMLGeometry(
            n_features=X_pca_fit.shape[1], hilbert_dim=self.hilbert_dim
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

    def compute_specific_heat_series(self, X: np.ndarray) -> np.ndarray:
        """Compute raw specific heat C(t) for each time step.

        Args:
            X: Feature matrix of shape (T, n_features).

        Returns:
            C_series: 1-D array of shape (T,) with specific heat values.
                      NaN where computation fails.
        """
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_specific_heat_series().")

        X_transformed = _transform_array(
            X, self._scaler, self._pca,
            normalization=self.normalization,
            train_norms=self._train_norms,
            train_std=self._train_std,
        )
        T = len(X_transformed)
        C_series = np.full(T, np.nan)

        for t in range(T):
            if self._snapshots:
                geo, xt = self._transform_point_at(X[t], t)
            else:
                geo, xt = self._geometry, X_transformed[t]

            try:
                eigenvalues = geo.full_spectrum(xt)
                thermo = compute_partition_thermodynamics(eigenvalues, self.beta)
                C_series[t] = thermo['C']
            except Exception as exc:
                logger.warning(f"Partition function failed at t={t}: {exc}")

        return C_series

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute regime scores as z-scored specific heat.

        Higher scores indicate higher specific heat (peak near phase transition).

        Args:
            X: Feature matrix of shape (T, n_features).

        Returns:
            scores: 1-D array of length T. Higher = more regime-transition-like.
        """
        C_series = self.compute_specific_heat_series(X)
        T = len(C_series)
        return _expanding_zscore(
            C_series,
            self.rolling_window,
            self.min_expanding,
            T,
            skip_nan_start=0,
        )
