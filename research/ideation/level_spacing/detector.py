"""
Level Spacing Ratio Detector for Financial Regime Detection

Measures the nearest-neighbor level spacing ratio <r> of the QCML error
Hamiltonian H(x_t) spectrum at each time step. In random matrix theory:

    - GOE (Gaussian Orthogonal Ensemble):   <r> ~ 0.5307
    - GUE (Gaussian Unitary Ensemble):      <r> ~ 0.5996
    - Poisson (integrable / uncorrelated):   <r> ~ 0.3863

The idea: during normal market conditions, eigenvalues may follow one
universality class; at crisis onset, the spectral statistics may shift
(e.g., from GOE-like level repulsion toward Poisson-like clustering,
or vice versa), reflecting a change in the effective "integrability"
of the market's quantum geometric structure.

Three observables per time step:

1. **Mean ratio** <r> = mean_n [min(s_n, s_{n+1}) / max(s_n, s_{n+1})]
   where s_n = E_{n+1} - E_n are the nearest-neighbor spacings.
   Summarizes the spectral correlation regime.

2. **Std ratio** sigma_r = std(r_n)
   Captures heterogeneity in level repulsion across the spectrum.

3. **Poisson fraction** f_Poisson = fraction of r_n < 0.386
   Measures clustering toward Poisson statistics (integrable limit).

All three are z-scored with an expanding window (no future data).

References:
    - Oganesyan, Huse (2007). Localization of interacting fermions at
      high temperature. Phys. Rev. B 75, 155111.
    - Atas et al. (2013). Distribution of the ratio of consecutive
      level spacings in random matrix ensembles. Phys. Rev. Lett. 110,
      084101.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qcml_geometry.core import QCMLGeometry

logger = logging.getLogger(__name__)

# RMT reference values
RMT_POISSON = 0.3863   # 2 ln 2 - 1
RMT_GOE = 0.5307       # 4 - 2 sqrt(3)
RMT_GUE = 0.5996       # 2 sqrt(3) / pi - 1/2


def compute_level_spacing_ratios(eigenvalues: np.ndarray) -> np.ndarray:
    """Compute nearest-neighbor level spacing ratios from sorted eigenvalues.

    Given sorted eigenvalues E_0 < E_1 < ... < E_{d-1}:
        s_n = E_{n+1} - E_n      for n = 0, ..., d-2
        r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1})  for n = 0, ..., d-3

    Args:
        eigenvalues: Sorted eigenvalues of shape (d,).

    Returns:
        ratios: Level spacing ratios of shape (d-2,). Each in [0, 1].
    """
    spacings = np.diff(eigenvalues)
    # Avoid division by zero for degenerate eigenvalues
    spacings = np.maximum(spacings, 1e-15)

    if len(spacings) < 2:
        return np.array([RMT_POISSON])  # Fallback: single ratio

    s_n = spacings[:-1]
    s_np1 = spacings[1:]

    min_s = np.minimum(s_n, s_np1)
    max_s = np.maximum(s_n, s_np1)

    ratios = min_s / max_s
    return ratios


def _apply_normalization(X_pca, mode, train_norms=None, train_std=None):
    """Apply post-PCA normalization."""
    is_1d = X_pca.ndim == 1
    if is_1d:
        X_pca = X_pca.reshape(1, -1)

    if mode == 'soft':
        median_norm = np.median(train_norms) if train_norms is not None else 1.0
        norms = np.linalg.norm(X_pca, axis=1, keepdims=True)
        result = X_pca / (norms + median_norm)
    elif mode == 'sphere':
        norms = np.linalg.norm(X_pca, axis=1, keepdims=True)
        result = X_pca / (norms + 1e-8)
    elif mode == 'none':
        result = X_pca
    else:
        result = X_pca

    return result.ravel() if is_1d else result


def _expanding_zscore(raw_values, rolling_window, min_expanding, T,
                      skip_nan_start=0):
    """Compute expanding-window z-score of rolling-mean values.

    Uses only past data at each point (no future leakage).

    Args:
        raw_values: Raw observable values of shape (T,).
        rolling_window: Rolling mean window.
        min_expanding: Minimum observations before computing z-score.
        T: Total length.
        skip_nan_start: Number of initial NaN values to skip.

    Returns:
        z_scores: Z-scored values of shape (T,), with NaN for warmup.
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


class LevelSpacingRatioDetector:
    """Regime detection via nearest-neighbor level spacing ratios of H(x).

    At each time step t, computes the full eigenvalue spectrum of the error
    Hamiltonian H(x_t), then extracts the level spacing ratios r_n. Three
    summary statistics are computed:

        1. mean_ratio: <r> -- mean level spacing ratio
        2. std_ratio: sigma_r -- standard deviation of ratios
        3. poisson_fraction: fraction of r_n < 0.386 (Poisson threshold)

    Each is z-scored using an expanding window for anomaly detection.

    Args:
        hilbert_dim: Hilbert space dimension (default 8 for 3-qubit).
        n_pca_components: Number of PCA components (default 8).
        operator_method: Method for constructing Hermitian operators.
        rolling_window: Rolling mean window for smoothing.
        min_expanding: Minimum observations before z-scoring.
        seed: Random seed for reproducibility.
        normalization: Post-PCA normalization mode.
        poisson_threshold: Threshold for Poisson fraction (default 0.386).
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
        poisson_threshold: float = 0.386,
    ):
        self.hilbert_dim = hilbert_dim
        self.n_pca_components = n_pca_components
        self.operator_method = operator_method
        self.scale_exponent = scale_exponent
        self.rolling_window = rolling_window
        self.min_expanding = min_expanding
        self.seed = seed
        self.normalization = normalization
        self.poisson_threshold = poisson_threshold

        self._geometry: Optional[QCMLGeometry] = None
        self._scaler: Optional[StandardScaler] = None
        self._pca: Optional[PCA] = None
        self._train_norms: Optional[np.ndarray] = None
        self._train_std: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return "Level Spacing Ratio"

    def fit(self, X: np.ndarray) -> 'LevelSpacingRatioDetector':
        """Fit scaler, PCA, and QCML operators on the training data.

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
        X_pca = _apply_normalization(
            X_pca_raw, self.normalization, self._train_norms, self._train_std,
        )

        self._geometry = QCMLGeometry(
            n_features=X_pca.shape[1], hilbert_dim=self.hilbert_dim
        )
        self._geometry.fit_operators(
            X_pca, method=self.operator_method,
            scale_exponent=self.scale_exponent,
        )
        return self

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Transform raw features through scaler + PCA + normalization.

        Args:
            X: Feature matrix (T, n_features).

        Returns:
            Transformed matrix (T, n_pca_components).
        """
        X_scaled = self._scaler.transform(X)
        X_pca_raw = self._pca.transform(X_scaled)
        return _apply_normalization(
            X_pca_raw, self.normalization, self._train_norms, self._train_std,
        )

    def compute_all_variants(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute all three level spacing observables (raw, not z-scored).

        Args:
            X: Feature matrix of shape (T, n_features).

        Returns:
            Dictionary with keys 'mean_ratio', 'std_ratio', 'poisson_fraction',
            each mapping to a 1-D array of length T.
        """
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_all_variants().")

        Xt = self._transform(X)
        T = len(Xt)

        mean_ratios = np.empty(T)
        std_ratios = np.empty(T)
        poisson_fractions = np.empty(T)

        for t in range(T):
            eigenvalues = self._geometry.full_spectrum(Xt[t])
            ratios = compute_level_spacing_ratios(eigenvalues)

            mean_ratios[t] = np.mean(ratios)
            std_ratios[t] = np.std(ratios, ddof=0)
            poisson_fractions[t] = np.mean(ratios < self.poisson_threshold)

        return {
            'mean_ratio': mean_ratios,
            'std_ratio': std_ratios,
            'poisson_fraction': poisson_fractions,
        }

    def compute_regime_scores(self, X: np.ndarray, variant: str = 'mean_ratio') -> np.ndarray:
        """Compute z-scored regime scores for a given variant.

        Args:
            X: Feature matrix of shape (T, n_features).
            variant: Which observable to score ('mean_ratio', 'std_ratio',
                'poisson_fraction').

        Returns:
            z_scores: Z-scored values of shape (T,).
        """
        variants = self.compute_all_variants(X)
        raw = variants[variant]
        return _expanding_zscore(raw, self.rolling_window, self.min_expanding, len(X))
