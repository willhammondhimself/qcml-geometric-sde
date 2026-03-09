"""
Ground State Energy Detector (Q0107)

Uses the reconstruction error E_0(x) = ground state energy of the error
Hamiltonian H(x) as a regime detection signal.

Physical intuition: The error Hamiltonian H(x) = 1/2 sum_k (A_k - x_k I)^2
measures how well the data point x is "explained" by the operators {A_k}.
Its ground state energy E_0(x) is the minimal reconstruction error.

During crises:
  - Market structure changes rapidly, moving data points away from the
    region the operators were calibrated on.
  - Cross-sectional correlations shift, changing effective feature
    distributions.
  - Both effects increase E_0(x), making it a natural anomaly score.

This is a distinct signal from geometric observables (metric, curvature,
Berry phase) because it directly measures the data-operator mismatch
rather than the geometry of the state manifold.

Two scoring modes:
  1. 'level': Raw E_0(x) z-scored. High energy = anomalous point.
  2. 'rate': |dE_0/dt| z-scored. Rapid energy changes = transition.
"""

import logging
from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from qcml_geometry.observables import (
    BaseRegimeDetector,
    ExpandingWindowMixin,
    _apply_normalization,
    _transform_array,
    _transform_point,
)
from qcml_geometry.core import QCMLGeometry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers (same as in detectors.py)
# ---------------------------------------------------------------------------

def _fit_pipeline(detector, X: np.ndarray):
    """Standard fit pipeline: scaler -> PCA -> normalization -> QCML geometry.

    Stores _scaler, _pca, _geometry, _train_norms, _train_std, _epsilon
    on the detector instance.
    """
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
        n_features=X_pca_fit.shape[1], hilbert_dim=detector.hilbert_dim
    )
    if detector.custom_operators is not None:
        detector._geometry.set_operators(detector.custom_operators)
    else:
        detector._geometry.fit_operators(
            X_pca_fit, method=detector.operator_method,
            scale_exponent=detector.scale_exponent,
        )


def _expanding_z_scores(raw_series, min_expanding=60):
    """Expanding-window z-scores (causal, no future leakage)."""
    T = len(raw_series)
    z = np.full(T, np.nan)
    for t in range(min_expanding, T):
        past = raw_series[:t]
        past_valid = past[~np.isnan(past)]
        if len(past_valid) < 10:
            continue
        mu = np.mean(past_valid)
        sigma = np.std(past_valid, ddof=1)
        if sigma > 1e-12:
            z[t] = (raw_series[t] - mu) / sigma
        else:
            z[t] = 0.0
    return z


# ---------------------------------------------------------------------------
# Default constructor kwargs
# ---------------------------------------------------------------------------

_DEFAULT_KWARGS = dict(
    hilbert_dim=8,
    n_pca_components=8,
    operator_method='random',
    scale_exponent=None,
    rolling_window=20,
    min_expanding=60,
    seed=42,
    causal_fit_length=None,
    expanding_refit_interval=None,
    normalization='soft',
    adaptive_epsilon=True,
    subsample=1,
    custom_operators=None,
)


# ===========================================================================
# Q0107 -- Ground State Energy Detector
# ===========================================================================

class GroundStateEnergyDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via ground state energy E_0(x) of the error Hamiltonian.

    The error Hamiltonian H(x) = 1/2 sum_k (A_k - x_k I)^2 has ground state
    energy E_0(x) that measures how well x is "explained" by operators {A_k}.

    Two scoring modes:

    - 'level': z-score of rolling-mean E_0(x). High energy = anomalous point.
    - 'rate': z-score of rolling-mean |dE_0/dt|. Rapid changes = transition.

    Args:
        score_mode: 'level' or 'rate'. Default 'level'.
        rate_window: Window for finite differences in 'rate' mode. Default 5.
        **kwargs: Standard detector kwargs (hilbert_dim, n_pca_components, etc.).
    """

    def __init__(self, score_mode: str = 'level', rate_window: int = 5, **kwargs):
        self.score_mode = score_mode
        self.rate_window = rate_window
        for k, v in {**_DEFAULT_KWARGS}.items():
            setattr(self, k, kwargs.get(k, v))
        self._geometry = None
        self._scaler = None
        self._pca = None
        self._snapshots = None
        self._train_norms = None
        self._train_std = None
        self._epsilon = 1e-5

    @property
    def name(self) -> str:
        return f"Ground State Energy ({self.score_mode})"

    def fit(self, X: np.ndarray, **kwargs) -> 'GroundStateEnergyDetector':
        if self.expanding_refit_interval is not None:
            return self._fit_expanding(X)
        _fit_pipeline(self, X)
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute regime scores based on ground state energy.

        Returns:
            1-D array of z-scored regime scores, length T.
            Higher values indicate crisis conditions.
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

        # Step 1: Compute E_0(x) for each time point
        energy_raw = np.full(T, np.nan)
        for t in range(0, T, self.subsample):
            if self._snapshots is not None:
                geo, xt = self._transform_point_at(X[t], t)
                _, E0 = geo.quasi_coherent_state(xt, return_energy=True)
            else:
                _, E0 = self._geometry.quasi_coherent_state(Xt[t], return_energy=True)
            energy_raw[t] = E0

        # Interpolate subsampled points
        if self.subsample > 1:
            energy_raw = pd.Series(energy_raw).interpolate(method='linear').values

        if self.score_mode == 'level':
            # Rolling mean of E_0, then z-score
            rolling_vals = (
                pd.Series(energy_raw)
                .rolling(window=self.rolling_window, min_periods=1)
                .mean()
                .values
            )
            # High energy = anomalous -> positive score
            return _expanding_z_scores(rolling_vals, self.min_expanding)

        elif self.score_mode == 'rate':
            # |dE_0/dt| = |E_0(t) - E_0(t - w)| / w
            rate_raw = np.full(T, np.nan)
            w = self.rate_window
            for t in range(w, T):
                if not np.isnan(energy_raw[t]) and not np.isnan(energy_raw[t - w]):
                    rate_raw[t] = abs(energy_raw[t] - energy_raw[t - w]) / w

            rolling_vals = (
                pd.Series(rate_raw)
                .rolling(window=self.rolling_window, min_periods=1)
                .mean()
                .values
            )
            # High rate = rapid transition -> positive score
            return _expanding_z_scores(rolling_vals, self.min_expanding)

        else:
            raise ValueError(f"Unknown score_mode: {self.score_mode}")
