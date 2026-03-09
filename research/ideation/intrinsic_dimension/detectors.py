"""
Intrinsic Dimension Detectors (Q0101-Q0104)

Four regime detectors based on intrinsic dimension estimation from the
quantum metric tensor eigenvalue spectrum:

1. IntrinsicDimensionDetector (Q0101): Spectral gap analysis to estimate
   the effective number of "active" directions in the data manifold.
2. SpectralGapRatioDetector (Q0102): Ratio of normal-to-tangent eigenvalues
   as a crisis severity measure.
3. DimensionRateDetector (Q0103): Time derivative |dd/dt| for early warning.
4. EffectiveDimensionDetector (Q0104): Participation ratio PR = (sum lambda)^2 / sum(lambda^2),
   a mathematically distinct measure from IPR = lambda_max / sum(lambda).
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
# Shared helpers
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


def _get_metric_eigenvalues(detector, Xt, X_raw, t):
    """Compute sorted (descending) non-negative eigenvalues of the metric at time t."""
    if detector._snapshots is not None:
        geo, xt = detector._transform_point_at(X_raw[t], t)
        g = geo.quantum_metric(xt)
    else:
        g = detector._geometry.quantum_metric(Xt[t])

    eigvals = np.linalg.eigvalsh(g)
    eigvals = np.maximum(eigvals, 0)
    # Sort descending
    eigvals = eigvals[::-1]
    return eigvals


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
# Default constructor kwargs shared across all four detectors
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
# Q0101 -- Intrinsic Dimension from spectral gap
# ===========================================================================

class IntrinsicDimensionDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via intrinsic dimension estimated from the quantum
    metric eigenvalue spectrum.

    Uses a continuous (soft) dimension estimate based on the spectral entropy
    of the normalized eigenvalue distribution:

        p_k = lambda_k / sum(lambda)
        d_eff(t) = exp(-sum(p_k * log(p_k)))

    This is the exponential of the Shannon entropy of the eigenvalue
    distribution.  It ranges continuously from 1 (single dominant eigenvalue)
    to n (uniform spectrum).  Unlike the hard spectral-gap approach, it
    responds smoothly to changes in the eigenvalue concentration.

    During crises the eigenvalue spectrum concentrates, so d_eff drops.
    Score = -expanding_z(rolling_mean(d_eff)) so drops produce positive scores.
    """

    def __init__(self, **kwargs):
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
        return "Intrinsic Dimension"

    def fit(self, X: np.ndarray, **kwargs) -> 'IntrinsicDimensionDetector':
        if self.expanding_refit_interval is not None:
            return self._fit_expanding(X)
        _fit_pipeline(self, X)
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        Xt = _transform_array(
            X, self._scaler, self._pca,
            normalization=self.normalization,
            train_norms=self._train_norms,
            train_std=self._train_std,
        )
        T = len(Xt)

        dim_raw = np.full(T, np.nan)
        for t in range(0, T, self.subsample):
            eigvals = _get_metric_eigenvalues(self, Xt, X, t)
            total = np.sum(eigvals)
            if total > 1e-15:
                p = eigvals / total
                p = p[p > 1e-30]  # avoid log(0)
                entropy = -np.sum(p * np.log(p))
                dim_raw[t] = np.exp(entropy)
            else:
                dim_raw[t] = 1.0

        # Interpolate subsampled points
        if self.subsample > 1:
            dim_raw = pd.Series(dim_raw).interpolate(method='linear').values

        rolling_vals = (
            pd.Series(dim_raw)
            .rolling(window=self.rolling_window, min_periods=1)
            .mean()
            .values
        )

        # Negate: dimension drops in crises -> positive score
        z = _expanding_z_scores(rolling_vals, self.min_expanding)
        return -z


# ===========================================================================
# Q0102 -- Spectral Gap Ratio
# ===========================================================================

class SpectralGapRatioDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via spectral gap ratio of the quantum metric.

    Splits eigenvalues at the spectral gap into "tangent" (above gap) and
    "normal" (below gap) groups.

    Score = abs(z-score of rolling-mean gap_sharpness), where
    gap_sharpness = lambda_{d} / lambda_{d+1} (ratio at gap position).

    A large gap means well-separated tangent/normal spaces (stable regime);
    gap collapse (ratio -> 1) indicates a crisis.  We negate so that crises
    produce high scores: score = -z(gap_sharpness), then take abs.
    """

    def __init__(self, **kwargs):
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
        return "Spectral Gap Ratio"

    def fit(self, X: np.ndarray, **kwargs) -> 'SpectralGapRatioDetector':
        if self.expanding_refit_interval is not None:
            return self._fit_expanding(X)
        _fit_pipeline(self, X)
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        Xt = _transform_array(
            X, self._scaler, self._pca,
            normalization=self.normalization,
            train_norms=self._train_norms,
            train_std=self._train_std,
        )
        T = len(Xt)

        ratio_raw = np.full(T, np.nan)
        for t in range(0, T, self.subsample):
            eigvals = _get_metric_eigenvalues(self, Xt, X, t)
            n = len(eigvals)
            if n < 2:
                ratio_raw[t] = 1.0
                continue

            # Find gap position
            best_k = 1
            best_gap = 0.0
            for k in range(1, n):
                denom = eigvals[k] if eigvals[k] > 1e-15 else 1e-15
                gap = eigvals[k - 1] / denom
                if gap > best_gap:
                    best_gap = gap
                    best_k = k

            # Gap sharpness = ratio at the gap
            denom = eigvals[best_k] if eigvals[best_k] > 1e-15 else 1e-15
            ratio_raw[t] = eigvals[best_k - 1] / denom

        if self.subsample > 1:
            ratio_raw = pd.Series(ratio_raw).interpolate(method='linear').values

        rolling_vals = (
            pd.Series(ratio_raw)
            .rolling(window=self.rolling_window, min_periods=1)
            .mean()
            .values
        )

        # Negate: low gap ratio = crisis -> high score
        z = _expanding_z_scores(rolling_vals, self.min_expanding)
        return -z


# ===========================================================================
# Q0103 -- Dimension Rate (dd/dt)
# ===========================================================================

class DimensionRateDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via rate of change of intrinsic dimension.

    Uses the same soft dimension estimate as IntrinsicDimensionDetector
    (spectral entropy dimension), then takes:

        |dd/dt| = |d_eff(t) - d_eff(t - rate_window)| / rate_window

    Rapid dimension changes signal regime transitions, potentially providing
    earlier warning than the level of d itself.

    Score = abs(expanding z-score of rolling |dd/dt|).
    """

    def __init__(self, rate_window: int = 5, **kwargs):
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
        return "Dimension Rate"

    def fit(self, X: np.ndarray, **kwargs) -> 'DimensionRateDetector':
        if self.expanding_refit_interval is not None:
            return self._fit_expanding(X)
        _fit_pipeline(self, X)
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        Xt = _transform_array(
            X, self._scaler, self._pca,
            normalization=self.normalization,
            train_norms=self._train_norms,
            train_std=self._train_std,
        )
        T = len(Xt)

        # Step 1: compute soft dimension time series (spectral entropy dimension)
        dim_raw = np.full(T, np.nan)
        for t in range(0, T, self.subsample):
            eigvals = _get_metric_eigenvalues(self, Xt, X, t)
            total = np.sum(eigvals)
            if total > 1e-15:
                p = eigvals / total
                p = p[p > 1e-30]
                entropy = -np.sum(p * np.log(p))
                dim_raw[t] = np.exp(entropy)
            else:
                dim_raw[t] = 1.0

        if self.subsample > 1:
            dim_raw = pd.Series(dim_raw).interpolate(method='linear').values

        # Step 2: |dd/dt|
        rate_raw = np.full(T, np.nan)
        w = self.rate_window
        for t in range(w, T):
            if not np.isnan(dim_raw[t]) and not np.isnan(dim_raw[t - w]):
                rate_raw[t] = abs(dim_raw[t] - dim_raw[t - w]) / w

        rolling_vals = (
            pd.Series(rate_raw)
            .rolling(window=self.rolling_window, min_periods=1)
            .mean()
            .values
        )

        return _expanding_z_scores(rolling_vals, self.min_expanding)


# ===========================================================================
# Q0104 -- Effective Dimension (Participation Ratio)
# ===========================================================================

class EffectiveDimensionDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via participation ratio of metric eigenvalues.

    PR = (sum lambda)^2 / sum(lambda^2)

    This is the proper participation ratio, counting the effective number of
    contributing eigenvalues.  It ranges from 1 (one dominant eigenvalue) to n
    (uniform spectrum).

    Contrast with IPR used by DimensionalityCollapseDetector:
        IPR = lambda_max / sum(lambda)   (range 1/n to 1)

    PR is a more sensitive measure because it responds to the full eigenvalue
    distribution, not just the top eigenvalue.

    During crises, the eigenvalue spectrum concentrates (fewer significant
    directions), so PR drops.

    Score = -expanding_z(rolling_mean(PR)) so that drops produce positive scores.
    """

    def __init__(self, **kwargs):
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
        return "Effective Dimension (PR)"

    def fit(self, X: np.ndarray, **kwargs) -> 'EffectiveDimensionDetector':
        if self.expanding_refit_interval is not None:
            return self._fit_expanding(X)
        _fit_pipeline(self, X)
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        Xt = _transform_array(
            X, self._scaler, self._pca,
            normalization=self.normalization,
            train_norms=self._train_norms,
            train_std=self._train_std,
        )
        T = len(Xt)

        pr_raw = np.full(T, np.nan)
        for t in range(0, T, self.subsample):
            eigvals = _get_metric_eigenvalues(self, Xt, X, t)
            total = np.sum(eigvals)
            sum_sq = np.sum(eigvals ** 2)
            if sum_sq > 1e-30:
                pr_raw[t] = (total ** 2) / sum_sq
            else:
                pr_raw[t] = 1.0

        if self.subsample > 1:
            pr_raw = pd.Series(pr_raw).interpolate(method='linear').values

        rolling_vals = (
            pd.Series(pr_raw)
            .rolling(window=self.rolling_window, min_periods=1)
            .mean()
            .values
        )

        # Negate: PR drops during crises -> high score
        z = _expanding_z_scores(rolling_vals, self.min_expanding)
        return -z
