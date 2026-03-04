"""
Information Geometry Detectors for Regime Detection

Statistical geometry on return distributions — complementary to QCML Hilbert
space geometry. These detectors measure distances and divergences between
windowed return distributions to detect distributional regime shifts.

Detectors:
1. FisherRaoDetector — Fisher-Rao distance between windowed Gaussian fits
2. WassersteinDetector — Wasserstein-1 distance between return windows
3. KLDivergenceDetector — Symmetrized KL divergence between windows
4. SinkhornDetector — Regularized optimal transport (Sinkhorn divergence)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from .observables import BaseRegimeDetector

logger = logging.getLogger(__name__)


class FisherRaoDetector(BaseRegimeDetector):
    """Regime detection via Fisher-Rao distance on the statistical manifold.

    For univariate Gaussian, the Fisher-Rao distance between N(mu1, sigma1)
    and N(mu2, sigma2) is:

        d_FR = sqrt(2) * arccosh(1 + ((mu1-mu2)^2 + 2*(sigma1-sigma2)^2) /
                                      (2 * sigma1 * sigma2))

    For practical estimation, we fit Gaussian parameters to rolling windows
    and compute the Fisher-Rao distance between consecutive windows.

    Score = abs(z-score of d_FR) smoothed over a rolling window.
    """

    def __init__(
        self,
        window: int = 30,
        lag: int = 10,
        rolling_window: int = 15,
        min_expanding: int = 60,
        seed: int = 42,
        **kwargs,
    ):
        self.window = window
        self.lag = lag
        self.rolling_window = rolling_window
        self.min_expanding = min_expanding
        self.seed = seed
        self._is_fitted = False

    @property
    def name(self) -> str:
        return "Fisher-Rao"

    def fit(self, X: np.ndarray, **kwargs) -> 'FisherRaoDetector':
        self._is_fitted = True
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        T, d = X.shape

        # Use first column (typically returns) for univariate Fisher-Rao
        returns = X[:, 0]

        fisher_rao = np.full(T, np.nan)
        w = self.window
        lag = self.lag

        for t in range(w + lag, T):
            # Current window
            win_curr = returns[t - w:t]
            # Lagged window
            win_prev = returns[t - w - lag:t - lag]

            mu1, sigma1 = np.mean(win_curr), np.std(win_curr, ddof=1)
            mu2, sigma2 = np.mean(win_prev), np.std(win_prev, ddof=1)

            sigma1 = max(sigma1, 1e-10)
            sigma2 = max(sigma2, 1e-10)

            # Fisher-Rao distance for univariate Gaussian
            numer = (mu1 - mu2) ** 2 + 2 * (sigma1 - sigma2) ** 2
            denom = 2 * sigma1 * sigma2
            arg = 1.0 + numer / denom
            fisher_rao[t] = np.sqrt(2) * np.arccosh(max(arg, 1.0))

        rolling_vals = (
            pd.Series(fisher_rao)
            .rolling(window=self.rolling_window, min_periods=1)
            .mean()
            .values
        )

        z_scores = np.full(T, np.nan)
        for t in range(self.min_expanding, T):
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


class WassersteinDetector(BaseRegimeDetector):
    """Regime detection via Wasserstein-1 distance between return windows.

    W_1 measures the minimal cost to transport one distribution to another.
    Robust to outliers and captures both location and shape changes.
    Uses scipy.stats.wasserstein_distance for 1D; extends to multivariate
    via sliced Wasserstein (average over random projections).

    Score = abs(z-score of W_1) smoothed over a rolling window.
    """

    def __init__(
        self,
        window: int = 30,
        lag: int = 10,
        rolling_window: int = 15,
        min_expanding: int = 60,
        n_projections: int = 20,
        seed: int = 42,
        **kwargs,
    ):
        self.window = window
        self.lag = lag
        self.rolling_window = rolling_window
        self.min_expanding = min_expanding
        self.n_projections = n_projections
        self.seed = seed
        self._is_fitted = False

    @property
    def name(self) -> str:
        return "Wasserstein"

    def fit(self, X: np.ndarray, **kwargs) -> 'WassersteinDetector':
        self._is_fitted = True
        return self

    def _sliced_wasserstein(self, X1: np.ndarray, X2: np.ndarray,
                            rng: np.random.Generator) -> float:
        """Compute sliced Wasserstein distance between two multivariate samples."""
        d = X1.shape[1]
        if d == 1:
            return float(stats.wasserstein_distance(X1.ravel(), X2.ravel()))

        total = 0.0
        for _ in range(self.n_projections):
            direction = rng.standard_normal(d)
            direction /= np.linalg.norm(direction)
            proj1 = X1 @ direction
            proj2 = X2 @ direction
            total += stats.wasserstein_distance(proj1, proj2)

        return total / self.n_projections

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        T, d = X.shape
        rng = np.random.default_rng(self.seed)

        wass = np.full(T, np.nan)
        w = self.window
        lag = self.lag

        for t in range(w + lag, T):
            win_curr = X[t - w:t]
            win_prev = X[t - w - lag:t - lag]

            if d == 1:
                wass[t] = float(stats.wasserstein_distance(
                    win_curr.ravel(), win_prev.ravel()
                ))
            else:
                wass[t] = self._sliced_wasserstein(win_curr, win_prev, rng)

        rolling_vals = (
            pd.Series(wass)
            .rolling(window=self.rolling_window, min_periods=1)
            .mean()
            .values
        )

        z_scores = np.full(T, np.nan)
        for t in range(self.min_expanding, T):
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


class KLDivergenceDetector(BaseRegimeDetector):
    """Regime detection via symmetrized KL divergence between return windows.

    Uses Jensen-Shannon divergence (symmetric, bounded version of KL):
        JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = 0.5*(P+Q)

    Estimated via histogram binning of windowed returns.

    Score = abs(z-score of JSD) smoothed over a rolling window.
    """

    def __init__(
        self,
        window: int = 30,
        lag: int = 10,
        rolling_window: int = 15,
        min_expanding: int = 60,
        n_bins: int = 30,
        seed: int = 42,
        **kwargs,
    ):
        self.window = window
        self.lag = lag
        self.rolling_window = rolling_window
        self.min_expanding = min_expanding
        self.n_bins = n_bins
        self.seed = seed
        self._is_fitted = False

    @property
    def name(self) -> str:
        return "KL Divergence"

    def fit(self, X: np.ndarray, **kwargs) -> 'KLDivergenceDetector':
        self._is_fitted = True
        return self

    def _jsd_1d(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute 1D Jensen-Shannon divergence via histogram."""
        combined = np.concatenate([x1, x2])
        lo, hi = np.min(combined), np.max(combined)
        if hi - lo < 1e-15:
            return 0.0

        bins = np.linspace(lo, hi, self.n_bins + 1)
        p, _ = np.histogram(x1, bins=bins, density=True)
        q, _ = np.histogram(x2, bins=bins, density=True)

        # Add small epsilon for numerical stability
        eps = 1e-10
        p = p + eps
        q = q + eps
        p = p / p.sum()
        q = q / q.sum()

        m = 0.5 * (p + q)
        jsd = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
        return float(jsd)

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        T, d = X.shape

        jsd_vals = np.full(T, np.nan)
        w = self.window
        lag = self.lag

        for t in range(w + lag, T):
            total_jsd = 0.0
            for col in range(d):
                win_curr = X[t - w:t, col]
                win_prev = X[t - w - lag:t - lag, col]
                total_jsd += self._jsd_1d(win_curr, win_prev)
            jsd_vals[t] = total_jsd / d

        rolling_vals = (
            pd.Series(jsd_vals)
            .rolling(window=self.rolling_window, min_periods=1)
            .mean()
            .values
        )

        z_scores = np.full(T, np.nan)
        for t in range(self.min_expanding, T):
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


class SinkhornDetector(BaseRegimeDetector):
    """Regime detection via Sinkhorn divergence (regularized optimal transport).

    Sinkhorn divergence S_eps(P, Q) = OT_eps(P, Q) - 0.5*OT_eps(P, P) - 0.5*OT_eps(Q, Q)
    where OT_eps is the entropy-regularized optimal transport cost.

    Falls back to Wasserstein-1 if POT library is unavailable.

    Score = abs(z-score of S_eps) smoothed over a rolling window.
    """

    def __init__(
        self,
        window: int = 30,
        lag: int = 10,
        rolling_window: int = 15,
        min_expanding: int = 60,
        reg: float = 0.1,
        seed: int = 42,
        **kwargs,
    ):
        self.window = window
        self.lag = lag
        self.rolling_window = rolling_window
        self.min_expanding = min_expanding
        self.reg = reg
        self.seed = seed
        self._is_fitted = False
        self._use_pot = False

    @property
    def name(self) -> str:
        return "Sinkhorn"

    def fit(self, X: np.ndarray, **kwargs) -> 'SinkhornDetector':
        try:
            import ot  # noqa: F401
            self._use_pot = True
        except ImportError:
            logger.info("POT not installed; Sinkhorn falls back to Wasserstein-1.")
            self._use_pot = False
        self._is_fitted = True
        return self

    def _sinkhorn_1d(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute 1D Sinkhorn divergence."""
        if self._use_pot:
            import ot

            n, m = len(x1), len(x2)
            a = np.ones(n) / n
            b = np.ones(m) / m

            # Cost matrix: squared Euclidean distance
            M = (x1.reshape(-1, 1) - x2.reshape(1, -1)) ** 2

            # Sinkhorn divergence = OT_eps(P,Q) - 0.5*OT_eps(P,P) - 0.5*OT_eps(Q,Q)
            ot_pq = ot.sinkhorn2(a, b, M, self.reg, numItermax=100)

            M_pp = (x1.reshape(-1, 1) - x1.reshape(1, -1)) ** 2
            ot_pp = ot.sinkhorn2(a, a, M_pp, self.reg, numItermax=100)

            M_qq = (x2.reshape(-1, 1) - x2.reshape(1, -1)) ** 2
            ot_qq = ot.sinkhorn2(b, b, M_qq, self.reg, numItermax=100)

            return float(max(ot_pq - 0.5 * ot_pp - 0.5 * ot_qq, 0.0))
        else:
            return float(stats.wasserstein_distance(x1, x2))

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        T, d = X.shape

        sink_vals = np.full(T, np.nan)
        w = self.window
        lag = self.lag

        for t in range(w + lag, T):
            total = 0.0
            for col in range(d):
                win_curr = X[t - w:t, col]
                win_prev = X[t - w - lag:t - lag, col]
                total += self._sinkhorn_1d(win_curr, win_prev)
            sink_vals[t] = total / d

        rolling_vals = (
            pd.Series(sink_vals)
            .rolling(window=self.rolling_window, min_periods=1)
            .mean()
            .values
        )

        z_scores = np.full(T, np.nan)
        for t in range(self.min_expanding, T):
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
