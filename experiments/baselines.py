"""
Classical baseline regime detectors for comparison with QCML observables.

All detectors implement the BaseRegimeDetector interface from
qcml_geometry.observables:
    - fit(X) -> self
    - compute_regime_scores(X) -> 1-D array of length T
    - name (property) -> str

Detectors:
    RollingVolatilityDetector  — 20-day rolling vol z-score
    CUSUMDetector              — Cumulative sum changepoint
    HMMRegimeDetector          — 2-state Gaussian HMM
    BOCPDDetector              — Bayesian Online Changepoint Detection
    IsolationForestDetector    — Isolation Forest anomaly scores
    RandomForestRegimeDetector — Supervised RF with fit_with_labels()
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest

from qcml_geometry.observables import BaseRegimeDetector

logger = logging.getLogger(__name__)


class RollingVolatilityDetector(BaseRegimeDetector):
    """Rolling volatility z-score detector.

    Score = |z-score| of rolling standard deviation of first feature.
    """

    def __init__(self, vol_window: int = 20, min_expanding: int = 60):
        self.vol_window = vol_window
        self.min_expanding = min_expanding

    @property
    def name(self) -> str:
        return "Rolling Vol Z"

    def fit(self, X: np.ndarray, **kwargs) -> 'RollingVolatilityDetector':
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        T = X.shape[0]

        vol = pd.Series(X[:, 0]).rolling(self.vol_window, min_periods=1).std().values

        z_scores = np.full(T, np.nan)
        for t in range(self.min_expanding, T):
            mu = np.nanmean(vol[:t])
            sigma = np.nanstd(vol[:t], ddof=1)
            if sigma > 1e-12:
                z_scores[t] = abs((vol[t] - mu) / sigma)
            else:
                z_scores[t] = 0.0
        return z_scores


class CUSUMDetector(BaseRegimeDetector):
    """Cumulative sum changepoint detector.

    Two-sided CUSUM on first feature: accumulates when |x - mu| > k*sigma.
    Score is the maximum of upper and lower CUSUM statistics.
    """

    def __init__(self, k: float = 0.5, burn_in: int = 60):
        self.k = k
        self.burn_in = burn_in
        self._mu = None
        self._sigma = None

    @property
    def name(self) -> str:
        return "CUSUM"

    def fit(self, X: np.ndarray, **kwargs) -> 'CUSUMDetector':
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        burn = min(self.burn_in, X.shape[0])
        self._mu = np.mean(X[:burn, 0])
        self._sigma = max(np.std(X[:burn, 0], ddof=1), 1e-12)
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._mu is None:
            raise RuntimeError("Call fit() first")
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        T = X.shape[0]
        scores = np.zeros(T)
        s_high = 0.0
        s_low = 0.0

        for t in range(T):
            z = (X[t, 0] - self._mu) / self._sigma
            s_high = max(0.0, s_high + z - self.k)
            s_low = max(0.0, s_low - z - self.k)
            scores[t] = max(s_high, s_low)

        return scores


class HMMRegimeDetector(BaseRegimeDetector):
    """2-state Gaussian HMM regime detector.

    Score = P(high-volatility state | data). The high-vol state is identified
    as the one with higher mean absolute value of features.
    """

    def __init__(self, n_states: int = 2, n_iter: int = 100,
                 covariance_type: str = 'full', seed: int = 42):
        self.n_states = n_states
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.seed = seed
        self._model = None
        self._high_vol_state = None

    @property
    def name(self) -> str:
        return "HMM"

    def fit(self, X: np.ndarray, **kwargs) -> 'HMMRegimeDetector':
        from hmmlearn.hmm import GaussianHMM

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self._model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.seed,
        )
        self._model.fit(X)

        # Identify high-vol state as the one with larger mean absolute value
        means_abs = np.abs(self._model.means_).sum(axis=1)
        self._high_vol_state = int(np.argmax(means_abs))

        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() first")
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        posteriors = self._model.predict_proba(X)
        return posteriors[:, self._high_vol_state]


class BOCPDDetector(BaseRegimeDetector):
    """Bayesian Online Changepoint Detection (Adams & MacKay 2007).

    Score = 1 - P(run_length > current_run), i.e. probability of
    recent changepoint.
    """

    def __init__(self, hazard_rate: float = 250.0, min_expanding: int = 30):
        self.hazard_rate = hazard_rate
        self.min_expanding = min_expanding

    @property
    def name(self) -> str:
        return "BOCPD"

    def fit(self, X: np.ndarray, **kwargs) -> 'BOCPDDetector':
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        T = X.shape[0]
        x = X[:, 0]

        # Online Bayesian changepoint detection
        # R[t, r] = P(run_length = r at time t)
        R = np.zeros((T + 1, T + 1))
        R[0, 0] = 1.0

        scores = np.full(T, np.nan)
        h = 1.0 / self.hazard_rate  # hazard function (constant)

        # Sufficient statistics for online mean/variance
        mu_0, kappa_0, alpha_0, beta_0 = 0.0, 1.0, 1.0, 1.0

        for t in range(T):
            # Predictive probability under each run length
            predprobs = np.zeros(t + 1)
            for r in range(t + 1):
                if r == 0:
                    # Prior predictive
                    predprobs[r] = _student_t_pdf(x[t], mu_0, beta_0 * (kappa_0 + 1) / (kappa_0 * alpha_0), 2.0 * alpha_0)
                else:
                    # Use running sufficient stats
                    start = t - r
                    seg = x[start:t + 1]
                    n = len(seg)
                    mean_seg = np.mean(seg)
                    var_seg = np.var(seg, ddof=0) if n > 1 else 0.0

                    kappa_n = kappa_0 + n
                    alpha_n = alpha_0 + n / 2.0
                    mu_n = (kappa_0 * mu_0 + n * mean_seg) / kappa_n
                    beta_n = beta_0 + 0.5 * n * var_seg + 0.5 * kappa_0 * n * (mean_seg - mu_0) ** 2 / kappa_n

                    predprobs[r] = _student_t_pdf(
                        x[t], mu_n, beta_n * (kappa_n + 1) / (kappa_n * alpha_n), 2.0 * alpha_n
                    )

            # Growth probabilities
            R[t + 1, 1:t + 2] = R[t, :t + 1] * predprobs * (1 - h)
            # Changepoint probability
            R[t + 1, 0] = np.sum(R[t, :t + 1] * predprobs * h)

            # Normalize
            evidence = np.sum(R[t + 1, :t + 2])
            if evidence > 0:
                R[t + 1, :t + 2] /= evidence

            # Score = P(changepoint in recent window)
            scores[t] = R[t + 1, 0] + np.sum(R[t + 1, 1:min(4, t + 2)])

        return scores


def _student_t_pdf(x, mu, scale, df):
    """Student-t PDF for BOCPD predictive distribution."""
    from scipy.special import gammaln

    scale = max(scale, 1e-12)
    z = (x - mu) / np.sqrt(scale)
    log_pdf = (
        gammaln((df + 1) / 2) - gammaln(df / 2)
        - 0.5 * np.log(df * np.pi * scale)
        - (df + 1) / 2 * np.log(1 + z ** 2 / df)
    )
    return np.exp(log_pdf)


class IsolationForestDetector(BaseRegimeDetector):
    """Isolation Forest anomaly detector.

    Score = anomaly score from sklearn IsolationForest, rescaled to [0, 1].
    """

    def __init__(self, n_estimators: int = 100, contamination: float = 0.05,
                 seed: int = 42):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.seed = seed
        self._model = None

    @property
    def name(self) -> str:
        return "Isolation Forest"

    def fit(self, X: np.ndarray, **kwargs) -> 'IsolationForestDetector':
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.seed,
        )
        self._model.fit(X)
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() first")
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # score_samples returns negative anomaly scores (more negative = more anomalous)
        raw_scores = -self._model.score_samples(X)
        # Rescale to [0, 1]
        mn, mx = raw_scores.min(), raw_scores.max()
        if mx - mn > 1e-12:
            return (raw_scores - mn) / (mx - mn)
        return np.zeros(len(X))


class RandomForestRegimeDetector(BaseRegimeDetector):
    """Supervised Random Forest baseline.

    Requires fit_with_labels(X, y) where y is binary crisis labels.
    fit(X) is a no-op; compute_regime_scores() raises if no labels provided.
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 6,
                 seed: int = 42, lookback: int = 20):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed
        self.lookback = lookback
        self._model = None

    @property
    def name(self) -> str:
        return "Random Forest"

    def fit(self, X: np.ndarray, **kwargs) -> 'RandomForestRegimeDetector':
        # No-op: supervised baseline needs labels via fit_with_labels
        return self

    def fit_with_labels(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestRegimeDetector':
        """Fit supervised RF with binary crisis labels.

        Args:
            X: Feature matrix (T, d).
            y: Binary labels (T,) where 1 = crisis.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=self.lookback)
        y_trimmed = y[self.lookback - 1:]

        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.seed,
            n_jobs=-1,
        )
        self._model.fit(X_enriched, y_trimmed)
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Must call fit_with_labels(X, y) before compute_regime_scores()")
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=self.lookback)
        proba = self._model.predict_proba(X_enriched)[:, 1]

        # Pad front with NaN to match original length
        pad = np.full(self.lookback - 1, np.nan)
        return np.concatenate([pad, proba])
