"""
Classical baseline regime detectors for comparison with QCML observables.

All detectors implement the BaseRegimeDetector interface from
qcml_geometry.observables:
    - fit(X) -> self
    - compute_regime_scores(X) -> 1-D array of length T
    - name (property) -> str

Detectors:
    RollingVolatilityDetector    — 20-day rolling vol z-score
    CUSUMDetector                — Cumulative sum changepoint
    HMMRegimeDetector            — 2-state Gaussian HMM
    BOCPDDetector                — Bayesian Online Changepoint Detection
    IsolationForestDetector      — Isolation Forest anomaly scores
    RandomForestRegimeDetector   — Supervised RF with fit_with_labels()
    RollingWindowRFDetector      — RF trained on rolling VIX > threshold labels
    VIXThresholdDetector         — Expanding z-score of raw VIX close
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

        # Handle single-class training (e.g., no crisis labels in causal window)
        if len(self._model.classes_) < 2:
            proba = np.zeros(len(X_enriched))
        else:
            proba = self._model.predict_proba(X_enriched)[:, 1]

        # Pad front with NaN to match original length
        pad = np.full(self.lookback - 1, np.nan)
        return np.concatenate([pad, proba])


class RollingWindowRFDetector(BaseRegimeDetector):
    """Random Forest trained on a rolling window with VIX > threshold labels.

    Unlike RandomForestRegimeDetector (leave-one-crisis-out with hand-labeled
    crises), this detector uses continuous VIX-based labels: any day where
    VIX > vix_threshold is labeled as crisis. Training uses a trailing window
    of `train_window` days before the evaluation point.

    VIX is used ONLY for labels, not as a feature.

    Args:
        n_estimators: Number of trees.
        max_depth: Max tree depth.
        seed: Random seed.
        lookback: Feature enrichment window (must match pipeline).
        train_window: Number of trailing days for training.
        vix_threshold: VIX level above which a day is labeled crisis.
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 6,
                 seed: int = 42, lookback: int = 20,
                 train_window: int = 250, vix_threshold: float = 25.0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed
        self.lookback = lookback
        self.train_window = train_window
        self.vix_threshold = vix_threshold
        self._model = None

    @property
    def name(self) -> str:
        return "Rolling RF (VIX)"

    def fit(self, X: np.ndarray, **kwargs) -> 'RollingWindowRFDetector':
        return self

    def fit_rolling(self, X_train: np.ndarray, vix_train: np.ndarray) -> 'RollingWindowRFDetector':
        """Fit RF on trailing window with VIX-based labels.

        Args:
            X_train: Enriched feature matrix for the training window.
            vix_train: VIX close values aligned to X_train rows.
        """
        if len(X_train) != len(vix_train):
            raise ValueError(
                f"X_train ({len(X_train)}) and vix_train ({len(vix_train)}) length mismatch"
            )

        # Create binary labels from VIX threshold
        valid_mask = ~np.isnan(vix_train)
        if np.sum(valid_mask) < 10:
            logger.warning("Rolling RF: fewer than 10 valid VIX values in training window")
            return self

        y = (vix_train[valid_mask] > self.vix_threshold).astype(float)
        X_valid = X_train[valid_mask]

        # Handle single-class edge case
        if len(np.unique(y)) < 2:
            logger.warning(
                f"Rolling RF: single class in training window "
                f"(all {'crisis' if y[0] == 1 else 'normal'}), model will predict constant"
            )

        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.seed,
            n_jobs=-1,
        )
        self._model.fit(X_valid, y)
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        """Return P(crisis) from the trained RF.

        Args:
            X: Enriched feature matrix (T, d). Must already be enriched.

        Returns:
            1-D array of length T with P(crisis) scores.
        """
        if self._model is None:
            raise RuntimeError("Must call fit_rolling(X_train, vix_train) before compute_regime_scores()")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if len(self._model.classes_) < 2:
            return np.zeros(len(X))

        return self._model.predict_proba(X)[:, 1]


class VIXThresholdDetector(BaseRegimeDetector):
    """Expanding z-score of raw VIX close.

    Oracle-like upper bound since VIX directly measures implied volatility.
    Score = expanding z-score of VIX, floored at 0 (only elevated VIX
    contributes to regime score).

    The feature matrix X is ignored; scores come from stored VIX values.

    Args:
        min_expanding: Minimum history before computing z-score.
    """

    def __init__(self, min_expanding: int = 60):
        self.min_expanding = min_expanding
        self._vix = None

    @property
    def name(self) -> str:
        return "VIX Level"

    def set_vix(self, vix_aligned: np.ndarray) -> 'VIXThresholdDetector':
        """Store VIX values aligned to evaluation dates.

        Args:
            vix_aligned: 1-D array of VIX close values, same length as X
                         that will be passed to compute_regime_scores().
        """
        self._vix = np.asarray(vix_aligned, dtype=float)
        return self

    def fit(self, X: np.ndarray, **kwargs) -> 'VIXThresholdDetector':
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute expanding z-score of stored VIX values.

        Args:
            X: Feature matrix (ignored, scores come from VIX).

        Returns:
            1-D array of length T with z-scores (NaN before min_expanding).
        """
        if self._vix is None:
            raise RuntimeError("Must call set_vix(vix_aligned) before compute_regime_scores()")

        T = len(self._vix)
        scores = np.full(T, np.nan)

        for t in range(self.min_expanding, T):
            past = self._vix[:t]
            valid = past[~np.isnan(past)]
            if len(valid) < 2:
                continue
            mu = np.mean(valid)
            sigma = np.std(valid, ddof=1)
            if sigma > 1e-12:
                scores[t] = max(0.0, (self._vix[t] - mu) / sigma)
            else:
                scores[t] = 0.0

        return scores
