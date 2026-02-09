"""
Classical Baseline Regime Detectors for Head-to-Head Comparison

Four classical/statistical baselines:
1. RollingVolatilityDetector — rolling sigma z-scored against expanding window
2. CUSUMDetector — cumulative sum of mean-adjusted absolute returns
3. HMMRegimeDetector — 2-state Gaussian HMM via hmmlearn
4. RandomForestRegimeDetector — P(crisis) from sklearn RF
"""

import logging
from typing import Optional

import numpy as np

from qcml_geometry import BaseRegimeDetector

logger = logging.getLogger(__name__)


class RollingVolatilityDetector(BaseRegimeDetector):
    """Rolling volatility z-scored against an expanding window.

    score[t] = (sigma_window[t] - mu_expanding[t]) / sigma_expanding[t]

    Args:
        vol_window: Window for computing rolling standard deviation.
        min_expanding: Minimum expanding window before scoring starts.
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
        if X.ndim == 2 and X.shape[1] > 1:
            returns = np.diff(X[:, 0])
        else:
            returns = np.diff(X.ravel())

        T = len(returns)
        scores = np.full(T + 1, np.nan)

        for t in range(self.vol_window, T):
            sigma_t = np.std(returns[t - self.vol_window:t], ddof=1)

            if t >= self.min_expanding:
                expanding_vols = []
                for j in range(self.vol_window, t + 1):
                    expanding_vols.append(
                        np.std(returns[j - self.vol_window:j], ddof=1)
                    )
                mu_exp = np.mean(expanding_vols)
                sigma_exp = np.std(expanding_vols, ddof=1)
                if sigma_exp > 1e-12:
                    scores[t + 1] = (sigma_t - mu_exp) / sigma_exp
                else:
                    scores[t + 1] = 0.0

        return scores


class CUSUMDetector(BaseRegimeDetector):
    """Cumulative sum (CUSUM) detector for mean shifts in absolute returns.

    S[t] = max(0, S[t-1] + |r[t]| - k)

    where k (the drift parameter) defaults to the mean of |returns| during
    a burn-in period.

    Args:
        burn_in: Number of observations to estimate k.
    """

    def __init__(self, burn_in: int = 60):
        self.burn_in = burn_in
        self._k: Optional[float] = None

    @property
    def name(self) -> str:
        return "CUSUM"

    def fit(self, X: np.ndarray, **kwargs) -> 'CUSUMDetector':
        if X.ndim == 2 and X.shape[1] > 1:
            returns = np.diff(X[:, 0])
        else:
            returns = np.diff(X.ravel())

        n = min(self.burn_in, len(returns))
        self._k = float(np.mean(np.abs(returns[:n])))
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._k is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        if X.ndim == 2 and X.shape[1] > 1:
            returns = np.diff(X[:, 0])
        else:
            returns = np.diff(X.ravel())

        T = len(returns)
        S = np.zeros(T + 1)
        for t in range(T):
            S[t + 1] = max(0.0, S[t] + np.abs(returns[t]) - self._k)

        return S


class HMMRegimeDetector(BaseRegimeDetector):
    """2-state Gaussian Hidden Markov Model for regime detection.

    Score = P(high-volatility state) at each time step.  The high-vol
    state is identified as the one with larger variance.

    Requires ``hmmlearn``.

    Args:
        n_iter: Maximum EM iterations.
        seed: Random seed for reproducibility.
    """

    def __init__(self, n_iter: int = 100, seed: int = 42):
        self.n_iter = n_iter
        self.seed = seed
        self._model = None
        self._high_vol_state: Optional[int] = None

    @property
    def name(self) -> str:
        return "HMM 2-state"

    def fit(self, X: np.ndarray, **kwargs) -> 'HMMRegimeDetector':
        from hmmlearn.hmm import GaussianHMM

        if X.ndim == 2 and X.shape[1] > 1:
            returns = np.diff(X[:, 0]).reshape(-1, 1)
        else:
            returns = np.diff(X.ravel()).reshape(-1, 1)

        sorted_rets = np.sort(returns.ravel())
        n = len(sorted_rets)
        low_half = sorted_rets[:n // 2]
        high_half = sorted_rets[n // 2:]

        self._model = GaussianHMM(
            n_components=2,
            covariance_type='full',
            n_iter=self.n_iter,
            random_state=self.seed,
            init_params='',
        )
        self._model.startprob_ = np.array([0.5, 0.5])
        self._model.transmat_ = np.array([[0.95, 0.05], [0.05, 0.95]])
        self._model.means_ = np.array([
            [np.mean(low_half)], [np.mean(high_half)]
        ])
        self._model.covars_ = np.array([
            [[np.var(low_half, ddof=1) + 1e-10]],
            [[np.var(high_half, ddof=1) + 1e-10]],
        ])
        self._model.fit(returns)

        variances = [self._model.covars_[i][0, 0] for i in range(2)]
        self._high_vol_state = int(np.argmax(variances))
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        if X.ndim == 2 and X.shape[1] > 1:
            returns = np.diff(X[:, 0]).reshape(-1, 1)
        else:
            returns = np.diff(X.ravel()).reshape(-1, 1)

        posteriors = self._model.predict_proba(returns)
        high_vol_prob = posteriors[:, self._high_vol_state]

        return np.concatenate([[np.nan], high_vol_prob])


class RandomForestRegimeDetector(BaseRegimeDetector):
    """Random Forest classifier for regime detection.

    Produces P(crisis) from a trained RF.  Training requires labeled data
    via ``fit_with_labels``.  The standard ``fit()`` is a no-op.

    Args:
        n_estimators: Number of trees.
        max_depth: Maximum tree depth.
        seed: Random seed.
        lookback: Rolling feature window for constructing ML features.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        seed: int = 42,
        lookback: int = 20,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed
        self.lookback = lookback
        self._model = None

    @property
    def name(self) -> str:
        return "Random Forest"

    def fit(self, X: np.ndarray, **kwargs) -> 'RandomForestRegimeDetector':
        return self

    def fit_with_labels(
        self, X: np.ndarray, y: np.ndarray
    ) -> 'RandomForestRegimeDetector':
        """Train the RF on labeled feature matrix.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Binary labels (0 = normal, 1 = crisis).
        """
        from sklearn.ensemble import RandomForestClassifier

        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.seed,
            class_weight='balanced',
        )
        X_feat = self._build_ml_features(X)
        y_trimmed = y[self.lookback - 1:]
        self._model.fit(X_feat, y_trimmed)
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError(
                "Call fit_with_labels() before compute_regime_scores()."
            )
        X_feat = self._build_ml_features(X)
        proba = self._model.predict_proba(X_feat)

        crisis_col = list(self._model.classes_).index(1)
        scores = proba[:, crisis_col]

        pad = np.full(len(X) - len(scores), np.nan)
        return np.concatenate([pad, scores])

    def _build_ml_features(self, X: np.ndarray) -> np.ndarray:
        """Build rolling features (mean, std, min, max) from raw X."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        T, d = X.shape
        features = []

        for t in range(self.lookback - 1, T):
            window = X[t - self.lookback + 1:t + 1]
            row = np.concatenate([
                np.mean(window, axis=0),
                np.std(window, axis=0),
                np.min(window, axis=0),
                np.max(window, axis=0),
            ])
            features.append(row)

        return np.array(features)
