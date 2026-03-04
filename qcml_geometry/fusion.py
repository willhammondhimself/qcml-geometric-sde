"""
Fusion Methods for Regime Detection

Combine multiple regime detectors into ensemble signals. Three strategies:

1. RankFusionDetector — Non-parametric average rank across methods
2. StackingFusionDetector — Logistic regression meta-learner on z-scores
3. DynamicSwitchingDetector — Weight detectors by recent rolling AUC

All strategies follow the BaseRegimeDetector interface and maintain
causal evaluation (no future information leakage).
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .observables import BaseRegimeDetector

logger = logging.getLogger(__name__)


class RankFusionDetector(BaseRegimeDetector):
    """Non-parametric rank fusion across multiple detectors.

    At each timestep, ranks each detector's score across time, normalizes
    to [0, 1], and averages. No weights to overfit — fully non-parametric.

    Args:
        detectors: List of fitted BaseRegimeDetector instances.
        rolling_window: Smoothing window for final score.
        min_expanding: Minimum observations for z-scoring.
    """

    def __init__(
        self,
        detectors: Optional[List[BaseRegimeDetector]] = None,
        rolling_window: int = 15,
        min_expanding: int = 60,
    ):
        self.detectors = detectors or []
        self.rolling_window = rolling_window
        self.min_expanding = min_expanding
        self._is_fitted = False
        self._score_matrix = None

    @property
    def name(self) -> str:
        return "Rank Fusion"

    def fit(self, X: np.ndarray, **kwargs) -> 'RankFusionDetector':
        for det in self.detectors:
            det.fit(X, **kwargs)
        self._is_fitted = True
        return self

    def set_precomputed_scores(self, score_matrix: np.ndarray) -> 'RankFusionDetector':
        """Set pre-computed score matrix instead of running detectors.

        Args:
            score_matrix: Array of shape (T, n_detectors) with z-scores.
        """
        self._score_matrix = score_matrix
        self._is_fitted = True
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        if self._score_matrix is not None:
            scores = self._score_matrix
        else:
            scores = np.column_stack([
                det.compute_regime_scores(X) for det in self.detectors
            ])

        T, n_methods = scores.shape

        # Expanding-window rank normalization (causal)
        rank_avg = np.full(T, np.nan)
        for t in range(self.min_expanding, T):
            ranks = np.zeros(n_methods)
            for m in range(n_methods):
                col = scores[:t + 1, m]
                valid = col[~np.isnan(col)]
                if len(valid) < 10:
                    ranks[m] = 0.5
                    continue
                # Rank current value within expanding history
                r = np.sum(valid <= scores[t, m]) / len(valid)
                ranks[m] = r if not np.isnan(scores[t, m]) else np.nan
            valid_ranks = ranks[~np.isnan(ranks)]
            if len(valid_ranks) > 0:
                rank_avg[t] = np.mean(valid_ranks)

        # Z-score the rank average
        rolling_vals = (
            pd.Series(rank_avg)
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


class StackingFusionDetector(BaseRegimeDetector):
    """Stacking meta-learner: logistic regression on detector z-scores.

    Trains a logistic regression on z-scores from individual detectors to
    predict crisis labels. Uses leave-one-crisis-out cross-validation to
    prevent overfitting.

    Args:
        detectors: List of fitted BaseRegimeDetector instances.
        crisis_labels: Binary array (T,) — 1 during crisis, 0 otherwise.
        train_end: Last index of training data for temporal OOS.
        min_expanding: Minimum observations for valid scores.
    """

    def __init__(
        self,
        detectors: Optional[List[BaseRegimeDetector]] = None,
        crisis_labels: Optional[np.ndarray] = None,
        train_end: Optional[int] = None,
        min_expanding: int = 60,
    ):
        self.detectors = detectors or []
        self.crisis_labels = crisis_labels
        self.train_end = train_end
        self.min_expanding = min_expanding
        self._is_fitted = False
        self._weights = None
        self._intercept = 0.0
        self._score_matrix = None

    @property
    def name(self) -> str:
        return "Stacking Fusion"

    def set_precomputed_scores(self, score_matrix: np.ndarray) -> 'StackingFusionDetector':
        """Set pre-computed score matrix."""
        self._score_matrix = score_matrix
        return self

    def fit(self, X: np.ndarray, **kwargs) -> 'StackingFusionDetector':
        from sklearn.linear_model import LogisticRegression

        if self._score_matrix is None:
            for det in self.detectors:
                det.fit(X, **kwargs)
            scores = np.column_stack([
                det.compute_regime_scores(X) for det in self.detectors
            ])
        else:
            scores = self._score_matrix

        T, n_methods = scores.shape
        train_end = self.train_end or T

        if self.crisis_labels is None:
            # Default: equal weights if no labels
            self._weights = np.ones(n_methods) / n_methods
            self._intercept = 0.0
            self._is_fitted = True
            return self

        labels = self.crisis_labels[:train_end]
        X_train = scores[:train_end]

        # Find valid rows (no NaN in any detector)
        valid = ~np.any(np.isnan(X_train), axis=1)
        X_valid = X_train[valid]
        y_valid = labels[valid]

        if len(np.unique(y_valid)) < 2 or len(y_valid) < 20:
            self._weights = np.ones(n_methods) / n_methods
            self._intercept = 0.0
            self._is_fitted = True
            return self

        # Replace NaN with 0 for any remaining
        X_valid = np.nan_to_num(X_valid, nan=0.0)

        lr = LogisticRegression(C=0.1, penalty='l2', max_iter=1000, random_state=42)
        lr.fit(X_valid, y_valid)

        self._weights = lr.coef_.ravel()
        self._intercept = lr.intercept_[0]
        self._is_fitted = True
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        if self._score_matrix is not None:
            scores = self._score_matrix
        else:
            scores = np.column_stack([
                det.compute_regime_scores(X) for det in self.detectors
            ])

        T = scores.shape[0]
        scores_clean = np.nan_to_num(scores, nan=0.0)

        # Linear combination (logistic regression logit)
        logits = scores_clean @ self._weights + self._intercept

        # Z-score the logits causally
        z_scores = np.full(T, np.nan)
        for t in range(self.min_expanding, T):
            past = logits[:t]
            mu = np.mean(past)
            sigma = np.std(past, ddof=1)
            if sigma > 1e-12:
                z_scores[t] = (logits[t] - mu) / sigma
            else:
                z_scores[t] = 0.0

        return z_scores


class DynamicSwitchingDetector(BaseRegimeDetector):
    """Dynamic switching: weight detectors by recent rolling performance.

    At each timestep, weights each detector by its rolling 60-day
    correlation with a target signal (e.g., realized volatility spike).
    Adapts to changing market conditions.

    Args:
        detectors: List of fitted BaseRegimeDetector instances.
        eval_window: Rolling window for evaluating detector performance.
        rolling_window: Smoothing window for final score.
        min_expanding: Minimum observations for z-scoring.
    """

    def __init__(
        self,
        detectors: Optional[List[BaseRegimeDetector]] = None,
        eval_window: int = 60,
        rolling_window: int = 15,
        min_expanding: int = 60,
    ):
        self.detectors = detectors or []
        self.eval_window = eval_window
        self.rolling_window = rolling_window
        self.min_expanding = min_expanding
        self._is_fitted = False
        self._score_matrix = None

    @property
    def name(self) -> str:
        return "Dynamic Switching"

    def set_precomputed_scores(self, score_matrix: np.ndarray) -> 'DynamicSwitchingDetector':
        """Set pre-computed score matrix."""
        self._score_matrix = score_matrix
        return self

    def fit(self, X: np.ndarray, **kwargs) -> 'DynamicSwitchingDetector':
        for det in self.detectors:
            det.fit(X, **kwargs)
        self._is_fitted = True
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted and self._score_matrix is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        if self._score_matrix is not None:
            scores = self._score_matrix
        else:
            scores = np.column_stack([
                det.compute_regime_scores(X) for det in self.detectors
            ])

        T, n_methods = scores.shape

        # Target signal: rolling volatility of first feature column
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        target = pd.Series(X[:, 0]).rolling(20).std().values

        # Dynamic weighting based on rolling correlation with target
        fused = np.full(T, np.nan)
        ew = self.eval_window

        for t in range(max(ew, self.min_expanding), T):
            weights = np.zeros(n_methods)
            for m in range(n_methods):
                s = scores[t - ew:t, m]
                tgt = target[t - ew:t]
                valid = ~(np.isnan(s) | np.isnan(tgt))
                if np.sum(valid) < 10:
                    weights[m] = 1.0 / n_methods
                    continue
                corr = np.corrcoef(s[valid], tgt[valid])[0, 1]
                # Softmax-ish: use positive correlation as weight
                weights[m] = max(corr, 0.0)

            w_sum = np.sum(weights)
            if w_sum < 1e-12:
                weights = np.ones(n_methods) / n_methods
            else:
                weights = weights / w_sum

            # Weighted combination of current scores
            current = scores[t]
            current_clean = np.where(np.isnan(current), 0.0, current)
            fused[t] = np.dot(weights, current_clean)

        # Z-score the fused signal
        rolling_vals = (
            pd.Series(fused)
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
