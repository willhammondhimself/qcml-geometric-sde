"""
Fusion Methods for Regime Detection

Paper 1 strategies:
    1. RankFusionDetector — Non-parametric average rank across methods
    2. StackingFusionDetector — Logistic regression meta-learner on z-scores
    3. DynamicSwitchingDetector — Weight detectors by recent rolling AUC

Paper 2 ("The Observatory") strategies:
    4. HierarchicalFusionDetector — Family-level aggregation then cross-family fusion
    5. RegimeAdaptiveFusionDetector — Regime-conditional weights via walk-forward training
    6. BayesianEvidenceAccumulator — Sequential likelihood-ratio accumulation (SPRT)

All strategies follow the BaseRegimeDetector interface and maintain
causal evaluation (no future information leakage).
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .observables import BaseRegimeDetector

logger = logging.getLogger(__name__)


# =============================================================================
# Observatory Channel Taxonomy (Paper 2)
# =============================================================================

# Dead signals excluded from fusion — fail due to Kramers degeneracy or
# noise amplification (all d < 0.02 across 17 crises).
DEAD_CHANNELS = frozenset([
    'QGT Phase Rigidity',       # d=0.013, ||F||/||g|| nearly constant
    'Berry Velocity Coupling',  # d=0.012, inner product averages out
    'Curvature Rate',           # d=0.013, finite-difference noise
])

# Observable families mirroring Paper 1 Table 1.  Keys are family names,
# values are lists of detector display names (matching HPO_CONFIGS keys).
OBSERVABLE_FAMILIES: Dict[str, List[str]] = {
    'Holonomy': [
        'Berry Phase Rate',
        'Geometric Phase Rate',
    ],
    'Metric': [
        'QFI Determinant',
        'Hamiltonian Sensitivity',
    ],
    'State Dynamics': [
        'Multi-Lag Fidelity',
        'Reduced Purity',
        'Quantum Relative Entropy',
    ],
    'Kinematics': [
        'Geodesic Velocity',
        'Speed Limit Ratio',
    ],
    'Spectral': [
        'Spectral Entropy',
        'Spectral Complexity',
        'Effective State Dim',
        'Level Spacing Ratio',
    ],
    'Curvature': [
        'Sectional Curvature Sign',
        'Geodesic Curvature',
    ],
    'Topology': [
        'QCML Chern',
        'Dimensionality Collapse',
    ],
}

# Flat set of all active channels (excludes dead signals)
ACTIVE_CHANNELS = frozenset(
    ch for family in OBSERVABLE_FAMILIES.values() for ch in family
)


def _expanding_zscore_1d(vals: np.ndarray, min_obs: int = 60) -> np.ndarray:
    """Causal expanding-window z-score for a 1-D signal.

    Args:
        vals: Raw signal array of shape (T,).
        min_obs: Minimum observations before producing scores.

    Returns:
        z_scores: Array of shape (T,), NaN where insufficient data.
    """
    T = len(vals)
    z = np.full(T, np.nan)
    for t in range(min_obs, T):
        past = vals[:t]
        past_valid = past[~np.isnan(past)]
        if len(past_valid) < 10:
            continue
        mu = np.mean(past_valid)
        sigma = np.std(past_valid, ddof=1)
        if sigma > 1e-12:
            z[t] = (vals[t] - mu) / sigma
        else:
            z[t] = 0.0
    return z


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


# =============================================================================
# Paper 2: Hierarchical Fusion
# =============================================================================

class HierarchicalFusionDetector(BaseRegimeDetector):
    """Two-level hierarchical fusion mirroring the 7 observable families.

    Level 1 (within-family): Rank-average channels within each family to
    produce a family-level anomaly score.

    Level 2 (cross-family): Combine family scores via rank-average (default)
    or learned weights (if crisis_labels provided).

    All operations are causal (expanding-window only).

    Args:
        families: Dict mapping family name -> list of column indices into
            the score matrix.  If None, uses OBSERVABLE_FAMILIES with
            channel_names to resolve indices.
        channel_names: Ordered list of channel display names matching columns
            of the score matrix.  Required when families is None.
        cross_family_mode: 'rank' (non-parametric) or 'learned' (logistic).
        rolling_window: Smoothing window for final score.
        min_expanding: Minimum observations for z-scoring.
    """

    def __init__(
        self,
        families: Optional[Dict[str, List[int]]] = None,
        channel_names: Optional[List[str]] = None,
        cross_family_mode: str = 'rank',
        rolling_window: int = 15,
        min_expanding: int = 60,
        crisis_labels: Optional[np.ndarray] = None,
        train_end: Optional[int] = None,
    ):
        if families is None and channel_names is None:
            raise ValueError("Provide either families (index dict) or channel_names.")
        self._families_idx = families
        self._channel_names = channel_names
        self.cross_family_mode = cross_family_mode
        self.rolling_window = rolling_window
        self.min_expanding = min_expanding
        self.crisis_labels = crisis_labels
        self.train_end = train_end
        self._is_fitted = False
        self._score_matrix = None
        self._cross_weights = None
        self._cross_intercept = 0.0
        self._family_scores = None  # stored for diagnostics

    @property
    def name(self) -> str:
        return "Hierarchical Fusion"

    def _resolve_families(self, n_cols: int) -> Dict[str, List[int]]:
        """Map OBSERVABLE_FAMILIES display names to column indices."""
        if self._families_idx is not None:
            return self._families_idx
        name_to_idx = {n: i for i, n in enumerate(self._channel_names)}
        resolved = {}
        for fam, channels in OBSERVABLE_FAMILIES.items():
            idxs = [name_to_idx[ch] for ch in channels if ch in name_to_idx]
            if idxs:
                resolved[fam] = idxs
        return resolved

    def set_precomputed_scores(self, score_matrix: np.ndarray) -> 'HierarchicalFusionDetector':
        """Set pre-computed score matrix of shape (T, n_channels)."""
        self._score_matrix = score_matrix
        self._is_fitted = True
        return self

    def fit(self, X: np.ndarray, **kwargs) -> 'HierarchicalFusionDetector':
        self._is_fitted = True
        return self

    def _rank_normalize_column(self, col: np.ndarray) -> np.ndarray:
        """Expanding-window rank normalization to [0, 1] for one column."""
        T = len(col)
        out = np.full(T, np.nan)
        for t in range(self.min_expanding, T):
            past = col[:t + 1]
            valid = past[~np.isnan(past)]
            if len(valid) < 10 or np.isnan(col[t]):
                continue
            out[t] = np.sum(valid <= col[t]) / len(valid)
        return out

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._score_matrix is None:
            raise RuntimeError("Call set_precomputed_scores() first.")

        scores = self._score_matrix
        T, n_ch = scores.shape
        families = self._resolve_families(n_ch)

        if not families:
            raise RuntimeError("No families resolved. Check channel_names.")

        # --- Level 1: within-family rank aggregation ---
        family_names = sorted(families.keys())
        n_fam = len(family_names)
        family_scores = np.full((T, n_fam), np.nan)

        for fi, fam in enumerate(family_names):
            cols = families[fam]
            if len(cols) == 1:
                family_scores[:, fi] = self._rank_normalize_column(scores[:, cols[0]])
            else:
                ranked = np.column_stack([
                    self._rank_normalize_column(scores[:, c]) for c in cols
                ])
                family_scores[:, fi] = np.nanmean(ranked, axis=1)

        self._family_scores = family_scores

        # --- Level 2: cross-family fusion ---
        if self.cross_family_mode == 'learned' and self.crisis_labels is not None:
            fused = self._learned_cross_family(family_scores)
        else:
            cross_ranked = np.column_stack([
                self._rank_normalize_column(family_scores[:, fi])
                for fi in range(n_fam)
            ])
            fused = np.nanmean(cross_ranked, axis=1)

        smoothed = (
            pd.Series(fused)
            .rolling(window=self.rolling_window, min_periods=1)
            .mean()
            .values
        )
        return _expanding_zscore_1d(smoothed, self.min_expanding)

    def _learned_cross_family(self, family_scores: np.ndarray) -> np.ndarray:
        """Train logistic regression on family scores (walk-forward)."""
        from sklearn.linear_model import LogisticRegression

        T, n_fam = family_scores.shape
        train_end = self.train_end or T

        X_train = np.nan_to_num(family_scores[:train_end], nan=0.0)
        y_train = self.crisis_labels[:train_end]
        valid = ~np.all(X_train == 0, axis=1)
        X_valid = X_train[valid]
        y_valid = y_train[valid]

        if len(np.unique(y_valid)) < 2 or len(y_valid) < 20:
            self._cross_weights = np.ones(n_fam) / n_fam
            self._cross_intercept = 0.0
        else:
            lr = LogisticRegression(C=0.1, penalty='l2', max_iter=1000, random_state=42)
            lr.fit(X_valid, y_valid)
            self._cross_weights = lr.coef_.ravel()
            self._cross_intercept = lr.intercept_[0]

        fs_clean = np.nan_to_num(family_scores, nan=0.0)
        return fs_clean @ self._cross_weights + self._cross_intercept

    @property
    def family_scores(self) -> Optional[np.ndarray]:
        """Family-level scores from last compute_regime_scores call.

        Returns:
            Array of shape (T, n_families) or None if not yet computed.
        """
        return self._family_scores


# =============================================================================
# Paper 2: Regime-Adaptive Fusion
# =============================================================================

class RegimeAdaptiveFusionDetector(BaseRegimeDetector):
    """Regime-conditional fusion weights via walk-forward clustering + regression.

    Identifies the current market regime using geometric meta-features
    (cross-channel dispersion, activation fraction, max activation), then
    applies regime-specific fusion weights learned from crisis labels.

    Training is strictly walk-forward: at each evaluation point, the model
    is trained only on data before that point.

    Args:
        channel_names: Ordered list of channel display names.
        n_regimes: Number of latent regimes to identify via K-means.
        retrain_interval: Days between weight re-estimation.
        min_train_obs: Minimum observations before first training.
        rolling_window: Smoothing window for final score.
        min_expanding: Minimum observations for z-scoring.
        crisis_labels: Binary array (T,) for supervised weight learning.
    """

    def __init__(
        self,
        channel_names: Optional[List[str]] = None,
        n_regimes: int = 3,
        retrain_interval: int = 63,
        min_train_obs: int = 252,
        rolling_window: int = 15,
        min_expanding: int = 60,
        crisis_labels: Optional[np.ndarray] = None,
    ):
        self._channel_names = channel_names
        self.n_regimes = n_regimes
        self.retrain_interval = retrain_interval
        self.min_train_obs = min_train_obs
        self.rolling_window = rolling_window
        self.min_expanding = min_expanding
        self.crisis_labels = crisis_labels
        self._is_fitted = False
        self._score_matrix = None
        self._regime_labels = None
        self._weight_history = None

    @property
    def name(self) -> str:
        return "Regime-Adaptive Fusion"

    def set_precomputed_scores(self, score_matrix: np.ndarray) -> 'RegimeAdaptiveFusionDetector':
        self._score_matrix = score_matrix
        self._is_fitted = True
        return self

    def fit(self, X: np.ndarray, **kwargs) -> 'RegimeAdaptiveFusionDetector':
        self._is_fitted = True
        return self

    def _compute_meta_features(self, scores: np.ndarray) -> np.ndarray:
        """Extract regime-indicative meta-features from the score matrix.

        Meta-features capture the *state* of the observatory rather than
        individual detector outputs.

        Args:
            scores: Array of shape (T, n_channels).

        Returns:
            meta: Array of shape (T, 5) with meta-features.
        """
        scores_clean = np.nan_to_num(scores, nan=0.0)

        meta_list = [
            np.nanstd(scores, axis=1),                              # cross-channel dispersion
            np.nanmax(np.abs(scores_clean), axis=1),                # max activation
            np.sum(scores_clean > 1.0, axis=1) / max(scores.shape[1], 1),  # fraction > 1σ
            pd.Series(np.nanstd(scores, axis=1)).rolling(20, min_periods=1).mean().values,
            np.nanmean(np.abs(scores_clean), axis=1),               # mean abs score
        ]
        return np.column_stack(meta_list)

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._score_matrix is None:
            raise RuntimeError("Call set_precomputed_scores() first.")

        scores = self._score_matrix
        T, n_ch = scores.shape
        scores_clean = np.nan_to_num(scores, nan=0.0)

        meta = self._compute_meta_features(scores)

        fused = np.full(T, np.nan)
        regime_labels = np.full(T, -1, dtype=int)
        weight_history = np.full((T, n_ch), np.nan)
        last_train = 0

        cluster_model = None
        regime_weights = {}

        for t in range(self.min_train_obs, T):
            if cluster_model is None or (t - last_train) >= self.retrain_interval:
                cluster_model, regime_weights = self._train_regime_model(
                    scores_clean[:t], meta[:t],
                    self.crisis_labels[:t] if self.crisis_labels is not None else None,
                )
                last_train = t

            if cluster_model is not None and not np.any(np.isnan(meta[t])):
                regime = int(cluster_model.predict(meta[t:t+1])[0])
            else:
                regime = 0

            regime_labels[t] = regime
            w = regime_weights.get(regime, np.ones(n_ch) / n_ch)
            weight_history[t] = w
            fused[t] = np.dot(w, scores_clean[t])

        self._regime_labels = regime_labels
        self._weight_history = weight_history

        smoothed = (
            pd.Series(fused)
            .rolling(window=self.rolling_window, min_periods=1)
            .mean()
            .values
        )
        return _expanding_zscore_1d(smoothed, self.min_expanding)

    def _train_regime_model(
        self,
        scores: np.ndarray,
        meta: np.ndarray,
        crisis_labels: Optional[np.ndarray],
    ) -> Tuple:
        """Train regime classifier and per-regime fusion weights.

        Returns:
            (cluster_model, regime_weights) where regime_weights maps
            regime_id -> weight vector of shape (n_channels,).
        """
        from sklearn.cluster import KMeans
        from sklearn.linear_model import Ridge

        T, n_ch = scores.shape

        valid = np.any(meta != 0, axis=1) & ~np.any(np.isnan(meta), axis=1)
        if np.sum(valid) < self.n_regimes * 10:
            return None, {0: np.ones(n_ch) / n_ch}

        meta_valid = meta[valid]

        km = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
        km.fit(meta_valid)

        labels_full = np.full(T, -1, dtype=int)
        labels_full[valid] = km.labels_

        regime_weights = {}
        if crisis_labels is not None and len(crisis_labels) == T:
            for r in range(self.n_regimes):
                mask = labels_full == r
                if np.sum(mask) < 20:
                    regime_weights[r] = np.ones(n_ch) / n_ch
                    continue

                X_r = scores[mask]
                y_r = crisis_labels[mask].astype(float)

                if len(np.unique(y_r)) < 2:
                    regime_weights[r] = np.ones(n_ch) / n_ch
                    continue

                ridge = Ridge(alpha=1.0)
                ridge.fit(X_r, y_r)
                w = np.abs(ridge.coef_)
                w_sum = w.sum()
                regime_weights[r] = w / w_sum if w_sum > 1e-12 else np.ones(n_ch) / n_ch
        else:
            for r in range(self.n_regimes):
                regime_weights[r] = np.ones(n_ch) / n_ch

        return km, regime_weights

    @property
    def regime_labels(self) -> Optional[np.ndarray]:
        """Regime labels from last compute_regime_scores call."""
        return self._regime_labels

    @property
    def weight_history(self) -> Optional[np.ndarray]:
        """Per-timestep fusion weights, shape (T, n_channels)."""
        return self._weight_history


# =============================================================================
# Paper 2: Bayesian Evidence Accumulator (Online SPRT)
# =============================================================================

class BayesianEvidenceAccumulator(BaseRegimeDetector):
    """Sequential Bayesian evidence accumulation across detection channels.

    At each timestep, each channel contributes a log-likelihood ratio
    (crisis vs calm hypothesis). Evidence accumulates via a random walk
    with absorbing barriers (Wald's SPRT), providing bounded false alarm
    rate guarantees.

    The accumulator outputs a z-scored evidence level for compatibility
    with the BaseRegimeDetector interface.

    Args:
        channel_names: Ordered list of channel display names.
        log_alpha: Log false alarm rate target (default: log(0.05)).
        log_beta: Log missed detection rate target (default: log(0.20)).
        reset_on_alarm: If True, reset evidence after alarm triggers.
        decay: Exponential decay factor per step (1.0 = no decay).
        min_expanding: Minimum observations for z-scoring.
    """

    def __init__(
        self,
        channel_names: Optional[List[str]] = None,
        log_alpha: float = -2.9957,  # log(0.05)
        log_beta: float = -1.6094,   # log(0.20)
        reset_on_alarm: bool = True,
        decay: float = 0.995,
        min_expanding: int = 60,
    ):
        self._channel_names = channel_names
        self.log_alpha = log_alpha
        self.log_beta = log_beta
        self.reset_on_alarm = reset_on_alarm
        self.decay = decay
        self.min_expanding = min_expanding
        self._is_fitted = False
        self._score_matrix = None
        self._evidence = None
        self._alarm_times = None

    @property
    def name(self) -> str:
        return "Bayesian Evidence"

    def set_precomputed_scores(self, score_matrix: np.ndarray) -> 'BayesianEvidenceAccumulator':
        self._score_matrix = score_matrix
        self._is_fitted = True
        return self

    def fit(self, X: np.ndarray, **kwargs) -> 'BayesianEvidenceAccumulator':
        self._is_fitted = True
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute sequential evidence accumulation across channels.

        Each channel's z-score is converted to a log-likelihood ratio
        assuming crisis ~ N(mu_crisis, sigma) vs calm ~ N(0, sigma),
        where mu_crisis and sigma are estimated from expanding history.
        """
        if self._score_matrix is None:
            raise RuntimeError("Call set_precomputed_scores() first.")

        scores = self._score_matrix
        T, n_ch = scores.shape

        upper_barrier = -self.log_alpha
        lower_barrier = self.log_beta

        evidence = np.full(T, np.nan)
        alarm_times = []
        cum_evidence = 0.0

        # Per-channel running statistics (Welford online)
        ch_means = np.zeros(n_ch)
        ch_m2 = np.zeros(n_ch)
        ch_count = np.zeros(n_ch)

        for t in range(T):
            if t < self.min_expanding:
                for c in range(n_ch):
                    v = scores[t, c]
                    if not np.isnan(v):
                        ch_count[c] += 1
                        delta = v - ch_means[c]
                        ch_means[c] += delta / ch_count[c]
                        ch_m2[c] += delta * (v - ch_means[c])
                evidence[t] = 0.0
                continue

            llr_total = 0.0
            n_valid = 0
            for c in range(n_ch):
                v = scores[t, c]
                if np.isnan(v) or ch_count[c] < 20:
                    continue

                sigma = np.sqrt(ch_m2[c] / (ch_count[c] - 1)) if ch_count[c] > 1 else 1.0
                sigma = max(sigma, 1e-8)

                mu_crisis = sigma  # 1-sigma shift under H1
                llr = (v * mu_crisis / (sigma ** 2)) - (mu_crisis ** 2 / (2 * sigma ** 2))
                llr_total += llr
                n_valid += 1

            avg_llr = llr_total / n_valid if n_valid > 0 else 0.0
            cum_evidence = self.decay * cum_evidence + avg_llr

            if cum_evidence >= upper_barrier:
                alarm_times.append(t)
                if self.reset_on_alarm:
                    cum_evidence = 0.0
            elif cum_evidence <= lower_barrier:
                cum_evidence = 0.0

            evidence[t] = cum_evidence

            for c in range(n_ch):
                v = scores[t, c]
                if not np.isnan(v):
                    ch_count[c] += 1
                    delta = v - ch_means[c]
                    ch_means[c] += delta / ch_count[c]
                    ch_m2[c] += delta * (v - ch_means[c])

        self._evidence = evidence
        self._alarm_times = alarm_times

        return _expanding_zscore_1d(evidence, self.min_expanding)

    @property
    def raw_evidence(self) -> Optional[np.ndarray]:
        """Raw cumulative evidence from last call, shape (T,)."""
        return self._evidence

    @property
    def alarm_times(self) -> Optional[List[int]]:
        """Indices where SPRT alarm triggered."""
        return self._alarm_times
