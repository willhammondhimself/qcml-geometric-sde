"""
Core Geometric Observables for Regime Detection

Three novel unsupervised observables derived from the metric tensor geometry:

1. BerryPhaseRateDetector - Rate of change of Berry curvature
2. QFIDeterminantDetector - Metric pseudo-determinant
3. MultiLagFidelityDetector - Multi-lag fidelity

Also provides the BaseRegimeDetector ABC and ExpandingWindowMixin.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class BaseRegimeDetector(ABC):
    """Common interface for all regime detection methods."""

    @staticmethod
    def build_enriched_features(X: np.ndarray, lookback: int = 20) -> np.ndarray:
        """Build rolling features (mean, std, min, max) from raw X.

        Args:
            X: Feature matrix (T, d).
            lookback: Rolling window size.

        Returns:
            Enriched feature matrix (T - lookback + 1, 4*d).
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        T, d = X.shape
        features = []

        for t in range(lookback - 1, T):
            window = X[t - lookback + 1 : t + 1]
            row = np.concatenate(
                [
                    np.mean(window, axis=0),
                    np.std(window, axis=0),
                    np.min(window, axis=0),
                    np.max(window, axis=0),
                ]
            )
            features.append(row)

        return np.array(features)

    @abstractmethod
    def fit(self, X: np.ndarray, **kwargs) -> "BaseRegimeDetector":
        """Fit the detector to feature matrix X (T, n_features)."""
        ...

    @abstractmethod
    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        """Produce a 1-D regime score time series of length T."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for display in comparison tables."""
        ...


def _apply_normalization(
    X_pca: np.ndarray,
    mode: str = "sphere",
    train_norms: Optional[np.ndarray] = None,
    train_std: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply post-PCA normalization to a matrix.

    Args:
        X_pca: PCA-transformed matrix (T, n_components) or (n_components,).
        mode: Normalization mode:
            - 'sphere': Project onto unit sphere (original behavior).
            - 'none': No normalization after PCA.
            - 'soft': Divide by (||x|| + median_train_norm).
            - 'clip': Clip at ±5σ per component, no normalization.
        train_norms: Precomputed training set norms for 'soft' mode.
        train_std: Precomputed per-component std for 'clip' mode.

    Returns:
        Normalized array, same shape as input.
    """
    is_1d = X_pca.ndim == 1
    if is_1d:
        X_pca = X_pca.reshape(1, -1)

    if mode == "sphere":
        norms = np.linalg.norm(X_pca, axis=1, keepdims=True)
        result = X_pca / (norms + 1e-8)
    elif mode == "none":
        result = X_pca
    elif mode == "soft":
        median_norm = np.median(train_norms) if train_norms is not None else 1.0
        norms = np.linalg.norm(X_pca, axis=1, keepdims=True)
        result = X_pca / (norms + median_norm)
    elif mode == "clip":
        if train_std is not None:
            clip_bound = 5.0 * train_std
            result = np.clip(X_pca, -clip_bound, clip_bound)
        else:
            result = X_pca
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    return result.ravel() if is_1d else result


def _transform_point(
    x: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
    normalization: str = "sphere",
    train_norms: Optional[np.ndarray] = None,
    train_std: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Transform a single data point through stored scaler and PCA.

    Args:
        x: Raw feature vector (n_features,).
        scaler: Fitted StandardScaler.
        pca: Fitted PCA.
        normalization: Post-PCA normalization mode.
        train_norms: Training norms for 'soft' mode.
        train_std: Training per-component std for 'clip' mode.

    Returns:
        PCA-transformed vector with specified normalization.
    """
    x_scaled = scaler.transform(x.reshape(1, -1))
    x_pca = pca.transform(x_scaled).ravel()
    return _apply_normalization(x_pca, normalization, train_norms, train_std)


def _transform_array(
    X: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
    normalization: str = "sphere",
    train_norms: Optional[np.ndarray] = None,
    train_std: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Transform an array through stored scaler and PCA.

    Args:
        X: Raw feature matrix (T, n_features).
        scaler: Fitted StandardScaler.
        pca: Fitted PCA.
        normalization: Post-PCA normalization mode.
        train_norms: Training norms for 'soft' mode.
        train_std: Training per-component std for 'clip' mode.

    Returns:
        PCA-transformed matrix (T, n_components) with specified normalization.
    """
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    return _apply_normalization(X_pca, normalization, train_norms, train_std)


class ExpandingWindowMixin:
    """Mixin that periodically refits scaler/PCA/operators on expanding windows.

    When ``expanding_refit_interval`` is set, the mixin builds a sequence of
    snapshots during ``fit()``, each fitted on an expanding prefix of the data.
    During ``compute_regime_scores()``, the detector uses the most recent
    snapshot that precedes each time point and transforms data on-the-fly
    using only past-fitted preprocessing.
    """

    def _fit_expanding(self, X: np.ndarray) -> "ExpandingWindowMixin":
        """Build expanding-window snapshots for periodic refitting.

        Each snapshot stores its own scaler, PCA, and geometry fitted only on
        data up to refit_idx. No future data is transformed during fit.
        """
        from .core import QCMLGeometry

        T = X.shape[0]
        interval = self.expanding_refit_interval
        min_fit = max(self.min_expanding, 30)
        n_components = min(self.n_pca_components, X.shape[1])
        norm_mode = getattr(self, "normalization", "sphere")

        refit_points = list(range(min_fit, T, interval))
        if not refit_points or refit_points[-1] != T:
            refit_points.append(T)

        self._snapshots = []
        for refit_idx in refit_points:
            np.random.seed(self.seed)

            scaler = StandardScaler()
            scaler.fit(X[:refit_idx])

            pca = PCA(n_components=n_components)
            X_scaled_prefix = scaler.transform(X[:refit_idx])
            pca.fit(X_scaled_prefix)

            X_pca_raw = pca.transform(X_scaled_prefix)
            train_norms = np.linalg.norm(X_pca_raw, axis=1)
            train_std = np.std(X_pca_raw, axis=0)
            X_pca_prefix = _apply_normalization(
                X_pca_raw,
                norm_mode,
                train_norms,
                train_std,
            )

            geometry = QCMLGeometry(n_features=X_pca_prefix.shape[1], hilbert_dim=self.hilbert_dim)
            custom_ops = getattr(self, "custom_operators", None)
            if custom_ops is not None:
                geometry.set_operators(custom_ops)
            else:
                scale_exp = getattr(self, "scale_exponent", None)
                geometry.fit_operators(
                    X_pca_prefix,
                    method=self.operator_method,
                    scale_exponent=scale_exp,
                )

            self._snapshots.append(
                {
                    "refit_idx": refit_idx,
                    "geometry": geometry,
                    "scaler": scaler,
                    "pca": pca,
                    "train_norms": train_norms,
                    "train_std": train_std,
                }
            )

        last = self._snapshots[-1]
        self._geometry = last["geometry"]
        self._scaler = last["scaler"]
        self._pca = last["pca"]
        self._train_norms = last["train_norms"]
        self._train_std = last["train_std"]

        return self

    def _get_snapshot_at(self, t: int):
        """Return the most recent snapshot that precedes time t."""
        best_snap = self._snapshots[0]
        for snap in self._snapshots:
            if snap["refit_idx"] <= t:
                best_snap = snap
            else:
                break
        return best_snap

    def _transform_point_at(self, x_raw: np.ndarray, t: int) -> tuple:
        """Transform a single raw point using the appropriate snapshot.

        Returns (geometry, x_transformed) for time t.
        """
        snap = self._get_snapshot_at(t)
        norm_mode = getattr(self, "normalization", "sphere")
        x_t = _transform_point(
            x_raw,
            snap["scaler"],
            snap["pca"],
            normalization=norm_mode,
            train_norms=snap.get("train_norms"),
            train_std=snap.get("train_std"),
        )
        return snap["geometry"], x_t


class QFIDeterminantDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via metric determinant det(g_ab).

    The determinant of the metric tensor is the volume element of
    the data manifold. Collapsing volume indicates approaching a phase
    boundary; diverging volume indicates rapid expansion of accessible
    state space.

    Score = abs(z-score of log(|det(g)|)) smoothed over a rolling window.
    """

    def __init__(
        self,
        hilbert_dim: int = 8,
        n_pca_components: int = 15,
        operator_method: str = "random",
        scale_exponent: Optional[float] = None,
        rolling_window: int = 20,
        min_expanding: int = 60,
        seed: int = 42,
        causal_fit_length: Optional[int] = None,
        expanding_refit_interval: Optional[int] = None,
        normalization: str = "sphere",
        qfi_mode: str = "logdet",
        adaptive_epsilon: bool = False,
        custom_operators: Optional[List[np.ndarray]] = None,
        adaptive_z_window: Optional[int] = None,
    ):
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
        self.qfi_mode = qfi_mode
        self.adaptive_epsilon = adaptive_epsilon
        self.custom_operators = custom_operators
        self.adaptive_z_window = adaptive_z_window
        self._geometry = None
        self._scaler = None
        self._pca = None
        self._snapshots = None
        self._train_norms = None
        self._train_std = None
        self._epsilon = 1e-5

    @property
    def name(self) -> str:
        return "QFI Determinant"

    def fit(self, X: np.ndarray, **kwargs) -> "QFIDeterminantDetector":
        if self.expanding_refit_interval is not None:
            return self._fit_expanding(X)

        from .core import QCMLGeometry

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
            X_pca_raw,
            self.normalization,
            self._train_norms,
            self._train_std,
        )

        if self.adaptive_epsilon:
            self._epsilon = 1e-3 * np.median(np.abs(X_pca_fit))
        else:
            self._epsilon = 1e-5

        self._geometry = QCMLGeometry(n_features=X_pca_fit.shape[1], hilbert_dim=self.hilbert_dim)
        if self.custom_operators is not None:
            self._geometry.set_operators(self.custom_operators)
        else:
            self._geometry.fit_operators(
                X_pca_fit,
                method=self.operator_method,
                scale_exponent=self.scale_exponent,
            )
        return self

    def _extract_metric_feature(self, g_ij: np.ndarray) -> float:
        """Extract a scalar from the metric tensor based on qfi_mode."""
        eigenvalue_tolerance = 1e-10
        eigenvalues = np.linalg.eigvalsh(g_ij)
        nonzero_eigs = eigenvalues[eigenvalues > eigenvalue_tolerance]

        if self.qfi_mode == "logdet":
            if len(nonzero_eigs) > 0:
                return np.sum(np.log(nonzero_eigs))
            return np.log(eigenvalue_tolerance) * len(eigenvalues)

        elif self.qfi_mode == "trace":
            return np.sum(nonzero_eigs) if len(nonzero_eigs) > 0 else 0.0

        elif self.qfi_mode == "max_eig":
            return nonzero_eigs[-1] if len(nonzero_eigs) > 0 else 0.0

        elif self.qfi_mode == "condition":
            if len(nonzero_eigs) >= 2:
                return np.log(nonzero_eigs[-1] / nonzero_eigs[0])
            return 0.0

        elif self.qfi_mode == "entropy":
            if len(nonzero_eigs) > 0:
                p = nonzero_eigs / np.sum(nonzero_eigs)
                return -np.sum(p * np.log(p + 1e-15))
            return 0.0

        else:
            raise ValueError(f"Unknown qfi_mode: {self.qfi_mode}")

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        Xt = _transform_array(
            X,
            self._scaler,
            self._pca,
            normalization=self.normalization,
            train_norms=self._train_norms,
            train_std=self._train_std,
        )
        T = len(Xt)
        eps = self._epsilon

        import pandas as pd

        metric_values = np.empty(T)

        for t in range(T):
            if self._snapshots is not None:
                geo, xt = self._transform_point_at(X[t], t)
                g_ij = geo.quantum_metric(xt, epsilon=eps)
            else:
                g_ij = self._geometry.quantum_metric(Xt[t], epsilon=eps)

            metric_values[t] = self._extract_metric_feature(g_ij)

        rolling_vals = (
            pd.Series(metric_values)
            .rolling(window=self.rolling_window, min_periods=1)
            .mean()
            .values
        )

        z_scores = np.full(T, np.nan)
        for t in range(self.min_expanding, T):
            if self.adaptive_z_window is not None:
                lookback = min(t, self.adaptive_z_window)
                window_vals = rolling_vals[t - lookback : t]
                mu = np.median(window_vals)
                sigma = 1.4826 * np.median(np.abs(window_vals - mu))
            else:
                mu = np.mean(rolling_vals[:t])
                sigma = np.std(rolling_vals[:t], ddof=1)
            if sigma > 1e-12:
                z_scores[t] = abs((rolling_vals[t] - mu) / sigma)
            else:
                z_scores[t] = 0.0

        return z_scores


class BerryPhaseRateDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via rate of change of Berry curvature.

    Measures geometric transition speed: rapid changes in Berry curvature
    indicate the system is crossing a phase boundary.

    Score = abs(diff(Berry_curvature)) smoothed and z-scored.
    """

    def __init__(
        self,
        hilbert_dim: int = 8,
        n_pca_components: int = 15,
        operator_method: str = "random",
        scale_exponent: Optional[float] = None,
        rolling_window: int = 20,
        min_expanding: int = 60,
        seed: int = 42,
        causal_fit_length: Optional[int] = None,
        expanding_refit_interval: Optional[int] = None,
        normalization: str = "sphere",
        berry_aggregation: str = "f01",
        adaptive_epsilon: bool = False,
        custom_operators: Optional[List[np.ndarray]] = None,
        adaptive_z_window: Optional[int] = None,
    ):
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
        self.berry_aggregation = berry_aggregation
        self.adaptive_epsilon = adaptive_epsilon
        self.custom_operators = custom_operators
        self.adaptive_z_window = adaptive_z_window
        self._geometry = None
        self._scaler = None
        self._pca = None
        self._snapshots = None
        self._train_norms = None
        self._train_std = None
        self._epsilon = 1e-5

    @property
    def name(self) -> str:
        return "Berry Phase Rate"

    def fit(self, X: np.ndarray, **kwargs) -> "BerryPhaseRateDetector":
        if self.expanding_refit_interval is not None:
            return self._fit_expanding(X)

        from .core import QCMLGeometry

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
            X_pca_raw,
            self.normalization,
            self._train_norms,
            self._train_std,
        )

        if self.adaptive_epsilon:
            self._epsilon = 1e-3 * np.median(np.abs(X_pca_fit))
        else:
            self._epsilon = 1e-5

        self._geometry = QCMLGeometry(n_features=X_pca_fit.shape[1], hilbert_dim=self.hilbert_dim)
        if self.custom_operators is not None:
            self._geometry.set_operators(self.custom_operators)
        else:
            self._geometry.fit_operators(
                X_pca_fit,
                method=self.operator_method,
                scale_exponent=self.scale_exponent,
            )
        return self

    def _compute_berry_scalar(self, geo, xt, eps) -> float:
        """Compute a scalar Berry curvature value using the configured aggregation."""
        if self.berry_aggregation == "f01":
            return geo.berry_curvature_2d(xt, indices=(0, 1), epsilon=eps)

        F = geo.berry_curvature(xt, epsilon=eps)
        if self.berry_aggregation == "frobenius":
            return np.sqrt(np.sum(F**2))
        elif self.berry_aggregation == "max":
            return np.max(np.abs(F))
        else:
            raise ValueError(f"Unknown berry_aggregation: {self.berry_aggregation}")

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        Xt = _transform_array(
            X,
            self._scaler,
            self._pca,
            normalization=self.normalization,
            train_norms=self._train_norms,
            train_std=self._train_std,
        )
        T = len(Xt)
        eps = self._epsilon

        import pandas as pd

        berry = np.empty(T)
        for t in range(T):
            if self._snapshots is not None:
                geo, xt = self._transform_point_at(X[t], t)
            else:
                geo, xt = self._geometry, Xt[t]
            berry[t] = self._compute_berry_scalar(geo, xt, eps)

        berry_rate = np.abs(np.diff(berry))

        rolling_rate = (
            pd.Series(berry_rate).rolling(window=self.rolling_window, min_periods=1).mean().values
        )

        n = len(rolling_rate)
        z_scores = np.full(n, np.nan)
        for t in range(self.min_expanding, n):
            if self.adaptive_z_window is not None:
                lookback = min(t, self.adaptive_z_window)
                window_vals = rolling_rate[t - lookback : t]
                mu = np.median(window_vals)
                sigma = 1.4826 * np.median(np.abs(window_vals - mu))
            else:
                mu = np.mean(rolling_rate[:t])
                sigma = np.std(rolling_rate[:t], ddof=1)
            if sigma > 1e-12:
                z_scores[t] = (rolling_rate[t] - mu) / sigma
            else:
                z_scores[t] = 0.0

        return np.concatenate([[np.nan], z_scores])


class MultiScaleBerryDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Berry curvature rate computed at multiple rolling windows, then fused.

    Computes Berry curvature once (shared geometry), applies abs(diff()),
    smooths at each window scale, z-scores independently, then aggregates.

    This captures both short-term curvature spikes (small windows) and
    sustained geometric deformations (large windows).

    Args:
        windows: List of rolling window sizes for multi-scale smoothing.
        aggregation: Fusion method: 'rms' (root mean square of z-scores),
            'max' (max z-score across scales), 'mean', or 'rank_avg'
            (mean of per-scale ranks normalized to [0,1]).
        **berry_kwargs: Passed to internal BerryPhaseRateDetector
            (hilbert_dim, n_pca_components, operator_method, etc.).

    Score = aggregated z-score across multiple smoothing scales.
    """

    def __init__(
        self,
        windows: Optional[List[int]] = None,
        aggregation: str = "rms",
        hilbert_dim: int = 8,
        n_pca_components: int = 15,
        operator_method: str = "random",
        scale_exponent: Optional[float] = None,
        min_expanding: int = 60,
        seed: int = 42,
        causal_fit_length: Optional[int] = None,
        expanding_refit_interval: Optional[int] = None,
        normalization: str = "sphere",
        berry_aggregation: str = "f01",
        adaptive_epsilon: bool = False,
        custom_operators: Optional[List[np.ndarray]] = None,
        adaptive_z_window: Optional[int] = None,
    ):
        self.windows = windows or [5, 10, 20, 40]
        self.aggregation = aggregation
        self.hilbert_dim = hilbert_dim
        self.n_pca_components = n_pca_components
        self.operator_method = operator_method
        self.scale_exponent = scale_exponent
        self.rolling_window = max(self.windows)  # for ExpandingWindowMixin compat
        self.min_expanding = min_expanding
        self.seed = seed
        self.causal_fit_length = causal_fit_length
        self.expanding_refit_interval = expanding_refit_interval
        self.normalization = normalization
        self.berry_aggregation = berry_aggregation
        self.adaptive_epsilon = adaptive_epsilon
        self.custom_operators = custom_operators
        self.adaptive_z_window = adaptive_z_window
        self._geometry = None
        self._scaler = None
        self._pca = None
        self._snapshots = None
        self._train_norms = None
        self._train_std = None
        self._epsilon = 1e-5

    @property
    def name(self) -> str:
        return f"Multi-Scale Berry ({self.aggregation})"

    def fit(self, X: np.ndarray, **kwargs) -> "MultiScaleBerryDetector":
        if self.expanding_refit_interval is not None:
            return self._fit_expanding(X)

        from .core import QCMLGeometry

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
            X_pca_raw,
            self.normalization,
            self._train_norms,
            self._train_std,
        )

        if self.adaptive_epsilon:
            self._epsilon = 1e-3 * np.median(np.abs(X_pca_fit))
        else:
            self._epsilon = 1e-5

        self._geometry = QCMLGeometry(n_features=X_pca_fit.shape[1], hilbert_dim=self.hilbert_dim)
        if self.custom_operators is not None:
            self._geometry.set_operators(self.custom_operators)
        else:
            self._geometry.fit_operators(
                X_pca_fit,
                method=self.operator_method,
                scale_exponent=self.scale_exponent,
            )
        return self

    def _compute_berry_scalar(self, geo, xt, eps) -> float:
        """Compute a scalar Berry curvature value using the configured aggregation."""
        if self.berry_aggregation == "f01":
            return geo.berry_curvature_2d(xt, indices=(0, 1), epsilon=eps)

        F = geo.berry_curvature(xt, epsilon=eps)
        if self.berry_aggregation == "frobenius":
            return np.sqrt(np.sum(F**2))
        elif self.berry_aggregation == "max":
            return np.max(np.abs(F))
        else:
            raise ValueError(f"Unknown berry_aggregation: {self.berry_aggregation}")

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        Xt = _transform_array(
            X,
            self._scaler,
            self._pca,
            normalization=self.normalization,
            train_norms=self._train_norms,
            train_std=self._train_std,
        )
        T = len(Xt)
        eps = self._epsilon

        import pandas as pd

        # Compute raw Berry curvature ONCE (shared geometry)
        berry = np.empty(T)
        for t in range(T):
            if self._snapshots is not None:
                geo, xt = self._transform_point_at(X[t], t)
            else:
                geo, xt = self._geometry, Xt[t]
            berry[t] = self._compute_berry_scalar(geo, xt, eps)

        berry_rate = np.abs(np.diff(berry))

        # Smooth at each window scale, z-score independently
        z_channels = []
        for w in self.windows:
            rolling_rate = pd.Series(berry_rate).rolling(window=w, min_periods=1).mean().values
            n = len(rolling_rate)
            z = np.full(n, np.nan)
            for t in range(self.min_expanding, n):
                if self.adaptive_z_window is not None:
                    lookback = min(t, self.adaptive_z_window)
                    window_vals = rolling_rate[t - lookback : t]
                    mu = np.median(window_vals)
                    sigma = 1.4826 * np.median(np.abs(window_vals - mu))
                else:
                    mu = np.mean(rolling_rate[:t])
                    sigma = np.std(rolling_rate[:t], ddof=1)
                if sigma > 1e-12:
                    z[t] = (rolling_rate[t] - mu) / sigma
                else:
                    z[t] = 0.0
            z_channels.append(z)

        z_matrix = np.column_stack(z_channels)  # (T-1, n_windows)

        # Aggregate across scales
        if self.aggregation == "rms":
            agg = np.sqrt(np.nanmean(z_matrix**2, axis=1))
        elif self.aggregation == "max":
            agg = np.nanmax(z_matrix, axis=1)
        elif self.aggregation == "mean":
            agg = np.nanmean(z_matrix, axis=1)
        elif self.aggregation == "rank_avg":
            from scipy.stats import rankdata

            ranks = np.column_stack(
                [
                    rankdata(z_matrix[:, i], nan_policy="omit") / np.sum(~np.isnan(z_matrix[:, i]))
                    for i in range(z_matrix.shape[1])
                ]
            )
            agg = np.nanmean(ranks, axis=1)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        agg[: self.min_expanding] = np.nan

        # Prepend NaN to align with original T (diff loses 1 element)
        return np.concatenate([[np.nan], agg])


class MultiLagFidelityDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via multi-lag fidelity.

    Fidelity at lags [1, 3, 5, 10] provides multi-scale sensitivity:
    short lags detect fast crises, long lags detect gradual transitions.
    Score = weighted average infidelity (1-F), z-scored.
    """

    def __init__(
        self,
        hilbert_dim: int = 8,
        n_pca_components: int = 15,
        operator_method: str = "random",
        scale_exponent: Optional[float] = None,
        lags: Optional[List[int]] = None,
        lag_weights: Optional[List[float]] = None,
        rolling_window: int = 20,
        min_expanding: int = 60,
        seed: int = 42,
        causal_fit_length: Optional[int] = None,
        expanding_refit_interval: Optional[int] = None,
        normalization: str = "sphere",
        adaptive_epsilon: bool = False,
        custom_operators: Optional[List[np.ndarray]] = None,
        adaptive_z_window: Optional[int] = None,
    ):
        self.hilbert_dim = hilbert_dim
        self.n_pca_components = n_pca_components
        self.operator_method = operator_method
        self.scale_exponent = scale_exponent
        self.lags = lags or [1, 3, 5, 10]
        self.lag_weights = lag_weights or [0.4, 0.3, 0.2, 0.1]
        self.rolling_window = rolling_window
        self.min_expanding = min_expanding
        self.seed = seed
        self.causal_fit_length = causal_fit_length
        self.expanding_refit_interval = expanding_refit_interval
        self.normalization = normalization
        self.adaptive_epsilon = adaptive_epsilon
        self.custom_operators = custom_operators
        self.adaptive_z_window = adaptive_z_window
        self._geometry = None
        self._scaler = None
        self._pca = None
        self._snapshots = None
        self._train_norms = None
        self._train_std = None
        self._epsilon = 1e-5

    @property
    def name(self) -> str:
        return "Multi-Lag Fidelity"

    def fit(self, X: np.ndarray, **kwargs) -> "MultiLagFidelityDetector":
        if self.expanding_refit_interval is not None:
            return self._fit_expanding(X)

        from .core import QCMLGeometry

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
            X_pca_raw,
            self.normalization,
            self._train_norms,
            self._train_std,
        )

        if self.adaptive_epsilon:
            self._epsilon = 1e-3 * np.median(np.abs(X_pca_fit))
        else:
            self._epsilon = 1e-5

        self._geometry = QCMLGeometry(n_features=X_pca_fit.shape[1], hilbert_dim=self.hilbert_dim)
        if self.custom_operators is not None:
            self._geometry.set_operators(self.custom_operators)
        else:
            self._geometry.fit_operators(
                X_pca_fit,
                method=self.operator_method,
                scale_exponent=self.scale_exponent,
            )
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        Xt = _transform_array(
            X,
            self._scaler,
            self._pca,
            normalization=self.normalization,
            train_norms=self._train_norms,
            train_std=self._train_std,
        )
        T = len(Xt)
        max_lag = max(self.lags)

        import pandas as pd

        states = []
        for t in range(T):
            if self._snapshots is not None:
                geo, xt = self._transform_point_at(X[t], t)
                psi = geo.quasi_coherent_state(xt)
            else:
                psi = self._geometry.quasi_coherent_state(Xt[t])
            states.append(psi)

        combined = np.full(T, np.nan)
        for t in range(max_lag, T):
            weighted_infidelity = 0.0
            for lag, w in zip(self.lags, self.lag_weights):
                if t >= lag:
                    overlap = np.abs(np.vdot(states[t], states[t - lag]))
                    fidelity = overlap**2
                    weighted_infidelity += w * (1.0 - fidelity)
            combined[t] = weighted_infidelity

        rolling_combined = (
            pd.Series(combined).rolling(window=self.rolling_window, min_periods=1).mean().values
        )

        z_scores = np.full(T, np.nan)
        for t in range(self.min_expanding, T):
            if self.adaptive_z_window is not None:
                lookback = min(t, self.adaptive_z_window)
                window_vals = rolling_combined[t - lookback : t]
                window_valid = window_vals[~np.isnan(window_vals)]
                if len(window_valid) < 10:
                    continue
                mu = np.median(window_valid)
                sigma = 1.4826 * np.median(np.abs(window_valid - mu))
            else:
                past = rolling_combined[:t]
                past_valid = past[~np.isnan(past)]
                if len(past_valid) < 10:
                    continue
                mu = np.mean(past_valid)
                sigma = np.std(past_valid, ddof=1)
            if sigma > 1e-12:
                z_scores[t] = (rolling_combined[t] - mu) / sigma
            else:
                z_scores[t] = 0.0

        return z_scores


class SpectralGapDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via inverse spectral gap 1/(E_1 - E_0).

    In physics, gap closing signals a phase transition. The inverse gap
    spikes when the two lowest energy levels become degenerate.

    Score = abs(z-score of 1/gap) smoothed over a rolling window.
    """

    def __init__(
        self,
        hilbert_dim: int = 8,
        n_pca_components: int = 15,
        operator_method: str = "random",
        scale_exponent: Optional[float] = None,
        rolling_window: int = 20,
        min_expanding: int = 60,
        seed: int = 42,
        causal_fit_length: Optional[int] = None,
        expanding_refit_interval: Optional[int] = None,
        normalization: str = "sphere",
        adaptive_epsilon: bool = False,
    ):
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
        self._geometry = None
        self._scaler = None
        self._pca = None
        self._snapshots = None
        self._train_norms = None
        self._train_std = None
        self._epsilon = 1e-5

    @property
    def name(self) -> str:
        return "Spectral Gap"

    def fit(self, X: np.ndarray, **kwargs) -> "SpectralGapDetector":
        if self.expanding_refit_interval is not None:
            return self._fit_expanding(X)

        from .core import QCMLGeometry

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
            X_pca_raw,
            self.normalization,
            self._train_norms,
            self._train_std,
        )

        if self.adaptive_epsilon:
            self._epsilon = 1e-3 * np.median(np.abs(X_pca_fit))
        else:
            self._epsilon = 1e-5

        self._geometry = QCMLGeometry(n_features=X_pca_fit.shape[1], hilbert_dim=self.hilbert_dim)
        self._geometry.fit_operators(
            X_pca_fit,
            method=self.operator_method,
            scale_exponent=self.scale_exponent,
        )
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        Xt = _transform_array(
            X,
            self._scaler,
            self._pca,
            normalization=self.normalization,
            train_norms=self._train_norms,
            train_std=self._train_std,
        )
        T = len(Xt)

        import pandas as pd

        inv_gap = np.empty(T)
        for t in range(T):
            if self._snapshots is not None:
                geo, xt = self._transform_point_at(X[t], t)
            else:
                geo, xt = self._geometry, Xt[t]
            gap = geo.spectral_gap(xt)
            inv_gap[t] = 1.0 / (gap + 1e-10)

        rolling_vals = (
            pd.Series(inv_gap).rolling(window=self.rolling_window, min_periods=1).mean().values
        )

        z_scores = np.full(T, np.nan)
        for t in range(self.min_expanding, T):
            mu = np.mean(rolling_vals[:t])
            sigma = np.std(rolling_vals[:t], ddof=1)
            if sigma > 1e-12:
                z_scores[t] = abs((rolling_vals[t] - mu) / sigma)
            else:
                z_scores[t] = 0.0

        return z_scores


class MetricConditionDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via metric tensor condition number.

    The condition number kappa(g) = lambda_max / lambda_min measures
    geometric anisotropy. During regime transitions, the manifold distorts
    preferentially in certain directions, causing kappa to spike.

    Score = abs(z-score of log(kappa)) smoothed over a rolling window.
    """

    def __init__(
        self,
        hilbert_dim: int = 8,
        n_pca_components: int = 15,
        operator_method: str = "random",
        scale_exponent: Optional[float] = None,
        rolling_window: int = 20,
        min_expanding: int = 60,
        seed: int = 42,
        causal_fit_length: Optional[int] = None,
        expanding_refit_interval: Optional[int] = None,
        normalization: str = "sphere",
        adaptive_epsilon: bool = False,
    ):
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
        self._geometry = None
        self._scaler = None
        self._pca = None
        self._snapshots = None
        self._train_norms = None
        self._train_std = None
        self._epsilon = 1e-5

    @property
    def name(self) -> str:
        return "Metric Condition"

    def fit(self, X: np.ndarray, **kwargs) -> "MetricConditionDetector":
        if self.expanding_refit_interval is not None:
            return self._fit_expanding(X)

        from .core import QCMLGeometry

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
            X_pca_raw,
            self.normalization,
            self._train_norms,
            self._train_std,
        )

        if self.adaptive_epsilon:
            self._epsilon = 1e-3 * np.median(np.abs(X_pca_fit))
        else:
            self._epsilon = 1e-5

        self._geometry = QCMLGeometry(n_features=X_pca_fit.shape[1], hilbert_dim=self.hilbert_dim)
        self._geometry.fit_operators(
            X_pca_fit,
            method=self.operator_method,
            scale_exponent=self.scale_exponent,
        )
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        Xt = _transform_array(
            X,
            self._scaler,
            self._pca,
            normalization=self.normalization,
            train_norms=self._train_norms,
            train_std=self._train_std,
        )
        T = len(Xt)
        eps = self._epsilon

        import pandas as pd

        log_kappa = np.empty(T)
        for t in range(T):
            if self._snapshots is not None:
                geo, xt = self._transform_point_at(X[t], t)
                g_ij = geo.quantum_metric(xt, epsilon=eps)
            else:
                g_ij = self._geometry.quantum_metric(Xt[t], epsilon=eps)

            eigenvalues = np.linalg.eigvalsh(g_ij)
            pos_eigs = eigenvalues[eigenvalues > 1e-10]
            if len(pos_eigs) >= 2:
                log_kappa[t] = np.log(pos_eigs[-1] / pos_eigs[0])
            else:
                log_kappa[t] = 0.0

        rolling_vals = (
            pd.Series(log_kappa).rolling(window=self.rolling_window, min_periods=1).mean().values
        )

        z_scores = np.full(T, np.nan)
        for t in range(self.min_expanding, T):
            mu = np.mean(rolling_vals[:t])
            sigma = np.std(rolling_vals[:t], ddof=1)
            if sigma > 1e-12:
                z_scores[t] = abs((rolling_vals[t] - mu) / sigma)
            else:
                z_scores[t] = 0.0

        return z_scores


class GeometricEnsembleDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via RMS z-score ensemble of geometric features.

    Combines Berry Frobenius norm rate, QFI log-det, QFI trace, inverse
    spectral gap, metric condition number, and multi-lag infidelity.
    Each feature is independently z-scored, then combined as
    sqrt(mean(z_i^2)).
    """

    def __init__(
        self,
        hilbert_dim: int = 8,
        n_pca_components: int = 15,
        operator_method: str = "random",
        scale_exponent: Optional[float] = None,
        lags: Optional[List[int]] = None,
        lag_weights: Optional[List[float]] = None,
        rolling_window: int = 20,
        min_expanding: int = 60,
        seed: int = 42,
        causal_fit_length: Optional[int] = None,
        expanding_refit_interval: Optional[int] = None,
        normalization: str = "sphere",
        adaptive_epsilon: bool = False,
    ):
        self.hilbert_dim = hilbert_dim
        self.n_pca_components = n_pca_components
        self.operator_method = operator_method
        self.scale_exponent = scale_exponent
        self.lags = lags or [1, 3, 5, 10]
        self.lag_weights = lag_weights or [0.4, 0.3, 0.2, 0.1]
        self.rolling_window = rolling_window
        self.min_expanding = min_expanding
        self.seed = seed
        self.causal_fit_length = causal_fit_length
        self.expanding_refit_interval = expanding_refit_interval
        self.normalization = normalization
        self.adaptive_epsilon = adaptive_epsilon
        self._geometry = None
        self._scaler = None
        self._pca = None
        self._snapshots = None
        self._train_norms = None
        self._train_std = None
        self._epsilon = 1e-5

    @property
    def name(self) -> str:
        return "Geometric Ensemble"

    def fit(self, X: np.ndarray, **kwargs) -> "GeometricEnsembleDetector":
        if self.expanding_refit_interval is not None:
            return self._fit_expanding(X)

        from .core import QCMLGeometry

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
            X_pca_raw,
            self.normalization,
            self._train_norms,
            self._train_std,
        )

        if self.adaptive_epsilon:
            self._epsilon = 1e-3 * np.median(np.abs(X_pca_fit))
        else:
            self._epsilon = 1e-5

        self._geometry = QCMLGeometry(n_features=X_pca_fit.shape[1], hilbert_dim=self.hilbert_dim)
        self._geometry.fit_operators(
            X_pca_fit,
            method=self.operator_method,
            scale_exponent=self.scale_exponent,
        )
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        Xt = _transform_array(
            X,
            self._scaler,
            self._pca,
            normalization=self.normalization,
            train_norms=self._train_norms,
            train_std=self._train_std,
        )
        T = len(Xt)
        max_lag = max(self.lags)
        eps = self._epsilon

        import pandas as pd

        berry_frob = np.empty(T)
        log_det = np.empty(T)
        trace_g = np.empty(T)
        inv_gap = np.empty(T)
        log_kappa = np.empty(T)
        states = []

        for t in range(T):
            if self._snapshots is not None:
                geo, xt = self._transform_point_at(X[t], t)
            else:
                geo, xt = self._geometry, Xt[t]

            F = geo.berry_curvature(xt, epsilon=eps)
            berry_frob[t] = np.sqrt(np.sum(F**2))

            g_ij = geo.quantum_metric(xt, epsilon=eps)
            eigenvalues = np.linalg.eigvalsh(g_ij)
            pos_eigs = eigenvalues[eigenvalues > 1e-10]

            if len(pos_eigs) > 0:
                log_det[t] = np.sum(np.log(pos_eigs))
                trace_g[t] = np.sum(pos_eigs)
            else:
                log_det[t] = -20.0
                trace_g[t] = 0.0

            if len(pos_eigs) >= 2:
                log_kappa[t] = np.log(pos_eigs[-1] / pos_eigs[0])
            else:
                log_kappa[t] = 0.0

            gap = geo.spectral_gap(xt)
            inv_gap[t] = 1.0 / (gap + 1e-10)

            psi = geo.quasi_coherent_state(xt)
            states.append(psi)

        infidelity = np.full(T, np.nan)
        for t in range(max_lag, T):
            weighted_inf = 0.0
            for lag, w in zip(self.lags, self.lag_weights):
                if t >= lag:
                    overlap = np.abs(np.vdot(states[t], states[t - lag]))
                    weighted_inf += w * (1.0 - overlap**2)
            infidelity[t] = weighted_inf

        def expanding_zscore(vals):
            smoothed = (
                pd.Series(vals).rolling(window=self.rolling_window, min_periods=1).mean().values
            )
            z = np.full(T, 0.0)
            for t in range(self.min_expanding, T):
                past = smoothed[:t]
                past_valid = past[~np.isnan(past)]
                if len(past_valid) < 10:
                    continue
                mu = np.mean(past_valid)
                sigma = np.std(past_valid, ddof=1)
                if sigma > 1e-12:
                    z[t] = abs((smoothed[t] - mu) / sigma)
            return z

        z_channels = np.column_stack(
            [
                expanding_zscore(np.abs(np.diff(berry_frob, prepend=berry_frob[0]))),
                expanding_zscore(log_det),
                expanding_zscore(trace_g),
                expanding_zscore(inv_gap),
                expanding_zscore(log_kappa),
                expanding_zscore(infidelity),
            ]
        )

        rms = np.sqrt(np.mean(z_channels**2, axis=1))
        rms[: self.min_expanding] = np.nan

        return rms


class RicciScalarDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via Ricci scalar curvature R(x).

    The Ricci scalar captures the average curvature of the data manifold.
    Diverging curvature signals geometric phase transitions. Computationally
    expensive; use small n_pca_components (3-4).

    Score = abs(z-score of R(x)) smoothed over a rolling window.
    """

    def __init__(
        self,
        hilbert_dim: int = 6,
        n_pca_components: int = 3,
        operator_method: str = "pca_inspired",
        scale_exponent: Optional[float] = None,
        rolling_window: int = 20,
        min_expanding: int = 60,
        seed: int = 42,
        causal_fit_length: Optional[int] = None,
        expanding_refit_interval: Optional[int] = None,
        normalization: str = "soft",
        adaptive_epsilon: bool = True,
        custom_operators: Optional[List[np.ndarray]] = None,
    ):
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
        self._geometry = None
        self._scaler = None
        self._pca = None
        self._snapshots = None
        self._train_norms = None
        self._train_std = None
        self._epsilon = 1e-5

    @property
    def name(self) -> str:
        return "Ricci Scalar"

    def fit(self, X: np.ndarray, **kwargs) -> "RicciScalarDetector":
        if self.expanding_refit_interval is not None:
            return self._fit_expanding(X)

        from .core import QCMLGeometry

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
            X_pca_raw,
            self.normalization,
            self._train_norms,
            self._train_std,
        )

        if self.adaptive_epsilon:
            self._epsilon = 1e-3 * np.median(np.abs(X_pca_fit))
        else:
            self._epsilon = 1e-5

        self._geometry = QCMLGeometry(n_features=X_pca_fit.shape[1], hilbert_dim=self.hilbert_dim)
        if self.custom_operators is not None:
            self._geometry.set_operators(self.custom_operators)
        else:
            self._geometry.fit_operators(
                X_pca_fit,
                method=self.operator_method,
                scale_exponent=self.scale_exponent,
            )
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        Xt = _transform_array(
            X,
            self._scaler,
            self._pca,
            normalization=self.normalization,
            train_norms=self._train_norms,
            train_std=self._train_std,
        )
        T = len(Xt)

        import pandas as pd

        ricci_vals = np.empty(T)
        for t in range(T):
            if self._snapshots is not None:
                geo, xt = self._transform_point_at(X[t], t)
            else:
                geo, xt = self._geometry, Xt[t]
            ricci_vals[t] = geo.ricci_scalar(xt)

        rolling_vals = (
            pd.Series(ricci_vals).rolling(window=self.rolling_window, min_periods=1).mean().values
        )

        z_scores = np.full(T, np.nan)
        for t in range(self.min_expanding, T):
            mu = np.mean(rolling_vals[:t])
            sigma = np.std(rolling_vals[:t], ddof=1)
            if sigma > 1e-12:
                z_scores[t] = abs((rolling_vals[t] - mu) / sigma)
            else:
                z_scores[t] = 0.0

        return z_scores


class SectionalCurvatureDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via sectional curvature K(e_0, e_1).

    Sectional curvature measures curvature in the (e_0, e_1) plane of
    the data manifold. Captures directional geometric distortion during
    regime transitions. Computationally expensive; use small n_pca_components.

    Two scoring modes:
    - 'abs_zscore' (default): abs(z-score of K) — original behavior.
    - 'neg_fraction': rolling fraction of negative curvature values.
      Negative sectional curvature indicates geodesic divergence (chaotic
      dynamics); a high fraction signals sustained instability.
    """

    def __init__(
        self,
        hilbert_dim: int = 6,
        n_pca_components: int = 3,
        operator_method: str = "pca_inspired",
        scale_exponent: Optional[float] = None,
        rolling_window: int = 20,
        min_expanding: int = 60,
        seed: int = 42,
        causal_fit_length: Optional[int] = None,
        expanding_refit_interval: Optional[int] = None,
        normalization: str = "soft",
        adaptive_epsilon: bool = True,
        curvature_indices: Optional[tuple] = None,
        custom_operators: Optional[List[np.ndarray]] = None,
        score_mode: str = "abs_zscore",
        neg_fraction_window: int = 20,
        subsample: int = 1,
    ):
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
        self.curvature_indices = curvature_indices or (0, 1)
        self.custom_operators = custom_operators
        self.score_mode = score_mode
        self.neg_fraction_window = neg_fraction_window
        self.subsample = subsample
        self._geometry = None
        self._scaler = None
        self._pca = None
        self._snapshots = None
        self._train_norms = None
        self._train_std = None
        self._epsilon = 1e-5

    @property
    def name(self) -> str:
        i, j = self.curvature_indices
        if self.score_mode == "neg_fraction":
            return f"Sectional Curvature Sign ({i},{j})"
        return f"Sectional Curvature ({i},{j})"

    def fit(self, X: np.ndarray, **kwargs) -> "SectionalCurvatureDetector":
        if self.expanding_refit_interval is not None:
            return self._fit_expanding(X)

        from .core import QCMLGeometry

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
            X_pca_raw,
            self.normalization,
            self._train_norms,
            self._train_std,
        )

        if self.adaptive_epsilon:
            self._epsilon = 1e-3 * np.median(np.abs(X_pca_fit))
        else:
            self._epsilon = 1e-5

        self._geometry = QCMLGeometry(n_features=X_pca_fit.shape[1], hilbert_dim=self.hilbert_dim)
        if self.custom_operators is not None:
            self._geometry.set_operators(self.custom_operators)
        else:
            self._geometry.fit_operators(
                X_pca_fit,
                method=self.operator_method,
                scale_exponent=self.scale_exponent,
            )
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        Xt = _transform_array(
            X,
            self._scaler,
            self._pca,
            normalization=self.normalization,
            train_norms=self._train_norms,
            train_std=self._train_std,
        )
        T = len(Xt)
        ci, cj = self.curvature_indices

        import pandas as pd

        curv_vals = np.full(T, np.nan)
        for t in range(0, T, self.subsample):
            if self._snapshots is not None:
                geo, xt = self._transform_point_at(X[t], t)
            else:
                geo, xt = self._geometry, Xt[t]
            curv_vals[t] = geo.sectional_curvature(xt, i=ci, j=cj)

        # Interpolate subsampled points
        if self.subsample > 1:
            curv_vals = pd.Series(curv_vals).interpolate(method="linear").values

        if self.score_mode == "neg_fraction":
            # Rolling fraction of negative curvature values
            is_negative = (curv_vals < 0).astype(float)
            is_negative[np.isnan(curv_vals)] = np.nan
            scores = (
                pd.Series(is_negative)
                .rolling(window=self.neg_fraction_window, min_periods=5)
                .mean()
                .values
            )
            return scores

        # Default: abs_zscore mode (original behavior)
        rolling_vals = (
            pd.Series(curv_vals).rolling(window=self.rolling_window, min_periods=1).mean().values
        )

        z_scores = np.full(T, np.nan)
        for t in range(self.min_expanding, T):
            mu = np.mean(rolling_vals[:t])
            sigma = np.std(rolling_vals[:t], ddof=1)
            if sigma > 1e-12:
                z_scores[t] = abs((rolling_vals[t] - mu) / sigma)
            else:
                z_scores[t] = 0.0

        return z_scores


class GeodesicVelocityDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via Fubini-Study velocity d_FS(psi_t, psi_{t-1}).

    Measures the rate of change of the quasi-coherent state on the Hilbert
    space projective manifold. Fast regime transitions cause large state
    velocities. Uses quantum_distance (Fubini-Study) for efficiency.

    Score = abs(z-score of velocity) smoothed over a rolling window.
    """

    def __init__(
        self,
        hilbert_dim: int = 8,
        n_pca_components: int = 8,
        operator_method: str = "pca_inspired",
        scale_exponent: Optional[float] = None,
        rolling_window: int = 15,
        min_expanding: int = 60,
        seed: int = 42,
        causal_fit_length: Optional[int] = None,
        expanding_refit_interval: Optional[int] = None,
        normalization: str = "sphere",
        adaptive_epsilon: bool = False,
        custom_operators: Optional[List[np.ndarray]] = None,
    ):
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
        self._geometry = None
        self._scaler = None
        self._pca = None
        self._snapshots = None
        self._train_norms = None
        self._train_std = None
        self._epsilon = 1e-5

    @property
    def name(self) -> str:
        return "Geodesic Velocity"

    def fit(self, X: np.ndarray, **kwargs) -> "GeodesicVelocityDetector":
        if self.expanding_refit_interval is not None:
            return self._fit_expanding(X)

        from .core import QCMLGeometry

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
            X_pca_raw,
            self.normalization,
            self._train_norms,
            self._train_std,
        )

        if self.adaptive_epsilon:
            self._epsilon = 1e-3 * np.median(np.abs(X_pca_fit))
        else:
            self._epsilon = 1e-5

        self._geometry = QCMLGeometry(n_features=X_pca_fit.shape[1], hilbert_dim=self.hilbert_dim)
        if self.custom_operators is not None:
            self._geometry.set_operators(self.custom_operators)
        else:
            self._geometry.fit_operators(
                X_pca_fit,
                method=self.operator_method,
                scale_exponent=self.scale_exponent,
            )
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        Xt = _transform_array(
            X,
            self._scaler,
            self._pca,
            normalization=self.normalization,
            train_norms=self._train_norms,
            train_std=self._train_std,
        )
        T = len(Xt)

        import pandas as pd

        # Compute quantum states for all time steps
        states = []
        for t in range(T):
            if self._snapshots is not None:
                geo, xt = self._transform_point_at(X[t], t)
                psi = geo.quasi_coherent_state(xt)
            else:
                psi = self._geometry.quasi_coherent_state(Xt[t])
            states.append(psi)

        # Fubini-Study velocity: arccos(|<psi_t|psi_{t-1}>|) per timestep
        velocity = np.full(T, np.nan)
        for t in range(1, T):
            overlap = np.abs(np.vdot(states[t], states[t - 1]))
            overlap = np.clip(overlap, 0.0, 1.0)
            velocity[t] = np.arccos(overlap)

        rolling_vals = (
            pd.Series(velocity).rolling(window=self.rolling_window, min_periods=1).mean().values
        )

        z_scores = np.full(T, np.nan)
        for t in range(self.min_expanding, T):
            past = rolling_vals[1:t]
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


class SpeedLimitRatioDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via quantum speed limit ratio r(t) = v(t) / Delta(t).

    The Mandelstam-Tamm bound constrains how fast a quantum state can evolve:
    the Fubini-Study velocity v(t) cannot exceed the spectral gap Delta(t).
    When r → 1, the system evolves at its maximum possible speed — a hallmark
    of diabatic (sudden) transitions characteristic of financial crises.

    Score = abs(z-score of rolling-mean speed limit ratio).
    """

    def __init__(
        self,
        hilbert_dim: int = 8,
        n_pca_components: int = 8,
        operator_method: str = "random",
        scale_exponent: Optional[float] = None,
        rolling_window: int = 20,
        min_expanding: int = 60,
        seed: int = 42,
        causal_fit_length: Optional[int] = None,
        expanding_refit_interval: Optional[int] = None,
        normalization: str = "soft",
        adaptive_epsilon: bool = True,
        custom_operators: Optional[List[np.ndarray]] = None,
    ):
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
        self._geometry = None
        self._scaler = None
        self._pca = None
        self._snapshots = None
        self._train_norms = None
        self._train_std = None
        self._epsilon = 1e-5

    @property
    def name(self) -> str:
        return "Speed Limit Ratio"

    def fit(self, X: np.ndarray, **kwargs) -> "SpeedLimitRatioDetector":
        if self.expanding_refit_interval is not None:
            return self._fit_expanding(X)

        from .core import QCMLGeometry

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
            X_pca_raw,
            self.normalization,
            self._train_norms,
            self._train_std,
        )

        if self.adaptive_epsilon:
            self._epsilon = 1e-3 * np.median(np.abs(X_pca_fit))
        else:
            self._epsilon = 1e-5

        self._geometry = QCMLGeometry(n_features=X_pca_fit.shape[1], hilbert_dim=self.hilbert_dim)
        if self.custom_operators is not None:
            self._geometry.set_operators(self.custom_operators)
        else:
            self._geometry.fit_operators(
                X_pca_fit,
                method=self.operator_method,
                scale_exponent=self.scale_exponent,
            )
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        Xt = _transform_array(
            X,
            self._scaler,
            self._pca,
            normalization=self.normalization,
            train_norms=self._train_norms,
            train_std=self._train_std,
        )
        T = len(Xt)

        import pandas as pd

        # Compute quantum states and spectral gaps
        states = []
        gaps = np.empty(T)
        for t in range(T):
            if self._snapshots is not None:
                geo, xt = self._transform_point_at(X[t], t)
                states.append(geo.quasi_coherent_state(xt))
                gaps[t] = geo.spectral_gap(xt)
            else:
                states.append(self._geometry.quasi_coherent_state(Xt[t]))
                gaps[t] = self._geometry.spectral_gap(Xt[t])

        # FS velocity: arccos(|<psi_t|psi_{t-1}>|)
        velocity = np.full(T, np.nan)
        for t in range(1, T):
            overlap = np.abs(np.vdot(states[t], states[t - 1]))
            overlap = np.clip(overlap, 0.0, 1.0)
            velocity[t] = np.arccos(overlap)

        # Speed limit ratio: v(t) / max(gap(t), epsilon)
        ratio = np.full(T, np.nan)
        valid = (~np.isnan(velocity)) & (gaps > 1e-6)
        ratio[valid] = velocity[valid] / gaps[valid]

        rolling_vals = (
            pd.Series(ratio).rolling(window=self.rolling_window, min_periods=1).mean().values
        )

        z_scores = np.full(T, np.nan)
        for t in range(self.min_expanding, T):
            past = rolling_vals[1:t]
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


class DimensionalityCollapseDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via inverse participation ratio (IPR) of metric eigenvalues.

    IPR = lambda_max / tr(g) measures how concentrated the quantum metric
    eigenvalue spectrum is. During crises, herding behaviour causes one
    principal direction to dominate (IPR → 1), collapsing the effective
    dimensionality of the data manifold.

    Score = abs(z-score of rolling-mean IPR).
    """

    def __init__(
        self,
        hilbert_dim: int = 8,
        n_pca_components: int = 8,
        operator_method: str = "random",
        scale_exponent: Optional[float] = None,
        rolling_window: int = 20,
        min_expanding: int = 60,
        seed: int = 42,
        causal_fit_length: Optional[int] = None,
        expanding_refit_interval: Optional[int] = None,
        normalization: str = "soft",
        adaptive_epsilon: bool = True,
        subsample: int = 1,
        custom_operators: Optional[List[np.ndarray]] = None,
    ):
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
        self.subsample = subsample
        self.custom_operators = custom_operators
        self._geometry = None
        self._scaler = None
        self._pca = None
        self._snapshots = None
        self._train_norms = None
        self._train_std = None
        self._epsilon = 1e-5

    @property
    def name(self) -> str:
        return "Dimensionality Collapse"

    def fit(self, X: np.ndarray, **kwargs) -> "DimensionalityCollapseDetector":
        if self.expanding_refit_interval is not None:
            return self._fit_expanding(X)

        from .core import QCMLGeometry

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
            X_pca_raw,
            self.normalization,
            self._train_norms,
            self._train_std,
        )

        if self.adaptive_epsilon:
            self._epsilon = 1e-3 * np.median(np.abs(X_pca_fit))
        else:
            self._epsilon = 1e-5

        self._geometry = QCMLGeometry(n_features=X_pca_fit.shape[1], hilbert_dim=self.hilbert_dim)
        if self.custom_operators is not None:
            self._geometry.set_operators(self.custom_operators)
        else:
            self._geometry.fit_operators(
                X_pca_fit,
                method=self.operator_method,
                scale_exponent=self.scale_exponent,
            )
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        Xt = _transform_array(
            X,
            self._scaler,
            self._pca,
            normalization=self.normalization,
            train_norms=self._train_norms,
            train_std=self._train_std,
        )
        T = len(Xt)

        import pandas as pd

        ipr_raw = np.full(T, np.nan)
        for t in range(0, T, self.subsample):
            if self._snapshots is not None:
                geo, xt = self._transform_point_at(X[t], t)
                g = geo.quantum_metric(xt)
            else:
                g = self._geometry.quantum_metric(Xt[t])

            eigvals = np.linalg.eigvalsh(g)
            eigvals = np.maximum(eigvals, 0)
            total = np.sum(eigvals)
            if total > 1e-15:
                ipr_raw[t] = eigvals[-1] / total

        # Interpolate subsampled points
        if self.subsample > 1:
            ipr_raw = pd.Series(ipr_raw).interpolate(method="linear").values

        rolling_vals = (
            pd.Series(ipr_raw).rolling(window=self.rolling_window, min_periods=1).mean().values
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


class SpectralFlowDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via spectral flow d(eigenvalues)/dt.

    Tracks the rate of change of the full Hamiltonian spectrum. Rapid
    spectral rearrangement (level crossings, avoided crossings) signals
    a regime transition. Captures more information than the spectral gap
    alone by monitoring all energy levels.

    Score = abs(z-score of ||dE/dt||) smoothed over a rolling window.
    """

    def __init__(
        self,
        hilbert_dim: int = 8,
        n_pca_components: int = 8,
        operator_method: str = "pca_inspired",
        scale_exponent: Optional[float] = None,
        rolling_window: int = 15,
        min_expanding: int = 60,
        seed: int = 42,
        causal_fit_length: Optional[int] = None,
        expanding_refit_interval: Optional[int] = None,
        normalization: str = "soft",
        adaptive_epsilon: bool = True,
        custom_operators: Optional[List[np.ndarray]] = None,
    ):
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
        self._geometry = None
        self._scaler = None
        self._pca = None
        self._snapshots = None
        self._train_norms = None
        self._train_std = None
        self._epsilon = 1e-5

    @property
    def name(self) -> str:
        return "Spectral Flow"

    def fit(self, X: np.ndarray, **kwargs) -> "SpectralFlowDetector":
        if self.expanding_refit_interval is not None:
            return self._fit_expanding(X)

        from .core import QCMLGeometry

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
            X_pca_raw,
            self.normalization,
            self._train_norms,
            self._train_std,
        )

        if self.adaptive_epsilon:
            self._epsilon = 1e-3 * np.median(np.abs(X_pca_fit))
        else:
            self._epsilon = 1e-5

        self._geometry = QCMLGeometry(n_features=X_pca_fit.shape[1], hilbert_dim=self.hilbert_dim)
        if self.custom_operators is not None:
            self._geometry.set_operators(self.custom_operators)
        else:
            self._geometry.fit_operators(
                X_pca_fit,
                method=self.operator_method,
                scale_exponent=self.scale_exponent,
            )
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        Xt = _transform_array(
            X,
            self._scaler,
            self._pca,
            normalization=self.normalization,
            train_norms=self._train_norms,
            train_std=self._train_std,
        )
        T = len(Xt)

        import pandas as pd

        # Compute full spectrum at each time step
        spectra = []
        for t in range(T):
            if self._snapshots is not None:
                geo, xt = self._transform_point_at(X[t], t)
                eigs = geo.full_spectrum(xt)
            else:
                eigs = self._geometry.full_spectrum(Xt[t])
            spectra.append(eigs)

        spectra = np.array(spectra)  # (T, hilbert_dim)

        # Spectral flow = L2 norm of eigenvalue differences
        flow = np.full(T, np.nan)
        for t in range(1, T):
            flow[t] = np.linalg.norm(spectra[t] - spectra[t - 1])

        rolling_vals = (
            pd.Series(flow).rolling(window=self.rolling_window, min_periods=1).mean().values
        )

        z_scores = np.full(T, np.nan)
        for t in range(self.min_expanding, T):
            past = rolling_vals[1:t]
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


class CommutatorNormDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via Hamiltonian commutator norm ||[H(x_t), H(x_{t-1})]||_F.

    Measures geometric incompatibility between consecutive data points.
    When the Hamiltonians at successive time steps commute, the system
    evolves smoothly. Large commutator norms indicate the system is
    traversing geometrically incompatible regions — a hallmark of regime
    transitions.

    Score = abs(z-score of commutator norm) smoothed over a rolling window.
    """

    def __init__(
        self,
        hilbert_dim: int = 8,
        n_pca_components: int = 8,
        operator_method: str = "pca_inspired",
        scale_exponent: Optional[float] = None,
        rolling_window: int = 15,
        min_expanding: int = 60,
        seed: int = 42,
        causal_fit_length: Optional[int] = None,
        expanding_refit_interval: Optional[int] = None,
        normalization: str = "soft",
        adaptive_epsilon: bool = True,
        custom_operators: Optional[List[np.ndarray]] = None,
    ):
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
        self._geometry = None
        self._scaler = None
        self._pca = None
        self._snapshots = None
        self._train_norms = None
        self._train_std = None
        self._epsilon = 1e-5

    @property
    def name(self) -> str:
        return "Commutator Norm"

    def fit(self, X: np.ndarray, **kwargs) -> "CommutatorNormDetector":
        if self.expanding_refit_interval is not None:
            return self._fit_expanding(X)

        from .core import QCMLGeometry

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
            X_pca_raw,
            self.normalization,
            self._train_norms,
            self._train_std,
        )

        if self.adaptive_epsilon:
            self._epsilon = 1e-3 * np.median(np.abs(X_pca_fit))
        else:
            self._epsilon = 1e-5

        self._geometry = QCMLGeometry(n_features=X_pca_fit.shape[1], hilbert_dim=self.hilbert_dim)
        if self.custom_operators is not None:
            self._geometry.set_operators(self.custom_operators)
        else:
            self._geometry.fit_operators(
                X_pca_fit,
                method=self.operator_method,
                scale_exponent=self.scale_exponent,
            )
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        Xt = _transform_array(
            X,
            self._scaler,
            self._pca,
            normalization=self.normalization,
            train_norms=self._train_norms,
            train_std=self._train_std,
        )
        T = len(Xt)

        import pandas as pd

        comm_norms = np.full(T, np.nan)
        for t in range(1, T):
            if self._snapshots is not None:
                geo, xt = self._transform_point_at(X[t], t)
                _, xt_prev = self._transform_point_at(X[t - 1], t - 1)
                comm_norms[t] = geo.hamiltonian_commutator_norm(xt, xt_prev)
            else:
                comm_norms[t] = self._geometry.hamiltonian_commutator_norm(Xt[t], Xt[t - 1])

        rolling_vals = (
            pd.Series(comm_norms).rolling(window=self.rolling_window, min_periods=1).mean().values
        )

        z_scores = np.full(T, np.nan)
        for t in range(self.min_expanding, T):
            past = rolling_vals[1:t]
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
# Shared fit logic for new observatory detectors
# =============================================================================


def _standard_qcml_fit(detector, X: np.ndarray):
    """Shared fit logic for ExpandingWindowMixin + BaseRegimeDetector subclasses."""
    if detector.expanding_refit_interval is not None:
        return detector._fit_expanding(X)

    from .core import QCMLGeometry

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
        X_pca_raw,
        detector.normalization,
        detector._train_norms,
        detector._train_std,
    )

    if detector.adaptive_epsilon:
        detector._epsilon = 1e-3 * np.median(np.abs(X_pca_fit))
    else:
        detector._epsilon = 1e-5

    detector._geometry = QCMLGeometry(
        n_features=X_pca_fit.shape[1], hilbert_dim=detector.hilbert_dim
    )
    custom_ops = getattr(detector, "custom_operators", None)
    if custom_ops is not None:
        detector._geometry.set_operators(custom_ops)
    else:
        detector._geometry.fit_operators(
            X_pca_fit,
            method=detector.operator_method,
            scale_exponent=detector.scale_exponent,
        )
    return detector


def _standard_init(detector, **kwargs):
    """Shared __init__ logic for new observatory detectors."""
    for k, v in kwargs.items():
        setattr(detector, k, v)
    detector._geometry = None
    detector._scaler = None
    detector._pca = None
    detector._snapshots = None
    detector._train_norms = None
    detector._train_std = None
    detector._epsilon = 1e-5


def _expanding_zscore(raw_values, rolling_window, min_expanding, T, skip_nan_start=0):
    """Compute expanding-window z-score of rolling-mean values."""
    import pandas as pd

    rolling_vals = pd.Series(raw_values).rolling(window=rolling_window, min_periods=1).mean().values
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


# =============================================================================
# New Observatory Detectors (10 new detection channels)
# =============================================================================


class SpectralEntropyDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via Shannon entropy of excitation energy weights.

    Score = z-score of rolling-mean spectral entropy.
    """

    def __init__(
        self,
        hilbert_dim=8,
        n_pca_components=8,
        operator_method="random",
        scale_exponent=None,
        rolling_window=20,
        min_expanding=60,
        seed=42,
        causal_fit_length=None,
        expanding_refit_interval=None,
        normalization="soft",
        adaptive_epsilon=True,
        custom_operators=None,
        c=1.0,
    ):
        _standard_init(
            self,
            hilbert_dim=hilbert_dim,
            n_pca_components=n_pca_components,
            operator_method=operator_method,
            scale_exponent=scale_exponent,
            rolling_window=rolling_window,
            min_expanding=min_expanding,
            seed=seed,
            causal_fit_length=causal_fit_length,
            expanding_refit_interval=expanding_refit_interval,
            normalization=normalization,
            adaptive_epsilon=adaptive_epsilon,
            custom_operators=custom_operators,
            c=c,
        )

    @property
    def name(self):
        return "Spectral Entropy"

    def fit(self, X, **kwargs):
        return _standard_qcml_fit(self, X)

    def compute_regime_scores(self, X):
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")
        Xt = _transform_array(
            X,
            self._scaler,
            self._pca,
            normalization=self.normalization,
            train_norms=self._train_norms,
            train_std=self._train_std,
        )
        T = len(Xt)
        vals = np.empty(T)
        for t in range(T):
            geo, xt = (
                self._transform_point_at(X[t], t) if self._snapshots else (self._geometry, Xt[t])
            )
            vals[t] = geo.spectral_entropy(xt, c=self.c)
        return _expanding_zscore(vals, self.rolling_window, self.min_expanding, T)


class GeometricPhaseRateDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via geometric phase rate (Berry-like phase per step).

    Score = z-score of abs(geometric phase rate), smoothed.
    """

    def __init__(
        self,
        hilbert_dim=8,
        n_pca_components=8,
        operator_method="random",
        scale_exponent=None,
        rolling_window=20,
        min_expanding=60,
        seed=42,
        causal_fit_length=None,
        expanding_refit_interval=None,
        normalization="soft",
        adaptive_epsilon=True,
        custom_operators=None,
    ):
        _standard_init(
            self,
            hilbert_dim=hilbert_dim,
            n_pca_components=n_pca_components,
            operator_method=operator_method,
            scale_exponent=scale_exponent,
            rolling_window=rolling_window,
            min_expanding=min_expanding,
            seed=seed,
            causal_fit_length=causal_fit_length,
            expanding_refit_interval=expanding_refit_interval,
            normalization=normalization,
            adaptive_epsilon=adaptive_epsilon,
            custom_operators=custom_operators,
        )

    @property
    def name(self):
        return "Geometric Phase Rate"

    def fit(self, X, **kwargs):
        return _standard_qcml_fit(self, X)

    def compute_regime_scores(self, X):
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")
        Xt = _transform_array(
            X,
            self._scaler,
            self._pca,
            normalization=self.normalization,
            train_norms=self._train_norms,
            train_std=self._train_std,
        )
        T = len(Xt)
        vals = np.full(T, np.nan)
        for t in range(1, T):
            if self._snapshots:
                geo, xt = self._transform_point_at(X[t], t)
                _, xt_prev = self._transform_point_at(X[t - 1], t - 1)
            else:
                geo, xt, xt_prev = self._geometry, Xt[t], Xt[t - 1]
            vals[t] = abs(geo.geometric_phase_rate(xt_prev, xt))
        return _expanding_zscore(vals, self.rolling_window, self.min_expanding, T, 1)


class HamiltonianSensitivityDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via Var_psi(DeltaH) — ground-state sensitivity.

    Score = z-score of Var_psi(DH), smoothed.
    """

    def __init__(
        self,
        hilbert_dim=8,
        n_pca_components=8,
        operator_method="random",
        scale_exponent=None,
        rolling_window=20,
        min_expanding=60,
        seed=42,
        causal_fit_length=None,
        expanding_refit_interval=None,
        normalization="soft",
        adaptive_epsilon=True,
        custom_operators=None,
    ):
        _standard_init(
            self,
            hilbert_dim=hilbert_dim,
            n_pca_components=n_pca_components,
            operator_method=operator_method,
            scale_exponent=scale_exponent,
            rolling_window=rolling_window,
            min_expanding=min_expanding,
            seed=seed,
            causal_fit_length=causal_fit_length,
            expanding_refit_interval=expanding_refit_interval,
            normalization=normalization,
            adaptive_epsilon=adaptive_epsilon,
            custom_operators=custom_operators,
        )

    @property
    def name(self):
        return "Hamiltonian Sensitivity"

    def fit(self, X, **kwargs):
        return _standard_qcml_fit(self, X)

    def compute_regime_scores(self, X):
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")
        Xt = _transform_array(
            X,
            self._scaler,
            self._pca,
            normalization=self.normalization,
            train_norms=self._train_norms,
            train_std=self._train_std,
        )
        T = len(Xt)
        vals = np.full(T, np.nan)
        for t in range(1, T):
            if self._snapshots:
                geo, xt = self._transform_point_at(X[t], t)
                _, xt_prev = self._transform_point_at(X[t - 1], t - 1)
            else:
                geo, xt, xt_prev = self._geometry, Xt[t], Xt[t - 1]
            vals[t] = geo.hamiltonian_sensitivity(xt_prev, xt)
        return _expanding_zscore(vals, self.rolling_window, self.min_expanding, T, 1)


class GeodesicCurvatureDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via geodesic curvature (covariant acceleration).

    Uses small n_pca_components (3) due to Christoffel symbol cost.
    Score = z-score of geodesic curvature, smoothed.
    """

    def __init__(
        self,
        hilbert_dim=6,
        n_pca_components=3,
        operator_method="pca_inspired",
        scale_exponent=None,
        rolling_window=20,
        min_expanding=60,
        seed=42,
        causal_fit_length=None,
        expanding_refit_interval=None,
        normalization="soft",
        adaptive_epsilon=True,
        custom_operators=None,
        subsample=5,
    ):
        _standard_init(
            self,
            hilbert_dim=hilbert_dim,
            n_pca_components=n_pca_components,
            operator_method=operator_method,
            scale_exponent=scale_exponent,
            rolling_window=rolling_window,
            min_expanding=min_expanding,
            seed=seed,
            causal_fit_length=causal_fit_length,
            expanding_refit_interval=expanding_refit_interval,
            normalization=normalization,
            adaptive_epsilon=adaptive_epsilon,
            custom_operators=custom_operators,
            subsample=subsample,
        )

    @property
    def name(self):
        return "Geodesic Curvature"

    def fit(self, X, **kwargs):
        return _standard_qcml_fit(self, X)

    def compute_regime_scores(self, X):
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")
        Xt = _transform_array(
            X,
            self._scaler,
            self._pca,
            normalization=self.normalization,
            train_norms=self._train_norms,
            train_std=self._train_std,
        )
        T = len(Xt)
        import pandas as pd

        vals = np.full(T, np.nan)
        ss = self.subsample
        for t in range(ss, T - ss, ss):
            if self._snapshots:
                geo, xt = self._transform_point_at(X[t], t)
                _, xt_prev = self._transform_point_at(X[t - ss], t - ss)
                _, xt_next = self._transform_point_at(X[t + ss], t + ss)
            else:
                geo = self._geometry
                xt, xt_prev, xt_next = Xt[t], Xt[t - ss], Xt[t + ss]
            vals[t] = geo.geodesic_curvature(xt_prev, xt, xt_next)
        if ss > 1:
            vals = pd.Series(vals).interpolate(method="linear").values
        return _expanding_zscore(vals, self.rolling_window, self.min_expanding, T, ss)


class EffectiveStateDimensionDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via IPR of time-averaged density matrix.

    D_eff = W^2 / sum|<psi_s|psi_{s'}>|^2.
    Score = z-score of rolling D_eff.
    """

    def __init__(
        self,
        hilbert_dim=8,
        n_pca_components=8,
        operator_method="random",
        scale_exponent=None,
        rolling_window=20,
        min_expanding=60,
        seed=42,
        causal_fit_length=None,
        expanding_refit_interval=None,
        normalization="soft",
        adaptive_epsilon=True,
        custom_operators=None,
        state_window=20,
    ):
        _standard_init(
            self,
            hilbert_dim=hilbert_dim,
            n_pca_components=n_pca_components,
            operator_method=operator_method,
            scale_exponent=scale_exponent,
            rolling_window=rolling_window,
            min_expanding=min_expanding,
            seed=seed,
            causal_fit_length=causal_fit_length,
            expanding_refit_interval=expanding_refit_interval,
            normalization=normalization,
            adaptive_epsilon=adaptive_epsilon,
            custom_operators=custom_operators,
            state_window=state_window,
        )

    @property
    def name(self):
        return "Effective State Dimension"

    def fit(self, X, **kwargs):
        return _standard_qcml_fit(self, X)

    def compute_regime_scores(self, X):
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")
        Xt = _transform_array(
            X,
            self._scaler,
            self._pca,
            normalization=self.normalization,
            train_norms=self._train_norms,
            train_std=self._train_std,
        )
        T = len(Xt)
        W = self.state_window
        states = []
        for t in range(T):
            geo, xt = (
                self._transform_point_at(X[t], t) if self._snapshots else (self._geometry, Xt[t])
            )
            states.append(geo.quasi_coherent_state(xt))
        vals = np.full(T, np.nan)
        for t in range(W, T):
            vals[t] = self._geometry.effective_state_dimension(states[t - W : t])
        return _expanding_zscore(vals, self.rolling_window, self.min_expanding, T, W)


class QGTPhaseRigidityDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via Berry-to-metric Frobenius ratio ||F||/||g||.

    Score = z-score of rolling-mean phase rigidity.
    """

    def __init__(
        self,
        hilbert_dim=8,
        n_pca_components=8,
        operator_method="random",
        scale_exponent=None,
        rolling_window=20,
        min_expanding=60,
        seed=42,
        causal_fit_length=None,
        expanding_refit_interval=None,
        normalization="soft",
        adaptive_epsilon=True,
        custom_operators=None,
    ):
        _standard_init(
            self,
            hilbert_dim=hilbert_dim,
            n_pca_components=n_pca_components,
            operator_method=operator_method,
            scale_exponent=scale_exponent,
            rolling_window=rolling_window,
            min_expanding=min_expanding,
            seed=seed,
            causal_fit_length=causal_fit_length,
            expanding_refit_interval=expanding_refit_interval,
            normalization=normalization,
            adaptive_epsilon=adaptive_epsilon,
            custom_operators=custom_operators,
        )

    @property
    def name(self):
        return "QGT Phase Rigidity"

    def fit(self, X, **kwargs):
        return _standard_qcml_fit(self, X)

    def compute_regime_scores(self, X):
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")
        Xt = _transform_array(
            X,
            self._scaler,
            self._pca,
            normalization=self.normalization,
            train_norms=self._train_norms,
            train_std=self._train_std,
        )
        T = len(Xt)
        eps = self._epsilon
        vals = np.empty(T)
        for t in range(T):
            geo, xt = (
                self._transform_point_at(X[t], t) if self._snapshots else (self._geometry, Xt[t])
            )
            vals[t] = geo.qgt_phase_rigidity(xt, epsilon=eps)
        return _expanding_zscore(vals, self.rolling_window, self.min_expanding, T)


class ReducedPurityDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via purity of reduced density matrix Tr(rho_A^2).

    Uses operator_method='random' to avoid Kramers degeneracy.
    Score = z-score of rolling-mean purity.
    """

    def __init__(
        self,
        hilbert_dim=8,
        n_pca_components=8,
        operator_method="random",
        scale_exponent=None,
        rolling_window=20,
        min_expanding=60,
        seed=42,
        causal_fit_length=None,
        expanding_refit_interval=None,
        normalization="soft",
        adaptive_epsilon=True,
        custom_operators=None,
        partition=(2, 4),
    ):
        _standard_init(
            self,
            hilbert_dim=hilbert_dim,
            n_pca_components=n_pca_components,
            operator_method=operator_method,
            scale_exponent=scale_exponent,
            rolling_window=rolling_window,
            min_expanding=min_expanding,
            seed=seed,
            causal_fit_length=causal_fit_length,
            expanding_refit_interval=expanding_refit_interval,
            normalization=normalization,
            adaptive_epsilon=adaptive_epsilon,
            custom_operators=custom_operators,
            partition=partition,
        )

    @property
    def name(self):
        return "Reduced Purity"

    def fit(self, X, **kwargs):
        return _standard_qcml_fit(self, X)

    def compute_regime_scores(self, X):
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")
        Xt = _transform_array(
            X,
            self._scaler,
            self._pca,
            normalization=self.normalization,
            train_norms=self._train_norms,
            train_std=self._train_std,
        )
        T = len(Xt)
        vals = np.empty(T)
        for t in range(T):
            geo, xt = (
                self._transform_point_at(X[t], t) if self._snapshots else (self._geometry, Xt[t])
            )
            vals[t] = geo.reduced_state_purity(xt, partition=self.partition)
        return _expanding_zscore(vals, self.rolling_window, self.min_expanding, T)


class SpectralComplexityDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via adaptive-temperature Gibbs entropy.

    Score = z-score of rolling-mean spectral complexity.
    """

    def __init__(
        self,
        hilbert_dim=8,
        n_pca_components=8,
        operator_method="random",
        scale_exponent=None,
        rolling_window=20,
        min_expanding=60,
        seed=42,
        causal_fit_length=None,
        expanding_refit_interval=None,
        normalization="soft",
        adaptive_epsilon=True,
        custom_operators=None,
        c=1.0,
    ):
        _standard_init(
            self,
            hilbert_dim=hilbert_dim,
            n_pca_components=n_pca_components,
            operator_method=operator_method,
            scale_exponent=scale_exponent,
            rolling_window=rolling_window,
            min_expanding=min_expanding,
            seed=seed,
            causal_fit_length=causal_fit_length,
            expanding_refit_interval=expanding_refit_interval,
            normalization=normalization,
            adaptive_epsilon=adaptive_epsilon,
            custom_operators=custom_operators,
            c=c,
        )

    @property
    def name(self):
        return "Spectral Complexity"

    def fit(self, X, **kwargs):
        return _standard_qcml_fit(self, X)

    def compute_regime_scores(self, X):
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")
        Xt = _transform_array(
            X,
            self._scaler,
            self._pca,
            normalization=self.normalization,
            train_norms=self._train_norms,
            train_std=self._train_std,
        )
        T = len(Xt)
        vals = np.empty(T)
        for t in range(T):
            geo, xt = (
                self._transform_point_at(X[t], t) if self._snapshots else (self._geometry, Xt[t])
            )
            vals[t] = geo.spectral_complexity(xt, c=self.c)
        return _expanding_zscore(vals, self.rolling_window, self.min_expanding, T)


class BerryVelocityCouplingDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via Berry curvature-velocity coupling ||iota_v F||_g.

    Score = z-score of rolling-mean coupling.
    """

    def __init__(
        self,
        hilbert_dim=8,
        n_pca_components=8,
        operator_method="random",
        scale_exponent=None,
        rolling_window=20,
        min_expanding=60,
        seed=42,
        causal_fit_length=None,
        expanding_refit_interval=None,
        normalization="soft",
        adaptive_epsilon=True,
        custom_operators=None,
    ):
        _standard_init(
            self,
            hilbert_dim=hilbert_dim,
            n_pca_components=n_pca_components,
            operator_method=operator_method,
            scale_exponent=scale_exponent,
            rolling_window=rolling_window,
            min_expanding=min_expanding,
            seed=seed,
            causal_fit_length=causal_fit_length,
            expanding_refit_interval=expanding_refit_interval,
            normalization=normalization,
            adaptive_epsilon=adaptive_epsilon,
            custom_operators=custom_operators,
        )

    @property
    def name(self):
        return "Berry Velocity Coupling"

    def fit(self, X, **kwargs):
        return _standard_qcml_fit(self, X)

    def compute_regime_scores(self, X):
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")
        Xt = _transform_array(
            X,
            self._scaler,
            self._pca,
            normalization=self.normalization,
            train_norms=self._train_norms,
            train_std=self._train_std,
        )
        T = len(Xt)
        eps = self._epsilon
        vals = np.full(T, np.nan)
        for t in range(1, T):
            if self._snapshots:
                geo, xt = self._transform_point_at(X[t], t)
                _, xt_prev = self._transform_point_at(X[t - 1], t - 1)
            else:
                geo, xt, xt_prev = self._geometry, Xt[t], Xt[t - 1]
            vals[t] = geo.berry_velocity_coupling(xt, xt_prev, epsilon=eps)
        return _expanding_zscore(vals, self.rolling_window, self.min_expanding, T, 1)


class CurvatureRateDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via Ricci scalar curvature rate |dR/dt|.

    Computationally expensive; use small n_pca_components.
    Score = z-score of rolling-mean |dR/dt|.
    """

    def __init__(
        self,
        hilbert_dim=6,
        n_pca_components=3,
        operator_method="pca_inspired",
        scale_exponent=None,
        rolling_window=20,
        min_expanding=60,
        seed=42,
        causal_fit_length=None,
        expanding_refit_interval=None,
        normalization="soft",
        adaptive_epsilon=True,
        custom_operators=None,
        subsample=10,
    ):
        _standard_init(
            self,
            hilbert_dim=hilbert_dim,
            n_pca_components=n_pca_components,
            operator_method=operator_method,
            scale_exponent=scale_exponent,
            rolling_window=rolling_window,
            min_expanding=min_expanding,
            seed=seed,
            causal_fit_length=causal_fit_length,
            expanding_refit_interval=expanding_refit_interval,
            normalization=normalization,
            adaptive_epsilon=adaptive_epsilon,
            custom_operators=custom_operators,
            subsample=subsample,
        )

    @property
    def name(self):
        return "Curvature Rate"

    def fit(self, X, **kwargs):
        return _standard_qcml_fit(self, X)

    def compute_regime_scores(self, X):
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")
        Xt = _transform_array(
            X,
            self._scaler,
            self._pca,
            normalization=self.normalization,
            train_norms=self._train_norms,
            train_std=self._train_std,
        )
        T = len(Xt)
        import pandas as pd

        ricci_vals = np.full(T, np.nan)
        for t in range(0, T, self.subsample):
            geo, xt = (
                self._transform_point_at(X[t], t) if self._snapshots else (self._geometry, Xt[t])
            )
            ricci_vals[t] = geo.ricci_scalar(xt)
        if self.subsample > 1:
            ricci_vals = pd.Series(ricci_vals).interpolate(method="linear").values
        rate = np.full(T, np.nan)
        for t in range(1, T):
            if not np.isnan(ricci_vals[t]) and not np.isnan(ricci_vals[t - 1]):
                rate[t] = abs(ricci_vals[t] - ricci_vals[t - 1])
        return _expanding_zscore(rate, self.rolling_window, self.min_expanding, T, 1)


# --- Random Matrix Theory constants ---
RMT_POISSON = 0.3863  # 2 ln 2 - 1 (integrable / uncorrelated)
RMT_GOE = 0.5307  # 4 - 2 sqrt(3) (Gaussian Orthogonal Ensemble)
RMT_GUE = 0.5996  # 2 sqrt(3) / pi - 1/2 (Gaussian Unitary Ensemble)


def compute_level_spacing_ratios(eigenvalues: np.ndarray) -> np.ndarray:
    """Compute nearest-neighbor level spacing ratios from sorted eigenvalues.

    Given sorted eigenvalues E_0 < E_1 < ... < E_{d-1}:
        s_n = E_{n+1} - E_n        (nearest-neighbor spacings)
        r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1})   (ratios, in [0,1])

    Args:
        eigenvalues: Sorted eigenvalues of shape (d,).

    Returns:
        ratios: Level spacing ratios of shape (d-2,).
    """
    spacings = np.diff(eigenvalues)
    spacings = np.maximum(spacings, 1e-15)
    if len(spacings) < 2:
        return np.array([RMT_POISSON])
    s_n = spacings[:-1]
    s_np1 = spacings[1:]
    return np.minimum(s_n, s_np1) / np.maximum(s_n, s_np1)


class LevelSpacingRatioDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via nearest-neighbor level spacing ratios of H(x).

    Measures the spectral statistics of the error Hamiltonian at each time
    step. Three summary observables:

        1. mean_ratio: <r> -- mean level spacing ratio
        2. std_ratio: sigma_r -- standard deviation of ratios
        3. poisson_fraction: fraction of r_n < 0.386 (Poisson threshold)

    The default variant ('mean_ratio') is z-scored with an expanding window.

    IMPORTANT: Must use operator_method='random'. The 'pca_inspired' method
    produces Kramers-degenerate spectra with only 2 usable ratios at d=8.

    Specialist detector: exceptional on systemic crises (GFC d=3.27, COVID
    d=3.44) but weak on localized ones (flash crash d=0.07). The bimodal
    response is itself scientifically interesting.

    References:
        Oganesyan & Huse (2007). Phys. Rev. B 75, 155111.
        Atas et al. (2013). Phys. Rev. Lett. 110, 084101.

    Args:
        hilbert_dim: Hilbert space dimension (default 8 for 3-qubit).
        n_pca_components: Number of PCA components.
        operator_method: Must be 'random' (see above).
        rolling_window: Rolling mean window for smoothing.
        min_expanding: Minimum observations before z-scoring.
        seed: Random seed.
        normalization: Post-PCA normalization mode.
        variant: Which observable to score ('mean_ratio', 'std_ratio',
            'poisson_fraction'). Default 'mean_ratio'.
        poisson_threshold: Threshold for Poisson fraction (default 0.386).
    """

    def __init__(
        self,
        hilbert_dim=8,
        n_pca_components=8,
        operator_method="random",
        scale_exponent=None,
        rolling_window=20,
        min_expanding=60,
        seed=42,
        causal_fit_length=None,
        expanding_refit_interval=None,
        normalization="soft",
        adaptive_epsilon=True,
        custom_operators=None,
        variant="mean_ratio",
        poisson_threshold=0.386,
    ):
        _standard_init(
            self,
            hilbert_dim=hilbert_dim,
            n_pca_components=n_pca_components,
            operator_method=operator_method,
            scale_exponent=scale_exponent,
            rolling_window=rolling_window,
            min_expanding=min_expanding,
            seed=seed,
            causal_fit_length=causal_fit_length,
            expanding_refit_interval=expanding_refit_interval,
            normalization=normalization,
            adaptive_epsilon=adaptive_epsilon,
            custom_operators=custom_operators,
            variant=variant,
            poisson_threshold=poisson_threshold,
        )

    @property
    def name(self):
        return "Level Spacing Ratio"

    def fit(self, X, **kwargs):
        return _standard_qcml_fit(self, X)

    def compute_regime_scores(self, X):
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")
        Xt = _transform_array(
            X,
            self._scaler,
            self._pca,
            normalization=self.normalization,
            train_norms=self._train_norms,
            train_std=self._train_std,
        )
        T = len(Xt)
        mean_ratios = np.empty(T)
        std_ratios = np.empty(T)
        poisson_fractions = np.empty(T)
        for t in range(T):
            geo, xt = (
                self._transform_point_at(X[t], t) if self._snapshots else (self._geometry, Xt[t])
            )
            eigenvalues = geo.full_spectrum(xt)
            ratios = compute_level_spacing_ratios(eigenvalues)
            mean_ratios[t] = np.mean(ratios)
            std_ratios[t] = np.std(ratios, ddof=0)
            poisson_fractions[t] = np.mean(ratios < self.poisson_threshold)
        raw = {
            "mean_ratio": mean_ratios,
            "std_ratio": std_ratios,
            "poisson_fraction": poisson_fractions,
        }[self.variant]
        return _expanding_zscore(raw, self.rolling_window, self.min_expanding, T)


# --- Quantum Relative Entropy helpers ---

_QRE_EPSILON = 1e-10  # Floor for log(rho_ref) eigenvalues


def _quantum_relative_entropy(psi, rho_ref, epsilon=_QRE_EPSILON):
    """Compute D(|psi><psi| || rho_ref) = -<psi| log(rho_ref) |psi>.

    Args:
        psi: Pure state vector of shape (d,).
        rho_ref: Reference density matrix of shape (d, d).
        epsilon: Floor for eigenvalues to avoid log(0).

    Returns:
        D: Quantum relative entropy in nats (non-negative).
    """
    eigenvalues, eigenvectors = np.linalg.eigh(rho_ref)
    eigenvalues_reg = np.maximum(eigenvalues, epsilon)
    log_eigenvalues = np.log(eigenvalues_reg)
    coeffs = eigenvectors.conj().T @ psi
    probs = np.abs(coeffs) ** 2
    D = -np.sum(log_eigenvalues * probs)
    return float(max(D, 0.0))


def _build_expanding_rho_ref(states, t, min_window=20):
    """Build expanding-window average density matrix.

    rho_ref = (1/t) * sum_{s=0}^{t-1} |psi_s><psi_s|

    Args:
        states: List of state vectors.
        t: Current time index.
        min_window: Minimum past states required.

    Returns:
        rho_ref or None if insufficient history.
    """
    if t < min_window:
        return None
    d = states[0].shape[0]
    rho = np.zeros((d, d), dtype=complex)
    for s in range(t):
        psi_s = states[s]
        rho += np.outer(psi_s, psi_s.conj())
    rho /= t
    return rho


class QuantumRelativeEntropyDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via quantum relative entropy D(rho_t || rho_ref).

    Computes D(|psi_t><psi_t| || rho_ref) where rho_ref is the expanding-
    window average density matrix of all past ground states. D measures how
    "surprising" the current state is relative to the long-term trajectory.

    Score = expanding z-score of rolling-mean relative entropy.

    Best variant is 'D_expanding' (median d=1.578 on smoke test), which uses
    all past states for rho_ref. This is more orthogonal (rho=0.495 with MLF)
    and more sensitive than rolling-window variants.

    References:
        Umegaki (1962). Conditional expectation in an operator algebra.
        Vedral (2002). The role of relative entropy in quantum information theory.

    Args:
        hilbert_dim: Hilbert space dimension.
        state_window: Rolling window for reference density matrix.
        n_pca_components: Number of PCA components.
        operator_method: Operator construction method ('random' recommended).
        rolling_window: Rolling mean smoothing window.
        min_expanding: Minimum observations before z-scoring.
        seed: Random seed.
        normalization: Post-PCA normalization mode.
    """

    def __init__(
        self,
        hilbert_dim=8,
        n_pca_components=8,
        operator_method="random",
        scale_exponent=None,
        rolling_window=20,
        min_expanding=60,
        seed=42,
        causal_fit_length=None,
        expanding_refit_interval=None,
        normalization="soft",
        adaptive_epsilon=True,
        custom_operators=None,
        state_window=20,
    ):
        _standard_init(
            self,
            hilbert_dim=hilbert_dim,
            n_pca_components=n_pca_components,
            operator_method=operator_method,
            scale_exponent=scale_exponent,
            rolling_window=rolling_window,
            min_expanding=min_expanding,
            seed=seed,
            causal_fit_length=causal_fit_length,
            expanding_refit_interval=expanding_refit_interval,
            normalization=normalization,
            adaptive_epsilon=adaptive_epsilon,
            custom_operators=custom_operators,
            state_window=state_window,
        )

    @property
    def name(self):
        return "Quantum Relative Entropy"

    def fit(self, X, **kwargs):
        return _standard_qcml_fit(self, X)

    def compute_regime_scores(self, X):
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")
        Xt = _transform_array(
            X,
            self._scaler,
            self._pca,
            normalization=self.normalization,
            train_norms=self._train_norms,
            train_std=self._train_std,
        )
        T = len(Xt)
        # Collect all ground states
        states = []
        for t in range(T):
            geo, xt = (
                self._transform_point_at(X[t], t) if self._snapshots else (self._geometry, Xt[t])
            )
            states.append(geo.quasi_coherent_state(xt))
        # Compute expanding-window QRE
        vals = np.full(T, np.nan)
        for t in range(T):
            rho_ref = _build_expanding_rho_ref(states, t, min_window=self.state_window)
            if rho_ref is None:
                continue
            vals[t] = _quantum_relative_entropy(states[t], rho_ref)
        return _expanding_zscore(
            vals, self.rolling_window, self.min_expanding, T, skip_nan_start=self.state_window
        )
