"""
Core Quantum Geometric Observables for Regime Detection

Three novel unsupervised observables derived from the QCML quantum metric tensor:

1. BerryPhaseRateDetector - Rate of change of Berry curvature
2. QFIDeterminantDetector - Quantum metric pseudo-determinant
3. MultiLagFidelityDetector - Multi-lag quantum fidelity

Also provides the BaseRegimeDetector ABC and ExpandingWindowMixin.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, List

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
            window = X[t - lookback + 1:t + 1]
            row = np.concatenate([
                np.mean(window, axis=0),
                np.std(window, axis=0),
                np.min(window, axis=0),
                np.max(window, axis=0),
            ])
            features.append(row)

        return np.array(features)

    @abstractmethod
    def fit(self, X: np.ndarray, **kwargs) -> 'BaseRegimeDetector':
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


class ExpandingWindowMixin:
    """Mixin that periodically refits scaler/PCA/operators on expanding windows.

    When ``expanding_refit_interval`` is set, the mixin builds a sequence of
    snapshots during ``fit()``, each fitted on an expanding prefix of the data.
    During ``compute_regime_scores()``, the detector uses the most recent
    snapshot that precedes each time point.
    """

    def _fit_expanding(self, X: np.ndarray) -> 'ExpandingWindowMixin':
        """Build expanding-window snapshots for periodic refitting."""
        from .core import QCMLGeometry

        T = X.shape[0]
        interval = self.expanding_refit_interval
        min_fit = max(self.min_expanding, 30)
        n_components = min(self.n_pca_components, X.shape[1])

        refit_points = list(range(min_fit, T, interval))
        if not refit_points or refit_points[-1] != T:
            refit_points.append(T)

        self._snapshots = []
        for refit_idx in refit_points:
            np.random.seed(self.seed)

            scaler = StandardScaler()
            scaler.fit(X[:refit_idx])
            X_scaled = scaler.transform(X)

            pca = PCA(n_components=n_components)
            pca.fit(X_scaled[:refit_idx])
            X_pca = pca.transform(X_scaled)
            X_pca = X_pca / (np.linalg.norm(X_pca, axis=1, keepdims=True) + 1e-8)

            geometry = QCMLGeometry(
                n_features=X_pca.shape[1], hilbert_dim=self.hilbert_dim
            )
            geometry.fit_operators(X_pca[:refit_idx], method=self.operator_method)

            self._snapshots.append({
                'refit_idx': refit_idx,
                'geometry': geometry,
                'X_transformed': X_pca,
            })

        last = self._snapshots[-1]
        self._geometry = last['geometry']
        self._X_transformed = last['X_transformed']

        return self

    def _get_snapshot_at(self, t: int):
        """Return (geometry, x_point) for time t using the most recent snapshot."""
        best_snap = self._snapshots[0]
        for snap in self._snapshots:
            if snap['refit_idx'] <= t:
                best_snap = snap
            else:
                break

        return best_snap['geometry'], best_snap['X_transformed'][t]


class QFIDeterminantDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via quantum metric determinant det(g_ab).

    The determinant of the quantum metric tensor is the volume element of
    the data manifold. Collapsing volume indicates approaching a phase
    boundary; diverging volume indicates rapid expansion of accessible
    state space.

    Score = abs(z-score of log(|det(g)|)) smoothed over a rolling window.

    Args:
        expanding_refit_interval: If set, periodically refit scaler/PCA/operators
            on expanding windows at this interval. Default None (single-shot fit).
    """

    def __init__(
        self,
        hilbert_dim: int = 8,
        n_pca_components: int = 15,
        operator_method: str = 'random',
        rolling_window: int = 20,
        min_expanding: int = 60,
        seed: int = 42,
        causal_fit_length: Optional[int] = None,
        expanding_refit_interval: Optional[int] = None,
    ):
        self.hilbert_dim = hilbert_dim
        self.n_pca_components = n_pca_components
        self.operator_method = operator_method
        self.rolling_window = rolling_window
        self.min_expanding = min_expanding
        self.seed = seed
        self.causal_fit_length = causal_fit_length
        self.expanding_refit_interval = expanding_refit_interval
        self._geometry = None
        self._X_transformed = None
        self._snapshots = None

    @property
    def name(self) -> str:
        return "QFI Determinant"

    def fit(self, X: np.ndarray, **kwargs) -> 'QFIDeterminantDetector':
        if self.expanding_refit_interval is not None:
            return self._fit_expanding(X)

        from .core import QCMLGeometry

        np.random.seed(self.seed)
        n_components = min(self.n_pca_components, X.shape[1])
        fit_end = self.causal_fit_length or X.shape[0]

        scaler = StandardScaler()
        scaler.fit(X[:fit_end])
        X_scaled = scaler.transform(X)
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled[:fit_end])
        X_pca = pca.transform(X_scaled)
        X_pca = X_pca / (np.linalg.norm(X_pca, axis=1, keepdims=True) + 1e-8)

        self._X_transformed = X_pca
        self._geometry = QCMLGeometry(
            n_features=X_pca.shape[1], hilbert_dim=self.hilbert_dim
        )
        self._geometry.fit_operators(X_pca[:fit_end], method=self.operator_method)
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        Xt = self._X_transformed
        T = len(Xt)

        import pandas as pd

        log_pseudodet = np.empty(T)
        eigenvalue_tolerance = 1e-10

        for t in range(T):
            if self._snapshots is not None:
                geo, xt = self._get_snapshot_at(t)
                g_ij = geo.quantum_metric(xt)
            else:
                g_ij = self._geometry.quantum_metric(Xt[t])

            eigenvalues = np.linalg.eigvalsh(g_ij)
            nonzero_eigs = eigenvalues[eigenvalues > eigenvalue_tolerance]

            if len(nonzero_eigs) > 0:
                log_pseudodet[t] = np.sum(np.log(nonzero_eigs))
            else:
                log_pseudodet[t] = np.log(eigenvalue_tolerance) * len(eigenvalues)

        rolling_logdet = (
            pd.Series(log_pseudodet)
            .rolling(window=self.rolling_window, min_periods=1)
            .mean()
            .values
        )

        z_scores = np.full(T, np.nan)
        for t in range(self.min_expanding, T):
            mu = np.mean(rolling_logdet[:t])
            sigma = np.std(rolling_logdet[:t], ddof=1)
            if sigma > 1e-12:
                z_scores[t] = abs((rolling_logdet[t] - mu) / sigma)
            else:
                z_scores[t] = 0.0

        return z_scores


class BerryPhaseRateDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via rate of change of Berry curvature.

    Measures topological transition speed: rapid changes in Berry curvature
    indicate the system is crossing a phase boundary.

    Score = abs(diff(Berry_curvature)) smoothed and z-scored.

    Args:
        expanding_refit_interval: If set, periodically refit scaler/PCA/operators
            on expanding windows at this interval. Default None (single-shot fit).
    """

    def __init__(
        self,
        hilbert_dim: int = 8,
        n_pca_components: int = 15,
        operator_method: str = 'random',
        rolling_window: int = 20,
        min_expanding: int = 60,
        seed: int = 42,
        causal_fit_length: Optional[int] = None,
        expanding_refit_interval: Optional[int] = None,
    ):
        self.hilbert_dim = hilbert_dim
        self.n_pca_components = n_pca_components
        self.operator_method = operator_method
        self.rolling_window = rolling_window
        self.min_expanding = min_expanding
        self.seed = seed
        self.causal_fit_length = causal_fit_length
        self.expanding_refit_interval = expanding_refit_interval
        self._geometry = None
        self._X_transformed = None
        self._snapshots = None

    @property
    def name(self) -> str:
        return "Berry Phase Rate"

    def fit(self, X: np.ndarray, **kwargs) -> 'BerryPhaseRateDetector':
        if self.expanding_refit_interval is not None:
            return self._fit_expanding(X)

        from .core import QCMLGeometry

        np.random.seed(self.seed)
        n_components = min(self.n_pca_components, X.shape[1])
        fit_end = self.causal_fit_length or X.shape[0]

        scaler = StandardScaler()
        scaler.fit(X[:fit_end])
        X_scaled = scaler.transform(X)
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled[:fit_end])
        X_pca = pca.transform(X_scaled)
        X_pca = X_pca / (np.linalg.norm(X_pca, axis=1, keepdims=True) + 1e-8)

        self._X_transformed = X_pca
        self._geometry = QCMLGeometry(
            n_features=X_pca.shape[1], hilbert_dim=self.hilbert_dim
        )
        self._geometry.fit_operators(X_pca[:fit_end], method=self.operator_method)
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        Xt = self._X_transformed
        T = len(Xt)

        import pandas as pd

        berry = np.empty(T)
        for t in range(T):
            if self._snapshots is not None:
                geo, xt = self._get_snapshot_at(t)
                berry[t] = geo.berry_curvature_2d(xt, indices=(0, 1))
            else:
                berry[t] = self._geometry.berry_curvature_2d(Xt[t], indices=(0, 1))

        berry_rate = np.abs(np.diff(berry))

        rolling_rate = (
            pd.Series(berry_rate)
            .rolling(window=self.rolling_window, min_periods=1)
            .mean()
            .values
        )

        n = len(rolling_rate)
        z_scores = np.full(n, np.nan)
        for t in range(self.min_expanding, n):
            mu = np.mean(rolling_rate[:t])
            sigma = np.std(rolling_rate[:t], ddof=1)
            if sigma > 1e-12:
                z_scores[t] = (rolling_rate[t] - mu) / sigma
            else:
                z_scores[t] = 0.0

        return np.concatenate([[np.nan], z_scores])


class MultiLagFidelityDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via multi-lag quantum fidelity.

    Fidelity at lags [1, 3, 5, 10] provides multi-scale sensitivity:
    short lags detect fast crises, long lags detect gradual transitions.
    Score = weighted average infidelity (1-F), z-scored.

    Args:
        expanding_refit_interval: If set, periodically refit scaler/PCA/operators
            on expanding windows at this interval. Default None (single-shot fit).
    """

    def __init__(
        self,
        hilbert_dim: int = 8,
        n_pca_components: int = 15,
        operator_method: str = 'random',
        lags: Optional[List[int]] = None,
        lag_weights: Optional[List[float]] = None,
        rolling_window: int = 20,
        min_expanding: int = 60,
        seed: int = 42,
        causal_fit_length: Optional[int] = None,
        expanding_refit_interval: Optional[int] = None,
    ):
        self.hilbert_dim = hilbert_dim
        self.n_pca_components = n_pca_components
        self.operator_method = operator_method
        self.lags = lags or [1, 3, 5, 10]
        self.lag_weights = lag_weights or [0.4, 0.3, 0.2, 0.1]
        self.rolling_window = rolling_window
        self.min_expanding = min_expanding
        self.seed = seed
        self.causal_fit_length = causal_fit_length
        self.expanding_refit_interval = expanding_refit_interval
        self._geometry = None
        self._X_transformed = None
        self._snapshots = None

    @property
    def name(self) -> str:
        return "Multi-Lag Fidelity"

    def fit(self, X: np.ndarray, **kwargs) -> 'MultiLagFidelityDetector':
        if self.expanding_refit_interval is not None:
            return self._fit_expanding(X)

        from .core import QCMLGeometry

        np.random.seed(self.seed)
        n_components = min(self.n_pca_components, X.shape[1])
        fit_end = self.causal_fit_length or X.shape[0]

        scaler = StandardScaler()
        scaler.fit(X[:fit_end])
        X_scaled = scaler.transform(X)
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled[:fit_end])
        X_pca = pca.transform(X_scaled)
        X_pca = X_pca / (np.linalg.norm(X_pca, axis=1, keepdims=True) + 1e-8)

        self._X_transformed = X_pca
        self._geometry = QCMLGeometry(
            n_features=X_pca.shape[1], hilbert_dim=self.hilbert_dim
        )
        self._geometry.fit_operators(X_pca[:fit_end], method=self.operator_method)
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        Xt = self._X_transformed
        T = len(Xt)
        max_lag = max(self.lags)

        import pandas as pd

        states = []
        for t in range(T):
            if self._snapshots is not None:
                geo, xt = self._get_snapshot_at(t)
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
                    fidelity = overlap ** 2
                    weighted_infidelity += w * (1.0 - fidelity)
            combined[t] = weighted_infidelity

        rolling_combined = (
            pd.Series(combined)
            .rolling(window=self.rolling_window, min_periods=1)
            .mean()
            .values
        )

        z_scores = np.full(T, np.nan)
        for t in range(self.min_expanding, T):
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
