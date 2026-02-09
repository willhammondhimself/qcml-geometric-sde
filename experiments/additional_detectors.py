"""
Additional Geometric Regime Detectors

These detectors wrap the core geometry with various derived observables:
1. QCMLChernDetector — rolling Chern number
2. QFISusceptibilityDetector — Fubini-Study distance
3. ScalarCurvatureDetector — Ricci scalar curvature
4. MultiScaleChernDetector — multi-scale Chern consensus
5. QuantumEnsembleDetector — ensemble of all indicators
6. GeometricConsensusDetector — persistence + voting across geometric methods
7. FastGeometricConsensusDetector — tuned for sudden transitions
8. SlowGeometricConsensusDetector — tuned for gradual transitions
9. ShockMagnitudeDetector — V-shaped crisis detection
10. MetricConditionNumberDetector — condition number of metric tensor
"""

import logging
from typing import Optional, List

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from qcml_geometry import BaseRegimeDetector
from qcml_geometry.observables import _transform_array

logger = logging.getLogger(__name__)


class QCMLChernDetector(BaseRegimeDetector):
    """Regime detection via rolling Chern number from geometry.

    Args:
        hilbert_dim: Hilbert space dimension.
        window_size: Rolling window for Chern computation.
        n_pca_components: PCA dimensions before geometry fitting.
        operator_method: Method for fitting Hermitian operators.
    """

    def __init__(
        self,
        hilbert_dim: int = 8,
        window_size: int = 20,
        n_pca_components: int = 15,
        operator_method: str = 'random',
        seed: int = 42,
    ):
        self.hilbert_dim = hilbert_dim
        self.window_size = window_size
        self.n_pca_components = n_pca_components
        self.operator_method = operator_method
        self.seed = seed
        self._geometry = None
        self._detector = None
        self._scaler = None
        self._pca = None

    @property
    def name(self) -> str:
        return "QCML Chern"

    def fit(self, X: np.ndarray, **kwargs) -> 'QCMLChernDetector':
        from qcml_geometry import QCMLGeometry
        from qcml_geometry.topology import TopologicalRegimeDetector

        np.random.seed(self.seed)
        n_components = min(self.n_pca_components, X.shape[1])

        self._scaler = StandardScaler()
        self._scaler.fit(X)
        self._pca = PCA(n_components=n_components)
        X_scaled = self._scaler.transform(X)
        self._pca.fit(X_scaled)
        X_pca = self._pca.transform(X_scaled)
        X_pca = X_pca / (np.linalg.norm(X_pca, axis=1, keepdims=True) + 1e-8)

        self._geometry = QCMLGeometry(
            n_features=X_pca.shape[1], hilbert_dim=self.hilbert_dim
        )
        self._geometry.fit_operators(X_pca, method=self.operator_method)
        self._detector = TopologicalRegimeDetector(
            geometry=self._geometry, window_size=self.window_size
        )
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._detector is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")
        X_pca = _transform_array(X, self._scaler, self._pca)
        chern = self._detector.rolling_chern_number(
            X_pca, window=self.window_size
        )
        pad = np.full(len(X_pca) - len(chern), np.nan)
        return np.concatenate([pad, chern])


class QFISusceptibilityDetector(BaseRegimeDetector):
    """Regime detection via Fisher Information susceptibility.

    Measures the realized QFI through the Fubini-Study distance between
    consecutive states.

    Args:
        hilbert_dim: Hilbert space dimension.
        n_pca_components: PCA dimensions before geometry fitting.
        operator_method: Method for fitting Hermitian operators.
        rolling_window: Window for smoothing the raw FS distance series.
        min_expanding: Minimum expanding window before z-scoring starts.
        seed: Random seed.
    """

    def __init__(
        self,
        hilbert_dim: int = 8,
        n_pca_components: int = 15,
        operator_method: str = 'random',
        rolling_window: int = 20,
        min_expanding: int = 60,
        seed: int = 42,
    ):
        self.hilbert_dim = hilbert_dim
        self.n_pca_components = n_pca_components
        self.operator_method = operator_method
        self.rolling_window = rolling_window
        self.min_expanding = min_expanding
        self.seed = seed
        self._geometry = None
        self._scaler = None
        self._pca = None

    @property
    def name(self) -> str:
        return "QFI Susceptibility"

    def fit(self, X: np.ndarray, **kwargs) -> 'QFISusceptibilityDetector':
        from qcml_geometry import QCMLGeometry

        np.random.seed(self.seed)
        n_components = min(self.n_pca_components, X.shape[1])

        self._scaler = StandardScaler()
        self._scaler.fit(X)
        self._pca = PCA(n_components=n_components)
        X_scaled = self._scaler.transform(X)
        self._pca.fit(X_scaled)
        X_pca = self._pca.transform(X_scaled)
        X_pca = X_pca / (np.linalg.norm(X_pca, axis=1, keepdims=True) + 1e-8)

        self._geometry = QCMLGeometry(
            n_features=X_pca.shape[1], hilbert_dim=self.hilbert_dim
        )
        self._geometry.fit_operators(X_pca, method=self.operator_method)
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        import pandas as pd

        Xt = _transform_array(X, self._scaler, self._pca)
        T = len(Xt)

        raw_dist = np.empty(T - 1)
        for t in range(T - 1):
            raw_dist[t] = self._geometry.quantum_distance(Xt[t], Xt[t + 1])

        rolling_dist = (
            pd.Series(raw_dist)
            .rolling(window=self.rolling_window, min_periods=1)
            .mean()
            .values
        )

        n = len(rolling_dist)
        z_scores = np.full(n, np.nan)
        for t in range(self.min_expanding, n):
            mu = np.mean(rolling_dist[:t])
            sigma = np.std(rolling_dist[:t], ddof=1)
            if sigma > 1e-12:
                z_scores[t] = (rolling_dist[t] - mu) / sigma
            else:
                z_scores[t] = 0.0

        return np.concatenate([[np.nan], z_scores])


class ScalarCurvatureDetector(BaseRegimeDetector):
    """Ricci scalar curvature of the metric manifold.

    Computes R(t) at each time point. Large |R| indicates regions where the
    data manifold is highly curved (nonlinear, unstable dynamics).

    Args:
        hilbert_dim: Hilbert space dimension.
        n_curvature_components: PCA dimensions (kept small — Ricci is O(n^2)).
        operator_method: Method for fitting Hermitian operators.
        rolling_window: Window for smoothing the raw |R| series.
        min_expanding: Minimum expanding window before z-scoring starts.
        seed: Random seed.
    """

    def __init__(
        self,
        hilbert_dim: int = 8,
        n_curvature_components: int = 8,
        operator_method: str = 'random',
        rolling_window: int = 40,
        min_expanding: int = 60,
        seed: int = 42,
    ):
        self.hilbert_dim = hilbert_dim
        self.n_curvature_components = n_curvature_components
        self.operator_method = operator_method
        self.rolling_window = rolling_window
        self.min_expanding = min_expanding
        self.seed = seed
        self._geometry = None
        self._scaler = None
        self._pca = None

    @property
    def name(self) -> str:
        return "Scalar Curvature"

    def fit(self, X: np.ndarray, **kwargs) -> 'ScalarCurvatureDetector':
        from qcml_geometry import QCMLGeometry

        np.random.seed(self.seed)
        n_components = min(self.n_curvature_components, X.shape[1])

        self._scaler = StandardScaler()
        self._scaler.fit(X)
        self._pca = PCA(n_components=n_components)
        X_scaled = self._scaler.transform(X)
        self._pca.fit(X_scaled)
        X_pca = self._pca.transform(X_scaled)
        X_pca = X_pca / (np.linalg.norm(X_pca, axis=1, keepdims=True) + 1e-8)

        self._geometry = QCMLGeometry(
            n_features=X_pca.shape[1], hilbert_dim=self.hilbert_dim
        )
        self._geometry.fit_operators(X_pca, method=self.operator_method)
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        import pandas as pd

        Xt = _transform_array(X, self._scaler, self._pca)
        T = len(Xt)

        raw_R = np.empty(T)
        for t in range(T):
            raw_R[t] = self._geometry.ricci_scalar(Xt[t])
            self._geometry.clear_cache()

        rolling_R = (
            pd.Series(raw_R)
            .rolling(window=self.rolling_window, min_periods=1)
            .mean()
            .values
        )

        z_scores = np.full(T, np.nan)
        for t in range(self.min_expanding, T):
            mu = np.mean(rolling_R[:t])
            sigma = np.std(rolling_R[:t], ddof=1)
            if sigma > 1e-12:
                z_scores[t] = abs((rolling_R[t] - mu) / sigma)
            else:
                z_scores[t] = 0.0

        return z_scores


class MultiScaleChernDetector(BaseRegimeDetector):
    """Multi-scale Chern consensus via BaseRegimeDetector interface.

    Computes Chern numbers at multiple time scales and produces a weighted
    consensus signal.

    Args:
        hilbert_dim: Hilbert space dimension.
        n_pca_components: PCA dimensions before geometry fitting.
        scales: Window sizes for multi-scale analysis.
        weights: Per-scale weights (default: equal).
        consensus_threshold: Minimum weighted consensus for transition detection.
        normalization_strategy: Method for normalizing Chern changes across scales.
        normalization_window: Window size for 'rolling_fixed' strategy.
        operator_method: Method for fitting Hermitian operators.
        seed: Random seed.
    """

    def __init__(
        self,
        hilbert_dim: int = 8,
        n_pca_components: int = 15,
        scales: Optional[List[int]] = None,
        weights: Optional[List[float]] = None,
        consensus_threshold: float = 0.3,
        normalization_strategy: str = 'percentile',
        normalization_window: Optional[int] = None,
        operator_method: str = 'random',
        seed: int = 42,
    ):
        self.hilbert_dim = hilbert_dim
        self.n_pca_components = n_pca_components
        self.scales = scales or [10, 20, 30, 50]
        self.weights = weights or [0.15, 0.30, 0.30, 0.25]
        self.consensus_threshold = consensus_threshold
        self.normalization_strategy = normalization_strategy
        self.normalization_window = normalization_window
        self.operator_method = operator_method
        self.seed = seed
        self._geometry = None
        self._consensus = None
        self._scaler = None
        self._pca = None

    @property
    def name(self) -> str:
        return "Multi-Scale Chern"

    def fit(self, X: np.ndarray, **kwargs) -> 'MultiScaleChernDetector':
        from qcml_geometry import QCMLGeometry
        from qcml_geometry.indicators import MultiScaleChernConsensus

        np.random.seed(self.seed)
        n_components = min(self.n_pca_components, X.shape[1])

        self._scaler = StandardScaler()
        self._scaler.fit(X)
        self._pca = PCA(n_components=n_components)
        X_scaled = self._scaler.transform(X)
        self._pca.fit(X_scaled)
        X_pca = self._pca.transform(X_scaled)
        X_pca = X_pca / (np.linalg.norm(X_pca, axis=1, keepdims=True) + 1e-8)

        self._geometry = QCMLGeometry(
            n_features=X_pca.shape[1],
            hilbert_dim=self.hilbert_dim,
        )
        self._geometry.fit_operators(X_pca, method=self.operator_method)

        self._consensus = MultiScaleChernConsensus(
            geometry=self._geometry,
            scales=self.scales,
            weights=self.weights,
            consensus_threshold=self.consensus_threshold,
            normalization_strategy=self.normalization_strategy,
            normalization_window=self.normalization_window,
        )

        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._geometry is None or self._consensus is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        X_pca = _transform_array(X, self._scaler, self._pca)
        result = self._consensus.compute_consensus(X_pca)
        consensus_scores = result.values

        pad_len = len(X) - len(consensus_scores)
        if pad_len > 0:
            return np.concatenate([np.full(pad_len, np.nan), consensus_scores])
        else:
            return consensus_scores


class QuantumEnsembleDetector(BaseRegimeDetector):
    """Ensemble of all geometric indicators.

    Combines spectral gap, ground state energy, fidelity decay, and
    multi-scale Chern consensus into a single composite score.

    Args:
        hilbert_dim: Hilbert space dimension.
        n_pca_components: PCA dimensions before geometry fitting.
        window_size: Rolling window for indicators.
        operator_method: Method for fitting Hermitian operators.
        seed: Random seed.
    """

    def __init__(
        self,
        hilbert_dim: int = 8,
        n_pca_components: int = 15,
        window_size: int = 20,
        operator_method: str = 'random',
        seed: int = 42,
    ):
        self.hilbert_dim = hilbert_dim
        self.n_pca_components = n_pca_components
        self.window_size = window_size
        self.operator_method = operator_method
        self.seed = seed
        self._geometry = None
        self._suite = None
        self._scaler = None
        self._pca = None

    @property
    def name(self) -> str:
        return "Quantum Ensemble"

    def fit(self, X: np.ndarray, **kwargs) -> 'QuantumEnsembleDetector':
        from qcml_geometry import QCMLGeometry
        from qcml_geometry.indicators import QuantumIndicatorSuite

        np.random.seed(self.seed)
        n_components = min(self.n_pca_components, X.shape[1])

        self._scaler = StandardScaler()
        self._scaler.fit(X)
        self._pca = PCA(n_components=n_components)
        X_scaled = self._scaler.transform(X)
        self._pca.fit(X_scaled)
        X_pca = self._pca.transform(X_scaled)
        X_pca = X_pca / (np.linalg.norm(X_pca, axis=1, keepdims=True) + 1e-8)

        self._geometry = QCMLGeometry(
            n_features=X_pca.shape[1],
            hilbert_dim=self.hilbert_dim,
        )
        self._geometry.fit_operators(X_pca, method=self.operator_method)

        self._suite = QuantumIndicatorSuite(
            geometry=self._geometry,
            window_size=self.window_size,
        )

        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._geometry is None or self._suite is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        X_pca = _transform_array(X, self._scaler, self._pca)
        composite, _ = self._suite.compute_composite_score(X_pca)

        pad_len = len(X) - len(composite)
        if pad_len > 0:
            return np.concatenate([np.full(pad_len, np.nan), composite])
        else:
            return composite


class GeometricConsensusDetector(BaseRegimeDetector):
    """Consensus detector: persistence + voting across 4 geometric methods.

    Combines QCMLChernDetector, MultiScaleChernDetector, QFISusceptibilityDetector,
    and ScalarCurvatureDetector via a 5-step pipeline.

    Args:
        hilbert_dim: Hilbert space dimension.
        n_pca_components: PCA dimensions before geometry fitting.
        n_curvature_components: PCA dimensions for scalar curvature.
        operator_method: Method for fitting Hermitian operators.
        min_persistence: Consecutive days a sub-detector must flag.
        min_agreement: Minimum sub-detectors that must agree (of 4).
        z_threshold: Individual method z-score threshold.
        min_expanding: Minimum expanding window for z-scoring.
        rolling_window: Rolling window for smoothing sub-detector scores.
        seed: Random seed.
    """

    def __init__(
        self,
        hilbert_dim: int = 8,
        n_pca_components: int = 8,
        n_curvature_components: int = 8,
        operator_method: str = 'random',
        min_persistence: int = 3,
        min_agreement: int = 2,
        z_threshold: float = 1.5,
        min_expanding: int = 60,
        rolling_window: int = 20,
        seed: int = 42,
    ):
        self.hilbert_dim = hilbert_dim
        self.n_pca_components = n_pca_components
        self.n_curvature_components = n_curvature_components
        self.operator_method = operator_method
        self.min_persistence = min_persistence
        self.min_agreement = min_agreement
        self.z_threshold = z_threshold
        self.min_expanding = min_expanding
        self.rolling_window = rolling_window
        self.seed = seed
        self._sub_detectors: Optional[List] = None

    @property
    def name(self) -> str:
        return "Geometric Consensus"

    def fit(self, X: np.ndarray, **kwargs) -> 'GeometricConsensusDetector':
        self._sub_detectors = [
            QCMLChernDetector(
                hilbert_dim=self.hilbert_dim,
                window_size=self.rolling_window,
                n_pca_components=self.n_pca_components,
                operator_method=self.operator_method,
                seed=self.seed,
            ),
            MultiScaleChernDetector(
                hilbert_dim=self.hilbert_dim,
                n_pca_components=self.n_pca_components,
                scales=[10, 20, 30, 50],
                consensus_threshold=0.3,
                normalization_strategy='percentile',
                operator_method=self.operator_method,
                seed=self.seed,
            ),
            QFISusceptibilityDetector(
                hilbert_dim=self.hilbert_dim,
                n_pca_components=self.n_pca_components,
                operator_method=self.operator_method,
                rolling_window=self.rolling_window,
                min_expanding=self.min_expanding,
                seed=self.seed,
            ),
            ScalarCurvatureDetector(
                hilbert_dim=self.hilbert_dim,
                n_curvature_components=self.n_curvature_components,
                operator_method=self.operator_method,
                rolling_window=self.rolling_window,
                min_expanding=self.min_expanding,
                seed=self.seed,
            ),
        ]
        for det in self._sub_detectors:
            det.fit(X)
        return self

    @staticmethod
    def _apply_persistence(
        detected_mask: np.ndarray, min_persistence: int = 3,
    ) -> np.ndarray:
        """Keep only runs of consecutive True values >= min_persistence."""
        out = np.zeros_like(detected_mask, dtype=bool)
        n = len(detected_mask)
        run_start = None
        for i in range(n):
            if detected_mask[i]:
                if run_start is None:
                    run_start = i
            else:
                if run_start is not None:
                    if i - run_start >= min_persistence:
                        out[run_start:i] = True
                    run_start = None
        if run_start is not None and n - run_start >= min_persistence:
            out[run_start:n] = True
        return out

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._sub_detectors is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        T = len(X)

        raw_scores_list = [det.compute_regime_scores(X) for det in self._sub_detectors]

        z_scores_list = []
        for raw in raw_scores_list:
            z = np.full(T, np.nan)
            for t in range(self.min_expanding, T):
                past = raw[:t]
                past_valid = past[~np.isnan(past)]
                if len(past_valid) < 10:
                    continue
                mu = np.mean(past_valid)
                sigma = np.std(past_valid, ddof=1)
                if sigma > 1e-12 and not np.isnan(raw[t]):
                    z[t] = (raw[t] - mu) / sigma
            z_scores_list.append(z)

        flags_list = [z > self.z_threshold for z in z_scores_list]

        persisted_list = [
            self._apply_persistence(flags, self.min_persistence)
            for flags in flags_list
        ]

        agreement_count = np.zeros(T)
        z_sum = np.zeros(T)
        for i in range(len(self._sub_detectors)):
            agreement_count += persisted_list[i].astype(float)
            z_contribution = np.where(
                persisted_list[i] & ~np.isnan(z_scores_list[i]),
                z_scores_list[i],
                0.0,
            )
            z_sum += z_contribution

        final_scores = np.zeros(T)
        vote_mask = agreement_count >= self.min_agreement
        final_scores[vote_mask] = z_sum[vote_mask] / agreement_count[vote_mask]

        return final_scores


class FastGeometricConsensusDetector(GeometricConsensusDetector):
    """Fast detector optimized for sudden regime transitions."""

    def __init__(self, **kwargs):
        kwargs.setdefault('rolling_window', 15)
        kwargs.setdefault('z_threshold', 2.0)
        kwargs.setdefault('min_persistence', 3)
        kwargs.setdefault('min_agreement', 2)
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "Fast Consensus"


class SlowGeometricConsensusDetector(GeometricConsensusDetector):
    """Slow detector optimized for gradual regime transitions."""

    def __init__(self, **kwargs):
        kwargs.setdefault('rolling_window', 40)
        kwargs.setdefault('z_threshold', 1.2)
        kwargs.setdefault('min_persistence', 2)
        kwargs.setdefault('min_agreement', 1)
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "Slow Consensus"


class ShockMagnitudeDetector(BaseRegimeDetector):
    """Shock detector for V-shaped crises (2020 COVID).

    Uses QFI susceptibility for shock detection with short window
    and high threshold. Zeros out recovery periods.

    Args:
        hilbert_dim: Hilbert space dimension.
        n_pca_components: PCA dimensions before geometry fitting.
        operator_method: Method for fitting Hermitian operators.
        shock_window: Window for shock detection (default: 10 days).
        shock_threshold: Z-score threshold for shocks (default: 2.5).
        recovery_window: Days to zero out after shock (default: 10 days).
        min_expanding: Minimum expanding window for z-scoring.
        seed: Random seed.
    """

    def __init__(
        self,
        hilbert_dim: int = 8,
        n_pca_components: int = 8,
        operator_method: str = 'random',
        shock_window: int = 10,
        shock_threshold: float = 2.5,
        recovery_window: int = 10,
        min_expanding: int = 60,
        seed: int = 42,
    ):
        self.hilbert_dim = hilbert_dim
        self.n_pca_components = n_pca_components
        self.operator_method = operator_method
        self.shock_window = shock_window
        self.shock_threshold = shock_threshold
        self.recovery_window = recovery_window
        self.min_expanding = min_expanding
        self.seed = seed
        self._qfi_detector = None

    @property
    def name(self) -> str:
        return "Shock Magnitude"

    def fit(self, X: np.ndarray, **kwargs) -> 'ShockMagnitudeDetector':
        self._qfi_detector = QFISusceptibilityDetector(
            hilbert_dim=self.hilbert_dim,
            n_pca_components=self.n_pca_components,
            operator_method=self.operator_method,
            rolling_window=self.shock_window,
            min_expanding=self.min_expanding,
            seed=self.seed,
        )
        self._qfi_detector.fit(X)
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._qfi_detector is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        scores = self._qfi_detector.compute_regime_scores(X)
        T = len(scores)

        shock_mask = scores > self.shock_threshold
        for i in np.where(shock_mask)[0]:
            end_idx = min(i + 1 + self.recovery_window, T)
            scores[i+1:end_idx] = 0.0

        return scores


class MetricConditionNumberDetector(BaseRegimeDetector):
    """Regime detection via condition number of the metric tensor.

    kappa(g) = lambda_max / lambda_min. High condition number = anisotropic
    manifold = regime transition. Score = log(kappa) smoothed and z-scored.
    """

    def __init__(
        self,
        hilbert_dim: int = 8,
        n_pca_components: int = 15,
        operator_method: str = 'random',
        rolling_window: int = 20,
        min_expanding: int = 60,
        seed: int = 42,
    ):
        self.hilbert_dim = hilbert_dim
        self.n_pca_components = n_pca_components
        self.operator_method = operator_method
        self.rolling_window = rolling_window
        self.min_expanding = min_expanding
        self.seed = seed
        self._geometry = None
        self._scaler = None
        self._pca = None

    @property
    def name(self) -> str:
        return "Metric Condition Number"

    def fit(self, X: np.ndarray, **kwargs) -> 'MetricConditionNumberDetector':
        from qcml_geometry import QCMLGeometry

        np.random.seed(self.seed)
        n_components = min(self.n_pca_components, X.shape[1])

        self._scaler = StandardScaler()
        self._scaler.fit(X)
        self._pca = PCA(n_components=n_components)
        X_scaled = self._scaler.transform(X)
        self._pca.fit(X_scaled)
        X_pca = self._pca.transform(X_scaled)
        X_pca = X_pca / (np.linalg.norm(X_pca, axis=1, keepdims=True) + 1e-8)

        self._geometry = QCMLGeometry(
            n_features=X_pca.shape[1], hilbert_dim=self.hilbert_dim
        )
        self._geometry.fit_operators(X_pca, method=self.operator_method)
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        import pandas as pd

        Xt = _transform_array(X, self._scaler, self._pca)
        T = len(Xt)

        raw_logkappa = np.empty(T)
        for t in range(T):
            g = self._geometry.quantum_metric(Xt[t])
            eigvals = np.linalg.eigvalsh(g)
            pos_eigvals = eigvals[eigvals > 1e-15]
            if len(pos_eigvals) >= 2:
                kappa = pos_eigvals[-1] / pos_eigvals[0]
            else:
                kappa = 1.0
            raw_logkappa[t] = np.log(kappa + 1e-30)

        rolling_kappa = (
            pd.Series(raw_logkappa)
            .rolling(window=self.rolling_window, min_periods=1)
            .mean()
            .values
        )

        z_scores = np.full(T, np.nan)
        for t in range(self.min_expanding, T):
            mu = np.mean(rolling_kappa[:t])
            sigma = np.std(rolling_kappa[:t], ddof=1)
            if sigma > 1e-12:
                z_scores[t] = (rolling_kappa[t] - mu) / sigma
            else:
                z_scores[t] = 0.0

        return z_scores
