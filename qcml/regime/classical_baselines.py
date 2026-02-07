"""
Classical Baseline Regime Detectors for Head-to-Head Comparison

Provides a common interface (BaseRegimeDetector) and 8 implementations:
1. QCMLChernDetector — wraps existing TopologicalRegimeDetector
2. RollingVolatilityDetector — rolling sigma z-scored against expanding window
3. CUSUMDetector — cumulative sum of mean-adjusted absolute returns
4. HMMRegimeDetector — 2-state Gaussian HMM via hmmlearn
5. RandomForestRegimeDetector — P(crisis) from sklearn RF
6. MultiScaleChernDetector — multi-scale Chern consensus (10-100 days)
7. QuantumEnsembleDetector — ensemble of all 4 quantum indicators
8. QFISusceptibilityDetector — QFI susceptibility tr(g_ab) from quantum metric

All detectors expose the same fit/compute_regime_scores interface so the
comparison experiment can apply identical statistical tests to each method's
output.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class BaseRegimeDetector(ABC):
    """Common interface for all regime detection methods."""

    @staticmethod
    def build_enriched_features(X: np.ndarray, lookback: int = 20) -> np.ndarray:
        """Build rolling features (mean, std, min, max) from raw X.

        Identical to RandomForestRegimeDetector._build_ml_features() so that
        QCML detectors receive the same rolling aggregation features that RF
        computes internally.  This is a pure data representation change — no
        labels are used.

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
        """Fit the detector to feature matrix X (T, n_features).

        Returns self for method chaining.
        """
        ...

    @abstractmethod
    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        """Produce a 1-D regime score time series of length T.

        Higher values should indicate a more 'stressed' or 'crisis-like'
        regime.  The exact scale differs by method; the comparison
        framework only cares about pre/post-crisis distributional shift.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for display in comparison tables."""
        ...


# ---------------------------------------------------------------------------
# 1. QCML Chern (wraps existing code)
# ---------------------------------------------------------------------------

class QCMLChernDetector(BaseRegimeDetector):
    """Regime detection via rolling Chern number from QCML geometry.

    Wraps TopologicalRegimeDetector.rolling_chern_number() with the same
    PCA / normalization pipeline used in rigorous_crisis_validation.py.

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

    @property
    def name(self) -> str:
        return "QCML Chern"

    def fit(self, X: np.ndarray, **kwargs) -> 'QCMLChernDetector':
        from qcml.qcml_geometry import QCMLGeometry
        from qcml.topological_regime import TopologicalRegimeDetector

        np.random.seed(self.seed)
        n_components = min(self.n_pca_components, X.shape[1])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        X_pca = X_pca / (np.linalg.norm(X_pca, axis=1, keepdims=True) + 1e-8)

        self._X_transformed = X_pca
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
        chern = self._detector.rolling_chern_number(
            self._X_transformed, window=self.window_size
        )
        # Pad front so output length == T
        pad = np.full(len(self._X_transformed) - len(chern), np.nan)
        return np.concatenate([pad, chern])


# ---------------------------------------------------------------------------
# 2. Rolling Volatility Z-score (pure numpy)
# ---------------------------------------------------------------------------

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
        # Stateless — nothing to fit.
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        # Use first principal component of returns as the "returns" proxy
        if X.ndim == 2 and X.shape[1] > 1:
            returns = np.diff(X[:, 0])
        else:
            returns = np.diff(X.ravel())

        T = len(returns)
        scores = np.full(T + 1, np.nan)  # +1 to match original X length

        # Rolling volatility
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


# ---------------------------------------------------------------------------
# 3. CUSUM (pure numpy)
# ---------------------------------------------------------------------------

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
        S = np.zeros(T + 1)  # +1 to match original X length
        for t in range(T):
            S[t + 1] = max(0.0, S[t] + np.abs(returns[t]) - self._k)

        return S


# ---------------------------------------------------------------------------
# 4. HMM 2-state Gaussian
# ---------------------------------------------------------------------------

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

        # Provide all initial parameters manually to avoid kmeans
        # initialization which can crash on certain sklearn/numpy combos.
        sorted_rets = np.sort(returns.ravel())
        n = len(sorted_rets)
        low_half = sorted_rets[:n // 2]
        high_half = sorted_rets[n // 2:]

        self._model = GaussianHMM(
            n_components=2,
            covariance_type='full',
            n_iter=self.n_iter,
            random_state=self.seed,
            init_params='',  # we set everything manually
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

        # Identify the high-vol state
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

        # Pad front so length == T
        return np.concatenate([[np.nan], high_vol_prob])


# ---------------------------------------------------------------------------
# 5. Random Forest
# ---------------------------------------------------------------------------

class RandomForestRegimeDetector(BaseRegimeDetector):
    """Random Forest classifier for regime detection.

    Produces P(crisis) from a trained RF.  Training requires labeled data
    via ``fit_with_labels``.  The standard ``fit()`` is a no-op (labels
    required separately).

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
        # Standard fit is a no-op; use fit_with_labels for supervised training.
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
        # Trim y to match feature length (lost lookback-1 rows)
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

        # Find the column index for the crisis class (1)
        crisis_col = list(self._model.classes_).index(1)
        scores = proba[:, crisis_col]

        # Pad front so length == T
        pad = np.full(len(X) - len(scores), np.nan)
        return np.concatenate([pad, scores])

    def _build_ml_features(self, X: np.ndarray) -> np.ndarray:
        """Build rolling features (mean, std, min, max) from raw X."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        T, d = X.shape
        n_out = T - self.lookback + 1
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


# ---------------------------------------------------------------------------
# 6. Multi-Scale Chern Consensus
# ---------------------------------------------------------------------------

class MultiScaleChernDetector(BaseRegimeDetector):
    """Multi-scale Chern consensus via BaseRegimeDetector interface.

    Wraps MultiScaleChernConsensus from quantum_indicators to provide
    multi-scale topological regime detection. Computes Chern numbers at
    multiple time scales and produces a weighted consensus signal.

    Windows: [10, 20, 30, 50] by default (100-day scale dropped as too noisy).

    Args:
        hilbert_dim: Hilbert space dimension.
        n_pca_components: PCA dimensions before geometry fitting.
        scales: Window sizes for multi-scale analysis.
        weights: Per-scale weights (default: equal). Must sum to 1 or will be normalized.
        consensus_threshold: Minimum weighted consensus for transition detection.
        normalization_strategy: Method for normalizing Chern changes across scales.
            Options: 'rolling_adaptive', 'rolling_fixed', 'percentile', 'zscore'.
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

    @property
    def name(self) -> str:
        return "Multi-Scale Chern"

    def fit(self, X: np.ndarray, **kwargs) -> 'MultiScaleChernDetector':
        from qcml.qcml_geometry import QCMLGeometry
        from qcml.regime.quantum_indicators import MultiScaleChernConsensus

        np.random.seed(self.seed)
        n_components = min(self.n_pca_components, X.shape[1])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        X_pca = X_pca / (np.linalg.norm(X_pca, axis=1, keepdims=True) + 1e-8)

        self._X_transformed = X_pca
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

        # Use the same transformed data from fit
        result = self._consensus.compute_consensus(self._X_transformed)
        consensus_scores = result.values

        # Pad front to match length T
        pad_len = len(X) - len(consensus_scores)
        if pad_len > 0:
            return np.concatenate([np.full(pad_len, np.nan), consensus_scores])
        else:
            return consensus_scores


# ---------------------------------------------------------------------------
# 7. Quantum Ensemble (all 4 indicators)
# ---------------------------------------------------------------------------

class QuantumEnsembleDetector(BaseRegimeDetector):
    """Ensemble of all quantum indicators.

    Combines spectral gap, ground state energy, fidelity decay, and
    multi-scale Chern consensus into a single composite score using
    QuantumIndicatorSuite.compute_composite_score().

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

    @property
    def name(self) -> str:
        return "Quantum Ensemble"

    def fit(self, X: np.ndarray, **kwargs) -> 'QuantumEnsembleDetector':
        from qcml.qcml_geometry import QCMLGeometry
        from qcml.regime.quantum_indicators import QuantumIndicatorSuite

        np.random.seed(self.seed)
        n_components = min(self.n_pca_components, X.shape[1])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        X_pca = X_pca / (np.linalg.norm(X_pca, axis=1, keepdims=True) + 1e-8)

        self._X_transformed = X_pca
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

        # Use the same transformed data from fit
        composite, _ = self._suite.compute_composite_score(self._X_transformed)

        # Pad front to match length T
        pad_len = len(X) - len(composite)
        if pad_len > 0:
            return np.concatenate([np.full(pad_len, np.nan), composite])
        else:
            return composite


# ---------------------------------------------------------------------------
# 8. QFI Susceptibility — realized quantum metric via Fubini-Study distance
# ---------------------------------------------------------------------------

class QFISusceptibilityDetector(BaseRegimeDetector):
    """Regime detection via Quantum Fisher Information susceptibility.

    Measures the realized QFI through the Fubini-Study distance between
    consecutive quantum states:

        d_FS(t) = arccos(|⟨ψ(x_t)|ψ(x_{t+1})⟩|)

    This equals √(g_ab Δx^a Δx^b) — the quantum metric tensor contracted
    with the actual market displacement.  During regime transitions the
    quantum state evolves rapidly, producing large d_FS.

    A rolling mean smooths the raw distance series, then z-scoring against
    an expanding window yields the final regime score.

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
        self._X_transformed = None

    @property
    def name(self) -> str:
        return "QFI Susceptibility"

    def fit(self, X: np.ndarray, **kwargs) -> 'QFISusceptibilityDetector':
        from qcml.qcml_geometry import QCMLGeometry

        np.random.seed(self.seed)
        n_components = min(self.n_pca_components, X.shape[1])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        X_pca = X_pca / (np.linalg.norm(X_pca, axis=1, keepdims=True) + 1e-8)

        self._X_transformed = X_pca
        self._geometry = QCMLGeometry(
            n_features=X_pca.shape[1], hilbert_dim=self.hilbert_dim
        )
        self._geometry.fit_operators(X_pca, method=self.operator_method)
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        Xt = self._X_transformed
        T = len(Xt)

        # Compute Fubini-Study distance between consecutive states
        raw_dist = np.empty(T - 1)
        for t in range(T - 1):
            raw_dist[t] = self._geometry.quantum_distance(Xt[t], Xt[t + 1])

        # Rolling mean to smooth
        import pandas as pd
        rolling_dist = (
            pd.Series(raw_dist)
            .rolling(window=self.rolling_window, min_periods=1)
            .mean()
            .values
        )

        # Z-score against expanding window
        n = len(rolling_dist)
        z_scores = np.full(n, np.nan)
        for t in range(self.min_expanding, n):
            mu = np.mean(rolling_dist[:t])
            sigma = np.std(rolling_dist[:t], ddof=1)
            if sigma > 1e-12:
                z_scores[t] = (rolling_dist[t] - mu) / sigma
            else:
                z_scores[t] = 0.0

        # Pad front so output length == T (lost 1 from diff, add 1 NaN)
        return np.concatenate([[np.nan], z_scores])


class ScalarCurvatureDetector(BaseRegimeDetector):
    """Ricci scalar curvature of the quantum metric manifold.

    Computes R(t) = g^{mu nu} R_{mu nu} at each time point, where R_{mu nu}
    is the Ricci tensor derived from the Levi-Civita connection of the
    quantum metric.  Large |R| indicates regions where the data manifold
    is highly curved (nonlinear, unstable dynamics).

    The score is |R(t)| z-scored against an expanding window, so that
    both positive and negative curvature anomalies are flagged.

    Args:
        hilbert_dim: Hilbert space dimension.
        n_curvature_components: PCA dimensions (kept small — Ricci is O(n^2) in metric evals).
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
        self._X_transformed = None

    @property
    def name(self) -> str:
        return "Scalar Curvature"

    def fit(self, X: np.ndarray, **kwargs) -> 'ScalarCurvatureDetector':
        from qcml.qcml_geometry import QCMLGeometry

        np.random.seed(self.seed)
        n_components = min(self.n_curvature_components, X.shape[1])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        X_pca = X_pca / (np.linalg.norm(X_pca, axis=1, keepdims=True) + 1e-8)

        self._X_transformed = X_pca
        self._geometry = QCMLGeometry(
            n_features=X_pca.shape[1], hilbert_dim=self.hilbert_dim
        )
        self._geometry.fit_operators(X_pca, method=self.operator_method)
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        Xt = self._X_transformed
        T = len(Xt)

        # Compute signed R(t) at each time point (both directions indicate anomaly)
        raw_R = np.empty(T)
        for t in range(T):
            raw_R[t] = self._geometry.ricci_scalar(Xt[t])
            self._geometry.clear_cache()  # prevent memory blowup

        # Rolling mean to smooth the signed series
        import pandas as pd
        rolling_R = (
            pd.Series(raw_R)
            .rolling(window=self.rolling_window, min_periods=1)
            .mean()
            .values
        )

        # Z-score the signed series against expanding window, then take abs(z)
        # Both positive and negative curvature anomalies indicate regime transitions
        z_scores = np.full(T, np.nan)
        for t in range(self.min_expanding, T):
            mu = np.mean(rolling_R[:t])
            sigma = np.std(rolling_R[:t], ddof=1)
            if sigma > 1e-12:
                z_scores[t] = abs((rolling_R[t] - mu) / sigma)
            else:
                z_scores[t] = 0.0

        return z_scores


# ---------------------------------------------------------------------------
# 9. Geometric Consensus — persistence + voting across geometric methods
# ---------------------------------------------------------------------------

class GeometricConsensusDetector(BaseRegimeDetector):
    """Consensus detector: persistence + voting across 4 geometric methods.

    Combines QCMLChernDetector, MultiScaleChernDetector, QFISusceptibilityDetector,
    and ScalarCurvatureDetector via a 5-step pipeline:

      1. Get raw scores from each sub-detector.
      2. Z-score each via expanding window (causal, no look-ahead).
      3. Flag where z > ``z_threshold`` (permissive individually).
      4. Apply persistence filter per method (``min_persistence`` consecutive days).
      5. Voting: final score = mean z-score of agreeing methods where
         >= ``min_agreement`` agree; 0.0 otherwise.

    Design rationale:
      - z_threshold=1.5 is permissive because persistence + voting do the
        heavy lifting to eliminate false positives.
      - min_persistence=3 filters single-day noise without missing fast
        crises (e.g., COVID).
      - min_agreement=2 of 4 requires independent corroboration without
        being so strict it kills recall.
      - scales=[10,20,30,50] — dropped 100-day scale for runtime when
        nested inside consensus.

    Args:
        hilbert_dim: Hilbert space dimension.
        n_pca_components: PCA dimensions before geometry fitting.
        n_curvature_components: PCA dimensions for scalar curvature (kept small).
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

        # Step 1: Get raw scores
        raw_scores_list = [det.compute_regime_scores(X) for det in self._sub_detectors]

        # Step 2: Z-score each via expanding window (causal)
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

        # Step 3: Flag where z > z_threshold
        flags_list = [z > self.z_threshold for z in z_scores_list]

        # Step 4: Apply persistence filter per method
        persisted_list = [
            self._apply_persistence(flags, self.min_persistence)
            for flags in flags_list
        ]

        # Step 5: Voting
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


# ---------------------------------------------------------------------------
# 10. Fast/Slow/Shock Specialized Detectors for Adaptive Ensemble
# ---------------------------------------------------------------------------

class FastGeometricConsensusDetector(GeometricConsensusDetector):
    """Fast detector optimized for sudden regime transitions (2008, 2010, 2011, 2015).

    Uses shorter windows and higher thresholds for precision on rapid crashes.

    Args:
        Same as GeometricConsensusDetector, but with default params tuned for
        sudden crises.
    """

    def __init__(self, **kwargs):
        # Override defaults for fast detection
        kwargs.setdefault('rolling_window', 15)       # Short window
        kwargs.setdefault('z_threshold', 2.0)         # High threshold (precise)
        kwargs.setdefault('min_persistence', 3)       # Standard persistence
        kwargs.setdefault('min_agreement', 2)         # 2/4 sub-detectors
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "Fast Consensus"


class SlowGeometricConsensusDetector(GeometricConsensusDetector):
    """Slow detector optimized for gradual regime transitions (2018, 2022).

    Uses longer windows and lower thresholds for sensitivity to slow trends.

    Args:
        Same as GeometricConsensusDetector, but with default params tuned for
        gradual crises.
    """

    def __init__(self, **kwargs):
        # Override defaults for slow detection
        kwargs.setdefault('rolling_window', 40)       # Long window
        kwargs.setdefault('z_threshold', 1.2)         # Low threshold (sensitive)
        kwargs.setdefault('min_persistence', 2)       # Short persistence
        kwargs.setdefault('min_agreement', 1)         # 1/4 is enough
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "Slow Consensus"


class ShockMagnitudeDetector(BaseRegimeDetector):
    """Shock detector for V-shaped crises (2020 COVID).

    Detects V-shaped crises by focusing on shock magnitude only.
    Ignores recovery period to avoid false negatives.

    Uses QFI susceptibility for shock detection with:
    - Short window (10 days) for rapid shock detection
    - High threshold (2.5σ) for shock magnitude
    - Recovery period zeroing (next 10 days after spike)

    Args:
        hilbert_dim: Hilbert space dimension.
        n_pca_components: PCA dimensions before geometry fitting.
        operator_method: Method for fitting Hermitian operators.
        shock_window: Window for shock detection (default: 10 days)
        shock_threshold: Z-score threshold for shocks (default: 2.5)
        recovery_window: Days to zero out after shock (default: 10 days)
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
        # Use QFI susceptibility detector as base
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

        # Get QFI susceptibility scores
        scores = self._qfi_detector.compute_regime_scores(X)
        T = len(scores)

        # Zero out recovery periods (next recovery_window days after spike)
        shock_mask = scores > self.shock_threshold
        for i in np.where(shock_mask)[0]:
            # Zero out next recovery_window days
            end_idx = min(i + 1 + self.recovery_window, T)
            scores[i+1:end_idx] = 0.0

        return scores


# ---------------------------------------------------------------------------
# 11-14. New Quantum Feature Detectors
# ---------------------------------------------------------------------------

class QFIDeterminantDetector(BaseRegimeDetector):
    """Regime detection via quantum metric determinant det(g_ab).

    The determinant of the quantum metric tensor is the volume element of
    the data manifold. Collapsing volume indicates approaching a phase
    boundary; diverging volume indicates rapid expansion of accessible
    state space. Both extremes signal regime transitions.

    Score = abs(z-score of log(|det(g)|)) smoothed over a rolling window.
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
        self._X_transformed = None

    @property
    def name(self) -> str:
        return "QFI Determinant"

    def fit(self, X: np.ndarray, **kwargs) -> 'QFIDeterminantDetector':
        from qcml.qcml_geometry import QCMLGeometry

        np.random.seed(self.seed)
        n_components = min(self.n_pca_components, X.shape[1])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        X_pca = X_pca / (np.linalg.norm(X_pca, axis=1, keepdims=True) + 1e-8)

        self._X_transformed = X_pca
        self._geometry = QCMLGeometry(
            n_features=X_pca.shape[1], hilbert_dim=self.hilbert_dim
        )
        self._geometry.fit_operators(X_pca, method=self.operator_method)
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        Xt = self._X_transformed
        T = len(Xt)

        import pandas as pd

        # Compute log-pseudodeterminant using eigenvalues
        log_pseudodet = np.empty(T)
        eigenvalue_tolerance = 1e-10  # Filter near-zero eigenvalues

        for t in range(T):
            # Get quantum metric tensor
            g_ij = self._geometry.quantum_metric(Xt[t])

            # Compute eigenvalues and filter near-zero ones
            eigenvalues = np.linalg.eigvalsh(g_ij)
            nonzero_eigs = eigenvalues[eigenvalues > eigenvalue_tolerance]

            # Pseudo-determinant: sum of log of non-zero eigenvalues
            if len(nonzero_eigs) > 0:
                log_pseudodet[t] = np.sum(np.log(nonzero_eigs))
            else:
                # Fallback: all eigenvalues near zero
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


class BerryPhaseRateDetector(BaseRegimeDetector):
    """Regime detection via rate of change of Berry curvature.

    Measures topological transition speed: rapid changes in Berry curvature
    indicate the system is crossing a phase boundary.

    Score = abs(diff(Berry_curvature)) smoothed and z-scored.
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
        self._X_transformed = None

    @property
    def name(self) -> str:
        return "Berry Phase Rate"

    def fit(self, X: np.ndarray, **kwargs) -> 'BerryPhaseRateDetector':
        from qcml.qcml_geometry import QCMLGeometry

        np.random.seed(self.seed)
        n_components = min(self.n_pca_components, X.shape[1])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        X_pca = X_pca / (np.linalg.norm(X_pca, axis=1, keepdims=True) + 1e-8)

        self._X_transformed = X_pca
        self._geometry = QCMLGeometry(
            n_features=X_pca.shape[1], hilbert_dim=self.hilbert_dim
        )
        self._geometry.fit_operators(X_pca, method=self.operator_method)
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        Xt = self._X_transformed
        T = len(Xt)

        import pandas as pd

        berry = np.empty(T)
        for t in range(T):
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


class MultiLagFidelityDetector(BaseRegimeDetector):
    """Regime detection via multi-lag quantum fidelity.

    Fidelity at lags [1, 3, 5, 10] provides multi-scale sensitivity:
    short lags detect fast crises, long lags detect gradual transitions.
    Score = weighted average infidelity (1-F), z-scored.
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
    ):
        self.hilbert_dim = hilbert_dim
        self.n_pca_components = n_pca_components
        self.operator_method = operator_method
        self.lags = lags or [1, 3, 5, 10]
        self.lag_weights = lag_weights or [0.4, 0.3, 0.2, 0.1]
        self.rolling_window = rolling_window
        self.min_expanding = min_expanding
        self.seed = seed
        self._geometry = None
        self._X_transformed = None

    @property
    def name(self) -> str:
        return "Multi-Lag Fidelity"

    def fit(self, X: np.ndarray, **kwargs) -> 'MultiLagFidelityDetector':
        from qcml.qcml_geometry import QCMLGeometry

        np.random.seed(self.seed)
        n_components = min(self.n_pca_components, X.shape[1])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        X_pca = X_pca / (np.linalg.norm(X_pca, axis=1, keepdims=True) + 1e-8)

        self._X_transformed = X_pca
        self._geometry = QCMLGeometry(
            n_features=X_pca.shape[1], hilbert_dim=self.hilbert_dim
        )
        self._geometry.fit_operators(X_pca, method=self.operator_method)
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


class MetricConditionNumberDetector(BaseRegimeDetector):
    """Regime detection via condition number of the quantum metric tensor.

    kappa(g) = lambda_max / lambda_min of the quantum metric tensor.
    High condition number = anisotropic manifold = regime transition.
    Score = log(kappa) smoothed and z-scored.
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
        self._X_transformed = None

    @property
    def name(self) -> str:
        return "Metric Condition Number"

    def fit(self, X: np.ndarray, **kwargs) -> 'MetricConditionNumberDetector':
        from qcml.qcml_geometry import QCMLGeometry

        np.random.seed(self.seed)
        n_components = min(self.n_pca_components, X.shape[1])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        X_pca = X_pca / (np.linalg.norm(X_pca, axis=1, keepdims=True) + 1e-8)

        self._X_transformed = X_pca
        self._geometry = QCMLGeometry(
            n_features=X_pca.shape[1], hilbert_dim=self.hilbert_dim
        )
        self._geometry.fit_operators(X_pca, method=self.operator_method)
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        Xt = self._X_transformed
        T = len(Xt)

        import pandas as pd

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
