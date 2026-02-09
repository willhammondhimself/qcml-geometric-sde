"""
Fused Multi-Observable QCML Regime Detectors

Phase A: FusedQCMLDetector — combines the top 3 independent QCML detectors
(Berry Phase Rate, QFI Determinant, Multi-Lag Fidelity) via z-score fusion.

Phase B: GeometryOptimizedDetector — single optimized geometry with 5 fused
observables extracted from the same manifold, with Optuna-tuned operator
scale factors.

Both detectors implement the BaseRegimeDetector interface for plug-in
compatibility with the evaluation pipeline in regime_comparison.py.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from qcml.regime.classical_baselines import (
    BaseRegimeDetector,
    BerryPhaseRateDetector,
    QFIDeterminantDetector,
    MultiLagFidelityDetector,
)

logger = logging.getLogger(__name__)


class FusedQCMLDetector(BaseRegimeDetector):
    """Fuse N QCML detectors via z-score combination.

    Combines Berry Phase Rate, QFI Determinant, and Multi-Lag Fidelity —
    three largely independent geometric observables (Spearman rho 0.18–0.46)
    that dominate on complementary crises.

    Fusion strategies:
        'max': score(t) = max(z1(t), z2(t), z3(t)) — strongest signal wins.
        'weighted_mean': score(t) = Σ w_i * z_i(t) — optimized weights.
        'rank_mean': rank each method independently, average ranks, z-score.

    Args:
        hilbert_dim: Hilbert space dimension for sub-detectors.
        n_pca_components: PCA dimensions for sub-detectors.
        operator_method: Method for fitting Hermitian operators.
        seed: Random seed for reproducibility.
        fusion_method: 'max', 'weighted_mean', or 'rank_mean'.
        weights: Weights for weighted_mean fusion (berry, qfi, multilag).
        min_expanding: Minimum expanding window before z-scoring starts.
        rolling_window: Rolling window for sub-detectors.
    """

    def __init__(
        self,
        hilbert_dim: int = 8,
        n_pca_components: int = 15,
        operator_method: str = 'random',
        seed: int = 42,
        fusion_method: str = 'max',
        weights: Optional[List[float]] = None,
        min_expanding: int = 60,
        rolling_window: int = 20,
    ):
        self.hilbert_dim = hilbert_dim
        self.n_pca_components = n_pca_components
        self.operator_method = operator_method
        self.seed = seed
        self.fusion_method = fusion_method
        self.weights = weights or [1.0 / 3, 1.0 / 3, 1.0 / 3]
        self.min_expanding = min_expanding
        self.rolling_window = rolling_window
        self._sub_detectors: Optional[List[BaseRegimeDetector]] = None

    @property
    def name(self) -> str:
        return f"Fused QCML ({self.fusion_method})"

    def fit(self, X: np.ndarray, **kwargs) -> 'FusedQCMLDetector':
        self._sub_detectors = [
            BerryPhaseRateDetector(
                hilbert_dim=self.hilbert_dim,
                n_pca_components=self.n_pca_components,
                operator_method=self.operator_method,
                rolling_window=self.rolling_window,
                min_expanding=self.min_expanding,
                seed=self.seed,
            ),
            QFIDeterminantDetector(
                hilbert_dim=self.hilbert_dim,
                n_pca_components=self.n_pca_components,
                operator_method=self.operator_method,
                rolling_window=self.rolling_window,
                min_expanding=self.min_expanding,
                seed=self.seed,
            ),
            MultiLagFidelityDetector(
                hilbert_dim=self.hilbert_dim,
                n_pca_components=self.n_pca_components,
                operator_method=self.operator_method,
                rolling_window=self.rolling_window,
                min_expanding=self.min_expanding,
                seed=self.seed,
            ),
        ]
        for det in self._sub_detectors:
            det.fit(X)
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._sub_detectors is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        T = len(X)

        # Step 1: Get raw scores from each sub-detector
        raw_scores = [det.compute_regime_scores(X) for det in self._sub_detectors]

        if self.fusion_method == 'rank_mean':
            return self._fuse_rank_mean(raw_scores, T)

        # Step 2: Expanding z-score each (causal, no lookahead)
        z_scores = []
        for raw in raw_scores:
            z = self._expanding_zscore(raw, self.min_expanding)
            z_scores.append(z)

        # Step 3: Fuse
        if self.fusion_method == 'max':
            return self._fuse_max(z_scores, T)
        elif self.fusion_method == 'weighted_mean':
            return self._fuse_weighted_mean(z_scores, T)
        else:
            raise ValueError(f"Unknown fusion_method: {self.fusion_method}")

    @staticmethod
    def _expanding_zscore(raw: np.ndarray, min_expanding: int) -> np.ndarray:
        """Causal expanding z-score with no lookahead."""
        T = len(raw)
        z = np.full(T, np.nan)
        for t in range(min_expanding, T):
            past = raw[:t]
            past_valid = past[~np.isnan(past)]
            if len(past_valid) < 10:
                continue
            mu = np.mean(past_valid)
            sigma = np.std(past_valid, ddof=1)
            if sigma > 1e-12 and not np.isnan(raw[t]):
                z[t] = (raw[t] - mu) / sigma
        return z

    def _fuse_max(self, z_scores: List[np.ndarray], T: int) -> np.ndarray:
        """Max fusion: take strongest signal at each timestep."""
        stacked = np.column_stack(z_scores)  # (T, 3)
        # Replace NaN with -inf for max
        stacked_safe = np.where(np.isnan(stacked), -np.inf, stacked)
        result = np.max(stacked_safe, axis=1)
        # Where all are NaN, output NaN
        all_nan = np.all(np.isnan(stacked), axis=1)
        result[all_nan] = np.nan
        return result

    def _fuse_weighted_mean(self, z_scores: List[np.ndarray], T: int) -> np.ndarray:
        """Weighted mean fusion with normalized weights."""
        w = np.array(self.weights)
        w = w / w.sum()  # normalize
        stacked = np.column_stack(z_scores)  # (T, 3)
        # Weighted mean, treating NaN as 0 contribution
        result = np.full(T, np.nan)
        for t in range(T):
            vals = stacked[t]
            valid = ~np.isnan(vals)
            if valid.any():
                w_valid = w[valid]
                w_valid = w_valid / w_valid.sum()
                result[t] = np.dot(w_valid, vals[valid])
        return result

    def _fuse_rank_mean(self, raw_scores: List[np.ndarray], T: int) -> np.ndarray:
        """Rank fusion: rank each method independently, average ranks, z-score."""
        from scipy.stats import rankdata

        ranked = []
        for raw in raw_scores:
            # Replace NaN with median for ranking purposes
            filled = raw.copy()
            valid = ~np.isnan(filled)
            if valid.any():
                filled[~valid] = np.nanmedian(filled)
            r = rankdata(filled, method='average') / len(filled)
            # Set NaN positions back to NaN
            r[~valid] = np.nan
            ranked.append(r)

        stacked = np.column_stack(ranked)  # (T, 3)
        mean_rank = np.nanmean(stacked, axis=1)

        # Z-score the mean rank series
        return self._expanding_zscore(mean_rank, self.min_expanding)


class GeometryOptimizedDetector(BaseRegimeDetector):
    """Single geometry with optimized operators, 5 fused observables.

    Optimizes the Hermitian operator scale factors (diagonal metric on
    operator space) via Optuna TPE, then extracts 5 independent geometric
    observables from the same geometry and fuses them with learned weights.

    Observables:
        1. Berry curvature rate of change
        2. QFI determinant (log)
        3. Multi-lag fidelity (weighted, lags=[1,3,5,10])
        4. Spectral gap of error Hamiltonian
        5. Metric condition number (log kappa)

    Args:
        hilbert_dim: Hilbert space dimension.
        n_pca_components: PCA dimensions.
        operator_scales: Scale factors for each PCA-basis operator.
        fusion_weights: Weights for combining 5 observables.
        rolling_window: Window for smoothing raw observables.
        min_expanding: Minimum expanding window for z-scoring.
        seed: Random seed.
    """

    def __init__(
        self,
        hilbert_dim: int = 8,
        n_pca_components: int = 8,
        operator_scales: Optional[np.ndarray] = None,
        fusion_weights: Optional[np.ndarray] = None,
        rolling_window: int = 20,
        min_expanding: int = 60,
        seed: int = 42,
    ):
        self.hilbert_dim = hilbert_dim
        self.n_pca_components = n_pca_components
        self.operator_scales = operator_scales
        self.fusion_weights = fusion_weights if fusion_weights is not None else np.ones(5) / 5
        self.rolling_window = rolling_window
        self.min_expanding = min_expanding
        self.seed = seed
        self._geometry = None
        self._X_transformed = None

    @property
    def name(self) -> str:
        return "Geometry Optimized"

    def fit(self, X: np.ndarray, **kwargs) -> 'GeometryOptimizedDetector':
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
        # Fit base operators using pca_inspired method
        self._geometry.fit_operators(X_pca, method='pca_inspired')

        # Apply operator scale factors if provided
        if self.operator_scales is not None:
            scaled_ops = []
            for k, op in enumerate(self._geometry.operators):
                if k < len(self.operator_scales):
                    scaled_ops.append(op * self.operator_scales[k])
                else:
                    scaled_ops.append(op)
            self._geometry.set_operators(scaled_ops)

        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        Xt = self._X_transformed
        T = len(Xt)

        # Extract 5 observables from the same geometry
        obs1 = self._compute_berry_rate(Xt, T)
        obs2 = self._compute_qfi_det(Xt, T)
        obs3 = self._compute_multilag_fidelity(Xt, T)
        obs4 = self._compute_spectral_gap(Xt, T)
        obs5 = self._compute_condition_number(Xt, T)

        # Z-score each observable
        observables = [obs1, obs2, obs3, obs4, obs5]
        z_scores = [
            FusedQCMLDetector._expanding_zscore(obs, self.min_expanding)
            for obs in observables
        ]

        # Fuse with weights
        w = np.array(self.fusion_weights)
        w = w / w.sum()

        stacked = np.column_stack(z_scores)  # (T, 5)
        result = np.full(T, np.nan)
        for t in range(T):
            vals = stacked[t]
            valid = ~np.isnan(vals)
            if valid.any():
                w_valid = w[valid]
                w_valid = w_valid / w_valid.sum()
                result[t] = np.dot(w_valid, vals[valid])

        return result

    def _compute_berry_rate(self, Xt: np.ndarray, T: int) -> np.ndarray:
        """Berry curvature rate of change."""
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
        return np.concatenate([[np.nan], rolling_rate])

    def _compute_qfi_det(self, Xt: np.ndarray, T: int) -> np.ndarray:
        """Log QFI determinant."""
        log_det = np.empty(T)
        eigenvalue_tolerance = 1e-10

        for t in range(T):
            g_ij = self._geometry.quantum_metric(Xt[t])
            eigenvalues = np.linalg.eigvalsh(g_ij)
            nonzero_eigs = eigenvalues[eigenvalues > eigenvalue_tolerance]

            if len(nonzero_eigs) > 0:
                log_det[t] = np.sum(np.log(nonzero_eigs))
            else:
                log_det[t] = np.log(eigenvalue_tolerance) * len(eigenvalues)

        return (
            pd.Series(log_det)
            .rolling(window=self.rolling_window, min_periods=1)
            .mean()
            .values
        )

    def _compute_multilag_fidelity(self, Xt: np.ndarray, T: int) -> np.ndarray:
        """Multi-lag fidelity (weighted, lags=[1,3,5,10])."""
        lags = [1, 3, 5, 10]
        lag_weights = [0.4, 0.3, 0.2, 0.1]
        max_lag = max(lags)

        states = [self._geometry.quasi_coherent_state(Xt[t]) for t in range(T)]

        combined = np.full(T, np.nan)
        for t in range(max_lag, T):
            weighted_infidelity = 0.0
            for lag, w in zip(lags, lag_weights):
                if t >= lag:
                    overlap = np.abs(np.vdot(states[t], states[t - lag]))
                    fidelity = overlap ** 2
                    weighted_infidelity += w * (1.0 - fidelity)
            combined[t] = weighted_infidelity

        return (
            pd.Series(combined)
            .rolling(window=self.rolling_window, min_periods=1)
            .mean()
            .values
        )

    def _compute_spectral_gap(self, Xt: np.ndarray, T: int) -> np.ndarray:
        """Spectral gap of error Hamiltonian (inverse — small gap = transition)."""
        gaps = np.empty(T)
        for t in range(T):
            gaps[t] = self._geometry.spectral_gap(Xt[t])

        # Inverse: small gap means transition, so regime score = 1/gap
        inv_gap = 1.0 / (gaps + 1e-12)
        return (
            pd.Series(inv_gap)
            .rolling(window=self.rolling_window, min_periods=1)
            .mean()
            .values
        )

    def _compute_condition_number(self, Xt: np.ndarray, T: int) -> np.ndarray:
        """Log condition number of quantum metric tensor."""
        log_kappa = np.empty(T)
        for t in range(T):
            g = self._geometry.quantum_metric(Xt[t])
            eigvals = np.linalg.eigvalsh(g)
            pos_eigvals = eigvals[eigvals > 1e-15]
            if len(pos_eigvals) >= 2:
                kappa = pos_eigvals[-1] / pos_eigvals[0]
            else:
                kappa = 1.0
            log_kappa[t] = np.log(kappa + 1e-30)

        return (
            pd.Series(log_kappa)
            .rolling(window=self.rolling_window, min_periods=1)
            .mean()
            .values
        )
