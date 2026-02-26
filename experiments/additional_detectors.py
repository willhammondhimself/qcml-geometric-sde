"""
Additional QCML-based regime detectors for the comparison pipeline.

These wrap the core qcml_geometry library to provide detectors that use
Chern numbers, multi-scale consensus, and ensemble voting.
"""

import logging
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from qcml_geometry.core import QCMLGeometry
from qcml_geometry.observables import BaseRegimeDetector, _apply_normalization

logger = logging.getLogger(__name__)


class QCMLChernDetector(BaseRegimeDetector):
    """Regime detection via rolling Chern number (curvature integral).

    Computes Berry curvature over sliding windows and aggregates into
    a regime score. Elevated curvature integrals indicate transitions.
    """

    def __init__(
        self,
        hilbert_dim: int = 8,
        window_size: int = 20,
        n_pca_components: int = 15,
        operator_method: str = 'pca_inspired',
        seed: int = 42,
        causal_fit_length: Optional[int] = None,
        normalization: str = 'sphere',
    ):
        self.hilbert_dim = hilbert_dim
        self.window_size = window_size
        self.n_pca_components = n_pca_components
        self.operator_method = operator_method
        self.seed = seed
        self.causal_fit_length = causal_fit_length
        self.normalization = normalization
        self._geometry = None
        self._scaler = None
        self._pca = None
        self._train_norms = None
        self._train_std = None

    @property
    def name(self) -> str:
        return "QCML Chern"

    def fit(self, X: np.ndarray, **kwargs) -> 'QCMLChernDetector':
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
            X_pca_raw, self.normalization, self._train_norms, self._train_std,
        )

        self._geometry = QCMLGeometry(
            n_features=X_pca_fit.shape[1], hilbert_dim=self.hilbert_dim
        )
        self._geometry.fit_operators(X_pca_fit, method=self.operator_method)
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        X_scaled = self._scaler.transform(X)
        X_pca_raw = self._pca.transform(X_scaled)
        X_pca = _apply_normalization(
            X_pca_raw, self.normalization, self._train_norms, self._train_std,
        )

        T = len(X_pca)
        scores = np.full(T, np.nan)

        for t in range(self.window_size, T):
            # Compute Berry curvature for points in the window
            curvatures = []
            for i in range(max(0, t - self.window_size), t):
                F_01 = self._geometry.berry_curvature_2d(X_pca[i], indices=(0, 1))
                curvatures.append(abs(F_01))

            if curvatures:
                scores[t] = np.mean(curvatures)

        return scores


class GeometricConsensusDetector(BaseRegimeDetector):
    """Consensus detector combining Berry curvature and metric determinant.

    Fires only when BOTH Berry curvature rate and metric determinant
    z-scores exceed their respective thresholds, reducing false alarms.
    """

    def __init__(
        self,
        hilbert_dim: int = 8,
        n_pca_components: int = 15,
        n_curvature_components: int = 5,
        operator_method: str = 'pca_inspired',
        rolling_window: int = 20,
        min_expanding: int = 60,
        seed: int = 42,
        threshold_quantile: float = 0.9,
        causal_fit_length: Optional[int] = None,
        normalization: str = 'sphere',
    ):
        self.hilbert_dim = hilbert_dim
        self.n_pca_components = n_pca_components
        self.n_curvature_components = n_curvature_components
        self.operator_method = operator_method
        self.rolling_window = rolling_window
        self.min_expanding = min_expanding
        self.seed = seed
        self.threshold_quantile = threshold_quantile
        self.causal_fit_length = causal_fit_length
        self.normalization = normalization
        self._geometry = None
        self._scaler = None
        self._pca = None
        self._train_norms = None
        self._train_std = None

    @property
    def name(self) -> str:
        return "Geometric Consensus"

    def fit(self, X: np.ndarray, **kwargs) -> 'GeometricConsensusDetector':
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
            X_pca_raw, self.normalization, self._train_norms, self._train_std,
        )

        self._geometry = QCMLGeometry(
            n_features=X_pca_fit.shape[1], hilbert_dim=self.hilbert_dim
        )
        self._geometry.fit_operators(X_pca_fit, method=self.operator_method)
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        import pandas as pd

        X_scaled = self._scaler.transform(X)
        X_pca_raw = self._pca.transform(X_scaled)
        X_pca = _apply_normalization(
            X_pca_raw, self.normalization, self._train_norms, self._train_std,
        )

        T = len(X_pca)

        # Compute Berry curvature rate
        berry = np.empty(T)
        for t in range(T):
            berry[t] = self._geometry.berry_curvature_2d(X_pca[t], indices=(0, 1))
        berry_rate = np.abs(np.diff(berry))
        berry_rate = np.concatenate([[0], berry_rate])

        # Compute metric determinant
        log_det = np.empty(T)
        for t in range(T):
            g = self._geometry.quantum_metric(X_pca[t])
            eigs = np.linalg.eigvalsh(g)
            pos_eigs = eigs[eigs > 1e-10]
            log_det[t] = np.sum(np.log(pos_eigs)) if len(pos_eigs) > 0 else -20.0

        # Z-score both with expanding window
        def expanding_zscore(vals):
            smoothed = pd.Series(vals).rolling(self.rolling_window, min_periods=1).mean().values
            z = np.full(T, np.nan)
            for t in range(self.min_expanding, T):
                mu = np.mean(smoothed[:t])
                sigma = np.std(smoothed[:t], ddof=1)
                if sigma > 1e-12:
                    z[t] = abs((smoothed[t] - mu) / sigma)
                else:
                    z[t] = 0.0
            return z

        z_berry = expanding_zscore(berry_rate)
        z_det = expanding_zscore(log_det)

        # Consensus: geometric mean of z-scores, zero if either is below threshold
        valid_mask = ~np.isnan(z_berry) & ~np.isnan(z_det)
        berry_thresh = np.nanpercentile(z_berry[valid_mask], self.threshold_quantile * 100) if np.any(valid_mask) else 1.5
        det_thresh = np.nanpercentile(z_det[valid_mask], self.threshold_quantile * 100) if np.any(valid_mask) else 1.5

        scores = np.zeros(T)
        for t in range(T):
            if valid_mask[t] and z_berry[t] > berry_thresh and z_det[t] > det_thresh:
                scores[t] = np.sqrt(z_berry[t] * z_det[t])

        return scores
