"""
Full-Manifold Chern Number Detector

Computes an aggregate Chern-like topological invariant by summing absolute
Berry curvature across all 2D planes of the PCA-reduced feature space.

Background
----------
The Chern number is the integral of Berry curvature F_ab over a closed 2D
manifold: C = (1/2pi) * integral F_ab da db.  For a d-dimensional parameter
space there are C(d,2) = d*(d-1)/2 independent 2D planes.  Rather than
integrating over a pre-defined grid (which requires a closed surface), we use
the pointwise absolute Berry curvature ||F||_1 = sum_{a<b} |F_ab| as a
detector signal.  This "topological charge density" spikes when the state
|psi(x)> is near a degeneracy (a place where the Berry curvature diverges),
which is precisely what happens during market regime transitions.

Method
------
1. Fit PCA (n_components) and StandardScaler on training data.
2. For each time step t, compute the full Berry curvature tensor F (n x n)
   via finite differences of the quasi-coherent state.
3. Aggregate: total_curvature(t) = sum_{a<b} |F_ab(t)|
4. Z-score against an expanding window of past values to get a regime score.

This follows the BaseRegimeDetector interface from qcml_geometry/observables.py.
"""

import logging
from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from qcml_geometry.core import QCMLGeometry
from qcml_geometry.observables import (
    BaseRegimeDetector,
    ExpandingWindowMixin,
    _apply_normalization,
    _transform_array,
    _transform_point,
)

logger = logging.getLogger(__name__)


class FullManifoldChernDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via aggregate Berry curvature over the full parameter manifold.

    Computes sum_{a<b} |F_ab(x)| at each time step — the sum of absolute Berry
    curvature across all 2D planes of the PCA feature space.  This captures
    total topological charge density without requiring a predefined closed surface.

    For a d-dimensional PCA space, this sums C(d,2) curvature components.
    Near a spectral degeneracy (market regime transition), Berry curvature
    diverges, so the signal spikes ahead of and during crises.

    Score = abs(z-score of rolling-smoothed total_curvature), computed with
    an expanding window to avoid look-ahead.

    Args:
        hilbert_dim: Hilbert space dimension (default 4 for 2-qubit system).
            Use 4 or 8.  Higher dims are slower but richer.
        n_pca_components: Number of PCA dimensions. C(n,2) = n*(n-1)/2 planes
            are summed.  4-8 gives 6-28 planes; >8 is slow.
        operator_method: Method for constructing Hermitian operators.
            'random' avoids Kramers degeneracy that 'pca_inspired' can trigger
            on qubit systems.
        scale_exponent: PCA eigenvalue scaling exponent for pca_* methods.
        rolling_window: Days to smooth raw curvature values.
        min_expanding: Minimum observations before z-scoring begins.
        seed: Random seed for reproducibility.
        causal_fit_length: If set, fit scaler/PCA on first N rows only.
        expanding_refit_interval: If set, periodically refit on expanding window.
        normalization: Post-PCA normalization mode ('sphere', 'none', 'soft', 'clip').
        epsilon: Step size for Berry curvature finite differences.
        adaptive_epsilon: If True, set epsilon = 1e-3 * median(|x_pca|).
    """

    def __init__(
        self,
        hilbert_dim: int = 4,
        n_pca_components: int = 6,
        operator_method: str = 'random',
        scale_exponent: Optional[float] = None,
        rolling_window: int = 20,
        min_expanding: int = 60,
        seed: int = 42,
        causal_fit_length: Optional[int] = None,
        expanding_refit_interval: Optional[int] = None,
        normalization: str = 'sphere',
        epsilon: float = 1e-4,
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
        self.epsilon = epsilon
        self.adaptive_epsilon = adaptive_epsilon
        self.custom_operators = custom_operators

        self._geometry: Optional[QCMLGeometry] = None
        self._scaler: Optional[StandardScaler] = None
        self._pca: Optional[PCA] = None
        self._snapshots = None
        self._train_norms: Optional[np.ndarray] = None
        self._train_std: Optional[np.ndarray] = None
        self._epsilon_used: float = epsilon

    @property
    def name(self) -> str:
        return "Full Manifold Chern"

    def fit(self, X: np.ndarray, **kwargs) -> 'FullManifoldChernDetector':
        """Fit scaler, PCA, and QCML operators on X.

        Args:
            X: Feature matrix (T, n_raw_features).

        Returns:
            self
        """
        if self.expanding_refit_interval is not None:
            return self._fit_expanding(X)

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

        if self.adaptive_epsilon:
            self._epsilon_used = 1e-3 * float(np.median(np.abs(X_pca_fit)))
            self._epsilon_used = max(self._epsilon_used, 1e-6)
        else:
            self._epsilon_used = self.epsilon

        self._geometry = QCMLGeometry(
            n_features=X_pca_fit.shape[1], hilbert_dim=self.hilbert_dim
        )
        if self.custom_operators is not None:
            self._geometry.set_operators(self.custom_operators)
        else:
            self._geometry.fit_operators(
                X_pca_fit,
                method=self.operator_method,
                scale_exponent=self.scale_exponent,
            )

        logger.info(
            "FullManifoldChernDetector fitted: "
            f"pca_dim={n_components}, "
            f"n_planes={n_components*(n_components-1)//2}, "
            f"epsilon={self._epsilon_used:.2e}"
        )
        return self

    def _compute_total_curvature(self, x_pca: np.ndarray, geometry: QCMLGeometry) -> float:
        """Compute aggregate absolute Berry curvature across all 2D planes.

        For a d-dimensional point, computes F_ab for all a < b using
        QCMLGeometry.berry_curvature(), then returns sum_{a<b} |F_ab|.

        This is the "topological charge density" at point x_pca — it
        integrates (in the distributional sense) the Berry curvature
        over all 2D subspaces simultaneously.

        Args:
            x_pca: PCA-transformed, normalized feature vector (n_components,).
            geometry: Fitted QCMLGeometry instance.

        Returns:
            total_curvature: sum_{a<b} |F_ab| (non-negative).
        """
        F = geometry.berry_curvature(x_pca, epsilon=self._epsilon_used)
        # F is antisymmetric; sum upper triangle
        n = F.shape[0]
        total = 0.0
        for a in range(n):
            for b in range(a + 1, n):
                total += abs(F[a, b])
        return total

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute z-scored log(aggregate Berry curvature) time series.

        The raw sum_{a<b} |F_ab| is heavy-tailed due to near-degeneracy events.
        We apply log(1 + curv) to compress the dynamic range, roll-smooth,
        then use a MAD-based robust z-score to prevent outlier normal points
        from swamping the crisis signal.

        Args:
            X: Feature matrix (T, n_raw_features). Same feature set as fit().

        Returns:
            z_scores: (T,) array. NaN for t < min_expanding.
                Large positive values indicate regime stress.
        """
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        T = X.shape[0]

        # Compute raw curvature values at each time step
        curvature_vals = np.empty(T)
        for t in range(T):
            if self._snapshots is not None:
                geo, x_t = self._transform_point_at(X[t], t)
            else:
                x_t = _transform_point(
                    X[t], self._scaler, self._pca,
                    normalization=self.normalization,
                    train_norms=self._train_norms,
                    train_std=self._train_std,
                )
                geo = self._geometry

            try:
                curvature_vals[t] = self._compute_total_curvature(x_t, geo)
            except Exception as exc:
                logger.warning(f"t={t}: curvature computation failed ({exc}), using 0.")
                curvature_vals[t] = 0.0

        # Log-compress (heavy-tailed signal): log(1 + sum|F_ab|)
        log_curv = np.log1p(curvature_vals)

        # Rolling smoothing
        rolling_curv = (
            pd.Series(log_curv)
            .rolling(window=self.rolling_window, min_periods=1)
            .mean()
            .values
        )

        # Expanding-window robust z-score using MAD (causal, no look-ahead).
        # MAD is more robust than std to the spikey baseline typical of
        # Berry curvature in normal markets.
        z_scores = np.full(T, np.nan)
        for t in range(self.min_expanding, T):
            past = rolling_curv[:t]
            mu = np.median(past)
            mad = np.median(np.abs(past - mu))
            sigma = 1.4826 * mad  # consistent estimator of std for Gaussian
            if sigma > 1e-12:
                z_scores[t] = abs((rolling_curv[t] - mu) / sigma)
            else:
                z_scores[t] = 0.0

        return z_scores
