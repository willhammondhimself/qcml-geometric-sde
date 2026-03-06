"""
Full Quantum Geometric Tensor (QGT) Detector for Financial Regime Detection

The quantum geometric tensor (QGT) is:

    Q_ab = <d_a psi|d_b psi> - <d_a psi|psi><psi|d_b psi>

Its real part is the Fubini-Study quantum metric g_ab, and its imaginary
part (divided by 2) is the Berry curvature F_ab / 2:

    Q_ab = g_ab + i * F_ab / 2

Existing QCML detectors use either g (QFIDeterminantDetector) or F
(BerryPhaseRateDetector) separately. This detector uses the full complex
matrix Q to extract observables that mix metric and curvature:

1. Off-diagonal coupling: sum_{a != b} |Q_ab|^2
   Captures feature-feature quantum geometric interactions.
   High coupling = the geometry couples together different feature directions,
   which is a signature of regime transitions where feature correlations
   reorganize.

2. QGT Frobenius norm: Tr(Q^dag Q) = ||g||_F^2 + ||F||_F^2 / 4
   Combines metric and curvature magnitudes.

3. |det(Q)| (complex determinant magnitude)
   Scales with the product of QGT singular values.

4. max eigenvalue of Q^dag Q
   Dominant geometric scale in parameter space.

Primary signal: off-diagonal coupling (item 1), z-scored with expanding window.
The other observables are computed and reported for diagnostics.

Reference: Provost & Vallee (1980), Zanardi et al. (2007)
"""

import logging
from typing import Optional, List

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import sys
import os

# Add project root to path for imports
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from qcml_geometry.core import QCMLGeometry
from qcml_geometry.observables import (
    BaseRegimeDetector,
    ExpandingWindowMixin,
    _apply_normalization,
    _transform_array,
    _transform_point,
)

logger = logging.getLogger(__name__)


def _expanding_zscore(raw_values: np.ndarray, rolling_window: int,
                      min_expanding: int, T: int, skip_nan_start: int = 0) -> np.ndarray:
    """Compute expanding-window z-score of rolling-mean values.

    Matches the pattern used by other observatory detectors.

    Args:
        raw_values: 1-D raw signal array (may contain NaN).
        rolling_window: Rolling mean window size.
        min_expanding: Minimum samples before z-scoring starts.
        T: Total time series length.
        skip_nan_start: Skip leading NaN samples.

    Returns:
        z_scores: 1-D array of z-scores, NaN before min_expanding.
    """
    import pandas as pd
    rolling_vals = (
        pd.Series(raw_values)
        .rolling(window=rolling_window, min_periods=1)
        .mean()
        .values
    )
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


def _standard_init(detector, **kwargs):
    """Shared __init__ logic for detector initialization."""
    for k, v in kwargs.items():
        setattr(detector, k, v)
    detector._geometry = None
    detector._scaler = None
    detector._pca = None
    detector._snapshots = None
    detector._train_norms = None
    detector._train_std = None
    detector._epsilon = 1e-5


def _standard_qcml_fit(detector, X: np.ndarray):
    """Shared fit logic for ExpandingWindowMixin + BaseRegimeDetector subclasses."""
    if detector.expanding_refit_interval is not None:
        return detector._fit_expanding(X)

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
        X_pca_raw, detector.normalization, detector._train_norms, detector._train_std,
    )

    if detector.adaptive_epsilon:
        detector._epsilon = 1e-3 * np.median(np.abs(X_pca_fit))
    else:
        detector._epsilon = 1e-5

    detector._geometry = QCMLGeometry(
        n_features=X_pca_fit.shape[1], hilbert_dim=detector.hilbert_dim
    )
    custom_ops = getattr(detector, 'custom_operators', None)
    if custom_ops is not None:
        detector._geometry.set_operators(custom_ops)
    else:
        detector._geometry.fit_operators(
            X_pca_fit, method=detector.operator_method,
            scale_exponent=detector.scale_exponent,
        )

    return detector


def compute_full_qgt(
    geometry: QCMLGeometry,
    x: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """Compute the full quantum geometric tensor Q_ab at point x.

    Q_ab = <d_a psi|d_b psi> - <d_a psi|psi><psi|d_b psi>
         = g_ab + i * F_ab / 2

    where g_ab is the quantum metric (real, symmetric, PSD) and F_ab is
    the Berry curvature tensor (real, antisymmetric).

    Args:
        geometry: Fitted QCMLGeometry instance.
        x: Feature vector of shape (n_features,).
        epsilon: Step size for numerical differentiation.

    Returns:
        Q: Complex QGT matrix of shape (n_features, n_features).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)

    psi = geometry.quasi_coherent_state(x)

    # Compute numerical derivatives of |psi> w.r.t. each parameter
    dpsi = np.zeros((n, geometry.hilbert_dim), dtype=np.complex128)
    for a in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[a] += epsilon
        x_minus[a] -= epsilon
        psi_plus = geometry.quasi_coherent_state(x_plus)
        psi_minus = geometry.quasi_coherent_state(x_minus)
        dpsi[a] = (psi_plus - psi_minus) / (2.0 * epsilon)

    # Q_ab = <d_a psi|d_b psi> - <d_a psi|psi><psi|d_b psi>
    # where <u|v> = u.conj() @ v (numpy vdot)
    Q = np.zeros((n, n), dtype=np.complex128)
    for a in range(n):
        overlap_a_psi = np.vdot(dpsi[a], psi)  # <d_a psi|psi>
        for b in range(n):
            inner_ab = np.vdot(dpsi[a], dpsi[b])          # <d_a psi|d_b psi>
            overlap_psi_b = np.vdot(psi, dpsi[b])          # <psi|d_b psi>
            Q[a, b] = inner_ab - overlap_a_psi * overlap_psi_b

    return Q


def compute_qgt_observables(Q: np.ndarray) -> dict:
    """Extract regime-detection observables from the full QGT matrix.

    Args:
        Q: Complex QGT matrix of shape (n, n).

    Returns:
        Dict containing:
            off_diagonal_coupling: sum_{a != b} |Q_ab|^2
                Feature-feature geometric interaction strength.
            frobenius_norm_sq: Tr(Q^dag Q) = ||g||_F^2 + (1/4)||F||_F^2
                Combined metric + curvature magnitude.
            det_magnitude: |det(Q)|
                Volume scaling of QGT.
            max_singular_value_sq: Largest eigenvalue of Q^dag Q
                Dominant geometric scale.
            metric_frob: ||Re(Q)||_F = ||g||_F
            curvature_frob: ||2*Im(Q)||_F = ||F||_F
    """
    n = Q.shape[0]
    abs_sq = np.abs(Q) ** 2

    # Off-diagonal coupling: entries where a != b
    off_diag_mask = ~np.eye(n, dtype=bool)
    off_diagonal_coupling = float(np.sum(abs_sq[off_diag_mask]))

    # Frobenius norm squared: Tr(Q^dag Q)
    QtQ = Q.conj().T @ Q
    frob_sq = float(np.real(np.trace(QtQ)))

    # Complex determinant magnitude
    try:
        det_magnitude = float(np.abs(np.linalg.det(Q)))
    except np.linalg.LinAlgError:
        det_magnitude = 0.0

    # Max eigenvalue of Q^dag Q (which equals max singular value squared of Q)
    eigvals_QtQ = np.linalg.eigvalsh(QtQ)
    max_sv_sq = float(max(np.max(eigvals_QtQ), 0.0))

    # Metric and curvature norms
    g = np.real(Q)           # real part = g_ab (quantum metric)
    F = 2.0 * np.imag(Q)    # imaginary part * 2 = F_ab (Berry curvature)
    metric_frob = float(np.linalg.norm(g, 'fro'))
    curvature_frob = float(np.linalg.norm(F, 'fro'))

    return {
        'off_diagonal_coupling': off_diagonal_coupling,
        'frobenius_norm_sq': frob_sq,
        'det_magnitude': det_magnitude,
        'max_singular_value_sq': max_sv_sq,
        'metric_frob': metric_frob,
        'curvature_frob': curvature_frob,
    }


class QGTFullDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via full Quantum Geometric Tensor (QGT) off-diagonal coupling.

    Uses the complete complex QGT Q_ab = g_ab + i*F_ab/2, extracting the
    off-diagonal coupling sum_{a!=b} |Q_ab|^2 as the primary regime signal.

    Off-diagonal coupling captures geometric interactions between different
    feature directions. During normal regimes, features evolve nearly
    independently (low coupling). During crises, features become coupled
    through their shared geometric structure (high off-diagonal coupling).

    Score = expanding z-score of rolling-mean off-diagonal coupling.

    Primary observable: off_diagonal_coupling = sum_{a != b} |Q_ab|^2
    """

    def __init__(
        self,
        hilbert_dim: int = 4,
        n_pca_components: int = 8,
        operator_method: str = 'random',
        scale_exponent: Optional[float] = None,
        rolling_window: int = 20,
        min_expanding: int = 60,
        seed: int = 42,
        causal_fit_length: Optional[int] = None,
        expanding_refit_interval: Optional[int] = None,
        normalization: str = 'soft',
        adaptive_epsilon: bool = True,
        custom_operators: Optional[List[np.ndarray]] = None,
        epsilon: float = 1e-5,
    ):
        """Initialize the QGT Full Detector.

        Args:
            hilbert_dim: Hilbert space dimension. Default 4 (2-qubit).
            n_pca_components: Number of PCA components to retain.
            operator_method: Operator construction method ('random' recommended).
            scale_exponent: PCA eigenvalue scaling exponent.
            rolling_window: Window for rolling mean smoothing.
            min_expanding: Minimum samples before z-scoring starts.
            seed: Random seed for reproducibility.
            causal_fit_length: Fit only on first N samples (None = all).
            expanding_refit_interval: Refit interval for expanding windows.
            normalization: Post-PCA normalization mode.
            adaptive_epsilon: Adapt epsilon to data scale.
            custom_operators: Override learned operators.
            epsilon: Step size for QGT numerical differentiation.
        """
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
        self.epsilon = epsilon

    @property
    def name(self) -> str:
        return "QGT Full (Off-Diagonal Coupling)"

    def fit(self, X: np.ndarray, **kwargs) -> 'QGTFullDetector':
        """Fit scaler, PCA, and QCML geometry from feature matrix.

        Args:
            X: Feature matrix of shape (T, n_features).

        Returns:
            self
        """
        return _standard_qcml_fit(self, X)

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute regime scores as z-scored QGT off-diagonal coupling.

        Args:
            X: Feature matrix of shape (T, n_features).

        Returns:
            scores: 1-D array of length T. Higher = more regime-like.
        """
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        Xt = _transform_array(
            X, self._scaler, self._pca,
            normalization=self.normalization,
            train_norms=self._train_norms,
            train_std=self._train_std,
        )
        T = len(Xt)
        raw_vals = np.full(T, np.nan)

        eps = self._epsilon if self.adaptive_epsilon else self.epsilon

        for t in range(T):
            if self._snapshots:
                geo, xt = self._transform_point_at(X[t], t)
            else:
                geo, xt = self._geometry, Xt[t]

            try:
                Q = compute_full_qgt(geo, xt, epsilon=eps)
                obs = compute_qgt_observables(Q)
                raw_vals[t] = obs['off_diagonal_coupling']
            except Exception as e:
                logger.warning(f"QGT computation failed at t={t}: {e}")
                raw_vals[t] = np.nan

        return _expanding_zscore(raw_vals, self.rolling_window, self.min_expanding, T)

    def compute_all_observables(self, X: np.ndarray) -> dict:
        """Compute all four QGT-derived observable time series.

        Returns four signals:
            off_diagonal_coupling, frobenius_norm_sq, det_magnitude,
            max_singular_value_sq

        Each is z-scored with an expanding window.

        Args:
            X: Feature matrix of shape (T, n_features).

        Returns:
            Dict mapping observable name -> z-scored 1-D array of length T.
        """
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_all_observables().")

        Xt = _transform_array(
            X, self._scaler, self._pca,
            normalization=self.normalization,
            train_norms=self._train_norms,
            train_std=self._train_std,
        )
        T = len(Xt)
        keys = [
            'off_diagonal_coupling', 'frobenius_norm_sq',
            'det_magnitude', 'max_singular_value_sq',
        ]
        raw = {k: np.full(T, np.nan) for k in keys}

        eps = self._epsilon if self.adaptive_epsilon else self.epsilon

        for t in range(T):
            if self._snapshots:
                geo, xt = self._transform_point_at(X[t], t)
            else:
                geo, xt = self._geometry, Xt[t]

            try:
                Q = compute_full_qgt(geo, xt, epsilon=eps)
                obs = compute_qgt_observables(Q)
                for k in keys:
                    raw[k][t] = obs[k]
            except Exception as e:
                logger.warning(f"QGT computation failed at t={t}: {e}")

        result = {}
        for k in keys:
            result[k] = _expanding_zscore(
                raw[k], self.rolling_window, self.min_expanding, T
            )
        return result
