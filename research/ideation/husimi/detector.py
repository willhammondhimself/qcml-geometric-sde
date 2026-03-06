"""
Husimi Q-function Detector for Financial Regime Detection

Implements a Husimi Q-function inspired regime signal using phase-space
quasi-probability projections onto random unit vectors (coherent-state
analogues in finite-dimensional Hilbert space).

For the QCML ground state |psi> in C^d, the standard Husimi Q-function
Q(alpha) = <alpha|rho|alpha> / pi for continuous coherent states |alpha>
is not directly available (no Heisenberg-Weyl group structure for d=4).
Instead we use the discrete analogue:

    Q_k = |<v_k|psi>|^2        k = 1, ..., M

where |v_k> are M i.i.d. Haar-random unit vectors in C^d. This recovers
the SIC-POVM / frame-theoretic generalisation of the Husimi function in
finite dimensions (Zauner 1999, Appleby 2005).

Two signals are derived from the empirical Q distribution:

1. Wehrl entropy (discrete):
       S_W = -sum_k Q_k * log(Q_k + eps)     (nats, normalised by log M)

   Higher entropy  =>  phase-space delocalization  =>  regime transition.

2. Inverse Participation Ratio (IPR):
       IPR = 1 / sum_k Q_k^2

   Higher IPR  =>  flatter, more delocalized state  =>  same signal.

Both are z-scored in an expanding window. The final score uses Wehrl
entropy by default (empirically more stable than IPR).

References:
- Husimi (1940) Proc. Phys. Math. Soc. Japan
- Wehrl (1978) Rev. Mod. Phys.
- Zauner (1999) PhD thesis; Appleby (2005) J. Math. Phys.
"""

import logging
import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Project root injection
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from qcml_geometry.core import QCMLGeometry
from qcml_geometry.observables import (
    BaseRegimeDetector,
    ExpandingWindowMixin,
    _apply_normalization,
    _transform_array,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Expanding z-score helper (identical pattern to other ideation detectors)
# ---------------------------------------------------------------------------

def _expanding_zscore(
    raw_values: np.ndarray,
    rolling_window: int,
    min_expanding: int,
    T: int,
    skip_nan_start: int = 0,
) -> np.ndarray:
    """Compute expanding-window z-score of rolling-mean values.

    Args:
        raw_values: 1-D array of length T, may contain NaN.
        rolling_window: Rolling mean window size.
        min_expanding: Minimum past observations before z-scoring begins.
        T: Total time series length.
        skip_nan_start: Skip leading NaN entries when building the expanding window.

    Returns:
        z_scores: 1-D array of length T. NaN for warm-up period.
    """
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


# ---------------------------------------------------------------------------
# Shared init / fit helpers (mirrors loschmidt_echo / otoc pattern)
# ---------------------------------------------------------------------------

def _standard_init(detector, **kwargs) -> None:
    """Shared __init__ logic for QCML-based detector initialization."""
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
    """Shared fit logic for QCML-based detectors.

    Applies StandardScaler -> PCA -> QCMLGeometry.fit_operators.
    Respects causal_fit_length if set; otherwise fits on full dataset.

    Args:
        detector: Detector instance with standard QCML attributes.
        X: Feature matrix of shape (T, n_features).

    Returns:
        detector (fitted in-place).
    """
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
        X_pca_raw, detector.normalization,
        detector._train_norms, detector._train_std,
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
            X_pca_fit,
            method=detector.operator_method,
            scale_exponent=detector.scale_exponent,
        )

    return detector


# ---------------------------------------------------------------------------
# Husimi core computation
# ---------------------------------------------------------------------------

def _sample_haar_random_unit_vectors(
    dim: int,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample Haar-random unit vectors from C^dim.

    A Haar-random unit vector is obtained by normalizing a complex Gaussian
    vector. This ensures uniformity over the complex projective space CP^{d-1}.

    Args:
        dim: Hilbert space dimension d.
        n_samples: Number of unit vectors M to draw.
        rng: NumPy random Generator (for reproducibility).

    Returns:
        probes: Complex array of shape (n_samples, dim), each row unit-norm.
    """
    # Real + imaginary parts i.i.d. N(0,1) gives Haar-uniform direction
    real_part = rng.standard_normal((n_samples, dim))
    imag_part = rng.standard_normal((n_samples, dim))
    probes = real_part + 1j * imag_part
    norms = np.linalg.norm(probes, axis=1, keepdims=True)
    return probes / (norms + 1e-30)


def compute_husimi_observables(
    psi: np.ndarray,
    probe_vectors: np.ndarray,
    eps: float = 1e-30,
) -> dict:
    """Compute Husimi Q-function observables for a single quantum state.

    Given the ground state |psi> and M probe vectors |v_k>, computes:
        Q_k = |<v_k|psi>|^2     (Husimi quasi-probability; sums to 1 by Parseval)

    Then derives:
        Wehrl entropy:  S_W = -sum_k Q_k * log(Q_k + eps)  (normalised by log M)
        IPR:            IPR  = 1 / sum_k Q_k^2             (participation ratio)

    Note: For M >> d (over-complete frame), sum_k Q_k = M/d * d = M * (1/d) * d = M/d
    correction. We normalise Q_k -> Q_k / sum(Q_k) so the distribution always
    sums to 1 regardless of M.

    Args:
        psi: Ground-state vector of shape (d,), complex, assumed unit-norm.
        probe_vectors: Array of shape (M, d), each row a Haar-random unit vector.
        eps: Small constant for log stability.

    Returns:
        dict with keys:
            'wehrl_entropy': float, normalised Wehrl entropy in [0, 1].
            'ipr': float, inverse participation ratio in [1, M].
            'q_dist': np.ndarray of shape (M,), normalised Q distribution.
    """
    psi = psi / (np.linalg.norm(psi) + 1e-30)

    # Projections: Q_k = |<v_k|psi>|^2 — shape (M,)
    overlaps = probe_vectors.conj() @ psi        # (M,) complex
    q_raw = np.abs(overlaps) ** 2                # (M,) real, non-negative

    # Normalise so distribution sums to 1 (frame-corrected)
    q_sum = q_raw.sum()
    if q_sum < 1e-30:
        # Degenerate: return maximally uniform values (shouldn't occur)
        M = len(q_raw)
        q = np.full(M, 1.0 / M)
    else:
        q = q_raw / q_sum

    M = len(q)
    log_max = np.log(M) if M > 1 else 1.0

    # Wehrl entropy (discrete), normalised to [0, 1]
    wehrl_entropy = -np.sum(q * np.log(q + eps)) / log_max

    # Inverse Participation Ratio: 1 / sum(Q_k^2), normalised to [1, M]
    ipr_raw = 1.0 / (np.sum(q ** 2) + eps)

    return {
        'wehrl_entropy': float(wehrl_entropy),
        'ipr': float(ipr_raw),
        'q_dist': q,
    }


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class HusimiQDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via Husimi Q-function phase-space delocalization.

    Computes phase-space quasi-probability Q_k = |<v_k|psi>|^2 for M
    Haar-random probe vectors |v_k> against the QCML ground state |psi(x_t)>.
    Two signals are derived:

    - Wehrl entropy S_W = -sum_k Q_k log Q_k  (normalised by log M)
    - Inverse Participation Ratio IPR = 1 / sum_k Q_k^2

    Higher values in both signals indicate phase-space delocalization,
    which is the signature of a quantum state undergoing a transition (the
    Q-function spreads across phase space near a critical point).

    By default the Wehrl entropy is used as the regime signal. Set
    ``signal='ipr'`` to use the IPR instead.

    Score = z-score of rolling-mean Wehrl entropy (or IPR).

    Reference: Husimi (1940), Wehrl (1978), Appleby (2005).

    Attributes:
        n_probes: Number of Haar-random probe vectors M.
        signal: Which observable to use as the regime score ('wehrl' or 'ipr').
        probe_vectors: Pre-sampled probe array of shape (n_probes, hilbert_dim).
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
        n_probes: int = 512,
        signal: str = 'wehrl',
    ):
        """Initialize Husimi Q-function Detector.

        Args:
            hilbert_dim: Hilbert space dimension (default 4, avoids Kramers degeneracy).
            n_pca_components: Number of PCA components for feature reduction.
            operator_method: Operator construction method ('random' recommended).
            scale_exponent: PCA eigenvalue scaling exponent (None = default 0.5).
            rolling_window: Rolling mean window size for smoothing.
            min_expanding: Minimum expanding window observations before z-scoring.
            seed: Random seed for reproducibility (controls probe vectors).
            causal_fit_length: Fit only on first N samples (None = all).
            expanding_refit_interval: Refit interval for expanding windows (None = static).
            normalization: Post-PCA normalization mode ('soft', 'sphere', 'none', 'clip').
            adaptive_epsilon: Adapt numerical epsilon to data scale.
            custom_operators: Override learned QCML operators with custom list.
            n_probes: Number of Haar-random probe vectors M (default 512).
                      Should be >> hilbert_dim for good phase-space coverage.
            signal: Observable to use as regime score: 'wehrl' (Wehrl entropy)
                    or 'ipr' (inverse participation ratio). Default: 'wehrl'.
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

        if signal not in ('wehrl', 'ipr'):
            raise ValueError(f"signal must be 'wehrl' or 'ipr', got '{signal}'")

        self.n_probes = n_probes
        self.signal = signal

        # Pre-sample probe vectors at construction time — seeded for reproducibility.
        # Probe vectors are fixed across all time steps so the Q-function is computed
        # in the same "basis" at every t, making the time series comparable.
        rng = np.random.default_rng(seed)
        self.probe_vectors = _sample_haar_random_unit_vectors(hilbert_dim, n_probes, rng)

    @property
    def name(self) -> str:
        return f"Husimi Q ({self.signal})"

    def fit(self, X: np.ndarray, **kwargs) -> 'HusimiQDetector':
        """Fit scaler, PCA, and QCML geometry from feature matrix.

        Args:
            X: Feature matrix of shape (T, n_features).

        Returns:
            self
        """
        return _standard_qcml_fit(self, X)

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute regime scores as z-scored Husimi Q-function entropy / IPR.

        For each time step t:
        1. Transform x_t through scaler + PCA + normalization.
        2. Compute QCML ground state |psi(x_t)> from error Hamiltonian.
        3. Project |psi(x_t)> onto M fixed probe vectors to get Q distribution.
        4. Compute Wehrl entropy or IPR of that distribution.

        Raw values are smoothed with a rolling mean then z-scored in an
        expanding window to yield the final regime score.

        Args:
            X: Feature matrix of shape (T, n_features).

        Returns:
            scores: 1-D array of length T. Higher = more phase-space delocalized
                    = more regime-like. NaN for warm-up period.
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
        vals = np.full(T, np.nan)

        for t in range(T):
            if self._snapshots:
                geo, xt = self._transform_point_at(X[t], t)
            else:
                geo, xt = self._geometry, Xt[t]

            try:
                # Obtain ground state |psi(x_t)>
                psi = geo.quasi_coherent_state(xt)

                # Compute Husimi observables
                husimi = compute_husimi_observables(
                    psi=psi,
                    probe_vectors=self.probe_vectors,
                )

                if self.signal == 'wehrl':
                    vals[t] = husimi['wehrl_entropy']
                else:
                    vals[t] = husimi['ipr']

            except Exception as e:
                logger.warning(f"Husimi Q computation failed at t={t}: {e}")
                vals[t] = np.nan

        return _expanding_zscore(vals, self.rolling_window, self.min_expanding, T)
