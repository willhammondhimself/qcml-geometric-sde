"""
Out-of-Time-Order Correlator (OTOC) Detector for Financial Regime Detection

Measures quantum information scrambling via:
    F(t) = <psi|W(t)^dagger V^dagger W(t) V|psi>

where W(t) = e^{iHt} W e^{-iHt} is the Heisenberg-evolved operator,
H = H(x) is the QCML error Hamiltonian at feature vector x, and
|psi> is the ground state of H.

Rapid OTOC decay (negative slope of log F vs t) indicates chaotic dynamics,
which in the financial context corresponds to information scrambling during
regime transitions and crises.

The scrambling rate (negative slope of log F vs tau) is used as the raw
signal; it is then z-scored in an expanding window for regime detection.

Reference: Larkin & Ovchinnikov (1969), Kitaev (2015), Maldacena et al. (2016)
"""

import logging
import os
import sys
from typing import List, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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


# ---------------------------------------------------------------------------
# Shared helpers (same pattern as loschmidt_echo/detector.py)
# ---------------------------------------------------------------------------

def _expanding_zscore(raw_values, rolling_window, min_expanding, T,
                      skip_nan_start=0):
    """Compute expanding-window z-score of rolling-mean values.

    Args:
        raw_values: 1-D array of raw signal values (length T, may contain NaN).
        rolling_window: Rolling mean window size.
        min_expanding: Minimum number of past observations before scoring.
        T: Total time series length.
        skip_nan_start: Skip initial NaN entries when building expanding window.

    Returns:
        z_scores: 1-D array of length T with NaN for warm-up period.
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

    Applies StandardScaler + PCA + QCMLGeometry.fit_operators.
    Uses causal_fit_length if set; otherwise fits on all data.
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
            X_pca_fit, method=detector.operator_method,
            scale_exponent=detector.scale_exponent,
        )

    return detector


# ---------------------------------------------------------------------------
# Core OTOC computation
# ---------------------------------------------------------------------------

def _random_hermitian(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a random Hermitian matrix of dimension dim.

    Args:
        dim: Matrix dimension.
        rng: NumPy random Generator.

    Returns:
        Hermitian matrix of shape (dim, dim), dtype complex128.
    """
    A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    return (A + A.conj().T) / 2.0


def compute_otoc_scrambling_rate(
    geometry: QCMLGeometry,
    x: np.ndarray,
    W: np.ndarray,
    V: np.ndarray,
    tau_values: Optional[np.ndarray] = None,
) -> float:
    """Compute OTOC scrambling rate for a single feature vector.

    The OTOC is defined as:
        F(tau) = <psi| W(tau)^dagger V^dagger W(tau) V |psi>

    where W(tau) = e^{iH*tau} W e^{-iH*tau} is the Heisenberg-evolved
    operator, H = H(x) is the QCML error Hamiltonian, and |psi> is the
    ground state of H.

    The scrambling rate is the negative slope of log|F(tau)| vs tau.
    A positive rate indicates faster decay (more scrambling), which is
    associated with chaotic/crisis dynamics.

    The Heisenberg evolution is computed efficiently using the spectral
    decomposition of H rather than matrix exponentials:
        exp(+/- iH*tau) = U diag(exp(+/- i*lambda*tau)) U^dagger

    Args:
        geometry: Fitted QCMLGeometry instance.
        x: Feature vector of shape (n_features,).
        W: Fixed Hermitian operator of shape (hilbert_dim, hilbert_dim).
        V: Fixed Hermitian operator of shape (hilbert_dim, hilbert_dim).
        tau_values: Array of time evolution durations. Default: linspace(0.1, 5.0, 5).

    Returns:
        scrambling_rate: Non-negative float. Higher = faster OTOC decay = more scrambling.
    """
    if tau_values is None:
        tau_values = np.linspace(0.1, 5.0, 5)

    x = np.asarray(x, dtype=float)

    H = geometry.error_hamiltonian(x)
    evals, evecs = np.linalg.eigh(H)

    # Ground state
    psi = evecs[:, 0]
    psi = psi / (np.linalg.norm(psi) + 1e-15)

    otoc_values = np.empty(len(tau_values))
    for i, tau in enumerate(tau_values):
        # Heisenberg evolution via spectral decomposition:
        # exp(+iH*tau) = evecs @ diag(exp(+i*evals*tau)) @ evecs^dagger
        phases_pos = np.exp(1j * evals * tau)
        phases_neg = np.exp(-1j * evals * tau)

        exp_pos = evecs @ np.diag(phases_pos) @ evecs.conj().T
        exp_neg = evecs @ np.diag(phases_neg) @ evecs.conj().T

        # W(tau) = exp(+iH*tau) @ W @ exp(-iH*tau)
        W_t = exp_pos @ W @ exp_neg

        # OTOC operator: W(tau)^dagger @ V^dagger @ W(tau) @ V
        otoc_op = W_t.conj().T @ V.conj().T @ W_t @ V

        F = psi.conj() @ otoc_op @ psi
        otoc_values[i] = max(np.abs(F), 1e-15)

    # Scrambling rate = negative slope of log|F| vs tau
    log_otoc = np.log(otoc_values)
    if len(tau_values) > 1:
        slope = np.polyfit(tau_values, log_otoc, 1)[0]
        # Positive rate means decay (negative slope of log|F|)
        scrambling_rate = max(-slope, 0.0)
    else:
        scrambling_rate = 0.0

    return float(scrambling_rate)


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class OTOCDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via Out-of-Time-Order Correlator (OTOC) scrambling rate.

    The OTOC F(tau) = <psi|W(tau)^dagger V^dagger W(tau) V|psi> measures
    quantum information scrambling. Its decay rate under Heisenberg evolution
    serves as a regime detection signal: faster decay (higher scrambling rate)
    is associated with chaotic dynamics characteristic of financial crises.

    W and V are fixed random Hermitian operators seeded at initialization.
    The Heisenberg evolution uses the QCML error Hamiltonian H(x_t).

    Score = z-score of rolling-mean OTOC scrambling rate.

    Attributes:
        tau_values: Time evolution durations for OTOC computation.
        n_tau: Number of tau values in linspace(0.1, 5.0, n_tau).
        W: Fixed Hermitian operator W.
        V: Fixed Hermitian operator V.
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
        n_tau: int = 5,
        tau_min: float = 0.1,
        tau_max: float = 5.0,
    ):
        """Initialize OTOC Detector.

        Args:
            hilbert_dim: Hilbert space dimension (default 4, avoids Kramers degeneracy).
            n_pca_components: Number of PCA components for feature reduction.
            operator_method: Operator construction method ('random' recommended for OTOC).
            scale_exponent: PCA eigenvalue scaling exponent (None = default 0.5).
            rolling_window: Rolling mean window size for smoothing.
            min_expanding: Minimum expanding window observations before z-scoring.
            seed: Random seed for reproducibility.
            causal_fit_length: Fit only on first N samples (None = all).
            expanding_refit_interval: Refit interval for expanding windows.
            normalization: Post-PCA normalization mode ('soft', 'sphere', 'none', 'clip').
            adaptive_epsilon: Adapt numerical differentiation epsilon to data scale.
            custom_operators: Override learned QCML operators with custom list.
            n_tau: Number of time evolution steps in [tau_min, tau_max].
            tau_min: Minimum time evolution duration.
            tau_max: Maximum time evolution duration.
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
        self.n_tau = n_tau
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.tau_values = np.linspace(tau_min, tau_max, n_tau)

        # Fixed random Hermitian W and V — seeded at construction for reproducibility.
        # Using seed+1 and seed+2 ensures W != V and neither depends on QCML operators.
        rng = np.random.default_rng(seed)
        self.W = _random_hermitian(hilbert_dim, rng)
        self.V = _random_hermitian(hilbert_dim, rng)

    @property
    def name(self) -> str:
        return "OTOC"

    def fit(self, X: np.ndarray, **kwargs) -> 'OTOCDetector':
        """Fit scaler, PCA, and QCML geometry from feature matrix.

        Args:
            X: Feature matrix of shape (T, n_features).

        Returns:
            self
        """
        return _standard_qcml_fit(self, X)

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute regime scores as z-scored OTOC scrambling rates.

        For each time step t, computes the OTOC scrambling rate using the
        QCML error Hamiltonian H(x_t) with fixed W and V operators.
        The raw scrambling rates are smoothed with a rolling mean and then
        z-scored in an expanding window.

        Args:
            X: Feature matrix of shape (T, n_features).

        Returns:
            scores: 1-D array of length T. Higher = more regime-like (more scrambling).
                    NaN for warm-up period.
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
                vals[t] = compute_otoc_scrambling_rate(
                    geometry=geo,
                    x=xt,
                    W=self.W,
                    V=self.V,
                    tau_values=self.tau_values,
                )
            except Exception as e:
                logger.warning(f"OTOC computation failed at t={t}: {e}")
                vals[t] = np.nan

        return _expanding_zscore(vals, self.rolling_window, self.min_expanding, T)
