"""
Loschmidt Echo Detector for Financial Regime Detection

Implements fidelity decay under Hamiltonian perturbation as a regime signal.

The Loschmidt echo measures sensitivity of quantum dynamics to perturbation:

    M(tau) = |<psi_0| e^{iH'tau} e^{-iHtau} |psi_0>|^2

where H = H(x_t) is the error Hamiltonian at feature vector x_t,
H' = H(x_t + delta) is the perturbed Hamiltonian, and |psi_0> is the
ground state of H.

In chaotic regimes (financial crises), the echo decays faster because the
Hamiltonian landscape is more sensitive to perturbation. The echo decay
rate serves as a regime detection signal.

Reference: Peres (1984), Jalabert & Pastawski (2001)
"""

import logging
from typing import Optional, List

import numpy as np
from scipy.linalg import expm
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


def _expanding_zscore(raw_values, rolling_window, min_expanding, T,
                      skip_nan_start=0):
    """Compute expanding-window z-score of rolling-mean values.

    Matches the pattern used by other observatory detectors.
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


def compute_loschmidt_echo(
    geometry: QCMLGeometry,
    x: np.ndarray,
    perturbation_scale: float = 0.01,
    tau_values: Optional[np.ndarray] = None,
    seed: int = 42,
) -> float:
    """Compute the Loschmidt echo decay rate for a single feature vector.

    The Loschmidt echo is M(tau) = |<psi_0| e^{iH'tau} e^{-iHtau} |psi_0>|^2
    where H = H(x), H' = H(x + delta), delta is a small random perturbation.

    The decay rate is estimated by fitting -log(M(tau)) vs tau and taking
    the slope. Higher decay rate means more sensitivity to perturbation,
    signaling a regime transition.

    Args:
        geometry: Fitted QCMLGeometry instance.
        x: Feature vector of shape (n_features,).
        perturbation_scale: Scale of perturbation relative to feature magnitude.
        tau_values: Array of time evolution durations. Default: [0.1, 0.5, 1.0, 2.0, 5.0].
        seed: Random seed for perturbation direction.

    Returns:
        decay_rate: Non-negative float. Higher = more sensitive to perturbation.
    """
    if tau_values is None:
        tau_values = np.array([0.1, 0.5, 1.0, 2.0, 5.0])

    x = np.asarray(x, dtype=float)

    # Build unperturbed Hamiltonian and ground state
    H = geometry.error_hamiltonian(x)
    psi_0 = geometry.quasi_coherent_state(x)

    # Generate perturbation: fixed direction, scale proportional to |x|
    rng = np.random.default_rng(seed)
    delta_dir = rng.standard_normal(len(x))
    delta_dir = delta_dir / (np.linalg.norm(delta_dir) + 1e-15)
    x_mag = np.linalg.norm(x)
    delta = perturbation_scale * max(x_mag, 1e-3) * delta_dir

    # Build perturbed Hamiltonian
    x_perturbed = x + delta
    H_prime = geometry.error_hamiltonian(x_perturbed)

    # Compute echo M(tau) for each tau
    echo_values = np.empty(len(tau_values))
    for i, tau in enumerate(tau_values):
        # Forward evolution: e^{-iHtau}|psi_0>
        U = expm(-1j * H * tau)
        # Backward evolution under perturbed Hamiltonian: e^{iH'tau}
        U_prime_dag = expm(1j * H_prime * tau)

        # Loschmidt echo
        evolved = U_prime_dag @ U @ psi_0
        overlap = np.abs(np.vdot(psi_0, evolved)) ** 2
        echo_values[i] = max(overlap, 1e-15)  # avoid log(0)

    # Fit decay rate: -log(M(tau)) ~ gamma * tau
    # Use linear regression of -log(M) vs tau
    neg_log_echo = -np.log(echo_values)

    # Weighted least squares (later tau values noisier)
    weights = 1.0 / (1.0 + tau_values)
    W = np.diag(weights)
    A = np.column_stack([tau_values, np.ones_like(tau_values)])
    AtWA = A.T @ W @ A
    AtWy = A.T @ W @ neg_log_echo

    try:
        params = np.linalg.solve(AtWA, AtWy)
        decay_rate = max(params[0], 0.0)  # slope = decay rate (non-negative)
    except np.linalg.LinAlgError:
        decay_rate = 0.0

    return float(decay_rate)


class LoschmidtEchoDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via Loschmidt echo (fidelity decay under perturbation).

    The Loschmidt echo M(tau) = |<psi_0|e^{iH'tau}e^{-iHtau}|psi_0>|^2
    measures how quickly quantum state fidelity decays when the Hamiltonian
    is slightly perturbed. In chaotic/unstable regimes, the echo decays
    faster, producing a higher decay rate signal.

    Score = z-score of rolling-mean echo decay rate.

    Attributes:
        perturbation_scale: Fraction of feature magnitude used for delta.
        tau_values: Time evolution durations for echo computation.
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
        perturbation_scale: float = 0.01,
        tau_values: Optional[np.ndarray] = None,
    ):
        """Initialize Loschmidt Echo Detector.

        Args:
            hilbert_dim: Hilbert space dimension (default 4, avoids Kramers degeneracy).
            n_pca_components: Number of PCA components.
            operator_method: Operator construction method ('random' recommended).
            scale_exponent: PCA eigenvalue scaling exponent.
            rolling_window: Window for rolling mean smoothing.
            min_expanding: Minimum expanding window for z-score.
            seed: Random seed for reproducibility.
            causal_fit_length: Fit only on first N samples (None = all).
            expanding_refit_interval: Refit interval for expanding windows.
            normalization: Post-PCA normalization mode.
            adaptive_epsilon: Adapt epsilon to data scale.
            custom_operators: Override learned operators.
            perturbation_scale: Scale of Hamiltonian perturbation (fraction of |x|).
            tau_values: Time evolution durations for echo. Default: [0.1, 0.5, 1.0, 2.0, 5.0].
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
        self.perturbation_scale = perturbation_scale
        self.tau_values = tau_values if tau_values is not None else np.array(
            [0.1, 0.5, 1.0, 2.0, 5.0]
        )

    @property
    def name(self) -> str:
        return "Loschmidt Echo"

    def fit(self, X: np.ndarray, **kwargs) -> 'LoschmidtEchoDetector':
        """Fit scaler, PCA, and QCML geometry from feature matrix.

        Args:
            X: Feature matrix of shape (T, n_features).

        Returns:
            self
        """
        return _standard_qcml_fit(self, X)

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute regime scores as z-scored Loschmidt echo decay rates.

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
        vals = np.full(T, np.nan)

        for t in range(T):
            if self._snapshots:
                geo, xt = self._transform_point_at(X[t], t)
            else:
                geo, xt = self._geometry, Xt[t]

            try:
                vals[t] = compute_loschmidt_echo(
                    geometry=geo,
                    x=xt,
                    perturbation_scale=self.perturbation_scale,
                    tau_values=self.tau_values,
                    seed=self.seed + t,  # different perturbation per time step
                )
            except Exception as e:
                logger.warning(f"Loschmidt echo failed at t={t}: {e}")
                vals[t] = np.nan

        return _expanding_zscore(vals, self.rolling_window, self.min_expanding, T)
