"""
Renyi Entropy Detector for Financial Regime Detection

Computes Renyi entropy S_alpha(rho_A) of the reduced density matrix obtained
by tracing out a subsystem of the QCML ground state.

Renyi entropy: S_alpha(rho) = (1 / (1 - alpha)) * log(Tr(rho^alpha))

Special cases:
  - alpha -> 1:  S_1 = -Tr(rho log rho)  (von Neumann entropy)
  - alpha = 2:   S_2 = -log(Tr(rho^2)) = -log(purity)
  - alpha = 0.5: S_0.5 = -2 * log(Tr(sqrt(rho)))  (emphasizes tail eigenvalues)
  - alpha = 3:   S_3 = -(1/2) * log(Tr(rho^3))  (emphasizes dominant eigenvalues)

CRITICAL NOTE on redundancy:
  Renyi-2 = -log(purity) is a monotone transform of purity. Since the existing
  ReducedPurityDetector already computes Tr(rho_A^2) and z-scores it, the z-scores
  of -log(purity) and purity are monotonically related but NOT identical (log is
  concave, so the z-score distributions differ). Nevertheless, the Cohen's d
  ranking may be very similar. The primary novelty targets are alpha=0.5 and
  alpha=3, which weight the eigenvalue spectrum differently.

Hypothesis:
  - alpha < 1 (e.g., 0.5) emphasizes small eigenvalues (tail states), potentially
    detecting early-stage entanglement changes before they dominate.
  - alpha > 1 (e.g., 3) emphasizes large eigenvalues (dominant states), tracking
    the leading mode of subsystem coupling.
  - Different alpha values may be complementary for different crisis types.

Design choices:
  - hilbert_dim=8 (3-qubit), partition=(4,2): trace out 1 qubit, keep 2
  - operator_method='random' to avoid Kramers degeneracy
  - Expanding-window z-score (causal, no lookahead)

Reference: Renyi (1961), "On measures of entropy and information."
"""

import logging
import os
import sys
from typing import Optional, List

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add project root to path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
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
# Core computation: Renyi entropy of reduced density matrix
# ---------------------------------------------------------------------------

def renyi_entropy(psi: np.ndarray, dim_A: int, dim_B: int, alpha: float) -> float:
    """Compute Renyi-alpha entropy of the reduced density matrix rho_A.

    Given a pure state |psi> in a bipartite Hilbert space H_A x H_B,
    computes rho_A = Tr_B(|psi><psi|) and returns:

        S_alpha(rho_A) = (1 / (1 - alpha)) * log(Tr(rho_A^alpha))

    For alpha=1 (von Neumann): S_1 = -Tr(rho_A log rho_A)

    Args:
        psi: State vector of shape (dim_A * dim_B,), complex128, normalized.
        dim_A: Dimension of subsystem A (kept).
        dim_B: Dimension of subsystem B (traced out).
        alpha: Renyi parameter. Must be > 0.

    Returns:
        S: Renyi entropy in nats (non-negative float).

    Raises:
        ValueError: If alpha <= 0 or psi has wrong shape.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")

    total_dim = dim_A * dim_B
    if psi.shape != (total_dim,):
        raise ValueError(f"psi shape {psi.shape} != ({total_dim},)")

    # Reshape into bipartite matrix: rows = A basis, cols = B basis
    psi_matrix = psi.reshape(dim_A, dim_B)

    # Reduced density matrix rho_A = Tr_B(|psi><psi|) = psi_matrix @ psi_matrix^dagger
    rho_A = psi_matrix @ psi_matrix.conj().T

    # Eigenvalues (real, non-negative, sum to 1 for normalized psi)
    eigenvalues = np.linalg.eigvalsh(rho_A)
    eigenvalues = np.maximum(eigenvalues.real, 0.0)

    # Remove numerical zeros
    mask = eigenvalues > 1e-15

    if not np.any(mask):
        # All eigenvalues zero -- degenerate state
        return 0.0

    eigs = eigenvalues[mask]

    if abs(alpha - 1.0) < 1e-10:
        # Von Neumann entropy: S_1 = -sum_k lambda_k * log(lambda_k)
        return float(-np.sum(eigs * np.log(eigs)))
    else:
        # Renyi entropy: S_alpha = (1/(1-alpha)) * log(sum_k lambda_k^alpha)
        tr_rho_alpha = np.sum(eigs ** alpha)
        return float((1.0 / (1.0 - alpha)) * np.log(tr_rho_alpha))


def compute_renyi_entropy(
    geometry: QCMLGeometry,
    x: np.ndarray,
    dim_A: int,
    dim_B: int,
    alpha: float,
) -> float:
    """Compute Renyi entropy of the QCML ground state at feature vector x.

    Args:
        geometry: Fitted QCMLGeometry with hilbert_dim = dim_A * dim_B.
        x: Feature vector of shape (n_features,).
        dim_A: Subsystem A dimension.
        dim_B: Subsystem B dimension.
        alpha: Renyi parameter.

    Returns:
        S_alpha: Renyi entropy in nats (non-negative).
    """
    psi = geometry.quasi_coherent_state(x)
    return renyi_entropy(psi, dim_A, dim_B, alpha)


# ---------------------------------------------------------------------------
# Expanding z-score helper
# ---------------------------------------------------------------------------

def _expanding_zscore(
    raw_values: np.ndarray,
    rolling_window: int,
    min_expanding: int,
    T: int,
    skip_nan_start: int = 0,
) -> np.ndarray:
    """Compute expanding-window z-score of rolling-mean values.

    Causal: z-score at time t uses only data up to t-1.

    Args:
        raw_values: Raw signal of shape (T,).
        rolling_window: Window for initial rolling mean smoothing.
        min_expanding: Minimum samples before z-score is computed.
        T: Total length.
        skip_nan_start: Skip first N entries when computing expanding stats.

    Returns:
        z_scores: Z-scored signal of shape (T,), NaN before min_expanding.
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


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class RenyiEntropyDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via Renyi entropy of the reduced density matrix.

    For a 3-qubit QCML embedding (hilbert_dim=8), traces out 1 qubit to obtain
    a 4x4 reduced density matrix rho_A, then computes:

        S_alpha(rho_A) = (1 / (1 - alpha)) * log(Tr(rho_A^alpha))

    Different alpha values weight the eigenvalue spectrum differently:
      - alpha=0.5: Emphasizes small eigenvalues (tail sensitivity)
      - alpha=1.0: Von Neumann entropy (standard information measure)
      - alpha=2.0: Related to purity (-log version); tests log-transform effect
      - alpha=3.0: Emphasizes dominant eigenvalue (leading-mode sensitivity)

    Score = expanding z-score of rolling-mean Renyi entropy.

    Attributes:
        alpha: Renyi parameter (> 0).
        dim_A: Subsystem A dimension (kept after partial trace).
        dim_B: Subsystem B dimension (traced out).
    """

    def __init__(
        self,
        hilbert_dim: int = 8,
        dim_A: int = 4,
        dim_B: int = 2,
        alpha: float = 0.5,
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
    ):
        """Initialize Renyi Entropy Detector.

        Args:
            hilbert_dim: Hilbert space dimension. Must equal dim_A * dim_B.
            dim_A: Subsystem A dimension (default 4: two qubits kept).
            dim_B: Subsystem B dimension (default 2: one qubit traced out).
            alpha: Renyi parameter. 0.5, 1.0, 2.0, 3.0 are standard choices.
            n_pca_components: Number of PCA components for feature reduction.
            operator_method: Operator construction method. 'random' recommended.
            scale_exponent: PCA eigenvalue scaling exponent (pca_* methods only).
            rolling_window: Window for rolling mean smoothing.
            min_expanding: Minimum expanding window size for z-score computation.
            seed: Random seed for reproducibility.
            causal_fit_length: Fit on first N samples only (None = all).
            expanding_refit_interval: Refit interval for expanding windows.
            normalization: Post-PCA normalization mode ('soft', 'sphere', etc.).
            adaptive_epsilon: Adapt numerical epsilon to data scale.
            custom_operators: Override learned operators with custom list.
        """
        if dim_A * dim_B != hilbert_dim:
            raise ValueError(
                f"dim_A ({dim_A}) * dim_B ({dim_B}) = {dim_A * dim_B} "
                f"!= hilbert_dim ({hilbert_dim})"
            )
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")

        self.hilbert_dim = hilbert_dim
        self.dim_A = dim_A
        self.dim_B = dim_B
        self.alpha = alpha
        self.n_pca_components = n_pca_components
        self.operator_method = operator_method
        self.scale_exponent = scale_exponent
        self.rolling_window = rolling_window
        self.min_expanding = min_expanding
        self.seed = seed
        self.causal_fit_length = causal_fit_length
        self.expanding_refit_interval = expanding_refit_interval
        self.normalization = normalization
        self.adaptive_epsilon = adaptive_epsilon
        self.custom_operators = custom_operators

        # Internal state (set during fit)
        self._geometry: Optional[QCMLGeometry] = None
        self._scaler: Optional[StandardScaler] = None
        self._pca: Optional[PCA] = None
        self._snapshots = None
        self._train_norms: Optional[np.ndarray] = None
        self._train_std: Optional[np.ndarray] = None
        self._epsilon = 1e-5

    @property
    def name(self) -> str:
        return f"Renyi-{self.alpha}"

    def fit(self, X: np.ndarray, **kwargs) -> 'RenyiEntropyDetector':
        """Fit scaler, PCA, and QCML geometry from feature matrix.

        Args:
            X: Feature matrix of shape (T, n_features).

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
            X_pca_raw, self.normalization,
            self._train_norms, self._train_std,
        )

        if self.adaptive_epsilon:
            self._epsilon = 1e-3 * np.median(np.abs(X_pca_fit))
        else:
            self._epsilon = 1e-5

        self._geometry = QCMLGeometry(
            n_features=X_pca_fit.shape[1],
            hilbert_dim=self.hilbert_dim,
        )

        if self.custom_operators is not None:
            self._geometry.set_operators(self.custom_operators)
        else:
            self._geometry.fit_operators(
                X_pca_fit,
                method=self.operator_method,
                scale_exponent=self.scale_exponent,
            )

        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute regime scores as z-scored Renyi entropy.

        Args:
            X: Feature matrix of shape (T, n_features).

        Returns:
            scores: 1-D array of length T. Higher absolute z-score = more anomalous.
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
                vals[t] = compute_renyi_entropy(
                    geometry=geo,
                    x=xt,
                    dim_A=self.dim_A,
                    dim_B=self.dim_B,
                    alpha=self.alpha,
                )
            except Exception as e:
                logger.warning(f"Renyi entropy (alpha={self.alpha}) failed at t={t}: {e}")
                vals[t] = np.nan

        return _expanding_zscore(vals, self.rolling_window, self.min_expanding, T)
