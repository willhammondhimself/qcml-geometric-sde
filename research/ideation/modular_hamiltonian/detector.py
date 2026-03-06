"""
Modular Hamiltonian Detector for Financial Regime Detection

Q13: "Does the modular Hamiltonian K = -log(rho_A) reveal sector structure
     that signals regime transitions?"

The modular Hamiltonian K = -log(rho_A) of the reduced density matrix encodes
the full entanglement structure of subsystem A. Unlike entanglement entropy (a
scalar), Tr(K^2) is the purity-weighted sum of squared log-eigenvalues and captures
the *spread* of the entanglement spectrum — more informative about structural changes.

Key properties:
- K = -log(rho_A) is Hermitian (rho_A is PSD)
- Tr(K^2) = sum_k (-log lambda_k)^2 — weighted spectral spread
- For a maximally mixed rho_A: K = log(dim_A) * I, Tr(K^2) = dim_A * log(dim_A)^2
- For a pure rho_A: lambda = (1, 0, ...), Tr(K^2) → infinity (eigenvalue hits 0)
- Crisis hypothesis: entanglement spectrum becomes more uniform → smaller Tr(K^2)
  (sectors couple and share information uniformly during contagion)

Implementation:
- dim=4, partition 2x2: rho_A is 2x2, K is 2x2
- Score = Tr(K^2) = sum_k (-log lambda_k)^2, eigenvalue-floored at 1e-12
- Z-score with expanding window (causal, no lookahead)

References:
    Casini, H., Huerta, M. (2011). Lectures on entanglement in quantum field theory.
    Haag, R. (1996). Local Quantum Physics.
    Nielsen & Chuang (2000), Chapter 11.
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
# Core computation: modular Hamiltonian spectral spread
# ---------------------------------------------------------------------------

def modular_hamiltonian_score(psi: np.ndarray, dim_A: int, dim_B: int) -> float:
    """Compute Tr(K^2) where K = -log(rho_A) is the modular Hamiltonian.

    For a pure bipartite state |psi> in HA ⊗ HB:
    1. Trace out B to get rho_A = Tr_B(|psi><psi|)
    2. Compute K = -log(rho_A) via diagonalization
    3. Return Tr(K^2) = sum_k (-log lambda_k)^2

    The eigenvalue floor at EPS prevents log(0) divergence. During crises,
    the entanglement spectrum tends toward uniformity, yielding smaller Tr(K^2)
    (less spread). During normal regimes, one eigenvalue dominates, giving
    a broader spectrum (larger Tr(K^2)).

    Args:
        psi: Normalized state vector of shape (dim_A * dim_B,), complex128.
        dim_A: Dimension of subsystem A.
        dim_B: Dimension of subsystem B.

    Returns:
        score: Tr(K^2) = sum_k (-log lambda_k)^2. Non-negative float.
    """
    EPS = 1e-12  # eigenvalue floor to prevent log(0) divergence

    # Reshape into bipartite amplitude matrix: C[i,j] = <i|_A <j|_B | psi>
    psi_matrix = psi.reshape(dim_A, dim_B)

    # Reduced density matrix: rho_A = psi_matrix @ psi_matrix†
    rho_A = psi_matrix @ psi_matrix.conj().T

    # Eigendecomposition of rho_A (Hermitian → real eigenvalues)
    eigenvalues = np.linalg.eigvalsh(rho_A)
    eigenvalues = np.maximum(eigenvalues.real, EPS)

    # Normalize so eigenvalues sum to 1 (numerical safeguard)
    eigenvalues = eigenvalues / eigenvalues.sum()
    eigenvalues = np.maximum(eigenvalues, EPS)

    # Modular Hamiltonian eigenvalues: k_i = -log(lambda_i)
    k_eigenvalues = -np.log(eigenvalues)

    # Tr(K^2) = sum_k k_i^2
    return float(np.sum(k_eigenvalues ** 2))


# ---------------------------------------------------------------------------
# Expanding z-score helper (mirrors entanglement_entropy pattern)
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

class ModularHamiltonianDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via Tr(K^2) of the modular Hamiltonian K = -log(rho_A).

    Computes the QCML ground state |psi(x)> at each time step, reduces to
    subsystem A by tracing out B, then measures Tr(K^2) = sum_k (-log lambda_k)^2
    of the modular Hamiltonian K = -log(rho_A).

    Regime hypothesis:
        - Normal markets: entanglement spectrum peaked at one dominant eigenvalue
          → large log values → large Tr(K^2)
        - Crisis markets: sectors become uniformly coupled → flatter spectrum
          → smaller log values → smaller Tr(K^2)
        - Score is negated so higher score = more crisis-like

    Implementation:
        hilbert_dim = dim_A * dim_B = 4 (2×2 bipartition)
        operator_method = 'random' avoids Kramers degeneracy
        Score = expanding z-score of -Tr(K^2) (negated so crises show as peaks)

    Attributes:
        dim_A: Subsystem A dimension (default 2).
        dim_B: Subsystem B dimension (default 2).
        hilbert_dim: Must equal dim_A * dim_B.
    """

    def __init__(
        self,
        hilbert_dim: int = 4,
        dim_A: int = 2,
        dim_B: int = 2,
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
        negate_score: bool = True,
    ):
        """Initialize Modular Hamiltonian Detector.

        Args:
            hilbert_dim: Hilbert space dimension. Must equal dim_A * dim_B.
            dim_A: Subsystem A dimension.
            dim_B: Subsystem B dimension.
            n_pca_components: Number of PCA components for feature reduction.
            operator_method: Operator construction method ('random' avoids Kramers degeneracy).
            scale_exponent: PCA eigenvalue scaling exponent (pca_* methods only).
            rolling_window: Window for rolling mean smoothing.
            min_expanding: Minimum expanding window size for z-score computation.
            seed: Random seed for reproducibility.
            causal_fit_length: Fit on first N samples only (None = all).
            expanding_refit_interval: Refit interval for expanding windows.
            normalization: Post-PCA normalization mode ('soft', 'sphere', etc.).
            adaptive_epsilon: Adapt numerical epsilon to data scale.
            custom_operators: Override learned operators with custom list.
            negate_score: If True, negate Tr(K^2) so crises appear as positive spikes
                (crisis → uniform spectrum → smaller Tr(K^2) → negative → negate → positive).
        """
        if dim_A * dim_B != hilbert_dim:
            raise ValueError(
                f"dim_A ({dim_A}) * dim_B ({dim_B}) = {dim_A * dim_B} "
                f"!= hilbert_dim ({hilbert_dim})"
            )

        self.hilbert_dim = hilbert_dim
        self.dim_A = dim_A
        self.dim_B = dim_B
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
        self.negate_score = negate_score

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
        return "Modular Hamiltonian"

    def fit(self, X: np.ndarray, **kwargs) -> 'ModularHamiltonianDetector':
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
        """Compute regime scores from modular Hamiltonian Tr(K^2).

        Args:
            X: Feature matrix of shape (T, n_features).

        Returns:
            scores: 1-D array of length T. Higher = more crisis-like.
                    (Tr(K^2) negated if negate_score=True, then z-scored.)
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
                psi = geo.quasi_coherent_state(xt)
                raw = modular_hamiltonian_score(psi, self.dim_A, self.dim_B)
                vals[t] = -raw if self.negate_score else raw
            except Exception as e:
                logger.warning(f"Modular Hamiltonian failed at t={t}: {e}")
                vals[t] = np.nan

        return _expanding_zscore(vals, self.rolling_window, self.min_expanding, T)
