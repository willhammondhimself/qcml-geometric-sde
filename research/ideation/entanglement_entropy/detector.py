"""
Entanglement Entropy Detector for Financial Contagion Detection

Partitions market sectors into two subsystems (A: tech/energy, B: financials/healthcare),
embeds joint returns into a multi-qubit Hilbert space via QCML, and measures von Neumann
entanglement entropy of the reduced density matrix rho_A = Tr_B(|psi><psi|).

Hypothesis: During contagion/crises, cross-sector correlations increase, driving
higher entanglement entropy between subsystems. Normal regimes have lower entropy
as sectors behave more independently.

Key design choices:
- method='random' to avoid Kramers degeneracy (pca_inspired on 2-qubit systems
  produces degenerate ground states with zero spectral gap → d=0.0)
- hilbert_dim >= 4 (2-qubit) to allow nontrivial entanglement structure
- 4-sector ETF partition: A=[XLK, XLE], B=[XLF, XLV]
- Von Neumann entropy S = -Tr(rho_A log rho_A) instead of purity (more sensitive)
- Z-score with expanding window (causal, no lookahead)

Reference: Nielsen & Chuang (2000), Chapters 2, 11.
"""

import logging
import os
import sys
from typing import Optional, List, Tuple

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
# Core computation: von Neumann entropy of reduced density matrix
# ---------------------------------------------------------------------------

def von_neumann_entropy(psi: np.ndarray, dim_A: int, dim_B: int) -> float:
    """Compute von Neumann entropy S = -Tr(rho_A log rho_A).

    Given a pure state |psi> in a bipartite system HA ⊗ HB, traces out
    subsystem B to get rho_A, then computes its von Neumann entropy.

    For a maximally mixed rho_A, S = log(dim_A) (maximum entanglement).
    For a product state |psi> = |a>|b>, S = 0 (no entanglement).

    Args:
        psi: State vector of shape (dim_A * dim_B,), complex128.
        dim_A: Dimension of subsystem A.
        dim_B: Dimension of subsystem B.

    Returns:
        S: Von Neumann entropy (non-negative float). Units: nats.
    """
    assert psi.shape == (dim_A * dim_B,), (
        f"psi shape {psi.shape} != ({dim_A * dim_B},)"
    )

    # Reshape into bipartite matrix: rows = A basis, cols = B basis
    # psi[i * dim_B + j] = amplitude for |i>_A |j>_B
    psi_matrix = psi.reshape(dim_A, dim_B)

    # Reduced density matrix rho_A = Tr_B(|psi><psi|) = psi_matrix @ psi_matrix†
    # Shape: (dim_A, dim_A)
    rho_A = psi_matrix @ psi_matrix.conj().T

    # Eigenvalues of rho_A (real, non-negative, sum to 1 for normalized |psi>)
    eigenvalues = np.linalg.eigvalsh(rho_A)
    eigenvalues = np.maximum(eigenvalues.real, 0.0)  # numerical floor

    # Von Neumann entropy: S = -sum_k lambda_k * log(lambda_k)
    # Treat 0*log(0) = 0 (limit)
    mask = eigenvalues > 1e-15
    S = -np.sum(eigenvalues[mask] * np.log(eigenvalues[mask]))
    return float(S)


def compute_entanglement_entropy(
    geometry: QCMLGeometry,
    x: np.ndarray,
    dim_A: int,
    dim_B: int,
) -> float:
    """Compute entanglement entropy of the QCML ground state at feature vector x.

    Embeds x into the Hilbert space, extracts the ground state |psi(x)>,
    and computes S = -Tr(rho_A log rho_A) where rho_A = Tr_B(|psi><psi|).

    Args:
        geometry: Fitted QCMLGeometry with hilbert_dim = dim_A * dim_B.
        x: Feature vector of shape (n_features,).
        dim_A: Subsystem A dimension.
        dim_B: Subsystem B dimension.

    Returns:
        S: Entanglement entropy in nats (non-negative).
    """
    psi = geometry.quasi_coherent_state(x)
    return von_neumann_entropy(psi, dim_A, dim_B)


# ---------------------------------------------------------------------------
# Expanding z-score helper (matches Loschmidt echo pattern)
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
    Rolling mean first smooths the raw signal.

    Args:
        raw_values: Raw signal of shape (T,).
        rolling_window: Window for initial rolling mean.
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

class EntanglementEntropyDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via von Neumann entanglement entropy of sector subsystems.

    Partitions the QCML Hilbert space into two subsystems (A and B) representing
    distinct market sectors. Computes the von Neumann entropy S = -Tr(rho_A log rho_A)
    of the reduced density matrix. During contagion, cross-sector coupling increases,
    driving higher entanglement entropy.

    Score = expanding z-score of rolling-mean entropy.

    Key implementation note:
        operator_method='random' avoids Kramers degeneracy that kills the signal
        when using pca_inspired operators on 2-qubit systems.

    Attributes:
        dim_A: Subsystem A dimension (default 2, one qubit).
        dim_B: Subsystem B dimension (default 2, one qubit).
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
    ):
        """Initialize Entanglement Entropy Detector.

        Args:
            hilbert_dim: Hilbert space dimension. Must equal dim_A * dim_B.
            dim_A: Subsystem A dimension (sectors in partition A).
            dim_B: Subsystem B dimension (sectors in partition B).
            n_pca_components: Number of PCA components for feature reduction.
            operator_method: Operator construction method. 'random' recommended to
                avoid Kramers degeneracy from pca_inspired on 2-qubit systems.
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

        # Store all hyperparameters (mirrors _standard_init from loschmidt_echo)
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
        return "Entanglement Entropy"

    def fit(self, X: np.ndarray, **kwargs) -> 'EntanglementEntropyDetector':
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
        """Compute regime scores as z-scored entanglement entropy.

        Args:
            X: Feature matrix of shape (T, n_features).

        Returns:
            scores: 1-D array of length T. Higher = more entanglement = more crisis-like.
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
                vals[t] = compute_entanglement_entropy(
                    geometry=geo,
                    x=xt,
                    dim_A=self.dim_A,
                    dim_B=self.dim_B,
                )
            except Exception as e:
                logger.warning(f"Entanglement entropy failed at t={t}: {e}")
                vals[t] = np.nan

        return _expanding_zscore(vals, self.rolling_window, self.min_expanding, T)
