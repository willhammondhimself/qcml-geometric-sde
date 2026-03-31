"""
Quantum Coherence Detector for Financial Regime Detection

Measures quantum coherence (l1-norm) of the QCML ground state |psi(x)> in
three different bases, each capturing a different aspect of regime change.

Key design constraint: since |psi(t)> IS the ground state of H(t), it has
ZERO coherence in H(t)'s own eigenbasis. We therefore measure coherence in
bases that differ from the instantaneous eigenbasis.

Three variants:

1. **Computational basis coherence** (C_comp):
   C_l1 = sum_{i!=j} |rho_{ij}| = sum_{i!=j} |psi_i * conj(psi_j)|
   in the standard computational basis {|0>, |1>, ..., |d-1>}.
   Note: C_l1 = 2*(1 - IPR) where IPR = sum|psi_i|^4 is the inverse
   participation ratio. This variant tests whether coherence adds anything
   beyond what IPR already captures.

2. **Reference-basis coherence** (C_ref):
   Fix H_ref = time-average of H(t) over the first 252 trading days.
   Diagonalize H_ref to get {|e_k>}. Measure C_l1 of |psi(t)> in this
   fixed basis. Captures how far the current state has drifted from the
   reference eigenbasis -- large coherence means the system's preferred
   states have rotated away from the initial equilibrium.

3. **Temporal coherence** (C_temp):
   At each step, measure C_l1 of |psi(t)> in the eigenbasis of H(t-1).
   Captures "state novelty" -- how much the system changed in one step.
   Zero if consecutive Hamiltonians share eigenstates, large if the
   eigenbasis underwent a rotation.

References:
    - Baumgratz, Cramer, Plenio (2014). Quantifying Coherence.
      Phys. Rev. Lett. 113, 140401.
    - Winter, Yang (2016). Operational resource theory of coherence.
      Phys. Rev. Lett. 116, 120404.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qcml_geometry.core import QCMLGeometry

logger = logging.getLogger(__name__)


def l1_coherence_computational(psi: np.ndarray) -> float:
    """Compute l1-norm of coherence in the computational basis.

    C_l1 = sum_{i!=j} |psi_i * conj(psi_j)| = sum_{i!=j} |psi_i| * |psi_j|
         = (sum_i |psi_i|)^2 - sum_i |psi_i|^2
         = (sum |psi_i|)^2 - 1  (for normalized states)

    This equals 2*(1 - IPR) where IPR = sum|psi_i|^4,
    so it is algebraically related to the inverse participation ratio.

    Args:
        psi: State vector of shape (d,), assumed normalized.

    Returns:
        C_l1: l1-norm of coherence (non-negative). Range [0, d-1].
    """
    abs_psi = np.abs(psi)
    return float(np.sum(abs_psi) ** 2 - 1.0)


def l1_coherence_in_basis(psi: np.ndarray, basis_vecs: np.ndarray) -> float:
    """Compute l1-norm of coherence of |psi> in a given orthonormal basis.

    Given basis {|e_k>}, expand |psi> = sum_k c_k |e_k> where c_k = <e_k|psi>.
    C_l1 = sum_{k!=l} |c_k * conj(c_l)| = (sum_k |c_k|)^2 - sum_k |c_k|^2

    Args:
        psi: State vector of shape (d,), assumed normalized.
        basis_vecs: Orthonormal basis, shape (d, d). Columns are basis vectors.
            basis_vecs[:, k] = |e_k>.

    Returns:
        C_l1: l1-norm of coherence in the given basis. Range [0, d-1].
    """
    # Expand psi in the new basis: c_k = <e_k | psi> = conj(basis_vecs[:, k]) . psi
    coeffs = basis_vecs.conj().T @ psi  # shape (d,)
    abs_c = np.abs(coeffs)
    return float(np.sum(abs_c) ** 2 - np.sum(abs_c ** 2))


def compute_ipr(psi: np.ndarray) -> float:
    """Compute inverse participation ratio IPR = sum|psi_i|^4.

    Args:
        psi: State vector of shape (d,), assumed normalized.

    Returns:
        IPR in [1/d, 1]. Low = delocalized, High = localized.
    """
    return float(np.sum(np.abs(psi) ** 4))


class QuantumCoherenceDetector:
    """Regime detection via quantum coherence in multiple bases.

    Computes three coherence variants for each time step:
    - computational: l1-coherence in standard basis (related to IPR)
    - reference: l1-coherence in eigenbasis of time-averaged Hamiltonian
    - temporal: l1-coherence in eigenbasis of previous Hamiltonian

    Each is converted to an expanding-window z-score for regime scoring.

    Args:
        hilbert_dim: Dimension of QCML Hilbert space.
        n_pca_components: Number of PCA components.
        operator_method: Method for QCML operator construction.
        rolling_window: Window for rolling mean smoothing.
        min_expanding: Minimum observations before z-scoring.
        seed: Random seed.
        normalization: Post-PCA normalization ('soft', 'sphere', 'none').
        reference_window: Number of initial days for reference Hamiltonian (default 252).
        variant: Which coherence variant to use for compute_regime_scores.
            'computational', 'reference', or 'temporal'.
    """

    def __init__(
        self,
        hilbert_dim: int = 8,
        n_pca_components: int = 8,
        operator_method: str = 'random',
        scale_exponent: Optional[float] = None,
        rolling_window: int = 20,
        min_expanding: int = 60,
        seed: int = 42,
        normalization: str = 'soft',
        reference_window: int = 252,
        variant: str = 'reference',
    ):
        self.hilbert_dim = hilbert_dim
        self.n_pca_components = n_pca_components
        self.operator_method = operator_method
        self.scale_exponent = scale_exponent
        self.rolling_window = rolling_window
        self.min_expanding = min_expanding
        self.seed = seed
        self.normalization = normalization
        self.reference_window = reference_window
        self.variant = variant

        self._geometry = None
        self._scaler = None
        self._pca = None
        self._train_norms = None
        self._train_std = None
        self._ref_basis = None  # Eigenvectors of H_ref

    @property
    def name(self) -> str:
        return f"Quantum Coherence ({self.variant})"

    def fit(self, X: np.ndarray) -> 'QuantumCoherenceDetector':
        """Fit the detector on feature matrix X.

        Fits PCA and QCML geometry, then builds the reference Hamiltonian
        from the first `reference_window` time steps.

        Args:
            X: Feature matrix of shape (T, n_features).

        Returns:
            self
        """
        np.random.seed(self.seed)
        n_components = min(self.n_pca_components, X.shape[1])

        self._scaler = StandardScaler()
        self._scaler.fit(X)

        self._pca = PCA(n_components=n_components)
        X_scaled = self._scaler.transform(X)
        self._pca.fit(X_scaled)

        X_pca_raw = self._pca.transform(X_scaled)
        self._train_norms = np.linalg.norm(X_pca_raw, axis=1)
        self._train_std = np.std(X_pca_raw, axis=0)

        from qcml_geometry.observables import _apply_normalization
        X_pca = _apply_normalization(
            X_pca_raw, self.normalization, self._train_norms, self._train_std,
        )

        self._geometry = QCMLGeometry(
            n_features=X_pca.shape[1], hilbert_dim=self.hilbert_dim
        )
        self._geometry.fit_operators(
            X_pca, method=self.operator_method,
            scale_exponent=self.scale_exponent,
        )

        # Build reference Hamiltonian: time-average of H(x_t) over first ref_window steps
        ref_end = min(self.reference_window, len(X_pca))
        H_ref = np.zeros(
            (self.hilbert_dim, self.hilbert_dim), dtype=np.complex128
        )
        for t in range(ref_end):
            H_ref += self._geometry.error_hamiltonian(X_pca[t])
        H_ref /= ref_end

        # Diagonalize to get reference basis
        _, ref_vecs = np.linalg.eigh(H_ref)
        self._ref_basis = ref_vecs  # columns are eigenvectors

        return self

    def _transform_point(self, x_raw: np.ndarray) -> np.ndarray:
        """Transform a single raw data point through scaler + PCA + normalization."""
        from qcml_geometry.observables import _apply_normalization
        x_scaled = self._scaler.transform(x_raw.reshape(1, -1))
        x_pca = self._pca.transform(x_scaled).ravel()
        return _apply_normalization(
            x_pca, self.normalization, self._train_norms, self._train_std,
        )

    def compute_all_variants(self, X: np.ndarray) -> dict:
        """Compute all three coherence variants and IPR for comparison.

        Returns raw (un-z-scored) time series for each variant.

        Args:
            X: Feature matrix of shape (T, n_features).

        Returns:
            dict with keys 'computational', 'reference', 'temporal', 'ipr',
            each mapping to np.ndarray of shape (T,).
        """
        if self._geometry is None:
            raise RuntimeError("Call fit() before compute_all_variants().")

        T = X.shape[0]
        comp_vals = np.empty(T)
        ref_vals = np.empty(T)
        temp_vals = np.full(T, np.nan)
        ipr_vals = np.empty(T)

        prev_eigvecs = None

        for t in range(T):
            x_t = self._transform_point(X[t])
            psi = self._geometry.quasi_coherent_state(x_t)

            # 1. Computational basis coherence
            comp_vals[t] = l1_coherence_computational(psi)

            # 2. Reference basis coherence
            ref_vals[t] = l1_coherence_in_basis(psi, self._ref_basis)

            # 3. Temporal coherence (needs H(t) eigenbasis for next step,
            #    and H(t-1) eigenbasis for current step)
            H_t = self._geometry.error_hamiltonian(x_t)
            _, curr_eigvecs = np.linalg.eigh(H_t)

            if prev_eigvecs is not None:
                temp_vals[t] = l1_coherence_in_basis(psi, prev_eigvecs)

            prev_eigvecs = curr_eigvecs

            # IPR for comparison
            ipr_vals[t] = compute_ipr(psi)

        return {
            'computational': comp_vals,
            'reference': ref_vals,
            'temporal': temp_vals,
            'ipr': ipr_vals,
        }

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute expanding-window z-scored coherence for the configured variant.

        Args:
            X: Feature matrix of shape (T, n_features).

        Returns:
            z_scores: Array of shape (T,) with NaN for warmup period.
        """
        all_variants = self.compute_all_variants(X)
        raw_vals = all_variants[self.variant]
        T = len(raw_vals)

        return _expanding_zscore(raw_vals, self.rolling_window, self.min_expanding, T)


def _expanding_zscore(raw_values, rolling_window, min_expanding, T,
                      skip_nan_start=0):
    """Compute expanding-window z-score of rolling-mean values.

    Identical to the pattern in qcml_geometry.observables._expanding_zscore.
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
