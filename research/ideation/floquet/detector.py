"""
Floquet Analysis Detector for Financial Regime Detection

Q15: "Do periodically-driven market dynamics reveal resonance-induced instabilities
     via quasi-energy level spacing?"

Floquet theory describes periodically-driven quantum systems. We treat the daily
sequence of QCML Hamiltonians H(x_t) as a periodic drive over a rolling window
of T_window trading days, constructing the Floquet operator:

    U_F = prod_{t in window} exp(-i * H(x_t) * dt)

where dt = 1/T_window (normalized). The quasi-energies epsilon_k are defined by:

    U_F |phi_k> = exp(-i * epsilon_k * T) |phi_k>

i.e., epsilon_k = -i/T * log(eigenvalue_k of U_F)

Key observables:
1. Quasi-energy gap: min spacing between adjacent quasi-energies (on the torus [-pi, pi])
   A small quasi-energy gap signals near-resonance and instability.
2. Quasi-energy spread: std dev of quasi-energies (measures uniformity of drive).
3. Score: level repulsion statistic r = <min_gap / max_gap> for the quasi-energy
   level spacing distribution (Wigner-Dyson vs Poisson statistics).

Market interpretation:
- Normal regimes: quasi-energies well-spaced (Poisson statistics, r ≈ 0.39)
  Markets behave quasi-randomly; no strong periodicity or resonance.
- Crisis regimes: level clustering, near-degeneracy (quasi GOE statistics, r ≈ 0.53)
  Synchronized Hamiltonians → drive becomes coherent → quasi-energy degeneracy.

Score = quasi-energy gap (negated: small gap = crisis), z-scored with expanding window.

Implementation notes:
- H(x_t) = sum_k x_{t,k} * O_k (linear combination of learned operators)
- exp(-i*H*dt) computed via matrix exponential (scipy.linalg.expm)
- Window = 20 trading days; product is ordered (non-commutative)
- Quasi-energies are wrapped to [-pi/T, pi/T]; gaps measured on the torus

References:
    Shirley, J. H. (1965). Solution of the Schrödinger Equation with a
        Hamiltonian Periodic in Time. Physical Review 138(4B):B979.
    Floquet, G. (1883). Sur les équations différentielles linéaires à coefficients
        périodiques. Ann. de l'Ecole Norm. 12, 47-88.
    Eckardt, A. (2017). Colloquium: Atomic quantum gases in periodically
        driven optical lattices. Rev. Mod. Phys. 89, 011004.
    D'Alessio, L., Rigol, M. (2014). Long-time behavior of isolated periodically
        driven interacting lattice systems. Phys. Rev. X 4, 041048.
"""

import logging
import os
import sys
from typing import Optional, List

import numpy as np
from scipy.linalg import expm
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
# Core computation: Floquet operator and quasi-energy analysis
# ---------------------------------------------------------------------------

def _build_hamiltonian(geometry: QCMLGeometry, x: np.ndarray) -> np.ndarray:
    """Build the QCML Hamiltonian H(x) = sum_k x_k * O_k.

    The QCML Hamiltonian is a linear combination of the learned Hermitian
    operators, with coefficients given by the feature vector x.

    Args:
        geometry: Fitted QCMLGeometry with operators O_k.
        x: Feature vector of shape (n_features,).

    Returns:
        H: Hermitian matrix of shape (hilbert_dim, hilbert_dim), complex128.
    """
    operators = geometry.operators  # List of (hilbert_dim, hilbert_dim) Hermitian arrays
    n_ops = len(operators)
    n_features = len(x)

    # Use as many operators as we have features (truncate or pad if needed)
    n_use = min(n_ops, n_features)
    H = np.zeros((operators[0].shape[0], operators[0].shape[0]), dtype=complex)

    for k in range(n_use):
        H += x[k] * operators[k]

    # Symmetrize to ensure exact Hermiticity
    H = 0.5 * (H + H.conj().T)
    return H


def floquet_analysis(
    geometry: QCMLGeometry,
    X_window: np.ndarray,
    floquet_window: int,
) -> dict:
    """Compute Floquet operator and quasi-energy statistics for a time window.

    Constructs U_F = exp(-i*H(x_{T-1})*dt) @ ... @ exp(-i*H(x_0)*dt),
    where dt = 2*pi/floquet_window (one full period normalized to 2*pi).

    Quasi-energies: epsilon_k = angle(eigenvalue_k of U_F) / (2*pi/floquet_window)
    mapped to the interval [-pi*floquet_window/(2*pi), +pi*floquet_window/(2*pi)].

    For the regime score we use:
    1. min_gap: minimum gap between adjacent quasi-energies on the circle
    2. level_spacing_ratio r: <r_n> where r_n = min(delta_n, delta_{n+1}) / max(...)
       r → 0.39 (Poisson, integrable), r → 0.53 (GOE, chaotic/crisis)

    Args:
        geometry: Fitted QCMLGeometry.
        X_window: Feature matrix for the window, shape (floquet_window, n_features).
        floquet_window: Number of time steps in the Floquet period.

    Returns:
        dict with keys:
            'quasi_energies': sorted quasi-energies (phases of U_F eigenvalues)
            'min_gap': minimum quasi-energy gap on the torus
            'gap_std': std deviation of quasi-energy gaps
            'level_spacing_ratio': mean r statistic (chaos indicator)
            'floquet_norm': |det(U_F)| (should be 1 for unitary)
    """
    dim = geometry.hilbert_dim
    dt = 2.0 * np.pi / floquet_window  # One full Floquet period = 2*pi

    # Build Floquet operator: ordered product U_F = U_T ... U_1
    U_F = np.eye(dim, dtype=complex)
    for t in range(floquet_window):
        x_t = X_window[t]
        H_t = _build_hamiltonian(geometry, x_t)
        U_t = expm(-1j * H_t * dt)
        U_F = U_t @ U_F  # Left-multiply: U_F = U_T @ ... @ U_1

    # Eigenvalues of U_F: lie on unit circle for unitary operator
    eigenvalues = np.linalg.eigvals(U_F)

    # Quasi-energies: phases of eigenvalues, mapped to [-pi, pi]
    quasi_energies = np.angle(eigenvalues)  # in [-pi, pi]
    quasi_energies = np.sort(quasi_energies.real)

    # Gaps between adjacent quasi-energies on the circle [-pi, pi]
    # Include wraparound gap: from quasi_energies[-1] to quasi_energies[0] + 2*pi
    n = len(quasi_energies)
    if n >= 2:
        linear_gaps = np.diff(quasi_energies)
        wrap_gap = (quasi_energies[0] + 2.0 * np.pi) - quasi_energies[-1]
        all_gaps = np.append(linear_gaps, wrap_gap)
        all_gaps = np.maximum(all_gaps, 0.0)  # numerical floor

        min_gap = float(np.min(all_gaps))
        gap_std = float(np.std(all_gaps))

        # Level spacing ratio r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1})
        # where s_n are the gaps (using circular arrangement)
        if n >= 3:
            s = all_gaps
            r_values = []
            for i in range(n):
                s_n = s[i]
                s_next = s[(i + 1) % n]
                if max(s_n, s_next) > 1e-15:
                    r_values.append(min(s_n, s_next) / max(s_n, s_next))
            level_spacing_ratio = float(np.mean(r_values)) if r_values else np.nan
        else:
            level_spacing_ratio = np.nan
    else:
        min_gap = np.nan
        gap_std = np.nan
        level_spacing_ratio = np.nan

    floquet_norm = float(np.abs(np.linalg.det(U_F)))

    return {
        'quasi_energies': quasi_energies.tolist(),
        'min_gap': min_gap,
        'gap_std': gap_std,
        'level_spacing_ratio': level_spacing_ratio,
        'floquet_norm': floquet_norm,
    }


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

class FloquetDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via Floquet quasi-energy analysis.

    Treats the rolling window of daily QCML Hamiltonians as a periodic drive,
    constructs the Floquet operator U_F, and extracts quasi-energy statistics.

    Crisis hypothesis:
        Normal markets have quasi-randomly varying Hamiltonians → Poisson-like
        quasi-energy statistics (large min_gap, low level-spacing ratio).
        Crisis markets have highly correlated, synchronized Hamiltonians →
        coherent drive → quasi-energy degeneracy / level clustering
        (small min_gap, high level-spacing ratio approaching GOE value 0.53).

    Score options:
        'min_gap': minimum quasi-energy gap (negated: small gap = crisis)
        'level_spacing_ratio': r statistic (crisis: r ↑ toward 0.53)
        'gap_std': std of quasi-energy gaps (negated: small spread = crisis)

    Attributes:
        floquet_window: Number of trading days in one Floquet period (default 20).
        score_field: Which Floquet statistic to use as the regime score.
        negate_score: Negate the raw score before z-scoring (if True, higher
            output score = more crisis-like; only applies to min_gap, gap_std).
    """

    def __init__(
        self,
        hilbert_dim: int = 4,
        n_pca_components: int = 8,
        operator_method: str = 'random',
        scale_exponent: Optional[float] = None,
        floquet_window: int = 20,
        rolling_window: int = 5,
        min_expanding: int = 60,
        seed: int = 42,
        causal_fit_length: Optional[int] = None,
        expanding_refit_interval: Optional[int] = None,
        normalization: str = 'soft',
        adaptive_epsilon: bool = True,
        custom_operators: Optional[List[np.ndarray]] = None,
        score_field: str = 'min_gap',
        negate_score: bool = True,
    ):
        """Initialize Floquet Detector.

        Args:
            hilbert_dim: Hilbert space dimension (default 4, 2-qubit).
            n_pca_components: Number of PCA components for feature reduction.
            operator_method: Operator construction ('random' recommended).
            scale_exponent: PCA eigenvalue scaling exponent.
            floquet_window: Length of one Floquet period in trading days (default 20).
            rolling_window: Window for secondary smoothing before z-score.
            min_expanding: Minimum expanding window size for z-score computation.
            seed: Random seed for reproducibility.
            causal_fit_length: Fit on first N samples only (None = all).
            expanding_refit_interval: Refit interval for expanding windows.
            normalization: Post-PCA normalization mode.
            adaptive_epsilon: Adapt numerical epsilon to data scale.
            custom_operators: Override learned operators.
            score_field: Floquet statistic to use as score:
                'min_gap': minimum quasi-energy gap (negate for crisis detection)
                'level_spacing_ratio': r statistic (higher = more GOE = crisis)
                'gap_std': std of quasi-energy gaps (negate for crisis detection)
            negate_score: Negate the raw score before z-scoring. Set True for
                'min_gap' and 'gap_std' (smaller = crisis), False for
                'level_spacing_ratio' (larger = crisis).
        """
        self.hilbert_dim = hilbert_dim
        self.n_pca_components = n_pca_components
        self.operator_method = operator_method
        self.scale_exponent = scale_exponent
        self.floquet_window = floquet_window
        self.rolling_window = rolling_window
        self.min_expanding = min_expanding
        self.seed = seed
        self.causal_fit_length = causal_fit_length
        self.expanding_refit_interval = expanding_refit_interval
        self.normalization = normalization
        self.adaptive_epsilon = adaptive_epsilon
        self.custom_operators = custom_operators
        self.score_field = score_field
        self.negate_score = negate_score

        # Internal state
        self._geometry: Optional[QCMLGeometry] = None
        self._scaler: Optional[StandardScaler] = None
        self._pca: Optional[PCA] = None
        self._snapshots = None
        self._train_norms: Optional[np.ndarray] = None
        self._train_std: Optional[np.ndarray] = None
        self._epsilon = 1e-5

    @property
    def name(self) -> str:
        return "Floquet Analysis"

    def fit(self, X: np.ndarray, **kwargs) -> 'FloquetDetector':
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
        """Compute regime scores from Floquet quasi-energy analysis.

        For each time t >= floquet_window, constructs the Floquet operator
        from the preceding floquet_window Hamiltonians and extracts quasi-energy
        statistics.

        Args:
            X: Feature matrix of shape (T, n_features).

        Returns:
            scores: 1-D array of length T. Higher = more crisis-like.
                    NaN for first floquet_window time steps.
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

        for t in range(self.floquet_window, T):
            # Use the window ending at t-1 (causal: no lookahead)
            window_start = t - self.floquet_window
            X_window = Xt[window_start:t]  # shape (floquet_window, n_pca)

            try:
                result = floquet_analysis(
                    geometry=self._geometry,
                    X_window=X_window,
                    floquet_window=self.floquet_window,
                )
                raw = result[self.score_field]
                if np.isnan(raw):
                    vals[t] = np.nan
                else:
                    vals[t] = -raw if self.negate_score else raw
            except Exception as e:
                logger.warning(f"Floquet analysis failed at t={t}: {e}")
                vals[t] = np.nan

        skip = self.floquet_window
        return _expanding_zscore(
            vals, self.rolling_window, self.min_expanding, T,
            skip_nan_start=skip,
        )
