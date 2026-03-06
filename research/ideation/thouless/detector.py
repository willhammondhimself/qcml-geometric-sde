"""
Thouless Energy Detector for Financial Regime Detection

Implements the Thouless energy (conductance) and level statistics of the
error Hamiltonian H(x_t) as regime detection signals.

Physical motivation:
--------------------
In disordered quantum systems, the Thouless energy

    g_T = delta_E / E_bandwidth

is the dimensionless conductance measuring the ratio of mean level spacing
to the single-particle bandwidth. The localization-delocalization transition
is captured by:

    g_T >> 1  →  extended (metallic, delocalized, GOE statistics)
    g_T ~  1  →  critical point (Anderson transition)
    g_T << 1  →  localized (insulating, Poisson statistics)

A complementary statistic is the consecutive level spacing ratio (Oganesyan
& Huse 2007):

    r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1})   where  s_n = E_{n+1} - E_n

This is bounded in [0, 1] with:

    <r>_GOE   ≈ 0.536  (Wigner-Dyson, extended/chaotic)
    <r>_Poisson ≈ 0.386  (localized/integrable)

Crisis hypothesis:
------------------
During market crises, correlations collapse and risk factors become
undiversified (localized). The Hamiltonian spectrum should show:

1. Reduced level repulsion (gap narrowing → g_T drops)
2. Transition toward Poisson statistics (r drops toward 0.386)
3. Spectral compression / bandwidth collapse

Both g_T and r are purely spectral (no eigenstate information needed),
making them fast and numerically stable.

References:
    Thouless (1974), Phys Rep 13, 93
    Oganesyan & Huse (2007), Phys Rev B 75, 155111
    Edwards & Thouless (1972), J Phys C 5, 807
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


# ---------------------------------------------------------------------------
# Shared helper utilities (mirrors loschmidt_echo/detector.py pattern)
# ---------------------------------------------------------------------------

def _expanding_zscore(
    raw_values: np.ndarray,
    rolling_window: int,
    min_expanding: int,
    T: int,
    skip_nan_start: int = 0,
) -> np.ndarray:
    """Compute expanding-window z-score of rolling-mean values.

    Matches the pattern used by other observatory detectors.

    Args:
        raw_values: Raw signal array of length T.
        rolling_window: Window for rolling mean smoothing.
        min_expanding: Minimum number of past observations for z-score.
        T: Total number of time steps.
        skip_nan_start: Skip this many leading entries when building the
            expanding past (accounts for mandatory NaN prefix).

    Returns:
        z_scores: Array of length T with NaN for the burn-in period.
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


def _standard_init(detector, **kwargs) -> None:
    """Shared __init__ logic: set all kwargs as attributes plus QCML state."""
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
    """Shared fit logic: scale → PCA → QCML geometry."""
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


# ---------------------------------------------------------------------------
# Thouless energy spectral functions
# ---------------------------------------------------------------------------

def compute_thouless_statistics(
    geometry: QCMLGeometry,
    x: np.ndarray,
    min_levels: int = 3,
) -> dict:
    """Compute Thouless energy and level spacing ratio for one feature vector.

    Builds the error Hamiltonian H(x) from the fitted QCML geometry, computes
    its real eigenvalue spectrum, and extracts:

    1. Thouless ratio g_T = delta_E / E_bandwidth
       where delta_E = mean consecutive level spacing,
       E_bandwidth = max(eigenvalues) - min(eigenvalues)

    2. Level spacing ratio r = mean_n [ min(s_n, s_{n+1}) / max(s_n, s_{n+1}) ]
       where s_n = E_{n+1} - E_n (Oganesyan & Huse 2007)

    3. Spectral bandwidth W = max(E) - min(E)

    Crisis signal: g_T decreases (localization); r decreases toward Poisson
    limit ≈ 0.386 (from GOE ≈ 0.536).

    Args:
        geometry: Fitted QCMLGeometry instance.
        x: Feature vector of shape (n_features,).
        min_levels: Minimum number of eigenvalues required. Returns NaN dict
            if the spectrum is too small.

    Returns:
        Dictionary with keys:
            'thouless_ratio': g_T = delta_E / W  (dimensionless, in [0, 1])
            'level_spacing_ratio': mean r_n  (in [0, 1])
            'bandwidth': W = max(E) - min(E)  (same units as eigenvalues)
            'mean_spacing': delta_E = W / (N - 1) where N = hilbert_dim
    """
    x = np.asarray(x, dtype=float)
    H = geometry.error_hamiltonian(x)

    # Eigenvalues only (no eigenvectors needed — much faster)
    eigenvalues = np.sort(np.linalg.eigvalsh(H).real)
    N = len(eigenvalues)

    nan_result = {
        'thouless_ratio': np.nan,
        'level_spacing_ratio': np.nan,
        'bandwidth': np.nan,
        'mean_spacing': np.nan,
    }

    if N < min_levels:
        return nan_result

    bandwidth = float(eigenvalues[-1] - eigenvalues[0])
    if bandwidth < 1e-14:
        # Completely degenerate spectrum — fully localized limit
        return {
            'thouless_ratio': 0.0,
            'level_spacing_ratio': 0.0,
            'bandwidth': 0.0,
            'mean_spacing': 0.0,
        }

    # Consecutive spacings s_n = E_{n+1} - E_n
    spacings = np.diff(eigenvalues)  # length N-1, all >= 0

    mean_spacing = float(np.mean(spacings))
    thouless_ratio = float(mean_spacing / bandwidth)  # = 1/(N-1) in uniform case

    # Level spacing ratio r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1})
    # Requires at least 2 spacings (N >= 3)
    if len(spacings) >= 2:
        s_curr = spacings[:-1]
        s_next = spacings[1:]
        s_min = np.minimum(s_curr, s_next)
        s_max = np.maximum(s_curr, s_next)
        # Guard against consecutive degenerate levels
        valid = s_max > 1e-16
        if np.any(valid):
            r_values = s_min[valid] / s_max[valid]
            level_spacing_ratio = float(np.mean(r_values))
        else:
            level_spacing_ratio = 0.0
    else:
        # Only one spacing — cannot compute r
        level_spacing_ratio = np.nan

    return {
        'thouless_ratio': thouless_ratio,
        'level_spacing_ratio': level_spacing_ratio,
        'bandwidth': bandwidth,
        'mean_spacing': mean_spacing,
    }


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class ThoulessEnergyDetector(ExpandingWindowMixin, BaseRegimeDetector):
    """Regime detection via Thouless energy and level spacing ratio.

    Computes two dimensionless spectral statistics of the QCML error
    Hamiltonian H(x_t) at each time step:

    (a) Thouless ratio:  g_T = delta_E / W
        where delta_E is the mean consecutive level spacing and W the
        full bandwidth. Localization (crisis) → g_T decreases.

    (b) Level spacing ratio:  r = <min(s_n, s_{n+1}) / max(s_n, s_{n+1})>
        GOE statistics (healthy, extended) → r ≈ 0.536
        Poisson statistics (localized, crisis) → r ≈ 0.386

    Both signals are z-scored with an expanding window. By default the
    detector uses the *negative* z-score of both (so a high score = more
    localized = more crisis-like).

    The combined score is:

        score_t = alpha * (-z[g_T]) + (1 - alpha) * (-z[r])

    where alpha controls the relative weight of the two statistics.

    Attributes:
        scoring_mode: One of 'thouless', 'lsr', 'combined'.
            'thouless'  → use only g_T signal
            'lsr'       → use only level spacing ratio r
            'combined'  → alpha-weighted combination (default)
        alpha: Weight for Thouless ratio in combined mode (default 0.5).
    """

    def __init__(
        self,
        hilbert_dim: int = 8,
        n_pca_components: int = 15,
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
        scoring_mode: str = 'combined',
        alpha: float = 0.5,
    ):
        """Initialize Thouless Energy Detector.

        Args:
            hilbert_dim: Hilbert space dimension. Larger = more spectral levels
                = richer g_T / r statistics. Must be >= 3 for r to be defined.
            n_pca_components: Number of PCA components.
            operator_method: Operator construction ('random' recommended to
                avoid Kramers degeneracy in even-dim qubit systems).
            scale_exponent: PCA eigenvalue scaling exponent.
            rolling_window: Window for rolling mean smoothing.
            min_expanding: Minimum expanding window for z-score.
            seed: Random seed.
            causal_fit_length: Fit only on first N samples.
            expanding_refit_interval: Refit interval for expanding windows.
            normalization: Post-PCA normalization mode.
            adaptive_epsilon: Adapt epsilon to data scale.
            custom_operators: Override learned operators.
            scoring_mode: 'thouless' | 'lsr' | 'combined'.
            alpha: Weight for Thouless ratio component in combined mode.
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
        if scoring_mode not in ('thouless', 'lsr', 'combined'):
            raise ValueError(
                f"scoring_mode must be 'thouless', 'lsr', or 'combined'. Got: {scoring_mode}"
            )
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1]. Got: {alpha}")

        self.scoring_mode = scoring_mode
        self.alpha = alpha

    @property
    def name(self) -> str:
        return f"Thouless Energy ({self.scoring_mode})"

    def fit(self, X: np.ndarray, **kwargs) -> 'ThoulessEnergyDetector':
        """Fit scaler, PCA, and QCML geometry from feature matrix.

        Args:
            X: Feature matrix of shape (T, n_features).

        Returns:
            self
        """
        return _standard_qcml_fit(self, X)

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute regime scores as z-scored Thouless / LSR statistics.

        Lower g_T and lower r indicate localization / crisis regime.
        The output is negated so that higher score = more crisis-like,
        consistent with all other observatory detectors.

        Args:
            X: Feature matrix of shape (T, n_features).

        Returns:
            scores: 1-D array of length T. Higher = more regime-like.
                NaN for the burn-in period.
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
        thouless_vals = np.full(T, np.nan)
        lsr_vals = np.full(T, np.nan)

        for t in range(T):
            if self._snapshots:
                geo, xt = self._transform_point_at(X[t], t)
            else:
                geo, xt = self._geometry, Xt[t]

            try:
                stats = compute_thouless_statistics(
                    geometry=geo,
                    x=xt,
                    min_levels=3,
                )
                thouless_vals[t] = stats['thouless_ratio']
                lsr_vals[t] = stats['level_spacing_ratio']
            except Exception as exc:
                logger.warning(f"Thouless stats failed at t={t}: {exc}")
                # Leave as NaN

        # Negate: localization = low g_T / low r = high crisis score
        neg_thouless = -thouless_vals
        neg_lsr = -lsr_vals

        if self.scoring_mode == 'thouless':
            raw = neg_thouless
            skip = 0
        elif self.scoring_mode == 'lsr':
            raw = neg_lsr
            skip = 0
        else:  # combined
            # Combine raw (pre-z-score) to avoid double-smoothing artifacts.
            # Use alpha weighting; fall back gracefully when one channel is NaN.
            combined = np.full(T, np.nan)
            for t in range(T):
                gt_ok = not np.isnan(neg_thouless[t])
                lr_ok = not np.isnan(neg_lsr[t])
                if gt_ok and lr_ok:
                    combined[t] = self.alpha * neg_thouless[t] + (1 - self.alpha) * neg_lsr[t]
                elif gt_ok:
                    combined[t] = neg_thouless[t]
                elif lr_ok:
                    combined[t] = neg_lsr[t]
            raw = combined
            skip = 0

        return _expanding_zscore(raw, self.rolling_window, self.min_expanding, T, skip)
