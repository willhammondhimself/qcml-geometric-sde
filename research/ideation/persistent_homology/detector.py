"""
Persistent Homology Detector for Financial Regime Detection

Uses Topological Data Analysis (TDA) on rolling correlation networks to detect
regime changes. The core insight is that during crises the correlation structure
of equity markets simplifies — assets all move together — which collapses the
topological complexity of the correlation network.

Background
----------
Given a rolling window of asset returns, we:
1. Compute the pairwise correlation matrix Σ ∈ R^{n×n}.
2. Convert to a geodesic distance matrix: D_ij = sqrt(2 * (1 - |ρ_ij|)).
   This is a proper metric on the space of correlation matrices and is the
   standard "correlation distance" used in financial network analysis
   (Mantegna 1999).
3. Run the Vietoris-Rips filtration over D and extract the persistence diagram
   via ripser (Bauer 2021).
4. Extract the H0 total persistence: sum of finite H0 bar lengths.
   H0 bars represent connected components merging as the filtration radius
   grows — shorter total H0 persistence means the network is already highly
   connected at small radii (i.e., all assets co-move tightly = crisis mode).
5. Z-score with an expanding window (causal, no look-ahead) to produce a
   standardized regime score.

During a crisis, assets co-move → correlation matrix becomes near-rank-1 →
distance matrix shrinks → assets all connect at small filtration radius →
shorter H0 bars → low total H0 persistence → negative deviation from baseline
→ inverted to produce a high positive regime score.

Note on H1 (loops):
With a small number of assets (n ≈ 8), H1 bars (1-cycles/loops) are almost
always zero: the correlation distance matrix on 8 points rarely forms
independent loops under Vietoris-Rips. H0 persistence is the robust signal
for this dimensionality.  For larger asset universes (n ≥ 30), H1 becomes
informative (Gidea & Katz 2018).

References
----------
- Mantegna (1999). Hierarchical structure in financial markets.
  European Physical Journal B 11, 193-197.
- Bauer (2021). Ripser: efficient computation of Vietoris-Rips persistence barcodes.
  Journal of Applied and Computational Topology 5(3), 391-423.
- Gidea & Katz (2018). Topological data analysis of financial time series.
  Physica A 491, 820-834.
"""

import logging
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Lazy import: ripser is required; guard with a helpful error.
try:
    from ripser import ripser as _ripser_compute
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False
    logger.warning("ripser not found. Install with: pip install ripser")


# =============================================================================
# Public interface: follows the BaseRegimeDetector ABC from observables.py
# but does not inherit it to keep this detector self-contained and importable
# without the QCML core library installed.
# =============================================================================


class PersistentHomologyDetector:
    """Regime detector based on persistent homology of the rolling correlation network.

    At each time step, computes a Vietoris-Rips filtration over the pairwise
    correlation distances of a multi-asset return window, then uses H0 total
    persistence as the primary regime signal.

    Lower raw H0 persistence → assets all connect at small filtration radius
    → near-rank-1 correlation matrix → crisis co-movement.

    We negate and z-score so that *higher* output scores indicate regime stress,
    consistent with other detectors in the observatory.

    Args:
        rolling_window: Number of trading days in each rolling returns window
            used to compute the correlation matrix.  60 gives ~3 months.
        min_expanding: Minimum observations (from rolling_window start) before
            z-scoring begins. Default 120 (2 rolling windows).
        maxdim: Maximum homology dimension computed by ripser.
            For n_assets ≤ 15, use maxdim=1; higher dims add cost with no gain.
        score_mode: Which persistence quantity to use as the raw signal.
            - 'h0_total'   : sum of finite H0 bar lengths (recommended for n ≤ 20)
            - 'h0h1_combo' : h0_weight*H0_total + h1_weight*H1_total
        h0_weight: Weight on H0 total when score_mode='h0h1_combo'.
        h1_weight: Weight on H1 total when score_mode='h0h1_combo'.
        corr_regularization: (unused; kept for API consistency)
        min_corr_assets: Minimum number of non-NaN assets required in a window.
        smooth_window: Rolling average applied to raw persistence scores before
            z-scoring.  Reduces day-to-day noise.  Set 1 to disable.
        seed: Random seed (not currently used; included for API consistency).
    """

    def __init__(
        self,
        rolling_window: int = 60,
        min_expanding: int = 120,
        maxdim: int = 1,
        score_mode: str = 'h0_total',
        h0_weight: float = 0.7,
        h1_weight: float = 0.3,
        corr_regularization: float = 1e-6,
        min_corr_assets: int = 4,
        smooth_window: int = 10,
        seed: int = 42,
    ):
        if not HAS_RIPSER:
            raise ImportError(
                "ripser is required for PersistentHomologyDetector. "
                "Install with: pip install ripser"
            )

        self.rolling_window = rolling_window
        self.min_expanding = min_expanding
        self.maxdim = maxdim
        self.score_mode = score_mode
        self.h0_weight = h0_weight
        self.h1_weight = h1_weight
        self.corr_regularization = corr_regularization
        self.min_corr_assets = min_corr_assets
        self.smooth_window = smooth_window
        self.seed = seed

        self._is_fitted: bool = False
        self._returns_df: Optional[pd.DataFrame] = None

    @property
    def name(self) -> str:
        return f"Persistent Homology ({self.score_mode})"

    def fit(self, returns_df: pd.DataFrame) -> 'PersistentHomologyDetector':
        """Store the multi-asset returns DataFrame for regime scoring.

        Unlike QCML detectors there is no parametric fit — the method is fully
        non-parametric and causal (each window uses only past data).

        Args:
            returns_df: DataFrame of shape (T, n_assets) with log-returns.
                Index must be a DatetimeIndex.  Columns are asset ticker symbols.

        Returns:
            self
        """
        if not isinstance(returns_df, pd.DataFrame):
            raise TypeError("returns_df must be a pandas DataFrame")
        if returns_df.shape[1] < self.min_corr_assets:
            raise ValueError(
                f"Need at least {self.min_corr_assets} asset columns; "
                f"got {returns_df.shape[1]}."
            )

        self._returns_df = returns_df.copy()
        self._is_fitted = True

        logger.info(
            "PersistentHomologyDetector fitted: "
            f"T={len(returns_df)}, n_assets={returns_df.shape[1]}, "
            f"rolling_window={self.rolling_window}, maxdim={self.maxdim}, "
            f"score_mode={self.score_mode}"
        )
        return self

    def _correlation_distance_matrix(self, window_returns: np.ndarray) -> Optional[np.ndarray]:
        """Compute pairwise correlation distance matrix for a return window.

        Uses Mantegna's (1999) correlation distance: D_ij = sqrt(2*(1 - |rho_ij|)).
        This satisfies the triangle inequality and maps [-1,+1] correlations
        to distances in [0, sqrt(2)].

        Args:
            window_returns: Array of shape (T_window, n_assets) with log-returns.

        Returns:
            Distance matrix of shape (n_assets_valid, n_assets_valid), or None
            if too few assets have nonzero variance.
        """
        std = np.std(window_returns, axis=0)
        valid = std > 1e-8
        if valid.sum() < self.min_corr_assets:
            return None

        w = window_returns[:, valid]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = np.corrcoef(w.T)

        # Clip to [-1, 1] to handle floating-point edge cases
        corr = np.clip(corr, -1.0, 1.0)

        # Mantegna correlation distance: sqrt(2 * (1 - |rho|))
        dist = np.sqrt(2.0 * (1.0 - np.abs(corr)))
        np.fill_diagonal(dist, 0.0)

        return dist

    def _persistence_components(self, dist_matrix: np.ndarray) -> Tuple[float, float]:
        """Run Vietoris-Rips filtration and return H0 and H1 total persistence.

        Args:
            dist_matrix: Square distance matrix of shape (n, n).

        Returns:
            Tuple of (h0_total, h1_total) where each is the sum of finite
            bar lengths in the corresponding persistence diagram.
        """
        result = _ripser_compute(
            dist_matrix,
            maxdim=self.maxdim,
            distance_matrix=True,
        )
        diagrams = result['dgms']

        # H0: connected components
        h0_total = 0.0
        if len(diagrams) > 0:
            h0_diag = diagrams[0]
            # The last H0 bar has infinite death (the single surviving component).
            # Exclude it — only finite bars carry information about merging distances.
            finite_h0 = h0_diag[np.isfinite(h0_diag[:, 1])]
            if len(finite_h0) > 0:
                h0_total = float(np.sum(finite_h0[:, 1] - finite_h0[:, 0]))

        # H1: loops (1-cycles)
        h1_total = 0.0
        if len(diagrams) > 1:
            h1_diag = diagrams[1]
            finite_h1 = h1_diag[np.isfinite(h1_diag[:, 1])]
            if len(finite_h1) > 0:
                h1_total = float(np.sum(finite_h1[:, 1] - finite_h1[:, 0]))

        return h0_total, h1_total

    def _aggregate_score(self, h0_total: float, h1_total: float) -> float:
        """Aggregate H0 and H1 persistence into a single raw score.

        Args:
            h0_total: Total H0 persistence (sum of finite bar lengths).
            h1_total: Total H1 persistence.

        Returns:
            Scalar raw persistence score.
        """
        if self.score_mode == 'h0_total':
            return h0_total
        elif self.score_mode == 'h0h1_combo':
            return self.h0_weight * h0_total + self.h1_weight * h1_total
        else:
            raise ValueError(f"Unknown score_mode: {self.score_mode!r}")

    def compute_regime_scores(self) -> Tuple[np.ndarray, pd.DatetimeIndex]:
        """Compute causal, z-scored persistence regime scores for the full series.

        For each time step t ≥ rolling_window, computes the persistence score
        from the returns in [t - rolling_window, t).  After smoothing, applies
        an expanding-window z-score (causal, no look-ahead).

        Crisis periods are characterized by *low* raw H0 persistence (topological
        simplification due to co-movement).  We negate so that *large positive*
        z-scores indicate regime stress, matching other detectors in the observatory.

        Returns:
            Tuple of:
                scores: np.ndarray of shape (T,), NaN for warm-up period.
                dates: DatetimeIndex of length T aligned with scores.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before compute_regime_scores().")

        df = self._returns_df
        returns = df.values   # shape (T, n_assets)
        T = len(returns)
        dates = df.index

        # Step 1: Compute raw persistence values at each time step
        # Only for t ≥ rolling_window (need a full window)
        raw_scores = np.full(T, np.nan)

        for t in range(self.rolling_window, T):
            window = returns[t - self.rolling_window:t]
            dist_mat = self._correlation_distance_matrix(window)
            if dist_mat is None:
                continue
            try:
                h0, h1 = self._persistence_components(dist_mat)
                raw_scores[t] = self._aggregate_score(h0, h1)
            except Exception as exc:
                logger.warning(f"t={t} ({dates[t]}): ripser failed ({exc}), skipping.")

        # Step 2: Smooth raw scores — only where defined (avoids NaN propagation)
        # We forward-fill over any isolated NaN gaps before smoothing.
        raw_series = pd.Series(raw_scores)
        if self.smooth_window > 1:
            smoothed_values = (
                raw_series
                .rolling(window=self.smooth_window, min_periods=1)
                .mean()
                .values
            )
        else:
            smoothed_values = raw_scores.copy()

        # Step 3: Expanding-window robust z-score (causal, no look-ahead).
        # Reference window starts at rolling_window (first valid raw score).
        # Negate: crisis = low persistence = negative deviation → positive z.
        z_scores = np.full(T, np.nan)

        # Collect valid indices: t where smoothed_values[t] is not NaN
        # and we have enough history.  The effective start is rolling_window +
        # min_expanding so we have min_expanding valid points in the past.
        effective_start = self.rolling_window + self.min_expanding

        for t in range(effective_start, T):
            # Past window: all points from rolling_window to t (exclusive)
            past = smoothed_values[self.rolling_window:t]
            past_valid = past[~np.isnan(past)]
            if len(past_valid) < 10:
                continue
            if np.isnan(smoothed_values[t]):
                continue

            mu = np.median(past_valid)
            mad = np.median(np.abs(past_valid - mu))
            sigma = 1.4826 * mad  # MAD → consistent std estimator for Gaussian
            if sigma < 1e-12:
                z_scores[t] = 0.0
            else:
                # Negate: crisis has low persistence → z-score is positive
                z_scores[t] = -(smoothed_values[t] - mu) / sigma

        return z_scores, dates
