"""
TDA (Topological Data Analysis) Baseline Regime Detector

Implements persistent homology-based regime detection using sliding
windows of financial time series.  For each window:
  1. Compute Vietoris-Rips persistence diagram.
  2. Extract H_0 (connected components) and H_1 (loops) Betti numbers.
  3. Score = total persistence = Σ(death - birth) for significant features.

Requires: ripser, persim (pip install ripser persim)

Author: QCML Research
"""

import logging
from typing import Optional

import numpy as np

from qcml.regime.classical_baselines import BaseRegimeDetector

logger = logging.getLogger(__name__)

try:
    from ripser import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False


class TDARegimeDetector(BaseRegimeDetector):
    """Persistent homology regime detector.

    Uses rolling window persistent homology on feature matrices.
    For each window of data points, computes Vietoris-Rips persistence
    diagrams and extracts total persistence as a regime score.

    High total persistence = complex topological structure = potential
    regime transition.

    Args:
        window_size: Sliding window size (number of data points).
        max_dim: Maximum homology dimension to compute (0 = H_0, 1 = H_1).
        max_edge_length: Maximum edge length for Vietoris-Rips complex.
        persistence_threshold: Minimum persistence (death - birth) to count.
        seed: Random seed (unused, for interface compatibility).
    """

    def __init__(
        self,
        window_size: int = 20,
        max_dim: int = 1,
        max_edge_length: float = np.inf,
        persistence_threshold: float = 0.0,
        seed: int = 42,
    ):
        if not HAS_RIPSER:
            raise ImportError(
                "ripser is required for TDARegimeDetector. "
                "Install with: pip install ripser"
            )
        self.window_size = window_size
        self.max_dim = max_dim
        self.max_edge_length = max_edge_length
        self.persistence_threshold = persistence_threshold
        self.seed = seed
        self._fitted = False

    @property
    def name(self) -> str:
        return "TDA Persistence"

    def fit(self, X: np.ndarray, **kwargs) -> 'TDARegimeDetector':
        """Fit is a no-op (TDA is non-parametric)."""
        self._fitted = True
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute total persistence for sliding windows.

        Args:
            X: Feature matrix (T, d).

        Returns:
            scores: (T,) array — total persistence at each time step.
                    NaN for the first window_size-1 steps.
        """
        T = X.shape[0]
        scores = np.full(T, np.nan)

        for t in range(self.window_size - 1, T):
            window = X[t - self.window_size + 1:t + 1]

            try:
                result = ripser(
                    window,
                    maxdim=self.max_dim,
                    thresh=self.max_edge_length,
                )
                diagrams = result["dgms"]

                total_persistence = 0.0
                for dim_dgm in diagrams:
                    for birth, death in dim_dgm:
                        if np.isinf(death):
                            continue
                        pers = death - birth
                        if pers > self.persistence_threshold:
                            total_persistence += pers

                scores[t] = total_persistence

            except Exception as e:
                logger.debug(f"TDA computation failed at t={t}: {e}")
                scores[t] = np.nan

        return scores

    def compute_betti_numbers(self, X: np.ndarray) -> dict:
        """Compute rolling Betti numbers for comparison with Chern numbers.

        Returns dict with:
            betti_0: (T,) — number of connected components.
            betti_1: (T,) — number of loops/cycles.
            total_persistence: (T,) — same as compute_regime_scores.
        """
        T = X.shape[0]
        betti_0 = np.full(T, np.nan)
        betti_1 = np.full(T, np.nan)
        total_pers = np.full(T, np.nan)

        for t in range(self.window_size - 1, T):
            window = X[t - self.window_size + 1:t + 1]

            try:
                result = ripser(
                    window,
                    maxdim=self.max_dim,
                    thresh=self.max_edge_length,
                )
                diagrams = result["dgms"]

                # H_0 Betti: count features with finite death
                if len(diagrams) > 0:
                    h0 = diagrams[0]
                    finite_h0 = h0[~np.isinf(h0[:, 1])]
                    betti_0[t] = len(finite_h0) + 1  # +1 for infinite component

                # H_1 Betti: count 1-cycles
                if len(diagrams) > 1:
                    h1 = diagrams[1]
                    finite_h1 = h1[~np.isinf(h1[:, 1])]
                    significant = finite_h1[
                        (finite_h1[:, 1] - finite_h1[:, 0]) > self.persistence_threshold
                    ]
                    betti_1[t] = len(significant)
                else:
                    betti_1[t] = 0

                # Total persistence
                total = 0.0
                for dim_dgm in diagrams:
                    for birth, death in dim_dgm:
                        if not np.isinf(death):
                            pers = death - birth
                            if pers > self.persistence_threshold:
                                total += pers
                total_pers[t] = total

            except Exception as e:
                logger.debug(f"TDA Betti computation failed at t={t}: {e}")

        return {
            "betti_0": betti_0,
            "betti_1": betti_1,
            "total_persistence": total_pers,
        }
