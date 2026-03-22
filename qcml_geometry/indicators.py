"""
Novel Geometric Indicators for Regime Detection

Four novel contributions for academic research on topological regime detection:

1. Spectral Gap Early Warning: Gap (E1 - E0) narrows before regime transitions,
   analogous to quantum phase transitions (Sachdev, "Quantum Phase Transitions").

2. Ground State Energy Evolution: E0(t) indicates regime "intensity" - higher
   ground state energy corresponds to more stressed market conditions.

3. Quantum Fidelity Decay Rate: Rapid fidelity decay F(t, t+dt) = |<psi(t)|psi(t+dt)>|^2
   indicates unstable regime. Measures how quickly the quantum state changes.

4. Multi-Scale Chern Consensus: Weighted consensus across multiple time scales
   with dynamic scale selection based on regime characteristics.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .core import QCMLGeometry
from .topology import TopologicalRegimeDetector

logger = logging.getLogger(__name__)


@dataclass
class IndicatorResult:
    """
    Result from a single quantum indicator computation.

    Attributes:
        name: Indicator name
        values: Time series of indicator values
        transitions: Indices where indicator signals a transition
        threshold: Threshold used for transition detection
        metadata: Additional metadata (statistics, parameters)
    """
    name: str
    values: np.ndarray
    transitions: List[int]
    threshold: float
    metadata: Dict = field(default_factory=dict)


class SpectralGapIndicator:
    """
    Spectral gap early warning indicator.

    The spectral gap Delta = E1 - E0 of the error Hamiltonian narrows before
    regime transitions, analogous to gap closing at quantum critical points.

    Attributes:
        geometry: Fitted QCMLGeometry instance
        window_size: Rolling window for gap computation
        collapse_threshold_std: Number of std devs below mean to signal collapse
    """

    def __init__(
        self,
        geometry: QCMLGeometry,
        window_size: int = 20,
        collapse_threshold_std: float = 2.0
    ):
        self.geometry = geometry
        self.window_size = window_size
        self.collapse_threshold_std = collapse_threshold_std

    def compute_spectral_gap_series(self, X: np.ndarray) -> np.ndarray:
        """Compute spectral gap time series.

        Args:
            X: Feature array of shape (T, n_features)

        Returns:
            gaps: Array of spectral gaps, shape (T,)
        """
        T = X.shape[0]
        gaps = np.zeros(T)
        for t in range(T):
            gaps[t] = self.geometry.spectral_gap(X[t])
        return gaps

    def compute_rolling_gap(self, X: np.ndarray) -> np.ndarray:
        """Compute rolling average spectral gap."""
        gaps = self.compute_spectral_gap_series(X)
        series = pd.Series(gaps)
        rolling = series.rolling(window=self.window_size, min_periods=1).mean()
        return rolling.values

    def detect_gap_collapse(self, X: np.ndarray) -> IndicatorResult:
        """Detect spectral gap collapse events."""
        gaps = self.compute_spectral_gap_series(X)
        series = pd.Series(gaps)

        rolling_mean = series.rolling(
            window=self.window_size, min_periods=self.window_size // 2
        ).mean()
        rolling_std = series.rolling(
            window=self.window_size, min_periods=self.window_size // 2
        ).std()

        threshold_series = rolling_mean - self.collapse_threshold_std * rolling_std
        collapse_mask = series < threshold_series

        transitions = []
        in_collapse = False
        for i in range(len(collapse_mask)):
            if collapse_mask.iloc[i] and not in_collapse:
                transitions.append(i)
                in_collapse = True
            elif not collapse_mask.iloc[i]:
                in_collapse = False

        return IndicatorResult(
            name="spectral_gap",
            values=gaps,
            transitions=transitions,
            threshold=float(self.collapse_threshold_std),
            metadata={
                "mean_gap": float(np.nanmean(gaps)),
                "std_gap": float(np.nanstd(gaps)),
                "min_gap": float(np.nanmin(gaps)),
                "n_collapses": len(transitions),
                "window_size": self.window_size,
            }
        )


class EnergyEvolutionIndicator:
    """
    Ground state energy evolution indicator.

    The ground state energy E0(t) increases during stressed market conditions.

    Attributes:
        geometry: Fitted QCMLGeometry instance
        window_size: Rolling window for energy statistics
        stress_threshold_std: Std devs above mean to signal stress
    """

    def __init__(
        self,
        geometry: QCMLGeometry,
        window_size: int = 20,
        stress_threshold_std: float = 2.0
    ):
        self.geometry = geometry
        self.window_size = window_size
        self.stress_threshold_std = stress_threshold_std

    def compute_energy_series(self, X: np.ndarray) -> np.ndarray:
        """Compute ground state energy time series."""
        T = X.shape[0]
        energies = np.zeros(T)
        for t in range(T):
            _, energy = self.geometry.quasi_coherent_state(X[t], return_energy=True)
            energies[t] = energy
        return energies

    def energy_regime_classification(self, X: np.ndarray) -> IndicatorResult:
        """Classify regime based on energy evolution."""
        energies = self.compute_energy_series(X)
        series = pd.Series(energies)

        rolling_mean = series.rolling(
            window=self.window_size, min_periods=self.window_size // 2
        ).mean()
        rolling_std = series.rolling(
            window=self.window_size, min_periods=self.window_size // 2
        ).std()

        threshold_series = rolling_mean + self.stress_threshold_std * rolling_std
        stress_mask = series > threshold_series

        transitions = []
        in_stress = False
        for i in range(len(stress_mask)):
            if stress_mask.iloc[i] and not in_stress:
                transitions.append(i)
                in_stress = True
            elif not stress_mask.iloc[i]:
                in_stress = False

        return IndicatorResult(
            name="ground_state_energy",
            values=energies,
            transitions=transitions,
            threshold=float(self.stress_threshold_std),
            metadata={
                "mean_energy": float(np.nanmean(energies)),
                "std_energy": float(np.nanstd(energies)),
                "max_energy": float(np.nanmax(energies)),
                "n_stress_events": len(transitions),
                "window_size": self.window_size,
            }
        )


class FidelityDecayIndicator:
    """
    Quantum fidelity decay rate indicator.

    Rapid fidelity decay F(t, t+dt) = |<psi(t)|psi(t+dt)>|^2
    indicates an unstable regime.

    Attributes:
        geometry: Fitted QCMLGeometry instance
        lag: Time lag dt for fidelity computation
        window_size: Rolling window for fidelity statistics
        instability_threshold_std: Std devs below mean to signal instability
    """

    def __init__(
        self,
        geometry: QCMLGeometry,
        lag: int = 1,
        window_size: int = 20,
        instability_threshold_std: float = 2.0
    ):
        self.geometry = geometry
        self.lag = lag
        self.window_size = window_size
        self.instability_threshold_std = instability_threshold_std

    def compute_fidelity_series(self, X: np.ndarray) -> np.ndarray:
        """Compute fidelity time series F(t, t+lag)."""
        T = X.shape[0]
        fidelities = np.zeros(T - self.lag)

        states = []
        for t in range(T):
            psi = self.geometry.quasi_coherent_state(X[t])
            states.append(psi)

        for t in range(T - self.lag):
            overlap = np.abs(np.vdot(states[t], states[t + self.lag]))
            fidelities[t] = overlap ** 2

        return fidelities

    def stability_index(self, X: np.ndarray) -> IndicatorResult:
        """Compute stability index based on fidelity decay."""
        fidelities = self.compute_fidelity_series(X)
        series = pd.Series(fidelities)

        rolling_mean = series.rolling(
            window=self.window_size, min_periods=self.window_size // 2
        ).mean()
        rolling_std = series.rolling(
            window=self.window_size, min_periods=self.window_size // 2
        ).std()

        threshold_series = rolling_mean - self.instability_threshold_std * rolling_std
        instability_mask = series < threshold_series

        transitions = []
        in_instability = False
        for i in range(len(instability_mask)):
            if instability_mask.iloc[i] and not in_instability:
                transitions.append(i)
                in_instability = True
            elif not instability_mask.iloc[i]:
                in_instability = False

        return IndicatorResult(
            name="fidelity_decay",
            values=fidelities,
            transitions=transitions,
            threshold=float(self.instability_threshold_std),
            metadata={
                "mean_fidelity": float(np.nanmean(fidelities)),
                "std_fidelity": float(np.nanstd(fidelities)),
                "min_fidelity": float(np.nanmin(fidelities)),
                "lag": self.lag,
                "n_instabilities": len(transitions),
                "window_size": self.window_size,
            }
        )


class MultiScaleChernConsensus:
    """
    Multi-scale Chern consensus indicator.

    Computes Chern numbers at multiple time scales and produces a
    weighted consensus signal.

    Attributes:
        geometry: Fitted QCMLGeometry instance
        scales: List of window sizes for multi-scale analysis
        weights: Initial weights for each scale
        consensus_threshold: Minimum weighted agreement for transition
    """

    def __init__(
        self,
        geometry: QCMLGeometry,
        scales: List[int] = None,
        weights: Optional[List[float]] = None,
        consensus_threshold: float = 0.6,
        normalization_strategy: str = 'rolling_adaptive',
        normalization_window: Optional[int] = None,
    ):
        self.geometry = geometry
        self.scales = scales or [10, 20, 30, 50, 100]
        self.consensus_threshold = consensus_threshold
        self.normalization_strategy = normalization_strategy
        self.normalization_window = normalization_window

        if weights is not None:
            if len(weights) != len(self.scales):
                raise ValueError(
                    f"weights length ({len(weights)}) must match "
                    f"scales length ({len(self.scales)})"
                )
            total = sum(weights)
            self.weights = [w / total for w in weights]
        else:
            n = len(self.scales)
            self.weights = [1.0 / n] * n

    def compute_multi_scale_chern(self, X: np.ndarray) -> Dict[int, np.ndarray]:
        """Compute Chern series at each scale."""
        chern_dict = {}
        for scale in self.scales:
            detector = TopologicalRegimeDetector(
                geometry=self.geometry,
                window_size=scale,
                chern_threshold=0.1
            )
            chern_values = detector.rolling_chern_number(X, window=scale)
            chern_dict[scale] = chern_values
        return chern_dict

    def compute_scale_signals(
        self,
        chern_dict: Dict[int, np.ndarray],
        threshold_std: float = 2.0
    ) -> Dict[int, np.ndarray]:
        """Compute normalized magnitude signals at each scale."""
        signal_dict = {}

        for scale, chern in chern_dict.items():
            delta = np.diff(chern, prepend=chern[0])
            series = pd.Series(np.abs(delta))

            if self.normalization_strategy == 'rolling_adaptive':
                window = max(60, scale * 3)
                rolling_std = series.rolling(window=window, min_periods=20).std()
                normalizer = rolling_std

            elif self.normalization_strategy == 'rolling_fixed':
                window = self.normalization_window or 60
                rolling_std = series.rolling(window=window, min_periods=20).std()
                normalizer = rolling_std

            elif self.normalization_strategy == 'percentile':
                p95 = series.rolling(window=100, min_periods=20).quantile(0.95)
                normalizer = p95

            elif self.normalization_strategy == 'zscore':
                normalizer = series.std()

            else:
                raise ValueError(
                    f"Unknown normalization_strategy: {self.normalization_strategy}"
                )

            # Backfill NaN normalizer values from rolling windows to prevent NaN propagation
            if isinstance(normalizer, pd.Series):
                normalizer = normalizer.bfill().ffill()
            signal_dict[scale] = (series / (normalizer + 1e-8)).values

        return signal_dict

    def compute_consensus(
        self,
        X: np.ndarray,
        threshold_std: float = 2.0
    ) -> IndicatorResult:
        """Compute weighted consensus across scales."""
        chern_dict = self.compute_multi_scale_chern(X)
        signal_dict = self.compute_scale_signals(chern_dict, threshold_std)

        min_len = min(len(s) for s in signal_dict.values())

        consensus = np.zeros(min_len)
        for scale, weight in zip(self.scales, self.weights):
            signal = signal_dict[scale]
            aligned = signal[len(signal) - min_len:]
            consensus += weight * aligned

        transitions = []
        above_threshold = False
        for i in range(len(consensus)):
            if consensus[i] >= self.consensus_threshold and not above_threshold:
                transitions.append(i)
                above_threshold = True
            elif consensus[i] < self.consensus_threshold:
                above_threshold = False

        scale_correlations = {}
        scale_list = list(signal_dict.keys())
        for i in range(len(scale_list)):
            for j in range(i + 1, len(scale_list)):
                s_i = signal_dict[scale_list[i]]
                s_j = signal_dict[scale_list[j]]
                common_len = min(len(s_i), len(s_j))
                s_i_aligned = s_i[len(s_i) - common_len:]
                s_j_aligned = s_j[len(s_j) - common_len:]
                if np.std(s_i_aligned) > 0 and np.std(s_j_aligned) > 0:
                    corr = np.corrcoef(s_i_aligned, s_j_aligned)[0, 1]
                else:
                    corr = 0.0
                scale_correlations[f"{scale_list[i]}_{scale_list[j]}"] = float(corr)

        return IndicatorResult(
            name="multi_scale_consensus",
            values=consensus,
            transitions=transitions,
            threshold=self.consensus_threshold,
            metadata={
                "scales": self.scales,
                "weights": self.weights,
                "n_transitions": len(transitions),
                "mean_consensus": float(np.nanmean(consensus)),
                "max_consensus": float(np.nanmax(consensus)),
                "scale_correlations": scale_correlations,
            }
        )


class GeometricIndicatorSuite:
    """
    Suite of all geometric indicators for comprehensive regime analysis.

    Combines spectral gap, energy evolution, fidelity decay, and
    multi-scale consensus into a unified analysis framework.
    """

    def __init__(
        self,
        geometry: QCMLGeometry,
        window_size: int = 20,
        fidelity_lag: int = 1,
        scales: Optional[List[int]] = None,
        threshold_std: float = 2.0
    ):
        self.geometry = geometry

        self.spectral_gap = SpectralGapIndicator(
            geometry, window_size=window_size,
            collapse_threshold_std=threshold_std
        )
        self.energy = EnergyEvolutionIndicator(
            geometry, window_size=window_size,
            stress_threshold_std=threshold_std
        )
        self.fidelity = FidelityDecayIndicator(
            geometry, lag=fidelity_lag, window_size=window_size,
            instability_threshold_std=threshold_std
        )
        self.consensus = MultiScaleChernConsensus(
            geometry, scales=scales
        )

    def compute_all(self, X: np.ndarray) -> Dict[str, IndicatorResult]:
        """Compute all geometric indicators."""
        results = {}

        logger.info("Computing spectral gap indicator...")
        results["spectral_gap"] = self.spectral_gap.detect_gap_collapse(X)

        logger.info("Computing ground state energy indicator...")
        results["ground_state_energy"] = self.energy.energy_regime_classification(X)

        logger.info("Computing fidelity decay indicator...")
        results["fidelity_decay"] = self.fidelity.stability_index(X)

        logger.info("Computing multi-scale consensus indicator...")
        results["multi_scale_consensus"] = self.consensus.compute_consensus(X)

        return results

    def compute_composite_score(
        self,
        X: np.ndarray,
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[np.ndarray, Dict[str, IndicatorResult]]:
        """Compute composite score combining all indicators."""
        if weights is None:
            weights = {
                "spectral_gap": 0.25,
                "ground_state_energy": 0.25,
                "fidelity_decay": 0.25,
                "multi_scale_consensus": 0.25,
            }

        results = self.compute_all(X)

        lengths = [len(r.values) for r in results.values()]
        min_len = min(lengths)

        composite = np.zeros(min_len)

        for name, result in results.items():
            vals = result.values
            aligned = vals[len(vals) - min_len:]

            mean_val = np.nanmean(aligned)
            std_val = np.nanstd(aligned)
            if std_val > 1e-10:
                z_scored = (aligned - mean_val) / std_val
            else:
                z_scored = np.zeros(min_len)

            if name in ("spectral_gap", "fidelity_decay"):
                z_scored = -z_scored

            # Replace any remaining NaN from windowed z-scoring
            z_scored = np.nan_to_num(z_scored, nan=0.0)

            composite += weights.get(name, 0.25) * z_scored

        return composite, results
