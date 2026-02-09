"""
Topological Regime Detection Module

Uses Berry curvature and Chern numbers to detect market regime transitions.
The key insight: topological invariants (Chern numbers) are INTEGERS and
therefore robust to noise - they distinguish "fundamentally different" regimes
from "extreme events within the same regime".

Key Innovation:
- delta_C = 0 -> same regime (even if extreme, like a flash crash)
- delta_C != 0 -> topological transition (true regime change)
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from enum import Enum
import warnings

from .core import QCMLGeometry


class RegimeType(Enum):
    """Classification of market regimes based on topology."""
    NORMAL = "normal"
    STRESSED = "stressed"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    TRANSITION = "transition"


@dataclass
class RegimeTransition:
    """Record of a detected regime transition."""
    start_idx: int
    end_idx: int
    start_time: Optional[float]
    end_time: Optional[float]
    chern_before: float
    chern_after: float
    delta_chern: float
    is_topological: bool
    confidence: float
    regime_before: Optional[RegimeType] = None
    regime_after: Optional[RegimeType] = None


@dataclass
class RegimeState:
    """Current state of regime detection."""
    current_chern: float
    current_regime: RegimeType
    curvature_history: List[float]
    chern_history: List[float]
    transitions: List[RegimeTransition]


class TopologicalRegimeDetector:
    """
    Detects market regime changes using topological invariants.

    Computes Berry curvature over rolling windows and integrates to obtain
    Chern numbers. Discontinuities in the Chern number signal topological
    transitions (true regime changes).

    Attributes:
        geometry: QCMLGeometry instance for computing Berry curvature
        window_size: Size of rolling window for Chern computation
        chern_threshold: Threshold for detecting topological transitions
    """

    def __init__(self, geometry: QCMLGeometry,
                 window_size: int = 50,
                 chern_threshold: float = 0.5,
                 smoothing_window: int = 5):
        self.geometry = geometry
        self.window_size = window_size
        self.chern_threshold = chern_threshold
        self.smoothing_window = smoothing_window
        self._state: Optional[RegimeState] = None

    def compute_berry_curvature_series(self, X: np.ndarray,
                                       indices: Tuple[int, int] = (0, 1),
                                       epsilon: float = 1e-5) -> np.ndarray:
        """Compute Berry curvature at each point in a time series."""
        X = np.asarray(X)
        T = X.shape[0]
        a, b = indices

        F = np.zeros(T)
        for t in range(T):
            F_full = self.geometry.berry_curvature(X[t], epsilon=epsilon)
            F[t] = F_full[a, b]

        return F

    def compute_local_chern(self, X: np.ndarray,
                           indices: Tuple[int, int] = (0, 1),
                           method: str = 'trapezoid') -> float:
        """Compute local Chern number for a data window."""
        X = np.asarray(X)
        n_points = X.shape[0]

        if n_points < 4:
            return 0.0

        a, b = indices

        if method == 'trapezoid':
            total = 0.0
            for i in range(n_points - 1):
                F_i = self.geometry.berry_curvature(X[i])
                F_ip1 = self.geometry.berry_curvature(X[i + 1])
                F_avg = 0.5 * (F_i[a, b] + F_ip1[a, b])
                dx = X[i + 1] - X[i]
                if len(dx) >= 2:
                    area = abs(dx[a]) * abs(dx[b])
                else:
                    area = abs(dx[0]) ** 2
                total += F_avg * area
            return total / (2 * np.pi)

        elif method == 'monte_carlo':
            n_samples = min(1000, n_points * 10)
            total = 0.0
            rng = np.random.default_rng(42)

            x_min = X[:, a].min()
            x_max = X[:, a].max()
            y_min = X[:, b].min()
            y_max = X[:, b].max()
            area_bbox = (x_max - x_min) * (y_max - y_min)

            for _ in range(n_samples):
                x_rand = rng.uniform(x_min, x_max)
                y_rand = rng.uniform(y_min, y_max)
                x_test = X[0].copy()
                x_test[a] = x_rand
                x_test[b] = y_rand
                F = self.geometry.berry_curvature(x_test)
                total += F[a, b]

            return total * area_bbox / (n_samples * 2 * np.pi)

        else:
            raise ValueError(f"Unknown method: {method}")

    def rolling_chern_number(self, X: np.ndarray,
                            window: Optional[int] = None,
                            indices: Tuple[int, int] = (0, 1),
                            stride: int = 1) -> np.ndarray:
        """Compute Chern number over rolling windows."""
        X = np.asarray(X)
        T = X.shape[0]
        window = window or self.window_size

        if T < window:
            warnings.warn(f"Data length {T} < window size {window}")
            return np.array([self.compute_local_chern(X, indices)])

        n_windows = (T - window) // stride + 1
        C = np.zeros(n_windows)

        for i in range(n_windows):
            start = i * stride
            end = start + window
            C[i] = self.compute_local_chern(X[start:end], indices)

        return C

    def detect_transitions(self, X: np.ndarray,
                          times: Optional[np.ndarray] = None,
                          indices: Tuple[int, int] = (0, 1),
                          min_separation: int = 10) -> List[RegimeTransition]:
        """Detect regime transitions from time series data."""
        X = np.asarray(X)
        T = X.shape[0]

        if times is None:
            times = np.arange(T, dtype=float)

        C = self.rolling_chern_number(X, indices=indices)

        if self.smoothing_window > 1 and len(C) > self.smoothing_window:
            kernel = np.ones(self.smoothing_window) / self.smoothing_window
            C_smooth = np.convolve(C, kernel, mode='valid')
            offset = (len(C) - len(C_smooth)) // 2
        else:
            C_smooth = C
            offset = 0

        transitions = []
        dC = np.diff(C_smooth)

        i = 0
        while i < len(dC):
            if abs(dC[i]) > self.chern_threshold:
                start_idx = (i + offset) * 1
                end_idx = start_idx + self.window_size
                start_idx = min(start_idx, T - 1)
                end_idx = min(end_idx, T - 1)
                confidence = min(1.0, abs(dC[i]) / 1.0)

                transition = RegimeTransition(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    start_time=float(times[start_idx]),
                    end_time=float(times[end_idx]),
                    chern_before=C_smooth[i],
                    chern_after=C_smooth[i + 1] if i + 1 < len(C_smooth) else C_smooth[i],
                    delta_chern=dC[i],
                    is_topological=abs(dC[i]) > self.chern_threshold,
                    confidence=confidence
                )
                transitions.append(transition)
                i += min_separation
            else:
                i += 1

        return transitions

    def classify_event(self, X_before: np.ndarray, X_event: np.ndarray,
                      X_after: np.ndarray,
                      indices: Tuple[int, int] = (0, 1)) -> Dict:
        """Classify whether an event is a regime change or extreme event."""
        C_before = self.compute_local_chern(X_before, indices)
        C_event = self.compute_local_chern(X_event, indices)
        C_after = self.compute_local_chern(X_after, indices)

        delta_C = C_after - C_before
        is_topological = abs(delta_C) > self.chern_threshold

        F_event = self.compute_berry_curvature_series(X_event, indices)
        F_std = np.std(F_event)
        F_max = np.max(np.abs(F_event))

        if is_topological:
            event_type = "REGIME_CHANGE"
            explanation = f"Topological transition detected: delta_C = {delta_C:.2f}"
        elif F_max > 3 * F_std:
            event_type = "EXTREME_EVENT"
            explanation = f"Extreme event within same regime: delta_C = {delta_C:.2f} < threshold"
        else:
            event_type = "NORMAL_FLUCTUATION"
            explanation = f"Normal market fluctuation: delta_C = {delta_C:.2f}"

        return {
            'event_type': event_type,
            'is_topological': is_topological,
            'chern_before': C_before,
            'chern_during': C_event,
            'chern_after': C_after,
            'delta_chern': delta_C,
            'curvature_volatility': F_std,
            'max_curvature': F_max,
            'explanation': explanation
        }

    def compute_regime_signature(self, X: np.ndarray,
                                indices: Tuple[int, int] = (0, 1)) -> Dict:
        """Compute topological signature of a market regime."""
        X = np.asarray(X)

        C = self.compute_local_chern(X, indices)

        F = self.compute_berry_curvature_series(X, indices)
        F_mean = np.mean(F)
        F_std = np.std(F)
        F_skew = np.mean(((F - F_mean) / (F_std + 1e-8)) ** 3)
        F_kurt = np.mean(((F - F_mean) / (F_std + 1e-8)) ** 4) - 3

        gaps = [self.geometry.spectral_gap(x) for x in X[::max(1, len(X)//10)]]
        gap_mean = np.mean(gaps)
        gap_std = np.std(gaps)

        g_traces = []
        g_dets = []
        for x in X[::max(1, len(X)//10)]:
            g = self.geometry.quantum_metric(x)
            g_traces.append(np.trace(g))
            g_dets.append(max(0, np.linalg.det(g)))

        return {
            'chern_number': C,
            'rounded_chern': round(C),
            'curvature_mean': F_mean,
            'curvature_std': F_std,
            'curvature_skewness': F_skew,
            'curvature_kurtosis': F_kurt,
            'spectral_gap_mean': gap_mean,
            'spectral_gap_std': gap_std,
            'metric_trace_mean': np.mean(g_traces),
            'metric_trace_std': np.std(g_traces),
            'metric_det_mean': np.mean(g_dets),
        }

    def online_update(self, x_new: np.ndarray,
                     indices: Tuple[int, int] = (0, 1)) -> Optional[RegimeTransition]:
        """Update detector with new observation (online mode)."""
        if self._state is None:
            self._state = RegimeState(
                current_chern=0.0,
                current_regime=RegimeType.NORMAL,
                curvature_history=[],
                chern_history=[],
                transitions=[]
            )

        F = self.geometry.berry_curvature(x_new)
        F_val = F[indices[0], indices[1]]
        self._state.curvature_history.append(F_val)

        max_history = 10 * self.window_size
        if len(self._state.curvature_history) > max_history:
            self._state.curvature_history = self._state.curvature_history[-max_history:]

        if len(self._state.curvature_history) >= self.window_size:
            recent = self._state.curvature_history[-self.window_size:]
            C_approx = np.sum(recent) / (2 * np.pi * self.window_size)
            self._state.chern_history.append(C_approx)

            if len(self._state.chern_history) >= 2:
                delta_C = C_approx - self._state.current_chern

                if abs(delta_C) > self.chern_threshold:
                    transition = RegimeTransition(
                        start_idx=len(self._state.chern_history) - 2,
                        end_idx=len(self._state.chern_history) - 1,
                        start_time=None,
                        end_time=None,
                        chern_before=self._state.current_chern,
                        chern_after=C_approx,
                        delta_chern=delta_C,
                        is_topological=True,
                        confidence=min(1.0, abs(delta_C))
                    )

                    self._state.transitions.append(transition)
                    self._state.current_chern = C_approx
                    return transition

            self._state.current_chern = C_approx

        return None

    def reset(self):
        """Reset online state."""
        self._state = None

    def get_state(self) -> Optional[RegimeState]:
        """Get current detector state."""
        return self._state


class MultiScaleRegimeDetector:
    """
    Multi-scale regime detection using multiple window sizes.

    Different time scales reveal different types of regime changes:
    - Short windows: Fast regime shifts (flash events)
    - Medium windows: Standard regime changes (sector rotations)
    - Long windows: Structural changes (paradigm shifts)
    """

    def __init__(self, geometry: QCMLGeometry,
                 window_sizes: List[int] = [20, 50, 100, 200],
                 chern_threshold: float = 0.5):
        self.detectors = [
            TopologicalRegimeDetector(
                geometry=geometry,
                window_size=w,
                chern_threshold=chern_threshold
            )
            for w in window_sizes
        ]
        self.window_sizes = window_sizes

    def analyze(self, X: np.ndarray,
               times: Optional[np.ndarray] = None,
               indices: Tuple[int, int] = (0, 1)) -> Dict:
        """Analyze data at multiple scales."""
        results = {}

        for detector, window in zip(self.detectors, self.window_sizes):
            transitions = detector.detect_transitions(X, times, indices)
            results[f'scale_{window}'] = {
                'window_size': window,
                'n_transitions': len(transitions),
                'transitions': transitions,
                'chern_series': detector.rolling_chern_number(X, indices=indices)
            }

        all_transitions = []
        for scale_result in results.values():
            all_transitions.extend(scale_result['transitions'])

        confirmed = []
        for t1 in all_transitions:
            confirmations = 0
            for t2 in all_transitions:
                if t1 is not t2:
                    if (t1.start_idx <= t2.end_idx and t2.start_idx <= t1.end_idx):
                        confirmations += 1
            if confirmations >= len(self.window_sizes) // 2:
                confirmed.append(t1)

        results['cross_scale'] = {
            'total_transitions': len(all_transitions),
            'confirmed_transitions': len(confirmed),
            'confirmed_list': confirmed
        }

        return results


def analyze_historical_crises(geometry: QCMLGeometry,
                             X: np.ndarray,
                             times: np.ndarray,
                             crisis_periods: List[Tuple[str, int, int]],
                             indices: Tuple[int, int] = (0, 1)) -> Dict:
    """Analyze historical crisis periods for topological transitions."""
    detector = TopologicalRegimeDetector(geometry, window_size=50)

    results = {}

    for name, start_idx, end_idx in crisis_periods:
        pre_start = max(0, start_idx - 100)
        post_end = min(len(X), end_idx + 100)

        X_before = X[pre_start:start_idx]
        X_during = X[start_idx:end_idx]
        X_after = X[end_idx:post_end]

        if len(X_before) < 10 or len(X_during) < 5 or len(X_after) < 10:
            results[name] = {'error': 'Insufficient data'}
            continue

        classification = detector.classify_event(X_before, X_during, X_after, indices)

        results[name] = {
            'period': (start_idx, end_idx),
            'classification': classification,
            'signature_before': detector.compute_regime_signature(X_before, indices),
            'signature_during': detector.compute_regime_signature(X_during, indices),
            'signature_after': detector.compute_regime_signature(X_after, indices)
        }

    return results
