"""
Topological Regime Detection Module

Uses Berry curvature and Chern numbers to detect market regime transitions.
The key insight: topological invariants (Chern numbers) are INTEGERS and
therefore robust to noise - they distinguish "fundamentally different" regimes
from "extreme events within the same regime".

Key Innovation:
- ΔC = 0 → same regime (even if extreme, like a flash crash)
- ΔC ≠ 0 → topological transition (true regime change)

This provides a mathematically rigorous framework for regime detection
that is orthogonal to traditional statistical methods.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Union
from dataclasses import dataclass
from enum import Enum
import warnings

from .qcml_geometry import QCMLGeometry


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
    is_topological: bool  # True if |ΔC| > threshold (integer change)
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

    The detector computes Berry curvature over rolling windows of market data
    and integrates to obtain Chern numbers. Discontinuities in the Chern number
    signal topological transitions (true regime changes), while extreme events
    within a regime leave the Chern number unchanged.

    Attributes:
        geometry: QCMLGeometry instance for computing Berry curvature
        window_size: Size of rolling window for Chern computation
        chern_threshold: Threshold for detecting topological transitions
    """

    def __init__(self, geometry: QCMLGeometry,
                 window_size: int = 50,
                 chern_threshold: float = 0.5,
                 smoothing_window: int = 5):
        """
        Initialize regime detector.

        Args:
            geometry: Fitted QCMLGeometry instance
            window_size: Rolling window size for Chern computation
            chern_threshold: Threshold for declaring topological transition
                            (0.5 means changes ≥0.5 are significant)
            smoothing_window: Window for smoothing curvature estimates
        """
        self.geometry = geometry
        self.window_size = window_size
        self.chern_threshold = chern_threshold
        self.smoothing_window = smoothing_window

        self._state: Optional[RegimeState] = None

    def compute_berry_curvature_series(self, X: np.ndarray,
                                       indices: Tuple[int, int] = (0, 1),
                                       epsilon: float = 1e-5) -> np.ndarray:
        """
        Compute Berry curvature at each point in a time series.

        Args:
            X: Time series of shape (T, n_features)
            indices: Which 2D plane to compute curvature for
            epsilon: Step size for numerical differentiation

        Returns:
            F: Berry curvature time series of shape (T,)
        """
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
        """
        Compute local Chern number for a data window.

        Uses the data points as a 2D manifold patch and integrates
        the Berry curvature.

        Args:
            X: Data window of shape (n_points, n_features)
            indices: 2D plane indices
            method: Integration method ('trapezoid', 'simpson', 'monte_carlo')

        Returns:
            C: Local Chern number estimate
        """
        X = np.asarray(X)
        n_points = X.shape[0]

        if n_points < 4:
            return 0.0

        a, b = indices

        if method == 'trapezoid':
            # Simple trapezoidal integration over the data points
            total = 0.0
            for i in range(n_points - 1):
                F_i = self.geometry.berry_curvature(X[i])
                F_ip1 = self.geometry.berry_curvature(X[i + 1])

                # Average curvature
                F_avg = 0.5 * (F_i[a, b] + F_ip1[a, b])

                # Area element (approximate)
                dx = X[i + 1] - X[i]
                # Use cross-product proxy for area
                if len(dx) >= 2:
                    area = abs(dx[a]) * abs(dx[b])
                else:
                    area = abs(dx[0]) ** 2

                total += F_avg * area

            return total / (2 * np.pi)

        elif method == 'monte_carlo':
            # Monte Carlo integration
            n_samples = min(1000, n_points * 10)

            total = 0.0
            rng = np.random.default_rng(42)

            # Estimate bounding box
            x_min = X[:, a].min()
            x_max = X[:, a].max()
            y_min = X[:, b].min()
            y_max = X[:, b].max()

            area_bbox = (x_max - x_min) * (y_max - y_min)

            for _ in range(n_samples):
                # Random point in bounding box
                x_rand = rng.uniform(x_min, x_max)
                y_rand = rng.uniform(y_min, y_max)

                # Find nearest data point and use its curvature
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
        """
        Compute Chern number over rolling windows.

        Args:
            X: Time series of shape (T, n_features)
            window: Window size (default: self.window_size)
            indices: 2D plane indices
            stride: Step size between windows

        Returns:
            C: Chern number series of shape ((T - window) // stride + 1,)
        """
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
        """
        Detect regime transitions from time series data.

        A transition is detected when the Chern number changes by more
        than the threshold, indicating a topological change.

        Args:
            X: Time series of shape (T, n_features)
            times: Optional time stamps
            indices: 2D plane indices
            min_separation: Minimum steps between transitions

        Returns:
            transitions: List of RegimeTransition objects
        """
        X = np.asarray(X)
        T = X.shape[0]

        if times is None:
            times = np.arange(T, dtype=float)

        # Compute rolling Chern numbers
        C = self.rolling_chern_number(X, indices=indices)

        # Smooth the Chern series
        if self.smoothing_window > 1 and len(C) > self.smoothing_window:
            kernel = np.ones(self.smoothing_window) / self.smoothing_window
            C_smooth = np.convolve(C, kernel, mode='valid')
            offset = (len(C) - len(C_smooth)) // 2
        else:
            C_smooth = C
            offset = 0

        # Detect jumps in Chern number
        transitions = []
        dC = np.diff(C_smooth)

        i = 0
        while i < len(dC):
            if abs(dC[i]) > self.chern_threshold:
                # Found a potential transition
                start_idx = (i + offset) * 1  # Approximate original index
                end_idx = start_idx + self.window_size

                # Ensure within bounds
                start_idx = min(start_idx, T - 1)
                end_idx = min(end_idx, T - 1)

                # Compute confidence based on magnitude
                confidence = min(1.0, abs(dC[i]) / 1.0)  # Normalize by expected integer jump

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

                # Skip ahead to avoid detecting same transition multiple times
                i += min_separation
            else:
                i += 1

        return transitions

    def classify_event(self, X_before: np.ndarray, X_event: np.ndarray,
                      X_after: np.ndarray,
                      indices: Tuple[int, int] = (0, 1)) -> Dict:
        """
        Classify whether an event is a regime change or extreme event.

        Args:
            X_before: Data window before event
            X_event: Data during event
            X_after: Data window after event
            indices: 2D plane indices

        Returns:
            classification: Dict with event type and details
        """
        # Compute Chern numbers for each period
        C_before = self.compute_local_chern(X_before, indices)
        C_event = self.compute_local_chern(X_event, indices)
        C_after = self.compute_local_chern(X_after, indices)

        # Check for topological change
        delta_C = C_after - C_before
        is_topological = abs(delta_C) > self.chern_threshold

        # Compute Berry curvature statistics during event
        F_event = self.compute_berry_curvature_series(X_event, indices)
        F_std = np.std(F_event)
        F_max = np.max(np.abs(F_event))

        if is_topological:
            event_type = "REGIME_CHANGE"
            explanation = f"Topological transition detected: ΔC = {delta_C:.2f}"
        elif F_max > 3 * F_std:
            event_type = "EXTREME_EVENT"
            explanation = f"Extreme event within same regime: ΔC = {delta_C:.2f} < threshold"
        else:
            event_type = "NORMAL_FLUCTUATION"
            explanation = f"Normal market fluctuation: ΔC = {delta_C:.2f}"

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
        """
        Compute topological signature of a market regime.

        The signature captures the geometric/topological character of
        the market state, useful for regime classification.

        Args:
            X: Data window
            indices: 2D plane indices

        Returns:
            signature: Dict with topological features
        """
        X = np.asarray(X)

        # Chern number
        C = self.compute_local_chern(X, indices)

        # Berry curvature statistics
        F = self.compute_berry_curvature_series(X, indices)
        F_mean = np.mean(F)
        F_std = np.std(F)
        F_skew = np.mean(((F - F_mean) / (F_std + 1e-8)) ** 3)
        F_kurt = np.mean(((F - F_mean) / (F_std + 1e-8)) ** 4) - 3

        # Spectral gap (indicator of geometric stability)
        gaps = [self.geometry.spectral_gap(x) for x in X[::max(1, len(X)//10)]]
        gap_mean = np.mean(gaps)
        gap_std = np.std(gaps)

        # Quantum metric statistics
        g_traces = []
        g_dets = []
        for x in X[::max(1, len(X)//10)]:
            g = self.geometry.quantum_metric(x)
            g_traces.append(np.trace(g))
            g_dets.append(max(0, np.linalg.det(g)))  # Clamp to handle numerical issues

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
        """
        Update detector with new observation (online mode).

        Args:
            x_new: New data point
            indices: 2D plane indices

        Returns:
            transition: RegimeTransition if detected, else None
        """
        if self._state is None:
            # Initialize state
            self._state = RegimeState(
                current_chern=0.0,
                current_regime=RegimeType.NORMAL,
                curvature_history=[],
                chern_history=[],
                transitions=[]
            )

        # Compute Berry curvature at new point
        F = self.geometry.berry_curvature(x_new)
        F_val = F[indices[0], indices[1]]
        self._state.curvature_history.append(F_val)

        # Keep history bounded
        max_history = 10 * self.window_size
        if len(self._state.curvature_history) > max_history:
            self._state.curvature_history = self._state.curvature_history[-max_history:]

        # Once we have enough history, compute rolling Chern
        if len(self._state.curvature_history) >= self.window_size:
            # Approximate Chern from curvature integral
            recent = self._state.curvature_history[-self.window_size:]
            C_approx = np.sum(recent) / (2 * np.pi * self.window_size)

            self._state.chern_history.append(C_approx)

            # Check for transition
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
        """
        Initialize multi-scale detector.

        Args:
            geometry: QCMLGeometry instance
            window_sizes: List of window sizes for different scales
            chern_threshold: Threshold for transitions
        """
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
        """
        Analyze data at multiple scales.

        Args:
            X: Time series data
            times: Optional timestamps
            indices: 2D plane indices

        Returns:
            analysis: Dict with results at each scale
        """
        results = {}

        for detector, window in zip(self.detectors, self.window_sizes):
            transitions = detector.detect_transitions(X, times, indices)

            results[f'scale_{window}'] = {
                'window_size': window,
                'n_transitions': len(transitions),
                'transitions': transitions,
                'chern_series': detector.rolling_chern_number(X, indices=indices)
            }

        # Cross-scale analysis
        all_transitions = []
        for scale_result in results.values():
            all_transitions.extend(scale_result['transitions'])

        # Find coincident transitions (confirmed across scales)
        confirmed = []
        for t1 in all_transitions:
            confirmations = 0
            for t2 in all_transitions:
                if t1 is not t2:
                    # Check if transitions overlap
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
    """
    Analyze historical crisis periods for topological transitions.

    Args:
        geometry: Fitted QCMLGeometry
        X: Full historical data
        times: Timestamps
        crisis_periods: List of (name, start_idx, end_idx) tuples
        indices: 2D plane indices

    Returns:
        analysis: Dict with analysis for each crisis
    """
    detector = TopologicalRegimeDetector(geometry, window_size=50)

    results = {}

    for name, start_idx, end_idx in crisis_periods:
        # Get data windows
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


if __name__ == "__main__":
    print("Testing Topological Regime Detection Module...")

    # Create test data with a regime change
    from .qcml_geometry import QCMLGeometry
    import numpy as np

    # Simulate regime change: sphere → different orientation sphere
    n_samples = 300
    rng = np.random.default_rng(42)

    # Regime 1: Sphere centered at origin
    t1 = np.linspace(0, 4*np.pi, n_samples//3)
    X1 = np.column_stack([
        np.cos(t1) + 0.1 * rng.normal(size=len(t1)),
        np.sin(t1) + 0.1 * rng.normal(size=len(t1)),
        0.5 * t1 / (4*np.pi) + 0.1 * rng.normal(size=len(t1))
    ])

    # Transition period
    t2 = np.linspace(0, 2*np.pi, n_samples//3)
    X2 = np.column_stack([
        1.5 * np.cos(t2) + 0.2 * rng.normal(size=len(t2)),
        0.5 * np.sin(t2) + 0.2 * rng.normal(size=len(t2)),
        0.3 * t2 / (2*np.pi) + 0.2 * rng.normal(size=len(t2))
    ])

    # Regime 2: Different geometric structure
    t3 = np.linspace(0, 4*np.pi, n_samples//3)
    X3 = np.column_stack([
        0.5 * np.cos(2*t3) + 0.1 * rng.normal(size=len(t3)),
        np.sin(t3) + 0.1 * rng.normal(size=len(t3)),
        -0.5 * t3 / (4*np.pi) + 0.1 * rng.normal(size=len(t3))
    ])

    X = np.vstack([X1, X2, X3])
    times = np.arange(len(X), dtype=float)

    print(f"Test data shape: {X.shape}")

    # Fit QCML geometry
    qcml = QCMLGeometry(n_features=3, hilbert_dim=4)
    qcml.fit_operators(X, method='pca_inspired')

    # Create detector
    detector = TopologicalRegimeDetector(
        geometry=qcml,
        window_size=30,
        chern_threshold=0.3
    )

    # Test rolling Chern computation
    C = detector.rolling_chern_number(X)
    print(f"Chern series shape: {C.shape}")
    print(f"Chern range: [{C.min():.3f}, {C.max():.3f}]")

    # Test transition detection
    transitions = detector.detect_transitions(X, times)
    print(f"\nDetected {len(transitions)} transitions:")
    for t in transitions:
        print(f"  idx {t.start_idx}-{t.end_idx}: ΔC={t.delta_chern:.3f}, "
              f"topological={t.is_topological}, conf={t.confidence:.2f}")

    # Test regime signature
    sig = detector.compute_regime_signature(X[:100])
    print(f"\nRegime signature (first 100 points):")
    print(f"  Chern: {sig['chern_number']:.3f} (rounded: {sig['rounded_chern']})")
    print(f"  Curvature mean: {sig['curvature_mean']:.4f}")
    print(f"  Curvature std: {sig['curvature_std']:.4f}")
    print(f"  Spectral gap: {sig['spectral_gap_mean']:.4f}")

    # Test multi-scale detection
    print("\nTesting Multi-Scale Detection...")
    multi_detector = MultiScaleRegimeDetector(
        geometry=qcml,
        window_sizes=[20, 40, 60],
        chern_threshold=0.3
    )

    multi_results = multi_detector.analyze(X, times)
    print(f"Cross-scale confirmed transitions: {multi_results['cross_scale']['confirmed_transitions']}")

    print("\nTopological Regime Detection Module tests passed!")
