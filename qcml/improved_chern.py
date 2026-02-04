"""
Improved Chern Number Computation

Track D: Better Chern computation methods to reduce false positives
while maintaining recall on true regime changes.

Improvements:
1. Higher threshold (3σ instead of 2σ)
2. Adaptive threshold based on rolling regime
3. Multi-scale Chern (windows: 10, 20, 50 days)
4. Quantization improvement: Force Chern toward integers
5. Confirmation logic: Require sustained signal

Author: QCML Research
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from enum import Enum

from .qcml_geometry import QCMLGeometry
from .topological_regime import TopologicalRegimeDetector


class ThresholdMethod(Enum):
    """Methods for computing spike detection threshold."""
    FIXED_STD = "fixed_std"       # Fixed multiplier of rolling std
    ADAPTIVE = "adaptive"         # Adapts based on regime
    QUANTILE = "quantile"         # Uses quantile threshold
    HYBRID = "hybrid"             # Combination of methods


@dataclass
class ImprovedChernResult:
    """Results from improved Chern computation."""
    chern_series: pd.Series
    delta_chern: pd.Series
    spikes: pd.Series
    threshold_series: pd.Series
    quantized_chern: pd.Series
    multi_scale_agreement: pd.Series


class ImprovedChernDetector:
    """
    Improved Chern number detector with enhanced spike detection.

    Key improvements over base TopologicalRegimeDetector:
    1. Configurable threshold methods
    2. Multi-scale analysis
    3. Quantization to enforce integer-like behavior
    4. Confirmation requirements for spikes
    """

    def __init__(
        self,
        geometry: QCMLGeometry,
        window_sizes: List[int] = [10, 20, 50],
        threshold_method: ThresholdMethod = ThresholdMethod.ADAPTIVE,
        base_threshold_std: float = 3.0,  # Higher default
        min_confirmation_days: int = 2,
        quantization_strength: float = 0.5
    ):
        """
        Initialize improved Chern detector.

        Args:
            geometry: Fitted QCMLGeometry instance
            window_sizes: List of window sizes for multi-scale analysis
            threshold_method: Method for computing spike threshold
            base_threshold_std: Base number of standard deviations for threshold
            min_confirmation_days: Minimum consecutive days for spike confirmation
            quantization_strength: How strongly to push toward integers (0-1)
        """
        self.geometry = geometry
        self.window_sizes = window_sizes
        self.threshold_method = threshold_method
        self.base_threshold_std = base_threshold_std
        self.min_confirmation_days = min_confirmation_days
        self.quantization_strength = quantization_strength

        # Create detectors for each scale
        self.detectors = {
            w: TopologicalRegimeDetector(geometry, window_size=w, chern_threshold=0.1)
            for w in window_sizes
        }

    def compute_chern_series(
        self,
        X: np.ndarray,
        primary_window: int = 20
    ) -> pd.Series:
        """
        Compute Chern number series using primary window.

        Args:
            X: Feature array of shape (T, n_features)
            primary_window: Primary window size

        Returns:
            chern_series: Series of Chern values
        """
        detector = self.detectors.get(primary_window)

        if detector is None:
            detector = TopologicalRegimeDetector(
                self.geometry,
                window_size=primary_window
            )

        chern_values = detector.rolling_chern_number(X, window=primary_window)

        return pd.Series(chern_values, name='chern')

    def compute_multi_scale_chern(
        self,
        X: np.ndarray
    ) -> Dict[int, pd.Series]:
        """
        Compute Chern series at multiple scales.

        Args:
            X: Feature array

        Returns:
            chern_dict: Dict mapping window size to Chern series
        """
        chern_dict = {}

        for window, detector in self.detectors.items():
            chern_values = detector.rolling_chern_number(X, window=window)
            chern_dict[window] = pd.Series(chern_values, name=f'chern_{window}')

        return chern_dict

    def compute_adaptive_threshold(
        self,
        delta_chern: pd.Series,
        lookback_short: int = 20,
        lookback_long: int = 120,
        regime_adjustment: float = 0.5
    ) -> pd.Series:
        """
        Compute adaptive threshold based on regime.

        In high-volatility regimes, use higher threshold to avoid
        false positives. In low-vol regimes, use standard threshold.

        Args:
            delta_chern: Series of Chern changes
            lookback_short: Short-term lookback for current vol
            lookback_long: Long-term lookback for baseline vol
            regime_adjustment: How much to adjust in high-vol regimes

        Returns:
            threshold: Adaptive threshold series
        """
        # Rolling statistics
        rolling_std_short = delta_chern.rolling(
            window=lookback_short,
            min_periods=lookback_short // 2
        ).std()

        rolling_std_long = delta_chern.rolling(
            window=lookback_long,
            min_periods=lookback_long // 2
        ).std()

        # Regime indicator: ratio of short-term to long-term vol
        vol_ratio = rolling_std_short / (rolling_std_long + 1e-10)

        # Adjust threshold based on regime
        # High vol_ratio (> 1) = elevated regime = higher threshold
        threshold_multiplier = self.base_threshold_std * (
            1 + regime_adjustment * (vol_ratio - 1).clip(lower=0)
        )

        threshold = threshold_multiplier * rolling_std_long

        return threshold

    def compute_quantile_threshold(
        self,
        delta_chern: pd.Series,
        quantile: float = 0.95,
        lookback: int = 252
    ) -> pd.Series:
        """
        Compute threshold based on rolling quantile.

        Args:
            delta_chern: Series of Chern changes
            quantile: Quantile for threshold (e.g., 0.95 for 95th percentile)
            lookback: Lookback window

        Returns:
            threshold: Quantile-based threshold
        """
        threshold = delta_chern.abs().rolling(
            window=lookback,
            min_periods=lookback // 2
        ).quantile(quantile)

        return threshold

    def compute_threshold(
        self,
        delta_chern: pd.Series
    ) -> pd.Series:
        """
        Compute spike detection threshold based on configured method.

        Args:
            delta_chern: Series of Chern changes

        Returns:
            threshold: Threshold series
        """
        if self.threshold_method == ThresholdMethod.FIXED_STD:
            rolling_std = delta_chern.rolling(window=60, min_periods=20).std()
            return self.base_threshold_std * rolling_std

        elif self.threshold_method == ThresholdMethod.ADAPTIVE:
            return self.compute_adaptive_threshold(delta_chern)

        elif self.threshold_method == ThresholdMethod.QUANTILE:
            return self.compute_quantile_threshold(delta_chern)

        elif self.threshold_method == ThresholdMethod.HYBRID:
            # Combine adaptive and quantile
            adaptive = self.compute_adaptive_threshold(delta_chern)
            quantile = self.compute_quantile_threshold(delta_chern)
            return np.maximum(adaptive, quantile)

        else:
            raise ValueError(f"Unknown threshold method: {self.threshold_method}")

    def quantize_chern(
        self,
        chern_series: pd.Series
    ) -> pd.Series:
        """
        Quantize Chern values toward integers.

        The Chern number is theoretically an integer for closed surfaces.
        This function applies soft quantization to push values toward
        integers while preserving trends.

        Args:
            chern_series: Raw Chern values

        Returns:
            quantized: Quantized Chern values
        """
        # Soft quantization: weighted average of raw and rounded
        rounded = chern_series.round()
        quantized = (
            (1 - self.quantization_strength) * chern_series +
            self.quantization_strength * rounded
        )

        return quantized

    def confirm_spikes(
        self,
        raw_spikes: pd.Series
    ) -> pd.Series:
        """
        Require multiple consecutive days for spike confirmation.

        This reduces false positives from single-day noise.

        Args:
            raw_spikes: Raw spike indicators

        Returns:
            confirmed_spikes: Confirmed spikes
        """
        if self.min_confirmation_days <= 1:
            return raw_spikes

        # Rolling sum of spikes
        spike_sum = raw_spikes.rolling(
            window=self.min_confirmation_days
        ).sum()

        # Confirmed if we have sustained signal
        confirmed = spike_sum >= self.min_confirmation_days

        return confirmed

    def compute_multi_scale_agreement(
        self,
        multi_scale_chern: Dict[int, pd.Series],
        threshold_std: float = 2.0
    ) -> pd.Series:
        """
        Compute agreement across multiple scales.

        A spike is more reliable if detected at multiple time scales.

        Args:
            multi_scale_chern: Dict of Chern series at different scales
            threshold_std: Standard deviations for spike at each scale

        Returns:
            agreement: Number of scales detecting spike (0 to n_scales)
        """
        scale_spikes = {}

        for window, chern_series in multi_scale_chern.items():
            delta = chern_series.diff().fillna(0)
            rolling_std = delta.rolling(window=60, min_periods=20).std()
            threshold = threshold_std * rolling_std
            scale_spikes[window] = (delta.abs() > threshold).astype(int)

        # Align all scales to common index
        common_idx = scale_spikes[self.window_sizes[0]].index

        for window in self.window_sizes[1:]:
            common_idx = common_idx.intersection(scale_spikes[window].index)

        # Sum agreements
        agreement = pd.Series(0, index=common_idx)

        for window in self.window_sizes:
            agreement += scale_spikes[window].reindex(common_idx, fill_value=0)

        return agreement

    def detect_improved_spikes(
        self,
        X: np.ndarray,
        times: Optional[pd.DatetimeIndex] = None,
        min_scale_agreement: int = 2
    ) -> ImprovedChernResult:
        """
        Detect spikes using all improved methods.

        Args:
            X: Feature array of shape (T, n_features)
            times: Optional datetime index
            min_scale_agreement: Minimum scales that must agree

        Returns:
            result: ImprovedChernResult with all computed series
        """
        # Compute multi-scale Chern
        multi_scale = self.compute_multi_scale_chern(X)

        # Use middle window as primary
        primary_window = self.window_sizes[len(self.window_sizes) // 2]
        chern_series = multi_scale[primary_window]

        # Apply timestamps if provided
        if times is not None:
            offset = primary_window - 1
            if len(times) > len(chern_series):
                chern_times = times[offset:offset + len(chern_series)]
            else:
                chern_times = times[:len(chern_series)]

            chern_series = pd.Series(chern_series.values, index=chern_times)

            # Update multi_scale with times
            for w in multi_scale:
                w_offset = w - 1
                if len(times) > len(multi_scale[w]):
                    w_times = times[w_offset:w_offset + len(multi_scale[w])]
                else:
                    w_times = times[:len(multi_scale[w])]
                multi_scale[w] = pd.Series(multi_scale[w].values, index=w_times)

        # Compute delta
        delta_chern = chern_series.diff().fillna(0)

        # Compute adaptive threshold
        threshold = self.compute_threshold(delta_chern)

        # Raw spikes
        raw_spikes = delta_chern.abs() > threshold

        # Confirm spikes
        confirmed_spikes = self.confirm_spikes(raw_spikes)

        # Multi-scale agreement
        agreement = self.compute_multi_scale_agreement(multi_scale)

        # Final spikes: confirmed AND sufficient scale agreement
        agreement_aligned = agreement.reindex(confirmed_spikes.index, fill_value=0)
        final_spikes = confirmed_spikes & (agreement_aligned >= min_scale_agreement)

        # Quantized Chern
        quantized = self.quantize_chern(chern_series)

        return ImprovedChernResult(
            chern_series=chern_series,
            delta_chern=delta_chern,
            spikes=final_spikes,
            threshold_series=threshold,
            quantized_chern=quantized,
            multi_scale_agreement=agreement
        )


def compare_detection_methods(
    X: np.ndarray,
    geometry: QCMLGeometry,
    known_events: Dict[str, str],
    times: pd.DatetimeIndex,
    event_window: int = 15
) -> Dict:
    """
    Compare original and improved detection methods.

    Args:
        X: Feature array
        geometry: Fitted QCMLGeometry
        known_events: Dict of event dates to names
        times: Datetime index
        event_window: Days around event for true positive

    Returns:
        comparison: Dict with metrics for each method
    """
    results = {}

    # Parse event dates
    event_dates = [pd.Timestamp(d) for d in known_events.keys()]

    def compute_metrics(spikes: pd.Series) -> Dict:
        """Compute precision, recall, F1."""
        spike_dates = spikes[spikes].index

        tp = 0
        detected_events = set()

        for spike_date in spike_dates:
            for event_date in event_dates:
                if abs((spike_date - event_date).days) <= event_window:
                    if event_date not in detected_events:
                        tp += 1
                        detected_events.add(event_date)
                    break

        fp = len(spike_dates) - tp
        fn = len(event_dates) - len(detected_events)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'spikes': int(len(spike_dates)),
            'tp': tp,
            'fp': fp,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    # Original method (2σ threshold)
    original_detector = TopologicalRegimeDetector(geometry, window_size=20, chern_threshold=0.1)
    chern_values = original_detector.rolling_chern_number(X, window=20)

    chern_times = times[19:19 + len(chern_values)]
    original_chern = pd.Series(chern_values, index=chern_times)
    original_delta = original_chern.diff().fillna(0)
    original_std = original_delta.rolling(window=60, min_periods=20).std()
    original_spikes = original_delta.abs() > (2.0 * original_std)

    results['original_2sigma'] = compute_metrics(original_spikes)

    # 3σ threshold
    spikes_3sigma = original_delta.abs() > (3.0 * original_std)
    results['fixed_3sigma'] = compute_metrics(spikes_3sigma)

    # Adaptive threshold
    improved_adaptive = ImprovedChernDetector(
        geometry,
        threshold_method=ThresholdMethod.ADAPTIVE,
        base_threshold_std=3.0
    )
    adaptive_result = improved_adaptive.detect_improved_spikes(X, times)
    results['adaptive'] = compute_metrics(adaptive_result.spikes)

    # Quantile threshold
    improved_quantile = ImprovedChernDetector(
        geometry,
        threshold_method=ThresholdMethod.QUANTILE,
        base_threshold_std=3.0
    )
    quantile_result = improved_quantile.detect_improved_spikes(X, times)
    results['quantile'] = compute_metrics(quantile_result.spikes)

    # Multi-scale with confirmation
    improved_full = ImprovedChernDetector(
        geometry,
        window_sizes=[10, 20, 50],
        threshold_method=ThresholdMethod.ADAPTIVE,
        base_threshold_std=2.5,
        min_confirmation_days=2
    )
    full_result = improved_full.detect_improved_spikes(X, times, min_scale_agreement=2)
    results['full_improved'] = compute_metrics(full_result.spikes)

    return results


if __name__ == "__main__":
    # Test the improved detector
    import os
    import sys
    from pathlib import Path
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from dotenv import load_dotenv

    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from qcml.data import PolygonDataSource, MinimalFeatureEngine

    load_dotenv(project_root / '.env')

    print("=" * 60)
    print("TRACK D: IMPROVED CHERN COMPUTATION TEST")
    print("=" * 60)

    # Fetch data
    api_key = os.getenv('POLYGON_API_KEY')
    source = PolygonDataSource(api_key=api_key)

    print("\nFetching SPY data...")
    spy_data = source.fetch_equities(['SPY'], '2006-01-01', '2024-06-30', timeframe='1d')
    spy_prices = spy_data['close'].unstack(level=0)['SPY'].ffill()

    # Create features
    engine = MinimalFeatureEngine(window=20)
    features = engine.create_feature_matrix(spy_prices.to_frame('SPY'), benchmark_col='SPY')
    features = features.dropna()

    X_raw = features.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    pca = PCA(n_components=min(15, X_raw.shape[1]))
    X = pca.fit_transform(X_scaled)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    # Fit geometry
    geometry = QCMLGeometry(n_features=X.shape[1], hilbert_dim=8)
    geometry.fit_operators(X, method='random')

    # Known events
    known_events = {
        '2008-09-15': 'Lehman',
        '2008-10-06': 'October Crash',
        '2010-05-06': 'Flash Crash',
        '2011-08-08': 'Downgrade',
        '2015-08-24': 'China',
        '2018-02-05': 'Volmageddon',
        '2018-12-24': 'Xmas Selloff',
        '2020-02-24': 'COVID Start',
        '2020-03-16': 'COVID Bottom',
        '2022-01-24': 'Fed Fear',
        '2022-03-16': 'Rate Hike',
        '2022-06-13': 'Bear Market',
        '2023-03-10': 'SVB',
    }

    # Compare methods
    print("\nComparing detection methods...")
    comparison = compare_detection_methods(X, geometry, known_events, features.index)

    print("\n" + "-" * 60)
    print("COMPARISON RESULTS")
    print("-" * 60)

    print(f"\n{'Method':<20} {'Spikes':<8} {'Precision':<12} {'Recall':<10} {'F1':<8}")
    print("-" * 60)

    for method, metrics in comparison.items():
        print(f"{method:<20} {metrics['spikes']:<8} {metrics['precision']:.1%}       "
              f"{metrics['recall']:.1%}      {metrics['f1']:.3f}")

    # Find best method
    best_method = max(comparison.items(), key=lambda x: x[1]['f1'])
    print(f"\nBest method by F1: {best_method[0]} (F1 = {best_method[1]['f1']:.3f})")

    # Check if any improved method beats original
    original_f1 = comparison['original_2sigma']['f1']
    improvements = [
        (m, c['f1'] - original_f1)
        for m, c in comparison.items()
        if m != 'original_2sigma'
    ]

    print("\n" + "-" * 40)
    print("IMPROVEMENT OVER ORIGINAL:")
    for method, improvement in improvements:
        symbol = "✓" if improvement > 0 else "✗"
        print(f"  {symbol} {method}: {improvement:+.3f}")

    # Save results
    import json
    output_dir = project_root / 'experiments' / 'outputs'
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / 'track_d_improved_chern_results.json', 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'track_d_improved_chern_results.json'}")
