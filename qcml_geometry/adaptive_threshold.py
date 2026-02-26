"""
Adaptive threshold calibration for regime detection scores.

Provides two complementary strategies:

1. RollingQuantileThreshold — Self-calibrating threshold based on recent
   score distribution. Adapts to changing volatility regimes without
   lookahead.

2. ScoreVelocityThreshold — Triggers on rapid score acceleration
   (first derivative). Catches fast regime shifts even when the absolute
   level hasn't breached a static threshold.

Both can be used independently or combined (logical OR) for maximum
detection coverage.
"""

import numpy as np
import pandas as pd


class RollingQuantileThreshold:
    """Adaptive threshold from rolling quantile of score history.

    At each time t, the threshold is the q-th quantile of scores
    over the trailing ``lookback`` days (excluding the most recent
    ``gap`` days to avoid self-referencing during rapid moves).

    An alarm fires when the score exceeds the rolling threshold for
    at least ``persistence`` consecutive days.

    Parameters:
        lookback: Rolling window size in trading days.
        quantile: Quantile for threshold (e.g. 0.95 = top 5%).
        persistence: Minimum consecutive days above threshold to fire.
        gap: Exclusion gap between current day and lookback window.
        min_history: Minimum days of history before producing thresholds.
    """

    def __init__(
        self,
        lookback: int = 252,
        quantile: float = 0.95,
        persistence: int = 3,
        gap: int = 5,
        min_history: int = 60,
    ):
        self.lookback = lookback
        self.quantile = quantile
        self.persistence = persistence
        self.gap = gap
        self.min_history = min_history

    def compute_thresholds(self, scores: np.ndarray) -> np.ndarray:
        """Compute rolling quantile thresholds for a score series.

        Args:
            scores: 1-D array of regime scores (may contain NaN).

        Returns:
            thresholds: 1-D array of same length, NaN where insufficient
                history.
        """
        T = len(scores)
        thresholds = np.full(T, np.nan)

        for t in range(self.min_history + self.gap, T):
            window_end = t - self.gap
            window_start = max(0, window_end - self.lookback)
            window = scores[window_start:window_end]
            valid = window[~np.isnan(window)]
            if len(valid) >= self.min_history // 2:
                thresholds[t] = np.percentile(valid, self.quantile * 100)

        return thresholds

    def detect(self, scores: np.ndarray) -> tuple:
        """Run detection: compute thresholds and apply persistence filter.

        Args:
            scores: 1-D array of regime scores.

        Returns:
            (alarm_mask, thresholds): Boolean alarm array and threshold
                array of same length as scores.
        """
        thresholds = self.compute_thresholds(scores)

        # Raw exceedances
        raw_alarm = np.zeros(len(scores), dtype=bool)
        valid = ~np.isnan(scores) & ~np.isnan(thresholds)
        raw_alarm[valid] = scores[valid] > thresholds[valid]

        # Apply persistence filter
        alarm = _persistence_filter(raw_alarm, self.persistence)

        return alarm, thresholds


class ScoreVelocityThreshold:
    """Threshold based on score rate of change (velocity).

    Detects rapid regime transitions by monitoring the first derivative
    of smoothed scores. A spike in velocity indicates the system is
    rapidly moving toward (or through) a phase boundary.

    Parameters:
        smoothing_window: Window for smoothing scores before differentiation.
        velocity_lookback: Rolling window for velocity baseline statistics.
        z_threshold: Z-score threshold for velocity alarm.
        persistence: Minimum consecutive days of elevated velocity.
        min_history: Minimum days before computing velocity statistics.
    """

    def __init__(
        self,
        smoothing_window: int = 5,
        velocity_lookback: int = 252,
        z_threshold: float = 2.0,
        persistence: int = 2,
        min_history: int = 60,
    ):
        self.smoothing_window = smoothing_window
        self.velocity_lookback = velocity_lookback
        self.z_threshold = z_threshold
        self.persistence = persistence
        self.min_history = min_history

    def compute_velocity(self, scores: np.ndarray) -> np.ndarray:
        """Compute score velocity (smoothed first derivative).

        Args:
            scores: 1-D array of regime scores.

        Returns:
            velocity: 1-D array of absolute score velocity.
        """
        smoothed = (
            pd.Series(scores)
            .rolling(self.smoothing_window, min_periods=1)
            .mean()
            .values
        )
        velocity = np.abs(np.diff(smoothed, prepend=smoothed[0]))
        return velocity

    def detect(self, scores: np.ndarray) -> tuple:
        """Run velocity-based detection.

        Args:
            scores: 1-D array of regime scores.

        Returns:
            (alarm_mask, velocity_z): Boolean alarm array and velocity
                z-scores.
        """
        velocity = self.compute_velocity(scores)
        T = len(velocity)

        # Expanding z-score of velocity
        velocity_z = np.full(T, np.nan)
        for t in range(self.min_history, T):
            window_start = max(0, t - self.velocity_lookback)
            past = velocity[window_start:t]
            valid = past[~np.isnan(past)]
            if len(valid) >= 10:
                mu = np.mean(valid)
                sigma = np.std(valid, ddof=1)
                if sigma > 1e-12:
                    velocity_z[t] = (velocity[t] - mu) / sigma

        # Threshold exceedance
        raw_alarm = np.zeros(T, dtype=bool)
        valid = ~np.isnan(velocity_z)
        raw_alarm[valid] = velocity_z[valid] > self.z_threshold

        alarm = _persistence_filter(raw_alarm, self.persistence)
        return alarm, velocity_z


class CombinedAdaptiveThreshold:
    """Combines rolling quantile and velocity thresholds.

    Fires alarm when EITHER the quantile threshold OR the velocity
    threshold triggers. This provides coverage of both:
    - Sustained elevated scores (quantile catches slow buildups)
    - Rapid score spikes (velocity catches fast transitions)

    Parameters:
        quantile_params: Dict of RollingQuantileThreshold parameters.
        velocity_params: Dict of ScoreVelocityThreshold parameters.
    """

    def __init__(
        self,
        quantile_params: dict = None,
        velocity_params: dict = None,
    ):
        self.quantile = RollingQuantileThreshold(**(quantile_params or {}))
        self.velocity = ScoreVelocityThreshold(**(velocity_params or {}))

    def detect(self, scores: np.ndarray) -> tuple:
        """Run combined detection.

        Args:
            scores: 1-D array of regime scores.

        Returns:
            (alarm_mask, details): Boolean alarm array and dict with
                per-strategy details.
        """
        q_alarm, q_thresholds = self.quantile.detect(scores)
        v_alarm, v_z_scores = self.velocity.detect(scores)

        combined_alarm = q_alarm | v_alarm

        details = {
            'quantile_alarm': q_alarm,
            'quantile_thresholds': q_thresholds,
            'velocity_alarm': v_alarm,
            'velocity_z_scores': v_z_scores,
        }
        return combined_alarm, details


def _persistence_filter(alarm_mask: np.ndarray, min_persistence: int) -> np.ndarray:
    """Remove alarm runs shorter than min_persistence consecutive days.

    Args:
        alarm_mask: Boolean array.
        min_persistence: Minimum run length to keep.

    Returns:
        Filtered boolean array.
    """
    if min_persistence <= 1:
        return alarm_mask.copy()

    mask = alarm_mask.copy()
    T = len(mask)
    result = np.zeros(T, dtype=bool)

    i = 0
    while i < T:
        if mask[i]:
            j = i
            while j < T and mask[j]:
                j += 1
            if (j - i) >= min_persistence:
                result[i:j] = True
            i = j
        else:
            i += 1

    return result
