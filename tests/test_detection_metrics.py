"""
Unit tests for detection metrics (F1 with tolerance, AUC-PR, delay, FAR).

Tests verify correctness of each core metric function using deterministic
synthetic data — no live API calls or pipeline runs.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Imports — will fail until implementation exists
# ---------------------------------------------------------------------------

from experiments.detection_metrics import (
    compute_f1_with_tolerance,
    compute_auc_pr,
    compute_detection_delay,
    compute_false_alarm_rate,
)


# ---------------------------------------------------------------------------
# TestF1WithTolerance
# ---------------------------------------------------------------------------


class TestF1WithTolerance:
    """F1 with tolerance windows following TCPDBench protocol."""

    def test_perfect_detection(self):
        """Alarms exactly at both boundaries -> F1 = 1.0."""
        T = 100
        scores = np.zeros(T)
        threshold = np.full(T, 0.5)
        crisis_start_idx = 40
        crisis_end_idx = 60

        # Place alarms exactly at boundaries
        scores[crisis_start_idx] = 1.0
        scores[crisis_end_idx] = 1.0

        result = compute_f1_with_tolerance(
            scores,
            threshold,
            crisis_start_idx,
            crisis_end_idx,
            tolerance_days=5,
        )
        assert result["precision"] == 1.0, "Only TP alarms, precision should be 1.0"
        assert result["recall"] == 1.0, "Both boundaries detected, recall should be 1.0"
        assert result["f1"] == 1.0
        assert result["n_tp"] == 2
        assert result["n_fp"] == 0
        assert result["n_boundaries_detected"] == 2

    def test_alarm_within_tolerance(self):
        """Alarm within tolerance of boundary still counts as TP."""
        T = 100
        scores = np.zeros(T)
        threshold = np.full(T, 0.5)
        crisis_start_idx = 40
        crisis_end_idx = 60

        # Place alarms near but not exactly at boundaries
        scores[crisis_start_idx + 3] = 1.0  # 3 days after start
        scores[crisis_end_idx - 2] = 1.0  # 2 days before end

        result = compute_f1_with_tolerance(
            scores,
            threshold,
            crisis_start_idx,
            crisis_end_idx,
            tolerance_days=5,
        )
        assert result["n_boundaries_detected"] == 2, "Both boundaries within tolerance"
        assert result["recall"] == 1.0
        assert result["f1"] > 0.9

    def test_alarm_outside_tolerance(self):
        """Alarm far from any boundary is a false positive."""
        T = 100
        scores = np.zeros(T)
        threshold = np.full(T, 0.5)
        crisis_start_idx = 40
        crisis_end_idx = 60

        # Alarm far from boundaries
        scores[10] = 1.0
        scores[90] = 1.0

        result = compute_f1_with_tolerance(
            scores,
            threshold,
            crisis_start_idx,
            crisis_end_idx,
            tolerance_days=5,
        )
        assert result["n_boundaries_detected"] == 0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0
        assert result["n_fp"] == 2

    def test_no_alarms(self):
        """No alarms anywhere -> recall = 0, F1 = 0."""
        T = 100
        scores = np.zeros(T)
        threshold = np.full(T, 0.5)

        result = compute_f1_with_tolerance(
            scores,
            threshold,
            crisis_start_idx=40,
            crisis_end_idx=60,
            tolerance_days=10,
        )
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0
        assert result["n_boundaries_detected"] == 0

    def test_all_alarms_everywhere(self):
        """Alarms at every timestep -> recall = 1.0, precision low."""
        T = 100
        scores = np.ones(T)
        threshold = np.full(T, 0.5)
        crisis_start_idx = 40
        crisis_end_idx = 60

        result = compute_f1_with_tolerance(
            scores,
            threshold,
            crisis_start_idx,
            crisis_end_idx,
            tolerance_days=5,
        )
        assert result["recall"] == 1.0, "Both boundaries detected among many alarms"
        assert (
            result["precision"] < 0.3
        ), f"Precision should be low with all alarms, got {result['precision']}"

    def test_each_boundary_matched_once(self):
        """Multiple alarms near same boundary count as 1 TP + N-1 FPs."""
        T = 100
        scores = np.zeros(T)
        threshold = np.full(T, 0.5)
        crisis_start_idx = 40
        crisis_end_idx = 60

        # 5 alarms all near start boundary, none near end
        for i in range(5):
            scores[crisis_start_idx + i] = 1.0

        result = compute_f1_with_tolerance(
            scores,
            threshold,
            crisis_start_idx,
            crisis_end_idx,
            tolerance_days=5,
        )
        assert result["n_boundaries_detected"] == 1, "Only start boundary matched"
        assert result["n_tp"] == 1, "One TP (closest to start boundary)"
        assert result["n_fp"] == 4, "Remaining 4 are FPs"
        assert result["recall"] == 0.5, "1 of 2 boundaries detected"

    def test_tolerance_zero(self):
        """tolerance_days=0 means only exact matches count."""
        T = 100
        scores = np.zeros(T)
        threshold = np.full(T, 0.5)
        crisis_start_idx = 40
        crisis_end_idx = 60

        # Alarm 1 day away from boundary
        scores[41] = 1.0

        result = compute_f1_with_tolerance(
            scores,
            threshold,
            crisis_start_idx,
            crisis_end_idx,
            tolerance_days=0,
        )
        assert result["n_boundaries_detected"] == 0


# ---------------------------------------------------------------------------
# TestAUCPR
# ---------------------------------------------------------------------------


class TestAUCPR:
    """Area under precision-recall curve."""

    def test_perfect_separation(self):
        """Scores perfectly separate crisis from normal -> AUC > 0.9."""
        T = 200
        crisis_mask = np.zeros(T, dtype=bool)
        crisis_mask[100:140] = True  # 40 crisis days

        scores = np.zeros(T)
        scores[crisis_mask] = 5.0  # high during crisis
        scores[~crisis_mask] = 0.1  # low during normal

        auc = compute_auc_pr(scores, crisis_mask)
        assert auc > 0.95, f"Perfect separation should give AUC > 0.95, got {auc}"

    def test_random_scores(self):
        """Random scores -> AUC near base rate."""
        T = 1000
        rng = np.random.default_rng(42)
        crisis_mask = np.zeros(T, dtype=bool)
        crisis_mask[400:500] = True  # 10% base rate

        scores = rng.standard_normal(T)
        auc = compute_auc_pr(scores, crisis_mask)
        # Base rate is 0.10, AUC should be near that
        assert auc < 0.3, f"Random scores AUC should be near base rate, got {auc}"

    def test_handles_nan_scores(self):
        """NaN scores should be handled gracefully."""
        T = 100
        crisis_mask = np.zeros(T, dtype=bool)
        crisis_mask[50:70] = True

        scores = np.ones(T)
        scores[crisis_mask] = 5.0
        scores[:10] = np.nan  # some NaN values

        auc = compute_auc_pr(scores, crisis_mask)
        assert np.isfinite(auc), "AUC should be finite even with NaN scores"
        assert auc > 0.5, "Should still get reasonable AUC after NaN handling"

    def test_no_crisis_days(self):
        """No crisis days -> AUC should be NaN or 0."""
        T = 100
        crisis_mask = np.zeros(T, dtype=bool)
        scores = np.random.default_rng(42).standard_normal(T)

        auc = compute_auc_pr(scores, crisis_mask)
        assert np.isnan(auc) or auc == 0.0, "No positives should give NaN or 0.0"


# ---------------------------------------------------------------------------
# TestDetectionDelay
# ---------------------------------------------------------------------------


class TestDetectionDelay:
    """Detection delay: days from crisis start to first alarm."""

    def test_alarm_at_crisis_start(self):
        """Alarm on crisis start day -> delay = 0."""
        T = 100
        scores = np.zeros(T)
        threshold = np.full(T, 0.5)
        scores[40] = 1.0  # alarm exactly at crisis start

        delay = compute_detection_delay(scores, threshold, crisis_start_idx=40, crisis_end_idx=60)
        assert delay == 0

    def test_alarm_five_days_after_start(self):
        """First alarm 5 days into crisis -> delay = 5."""
        T = 100
        scores = np.zeros(T)
        threshold = np.full(T, 0.5)
        scores[45] = 1.0  # alarm 5 days after crisis start

        delay = compute_detection_delay(scores, threshold, crisis_start_idx=40, crisis_end_idx=60)
        assert delay == 5

    def test_no_alarm_in_crisis_window(self):
        """No alarms during crisis -> delay = NaN."""
        T = 100
        scores = np.zeros(T)
        threshold = np.full(T, 0.5)
        # Alarm before crisis, not during
        scores[10] = 1.0

        delay = compute_detection_delay(scores, threshold, crisis_start_idx=40, crisis_end_idx=60)
        assert np.isnan(delay), "No alarm in crisis window should give NaN"

    def test_alarm_before_crisis_not_counted(self):
        """Alarm before crisis start index should not be counted."""
        T = 100
        scores = np.zeros(T)
        threshold = np.full(T, 0.5)
        scores[35] = 1.0  # 5 days BEFORE crisis start

        delay = compute_detection_delay(scores, threshold, crisis_start_idx=40, crisis_end_idx=60)
        assert np.isnan(delay), "Alarm before crisis should not count"

    def test_first_alarm_used_not_strongest(self):
        """Delay uses earliest alarm, not largest score."""
        T = 100
        scores = np.zeros(T)
        threshold = np.full(T, 0.5)
        scores[42] = 0.8  # first alarm (weaker)
        scores[50] = 5.0  # stronger alarm but later

        delay = compute_detection_delay(scores, threshold, crisis_start_idx=40, crisis_end_idx=60)
        assert delay == 2, "Should use first alarm (day 42), not strongest"

    def test_scalar_threshold(self):
        """Should work when threshold is a scalar."""
        T = 100
        scores = np.zeros(T)
        scores[43] = 1.0

        delay = compute_detection_delay(scores, 0.5, crisis_start_idx=40, crisis_end_idx=60)
        assert delay == 3


# ---------------------------------------------------------------------------
# TestFalseAlarmRate
# ---------------------------------------------------------------------------


class TestFalseAlarmRate:
    """False alarm rate: alarms per year outside crisis windows."""

    def test_alarms_only_during_crisis(self):
        """No alarms outside crisis -> FAR = 0."""
        T = 252  # one trading year
        crisis_mask = np.zeros(T, dtype=bool)
        crisis_mask[100:130] = True  # 30-day crisis

        scores = np.zeros(T)
        scores[100:130] = 1.0  # alarms only during crisis
        threshold = np.full(T, 0.5)

        far = compute_false_alarm_rate(scores, threshold, crisis_mask)
        assert far == 0.0

    def test_one_false_alarm_per_year(self):
        """Exactly 1 alarm outside crisis in 252 normal days -> FAR ~ 1.0/yr."""
        T = 282  # 252 normal + 30 crisis
        crisis_mask = np.zeros(T, dtype=bool)
        crisis_mask[100:130] = True

        scores = np.zeros(T)
        scores[10] = 1.0  # one false alarm
        threshold = np.full(T, 0.5)

        far = compute_false_alarm_rate(scores, threshold, crisis_mask)
        # 1 false alarm / 252 normal days * 252 = 1.0
        assert abs(far - 1.0) < 0.01, f"Expected FAR ~ 1.0, got {far}"

    def test_scalar_threshold(self):
        """Should work with scalar threshold."""
        T = 252
        crisis_mask = np.zeros(T, dtype=bool)
        crisis_mask[100:130] = True

        scores = np.zeros(T)
        scores[10] = 1.0
        scores[200] = 1.0

        far = compute_false_alarm_rate(scores, 0.5, crisis_mask)
        normal_days = np.sum(~crisis_mask)
        expected = (2 / normal_days) * 252
        assert abs(far - expected) < 0.01

    def test_handles_nan_scores(self):
        """NaN scores should not be counted as alarms."""
        T = 100
        crisis_mask = np.zeros(T, dtype=bool)
        crisis_mask[40:60] = True

        scores = np.full(T, np.nan)  # all NaN
        threshold = np.full(T, 0.5)

        far = compute_false_alarm_rate(scores, threshold, crisis_mask)
        assert far == 0.0, "NaN scores should not trigger alarms"
