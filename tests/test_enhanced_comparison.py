"""
Unit tests for the enhanced comparison pipeline output format.

Tests verify the output structure (scores, thresholds, crisis_masks,
Cohen's d) using a mock fixture — no live API calls or full pipeline runs.
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixture: Synthetic enhanced comparison output
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_enhanced_output():
    """Build a mock output dict matching enhanced_comparison.py's format.

    Simulates 2 methods x 2 crises with 100-point time series.
    """
    T = 100
    rng = np.random.default_rng(42)

    dates = [f"2020-01-{d+1:02d}" for d in range(T)]
    mask_2008 = [False] * 70 + [True] * 20 + [False] * 10
    mask_2020 = [False] * 40 + [True] * 30 + [False] * 30

    methods = ['Berry Phase Rate', 'QFI Determinant']
    crises = ['2008_gfc', '2020_covid']

    results = {}
    for m in methods:
        results[m] = {}
        for ck in crises:
            scores = rng.standard_normal(T).tolist()
            threshold = (np.sort(rng.uniform(0.5, 2.5, size=T))).tolist()
            results[m][ck] = {
                'd': round(rng.uniform(0.3, 1.5), 3),
                'ci_lo': round(rng.uniform(0.1, 0.5), 3),
                'ci_hi': round(rng.uniform(1.0, 2.0), 3),
                'scores': scores,
                'threshold': threshold,
            }

    crisis_masks = {
        '2008_gfc': {
            'mask': mask_2008,
            'dates': dates,
            'crisis_start': '2008-09-01',
            'crisis_end': '2009-03-31',
        },
        '2020_covid': {
            'mask': mask_2020,
            'dates': dates,
            'crisis_start': '2020-02-20',
            'crisis_end': '2020-04-30',
        },
    }

    return {
        'timestamp': '2026-02-23T12:00:00',
        'config': {
            'causal': True,
            'enhanced': True,
            'causal_method': 'per_crisis_cutoff',
            'window_size': 10,
            'n_bootstrap': 1000,
            'quick': True,
            'full': False,
            'n_crises': 2,
            'n_methods': 2,
        },
        'hpo_params': {},
        'results': results,
        'crisis_masks': crisis_masks,
        'summary': {
            'median_d': {'Berry Phase Rate': 0.85, 'QFI Determinant': 0.72},
            'friedman_chi_sq': 4.5,
            'friedman_p': 0.034,
            'mean_ranks': {'Berry Phase Rate': 1.3, 'QFI Determinant': 1.7},
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEnhancedOutputStructure:
    """Verify the enhanced comparison output has all required fields."""

    def test_scores_present_and_nonempty(self, mock_enhanced_output):
        """Each method-crisis pair must have a non-empty 'scores' list."""
        results = mock_enhanced_output['results']
        for method, crises in results.items():
            for ck, data in crises.items():
                assert 'scores' in data, f"Missing 'scores' for {method}/{ck}"
                assert isinstance(data['scores'], list), (
                    f"'scores' should be a list for {method}/{ck}"
                )
                assert len(data['scores']) > 0, (
                    f"'scores' should be non-empty for {method}/{ck}"
                )

    def test_threshold_present_and_same_length_as_scores(self, mock_enhanced_output):
        """Each method-crisis pair must have 'threshold' with same length as 'scores'."""
        results = mock_enhanced_output['results']
        for method, crises in results.items():
            for ck, data in crises.items():
                assert 'threshold' in data, f"Missing 'threshold' for {method}/{ck}"
                assert isinstance(data['threshold'], list), (
                    f"'threshold' should be a list for {method}/{ck}"
                )
                assert len(data['threshold']) == len(data['scores']), (
                    f"'threshold' length ({len(data['threshold'])}) != "
                    f"'scores' length ({len(data['scores'])}) for {method}/{ck}"
                )

    def test_crisis_masks_structure(self, mock_enhanced_output):
        """crisis_masks must have mask, dates, crisis_start, crisis_end per crisis."""
        crisis_masks = mock_enhanced_output['crisis_masks']
        assert len(crisis_masks) > 0, "crisis_masks should not be empty"

        for ck, cm in crisis_masks.items():
            assert 'mask' in cm, f"Missing 'mask' in crisis_masks[{ck}]"
            assert 'dates' in cm, f"Missing 'dates' in crisis_masks[{ck}]"
            assert 'crisis_start' in cm, f"Missing 'crisis_start' in crisis_masks[{ck}]"
            assert 'crisis_end' in cm, f"Missing 'crisis_end' in crisis_masks[{ck}]"
            assert isinstance(cm['mask'], list), f"'mask' should be a list for {ck}"
            assert isinstance(cm['dates'], list), f"'dates' should be a list for {ck}"
            assert isinstance(cm['crisis_start'], str), f"'crisis_start' should be str for {ck}"
            assert isinstance(cm['crisis_end'], str), f"'crisis_end' should be str for {ck}"

    def test_scores_and_mask_same_length(self, mock_enhanced_output):
        """Scores and crisis mask arrays must have the same length (same dates_enriched)."""
        results = mock_enhanced_output['results']
        crisis_masks = mock_enhanced_output['crisis_masks']

        for method, crises in results.items():
            for ck, data in crises.items():
                if ck in crisis_masks:
                    mask_len = len(crisis_masks[ck]['mask'])
                    scores_len = len(data['scores'])
                    assert scores_len == mask_len, (
                        f"scores length ({scores_len}) != mask length ({mask_len}) "
                        f"for {method}/{ck}"
                    )

    def test_cohens_d_and_cis_present(self, mock_enhanced_output):
        """Each method-crisis pair must still have d, ci_lo, ci_hi."""
        results = mock_enhanced_output['results']
        for method, crises in results.items():
            for ck, data in crises.items():
                assert 'd' in data, f"Missing 'd' for {method}/{ck}"
                assert 'ci_lo' in data, f"Missing 'ci_lo' for {method}/{ck}"
                assert 'ci_hi' in data, f"Missing 'ci_hi' for {method}/{ck}"

    def test_config_has_enhanced_flag(self, mock_enhanced_output):
        """Config must have enhanced=True to distinguish from base pipeline."""
        config = mock_enhanced_output['config']
        assert config.get('enhanced') is True, "config['enhanced'] should be True"
        assert config.get('causal') is True, "config['causal'] should be True"

    def test_summary_present(self, mock_enhanced_output):
        """Summary must have median_d, friedman_chi_sq, friedman_p."""
        summary = mock_enhanced_output['summary']
        assert 'median_d' in summary
        assert 'friedman_chi_sq' in summary
        assert 'friedman_p' in summary

    def test_dates_in_crisis_masks_same_length_as_mask(self, mock_enhanced_output):
        """dates[] and mask[] within each crisis_masks entry must have the same length."""
        for ck, cm in mock_enhanced_output['crisis_masks'].items():
            assert len(cm['dates']) == len(cm['mask']), (
                f"dates length ({len(cm['dates'])}) != mask length ({len(cm['mask'])}) "
                f"for crisis_masks[{ck}]"
            )

    def test_mask_contains_booleans(self, mock_enhanced_output):
        """Crisis mask values should all be booleans."""
        for ck, cm in mock_enhanced_output['crisis_masks'].items():
            for i, val in enumerate(cm['mask']):
                assert isinstance(val, bool), (
                    f"mask[{i}] is {type(val).__name__}, expected bool, "
                    f"in crisis_masks[{ck}]"
                )

    def test_scores_are_finite_numbers(self, mock_enhanced_output):
        """All scores should be finite (no NaN or inf after conversion)."""
        for method, crises in mock_enhanced_output['results'].items():
            for ck, data in crises.items():
                for i, val in enumerate(data['scores']):
                    assert np.isfinite(val), (
                        f"scores[{i}] = {val} is not finite for {method}/{ck}"
                    )


class TestHelperFunctions:
    """Test helper functions from enhanced_comparison module."""

    def test_compute_adaptive_threshold_length(self):
        """Adaptive threshold output must match input length."""
        from experiments.enhanced_comparison import compute_adaptive_threshold
        scores = np.random.randn(200)
        thresholds = compute_adaptive_threshold(scores, min_expanding=20, quantile=0.95)
        assert len(thresholds) == len(scores)

    def test_compute_adaptive_threshold_nan_prefix(self):
        """First min_expanding entries should be NaN."""
        from experiments.enhanced_comparison import compute_adaptive_threshold
        scores = np.random.randn(100)
        thresholds = compute_adaptive_threshold(scores, min_expanding=20, quantile=0.95)
        assert np.all(np.isnan(thresholds[:20]))
        assert not np.isnan(thresholds[20])

    def test_apply_persistence_filter_short_runs(self):
        """Short True runs should be removed."""
        from experiments.enhanced_comparison import apply_persistence_filter
        mask = np.array([False, True, True, False, True, True, True, True, False])
        filtered = apply_persistence_filter(mask, min_persistence=3)
        assert not filtered[1]  # 2-run removed
        assert not filtered[2]
        assert filtered[4]      # 4-run kept
        assert filtered[5]
        assert filtered[6]
        assert filtered[7]

    def test_scores_to_list_replaces_nan(self):
        """_scores_to_list should replace NaN with 0.0."""
        from experiments.enhanced_comparison import _scores_to_list
        scores = np.array([1.0, np.nan, 3.0, np.nan])
        result = _scores_to_list(scores)
        assert result == [1.0, 0.0, 3.0, 0.0]

    def test_threshold_to_list_replaces_nan(self):
        """_threshold_to_list should replace NaN with 0.0."""
        from experiments.enhanced_comparison import _threshold_to_list
        thresholds = np.array([np.nan, np.nan, 1.5, 1.6])
        result = _threshold_to_list(thresholds)
        assert result == [0.0, 0.0, 1.5, 1.6]
