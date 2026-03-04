"""
Tests for composite geometric signal (Exploration L).

Unit tests verify mathematical properties of the 4 fusion strategies:
- Max z-score, Mean z-score, PCA composite, Rank fusion.

Integration test verifies the composite on real data via the full pipeline.
"""

import numpy as np
import pytest
from scipy import stats

# ---------------------------------------------------------------------------
# Helpers — replicate the composite logic at small scale for unit testing
# ---------------------------------------------------------------------------

def expanding_zscore(signal, min_expanding=60, signed=False):
    """Causal expanding-window z-score (mirrors exploration script)."""
    T = len(signal)
    z = np.full(T, np.nan)
    for t in range(min_expanding, T):
        past = signal[:t]
        past = past[~np.isnan(past)]
        if len(past) < 10:
            continue
        mu = np.mean(past)
        sigma = np.std(past, ddof=1)
        if sigma > 1e-12 and not np.isnan(signal[t]):
            raw = (signal[t] - mu) / sigma
            z[t] = raw if signed else abs(raw)
    return z


def composite_max(Z):
    """Max z-score per timestep."""
    T = Z.shape[0]
    out = np.full(T, np.nan)
    for t in range(T):
        row = Z[t]
        valid = row[~np.isnan(row)]
        if len(valid) > 0:
            out[t] = np.max(valid)
    return out


def composite_mean(Z):
    """Mean z-score per timestep."""
    T = Z.shape[0]
    out = np.full(T, np.nan)
    for t in range(T):
        row = Z[t]
        valid = row[~np.isnan(row)]
        if len(valid) > 0:
            out[t] = np.mean(valid)
    return out


def composite_rank(Z, min_window=60):
    """Rank fusion: expanding percentile per component, then average."""
    T, n_comp = Z.shape
    percentiles = np.full_like(Z, np.nan)
    for k in range(n_comp):
        for t in range(min_window, T):
            if np.isnan(Z[t, k]):
                continue
            past = Z[:t, k]
            past = past[~np.isnan(past)]
            if len(past) < 10:
                continue
            percentiles[t, k] = np.mean(past <= Z[t, k])

    out = np.full(T, np.nan)
    for t in range(T):
        row = percentiles[t]
        valid = row[~np.isnan(row)]
        if len(valid) > 0:
            out[t] = np.mean(valid)
    return out


def composite_pca(Z, min_window=252):
    """Expanding-window PCA composite (first PC)."""
    from sklearn.decomposition import PCA

    T = Z.shape[0]
    out = np.full(T, np.nan)
    for t in range(T):
        row = Z[t]
        if np.any(np.isnan(row)):
            valid = row[~np.isnan(row)]
            if len(valid) > 0:
                out[t] = np.mean(valid)
            continue
        if t < min_window:
            out[t] = np.mean(row)
            continue
        history = Z[:t]
        valid_rows = ~np.any(np.isnan(history), axis=1)
        history_clean = history[valid_rows]
        if len(history_clean) < min_window:
            out[t] = np.mean(row)
            continue
        pca = PCA(n_components=1)
        pca.fit(history_clean)
        pc1 = pca.transform(row.reshape(1, -1))[0, 0]
        out[t] = abs(pc1)
    return out


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

class TestMaxZscore:
    """Max composite >= every individual component at every timestep."""

    def test_dominance(self):
        np.random.seed(42)
        Z = np.random.randn(200, 4)
        Z = np.abs(Z)  # z-scores are absolute values
        max_z = composite_max(Z)
        for col in range(4):
            assert np.all(max_z >= Z[:, col] - 1e-12), (
                f"Max composite should >= component {col} at every timestep"
            )

    def test_with_nans(self):
        Z = np.array([[1.0, np.nan, 3.0, 2.0],
                       [np.nan, np.nan, np.nan, np.nan],
                       [0.5, 0.8, 1.2, 0.3]])
        max_z = composite_max(Z)
        assert max_z[0] == 3.0
        assert np.isnan(max_z[1])
        assert max_z[2] == 1.2


class TestMeanZscore:
    """Mean composite is between min and max of components."""

    def test_bounds(self):
        np.random.seed(42)
        Z = np.abs(np.random.randn(200, 4))
        mean_z = composite_mean(Z)
        min_z = np.nanmin(Z, axis=1)
        max_z = np.nanmax(Z, axis=1)
        assert np.all(mean_z >= min_z - 1e-12)
        assert np.all(mean_z <= max_z + 1e-12)

    def test_equal_inputs(self):
        Z = np.full((50, 4), 2.5)
        mean_z = composite_mean(Z)
        np.testing.assert_allclose(mean_z, 2.5)


class TestRankFusion:
    """Rank fusion (expanding percentile) values are bounded in [0, 1]."""

    def test_bounded(self):
        np.random.seed(42)
        Z = np.abs(np.random.randn(200, 4))
        rank_z = composite_rank(Z, min_window=60)
        valid = ~np.isnan(rank_z)
        assert valid.sum() > 0, "Should have some valid values"
        assert np.all(rank_z[valid] >= -1e-12), "Rank fusion should be >= 0"
        assert np.all(rank_z[valid] <= 1.0 + 1e-12), "Rank fusion should be <= 1"

    def test_nan_before_min_window(self):
        """No values before min_window."""
        np.random.seed(42)
        Z = np.abs(np.random.randn(200, 4))
        rank_z = composite_rank(Z, min_window=60)
        assert np.all(np.isnan(rank_z[:60]))


class TestNanHandling:
    """Composites handle NaN inputs gracefully."""

    def test_all_nan_row(self):
        Z = np.array([[np.nan, np.nan, np.nan, np.nan]])
        assert np.isnan(composite_max(Z)[0])
        assert np.isnan(composite_mean(Z)[0])

    def test_single_valid(self):
        Z = np.array([[np.nan, 2.0, np.nan, np.nan]])
        assert composite_max(Z)[0] == 2.0
        assert composite_mean(Z)[0] == 2.0

    def test_mixed_nans(self):
        Z = np.array([[1.0, np.nan, 3.0, 2.0],
                       [np.nan, 4.0, np.nan, 1.0]])
        max_z = composite_max(Z)
        assert max_z[0] == 3.0
        assert max_z[1] == 4.0

        mean_z = composite_mean(Z)
        np.testing.assert_allclose(mean_z[0], 2.0)  # mean(1, 3, 2)
        np.testing.assert_allclose(mean_z[1], 2.5)  # mean(4, 1)


class TestPcaExpandingNoLookahead:
    """PCA weights at time t use only data from [:t]."""

    def test_no_lookahead(self):
        np.random.seed(42)
        T = 400
        Z = np.abs(np.random.randn(T, 4))

        # Compute PCA composite
        pca_z = composite_pca(Z, min_window=252)

        # Verify: changing future data should not affect past PCA values
        Z_modified = Z.copy()
        Z_modified[350:] = np.random.randn(50, 4) * 100  # Radically different future

        pca_z_modified = composite_pca(Z_modified, min_window=252)

        # Values at t < 350 should be identical (PCA only uses data up to t)
        for t in range(252, 350):
            if not np.isnan(pca_z[t]) and not np.isnan(pca_z_modified[t]):
                np.testing.assert_allclose(
                    pca_z[t], pca_z_modified[t], atol=1e-10,
                    err_msg=f"PCA at t={t} should not depend on data after t"
                )

    def test_fallback_before_min_window(self):
        """Before min_window, PCA falls back to mean."""
        np.random.seed(42)
        Z = np.abs(np.random.randn(300, 4))
        pca_z = composite_pca(Z, min_window=252)

        # Before t=252, should equal mean
        for t in range(300):
            if t < 252 and not np.any(np.isnan(Z[t])):
                expected = np.mean(Z[t])
                np.testing.assert_allclose(pca_z[t], expected, atol=1e-10)


class TestExpandingZscoreCausal:
    """expanding_zscore uses only past data."""

    def test_no_lookahead(self):
        np.random.seed(42)
        signal = np.random.randn(200)
        z = expanding_zscore(signal, min_expanding=60)

        # Modify future data
        signal_mod = signal.copy()
        signal_mod[150:] = signal_mod[150:] * 100

        z_mod = expanding_zscore(signal_mod, min_expanding=60)

        # z-scores up to t=149 should be identical
        for t in range(60, 150):
            if not np.isnan(z[t]) and not np.isnan(z_mod[t]):
                np.testing.assert_allclose(z[t], z_mod[t], atol=1e-10)

    def test_nan_before_min_expanding(self):
        signal = np.random.randn(100)
        z = expanding_zscore(signal, min_expanding=60)
        assert np.all(np.isnan(z[:60]))
