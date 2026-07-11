"""Unit tests for the pure overfitting-statistics functions (no network/detectors)."""

import numpy as np
import pytest

pytest.importorskip("scipy")

from experiments.overfitting_stats import (  # noqa: E402
    deflate_d,
    effective_n_trials,
    gap_stats,
    multiplicity,
    pbo_from_matrix,
)

# --- PBO / CSCV ------------------------------------------------------------- #


def test_pbo_dominant_config_is_zero():
    # One config beats every other on every crisis → IS-best is always OOS-best.
    M = np.zeros((5, 8))
    M[0, :] = 1.0
    res = pbo_from_matrix(M, n_partitions=8)
    assert res["pbo"] == 0.0
    assert res["n_splits"] == 70  # C(8,4)


def test_pbo_noise_is_near_half():
    rng = np.random.default_rng(0)
    M = rng.normal(size=(20, 8))
    res = pbo_from_matrix(M, n_partitions=8)
    assert 0.0 <= res["pbo"] <= 1.0
    assert 0.2 < res["pbo"] < 0.8  # no real skill → IS-best ~coin-flip OOS


def test_pbo_drops_nan_config_rows():
    M = np.ones((4, 6))
    M[2, 0] = np.nan  # this config has an incomplete row → dropped
    res = pbo_from_matrix(M, n_partitions=6)
    assert res["n_configs"] == 3


# --- effective N ------------------------------------------------------------ #


def test_effective_n_identical_configs_is_one():
    base = np.linspace(0, 1, 8)
    M = np.vstack([base + 1e-9 * i for i in range(6)])  # near-perfectly correlated
    assert effective_n_trials(M) < 2.0


def test_effective_n_independent_configs_is_large():
    rng = np.random.default_rng(1)
    M = rng.normal(size=(10, 40))  # ~independent rows
    assert effective_n_trials(M) > 5.0


# --- deflation -------------------------------------------------------------- #


def test_deflated_d_never_exceeds_observed():
    ds = np.array([0.1, 0.3, 0.5, 0.7, 0.6, 0.2])
    res = deflate_d(0.7, ds, n_eff=6)
    assert 0.0 <= res["deflated_d"] <= 0.7
    assert res["expected_max_under_null"] >= 0.0


def test_deflation_zero_dispersion_is_noop():
    res = deflate_d(0.5, np.full(8, 0.5), n_eff=8)
    assert res["deflated_d"] == 0.5


# --- gap significance ------------------------------------------------------- #


def test_gap_stats_detects_consistent_gap():
    # 8 windows: a consistent ~0.4 gap is reachable by the sign-flip permutation
    # (min p ≈ 2/2^8 ≈ 0.008). With <6 windows it is unreachable by construction.
    in_s = {k: 0.8 for k in "abcdefgh"}
    oos = {k: 0.4 for k in "abcdefgh"}
    res = gap_stats(in_s, oos, n_boot=2000, n_perm=4000, seed=0)
    assert res["mean_gap"] == pytest.approx(0.4, abs=0.05)
    assert res["gap_ci95"][0] > 0  # CI excludes zero
    assert res["significant"] is True


def test_gap_stats_no_gap_not_significant():
    d = {"a": 0.5, "b": 0.6, "c": 0.55}
    res = gap_stats(d, dict(d), n_boot=2000, n_perm=2000, seed=0)
    assert res["mean_gap"] == pytest.approx(0.0, abs=1e-9)
    assert res["significant"] is False


# --- multiplicity ----------------------------------------------------------- #


def test_multiplicity_adjusts_and_flags():
    res = multiplicity({"m1": 0.001, "m2": 0.04, "m3": 0.5})
    assert res["m1"]["holm_adjusted_p"] >= res["m1"]["raw_p"]
    assert res["m1"]["bh_adjusted_p"] >= res["m1"]["raw_p"]
    assert res["m1"]["holm_rejected"] is True
    assert res["m3"]["bh_rejected"] is False
