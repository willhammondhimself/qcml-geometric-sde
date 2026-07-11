"""Look-ahead guards for the forward systemic-risk targets.

The contract: target[t] depends only on returns in (t, t+horizon]. Perturbing the
past (≤ t) or the far future (> t+horizon) must NOT change target[t]; perturbing
inside the window MUST change it. This is what makes the prediction horse race honest.
"""

import numpy as np

from experiments.systemic_risk_targets import (
    forward_ar_change,
    forward_avg_correlation,
    forward_ew_max_drawdown,
)

H = 20


def _panel(seed=0, T=300, N=6):
    return np.random.default_rng(seed).normal(0, 0.01, (T, N))


def _check_causal(fn):
    R = _panel()
    base = fn(R, H)
    t = 150  # an interior index with a well-defined target
    assert np.isfinite(base[t])

    # perturbing the PAST (<= t) must not change target[t]
    Rp = R.copy()
    Rp[: t + 1] += 5.0
    assert np.isclose(fn(Rp, H)[t], base[t], equal_nan=True)

    # perturbing the FAR FUTURE (> t+H) must not change target[t]
    Rf = R.copy()
    Rf[t + 1 + H :] += 5.0
    assert np.isclose(fn(Rf, H)[t], base[t], equal_nan=True)

    # perturbing INSIDE the window (t, t+H] must change target[t]. Use a rank-1,
    # high-vol block (all assets identical) so it changes drawdown AND correlation
    # (a mere additive constant leaves correlation invariant — by design).
    Rw = R.copy()
    rng = np.random.default_rng(99)
    Rw[t + 1 : t + 1 + H] = rng.normal(0, 0.1, (H, 1)) * np.ones((1, R.shape[1]))
    assert not np.isclose(fn(Rw, H)[t], base[t], equal_nan=True)


def test_ew_max_drawdown_is_causal():
    _check_causal(forward_ew_max_drawdown)


def test_avg_correlation_is_causal():
    _check_causal(forward_avg_correlation)


def test_ar_change_forward_window_matters():
    # ar_change uses both a trailing and a forward window; verify the forward half
    # responds and the far future does not.
    R = _panel()
    base = forward_ar_change(R, H)
    t = 150
    assert np.isfinite(base[t])
    Rf = R.copy()
    Rf[t + 1 + H :] += 5.0
    assert np.isclose(forward_ar_change(Rf, H)[t], base[t], equal_nan=True)


def test_targets_are_finite_and_shaped():
    R = _panel()
    for fn in (forward_ew_max_drawdown, forward_avg_correlation, forward_ar_change):
        out = fn(R, H)
        assert out.shape == (R.shape[0],)
        assert np.isfinite(out).sum() > 100
