"""Forward systemic-risk targets for the multi-asset prediction horse race.

All targets at time t use ONLY returns strictly after t (the [t+1, t+1+horizon]
window), so they never leak into causal features computed up to t. The look-ahead
guard is unit-tested in tests/test_systemic_targets.py.

Targets (predict the future, controlling for what current vol/AR already know):
  * forward_ew_max_drawdown — max drawdown of the equal-weight panel over the next
    `horizon` days. The headline systemic tail-risk target.
  * forward_avg_correlation — mean |pairwise correlation| over the next `horizon`
    days (forward correlation-regime / diversification breakdown).
  * forward_ar_change — change in Absorption Ratio (top-k eigenvalue fraction) from
    the trailing window to the forward window (forward systemic concentration).
"""

from __future__ import annotations

import numpy as np


def forward_ew_max_drawdown(R: np.ndarray, horizon: int = 20) -> np.ndarray:
    """Max drawdown of the equal-weight portfolio over (t, t+horizon]. Positive = worse."""
    ew = np.asarray(R, dtype=float).mean(axis=1)
    T = len(ew)
    out = np.full(T, np.nan)
    for t in range(T - horizon):
        fr = ew[t + 1 : t + 1 + horizon]
        if not np.all(np.isfinite(fr)):
            continue
        cum = np.cumsum(fr)
        peak = np.maximum.accumulate(cum)
        out[t] = float(np.max(peak - cum))
    return out


def forward_avg_correlation(R: np.ndarray, horizon: int = 20) -> np.ndarray:
    """Mean |pairwise correlation| of asset returns over (t, t+horizon]."""
    R = np.asarray(R, dtype=float)
    T, N = R.shape
    iu = np.triu_indices(N, k=1)
    out = np.full(T, np.nan)
    for t in range(T - horizon):
        w = R[t + 1 : t + 1 + horizon]
        if w.shape[0] < 3 or not np.all(np.isfinite(w)):
            continue
        c = np.corrcoef(w, rowvar=False)
        out[t] = float(np.nanmean(np.abs(c[iu])))
    return out


def _absorption_ratio(window: np.ndarray, k: int = 2) -> float:
    c = np.corrcoef(window, rowvar=False)
    c = np.nan_to_num(c, nan=0.0)
    ev = np.sort(np.clip(np.linalg.eigvalsh(c), 0, None))[::-1]
    tot = ev.sum()
    return float(ev[:k].sum() / tot) if tot > 0 else np.nan


def forward_ar_change(R: np.ndarray, horizon: int = 20, k: int = 2) -> np.ndarray:
    """Forward minus trailing Absorption Ratio (systemic concentration change)."""
    R = np.asarray(R, dtype=float)
    T, N = R.shape
    out = np.full(T, np.nan)
    for t in range(horizon, T - horizon):
        past = _absorption_ratio(R[t - horizon : t], k)
        fut = _absorption_ratio(R[t + 1 : t + 1 + horizon], k)
        if np.isfinite(past) and np.isfinite(fut):
            out[t] = fut - past
    return out


TARGETS = {
    "fwd_ew_maxdd": forward_ew_max_drawdown,
    "fwd_avg_corr": forward_avg_correlation,
    "fwd_ar_change": forward_ar_change,
}
