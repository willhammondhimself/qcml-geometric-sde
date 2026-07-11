"""Does any geometric observable carry INCREMENTAL predictive information about
forward tail risk, beyond what current volatility already provides?

This reframes the question from "detect crisis windows" (which failed rigorous
controls) to "predict forward volatility / drawdown with information volatility
does not have." For each detector we compute the partial correlation of its
(causal) score with forward 20-day realized vol and forward 20-day max drawdown,
*controlling for current 20-day vol*. A non-trivial partial correlation = the
geometry adds predictive content beyond volatility. Significance uses a
moving-block bootstrap (overlapping forward windows are strongly autocorrelated).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import experiments.walk_forward_hpo as wf  # noqa: E402
from experiments.data_loader import fetch_data  # noqa: E402
from qcml_geometry import (  # noqa: E402
    BerryPhaseRateDetector,
    MultiLagFidelityDetector,
    ReducedPurityDetector,
    SpectralEntropyDetector,
    SpectralGapDetector,
)

CONFIGS = {
    "Berry Phase Rate": (
        BerryPhaseRateDetector,
        dict(hilbert_dim=6, n_pca_components=8, rolling_window=15, operator_method="random",
             seed=42, normalization="sphere", berry_aggregation="f01"),
    ),
    "Multi-Lag Fidelity": (
        MultiLagFidelityDetector,
        dict(hilbert_dim=4, n_pca_components=8, rolling_window=20, operator_method="pca_inspired",
             seed=42, normalization="sphere"),
    ),
    "Spectral Gap": (
        SpectralGapDetector,
        dict(hilbert_dim=8, n_pca_components=12, rolling_window=20, operator_method="random",
             seed=42, normalization="soft", adaptive_epsilon=True),
    ),
    "Spectral Entropy": (
        SpectralEntropyDetector,
        dict(hilbert_dim=8, n_pca_components=8, rolling_window=20, operator_method="random",
             seed=42, normalization="soft", adaptive_epsilon=True),
    ),
    "Reduced Purity": (
        ReducedPurityDetector,
        dict(hilbert_dim=8, n_pca_components=8, rolling_window=20, operator_method="random",
             seed=42, normalization="soft", adaptive_epsilon=True),
    ),
}


def forward_targets(ret, horizon=20):
    T = len(ret)
    fwd_vol = np.full(T, np.nan)
    fwd_dd = np.full(T, np.nan)
    for t in range(T - horizon):
        fr = ret[t + 1 : t + 1 + horizon]
        fr = fr[np.isfinite(fr)]
        if len(fr) < horizon // 2:
            continue
        fwd_vol[t] = np.std(fr)
        cum = np.cumsum(fr)
        peak = np.maximum.accumulate(cum)
        fwd_dd[t] = float(np.max(peak - cum))  # max drawdown of cum returns
    return fwd_vol, fwd_dd


def _residualize(y, z):
    A = np.vstack([np.ones_like(z), z]).T
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def partial_corr(x, y, z, mask, n_boot=2000, block=60, seed=0):
    """Partial corr(x, y | z) over mask, with moving-block-bootstrap p-value."""
    m = mask & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    X, Y, Z = x[m], y[m], z[m]
    if len(X) < 100:
        return None
    rx, ry = _residualize(X, Z), _residualize(Y, Z)
    r = float(np.corrcoef(rx, ry)[0, 1])

    rng = np.random.default_rng(seed)
    n = len(rx)
    n_blocks = int(np.ceil(n / block))
    boot = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        # shuffle ry blocks against rx to build the null of zero partial corr
        idy = np.concatenate([np.arange(s, s + block) for s in rng.integers(0, n - block + 1, size=n_blocks)])[:n]
        boot[b] = np.corrcoef(rx[idx], ry[idy])[0, 1]
    p = float((np.sum(np.abs(boot) >= abs(r)) + 1) / (n_boot + 1))
    return {"partial_r": r, "p_block_boot": p, "n": int(n)}


def main():
    print("Preparing data...")
    Xe, dates = wf.prepare_data()
    raw = fetch_data(["SPY", "DIA"], "1995-01-01", "2024-12-31")
    close = raw["close"].unstack("symbol").dropna()
    spy = close["SPY"].reindex(pd.to_datetime([str(d.date()) for d in dates]))
    ret = np.log(spy.values.astype(float))
    ret = np.concatenate([[np.nan], np.diff(ret)])
    T = len(dates)

    cur_vol = pd.Series(ret).rolling(20).std().values
    fwd_vol, fwd_dd = forward_targets(ret, horizon=20)

    fit_end = int(0.4 * T)
    causal = np.zeros(T, bool)
    causal[fit_end : T - 20] = True  # causal region, exclude the last horizon

    # baselines: volatility's own predictive content (persistence)
    base_vol = partial_corr(cur_vol, fwd_vol, np.zeros(T), causal)  # corr(cur_vol, fwd_vol)
    base_dd = partial_corr(cur_vol, fwd_dd, np.zeros(T), causal)
    print("\nBASELINE (volatility persistence):")
    print(f"  cur_vol -> fwd_vol : r={base_vol['partial_r']:.3f} (p={base_vol['p_block_boot']:.3f})")
    print(f"  cur_vol -> fwd_dd  : r={base_dd['partial_r']:.3f} (p={base_dd['p_block_boot']:.3f})")

    print("\nGEOMETRIC OBSERVABLES (causal scores; partial corr controlling for current vol):")
    print(f"{'detector':20s} {'contemp r(cur_vol)':>18s} {'fwd_vol|vol':>14s} {'fwd_dd|vol':>14s}")
    for name, (cls, params) in CONFIGS.items():
        try:
            det = cls(causal_fit_length=fit_end, **params)
            det.fit(Xe)
            s = np.asarray(det.compute_regime_scores(Xe), dtype=float)
        except Exception as exc:
            print(f"{name:20s} ERROR {type(exc).__name__}: {exc}")
            continue
        contemp = partial_corr(s, cur_vol, np.zeros(T), causal)
        pv = partial_corr(s, fwd_vol, cur_vol, causal)
        pd_ = partial_corr(s, fwd_dd, cur_vol, causal)

        def fmt(d):
            return "n/a" if d is None else f"{d['partial_r']:+.3f}(p{d['p_block_boot']:.2f})"

        print(f"{name:20s} {fmt(contemp):>18s} {fmt(pv):>14s} {fmt(pd_):>14s}")


if __name__ == "__main__":
    main()
