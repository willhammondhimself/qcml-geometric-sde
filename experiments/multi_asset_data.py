"""Multi-asset universe, pre-registered slices, and correlation-manifold features.

The whole point: give the geometry a genuinely high-dimensional correlation manifold
(many weakly-correlated assets), not the ~95%-correlated SPY+DIA pair it has always seen.

Two representations are produced from the same panel, so baselines and geometry see the
same underlying information:
  * returns_matrix  — (T, N) per-asset log returns; what AbsorptionRatio / Turbulence /
    Dispersion consume directly (they form the cross-asset correlation matrix).
  * correlation_features — (T, M) rolling pairwise correlations (lower triangle) plus
    correlation-matrix eigen-summaries; the geometry's input (→ PCA → manifold).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.data_loader import fetch_data  # noqa: E402

# Master universe
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC"]
SECTORS_CORE = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB"]  # pre-2015
MACRO = ["TLT", "IEF", "HYG", "LQD", "GLD", "DBC", "UUP", "EEM", "VNQ"]
CRYPTO = ["BTC-USD", "ETH-USD"]

# Pre-registered slices (see research/multi_asset_preregistration.md). Fixed; no additions.
SLICES = {
    "equity_sectors": SECTORS_CORE,
    "macro_crossasset": MACRO,
    "sectors+macro": SECTORS_CORE + MACRO,
    "full": SECTORS + MACRO,
    "full+crypto": SECTORS + MACRO + CRYPTO,
}


def fetch_panel(symbols, start="2005-01-01", end="2024-12-31"):
    """Aligned close-price panel (max common history) for the given symbols.

    Guards against a cached *partial* fetch (yfinance occasionally returns only a
    subset of a batch and that gets cached): if any requested symbol is missing or
    near-empty, re-fetch once bypassing the cache.
    """

    def _fetch(use_cache):
        raw = fetch_data(list(symbols), start, end, use_cache=use_cache)
        prices = raw["close"].unstack("symbol")
        return prices[[s for s in symbols if s in prices.columns]]

    prices = _fetch(True)
    complete = [s for s in symbols if s in prices.columns and prices[s].notna().sum() > 100]
    if len(complete) < len(symbols):
        prices = _fetch(False)  # cached fetch was partial → refresh
    prices = prices[[s for s in symbols if s in prices.columns]].dropna()
    return prices


def returns_matrix(prices: pd.DataFrame):
    """(R, dates): per-asset daily log returns, NaN warmup dropped."""
    logret = np.log(prices / prices.shift(1)).dropna()
    return logret.values, logret.index


def correlation_features(prices: pd.DataFrame, window: int = 20, eig_k: int = 3):
    """(C, dates): the correlation manifold the geometry sees.

    Per date t (using only the trailing ``window`` of returns, strictly causal):
      * lower-triangle of the N×N rolling correlation matrix  → N(N-1)/2 features
      * top-``eig_k`` eigenvalue variance fractions + participation ratio of the
        correlation spectrum (compact systemic-structure summary)
    """
    logret = np.log(prices / prices.shift(1)).dropna()
    R = logret.values
    dates = logret.index
    T, N = R.shape
    tril_i, tril_j = np.tril_indices(N, k=-1)
    n_pairs = len(tril_i)

    feats = np.full((T, n_pairs + eig_k + 1), np.nan)
    for t in range(window, T):
        w = R[t - window : t]  # strictly trailing → causal
        c = np.corrcoef(w, rowvar=False)
        c = np.nan_to_num(c, nan=0.0)
        feats[t, :n_pairs] = c[tril_i, tril_j]
        ev = np.sort(np.clip(np.linalg.eigvalsh(c), 0, None))[::-1]
        tot = ev.sum()
        if tot > 0:
            feats[t, n_pairs : n_pairs + eig_k] = np.cumsum(ev[:eig_k]) / tot
            feats[t, -1] = (tot**2) / np.sum(ev**2)  # participation ratio
    mask = ~np.any(np.isnan(feats), axis=1)
    return feats[mask], dates[mask]


def slice_diversity(prices: pd.DataFrame) -> float:
    """Pre-registered diversity metric: 1 − mean |pairwise correlation| over the slice."""
    logret = np.log(prices / prices.shift(1)).dropna()
    c = np.corrcoef(logret.values, rowvar=False)
    N = c.shape[0]
    off = c[np.tril_indices(N, k=-1)]
    return float(1.0 - np.mean(np.abs(off)))


if __name__ == "__main__":
    # quick descriptive smoke (no evaluation): slice sizes, history, diversity
    for name, members in SLICES.items():
        try:
            p = fetch_panel(members)
            R, d = returns_matrix(p)
            div = slice_diversity(p)
            print(
                f"{name:18s} N={p.shape[1]:2d}  {d[0].date()}..{d[-1].date()}  "
                f"rows={len(d):5d}  diversity={div:.3f}"
            )
        except Exception as exc:
            print(f"{name:18s} ERROR {type(exc).__name__}: {exc}")
