"""
Smoke Test: Persistent Homology Detector

Tests PersistentHomologyDetector on 4 crises using 8 assets from 2005-2025.
Saves results to smoke_results.json.

Usage:
    cd qcml-geometric-sde
    python research/ideation/persistent_homology/smoke_test.py
"""

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)

from experiments.data_loader import fetch_data, ALL_CRISES
from experiments.evaluation import compute_cohens_d_with_ci
from research.ideation.persistent_homology.detector import PersistentHomologyDetector

# ── Configuration ─────────────────────────────────────────────────────────────
SYMBOLS = ['SPY', 'DIA', 'QQQ', 'IWM', 'TLT', 'GLD', 'XLF', 'XLK']
START_DATE = '2005-01-01'
END_DATE = '2025-01-01'
TARGET_CRISES = ['2008_gfc', '2020_covid', '2022_rates', '2023_svb']
N_BOOTSTRAP = 1000
SEED = 42

DETECTOR_KWARGS = dict(
    rolling_window=60,
    min_expanding=120,
    maxdim=1,
    score_mode='h0_total',
    smooth_window=10,
    seed=SEED,
)


def build_returns(symbols: list, start: str, end: str) -> pd.DataFrame:
    """Fetch close prices and compute log-returns.

    Args:
        symbols: List of ticker symbols.
        start: Start date string.
        end: End date string.

    Returns:
        DataFrame of log-returns, shape (T-1, n_assets), DatetimeIndex.
    """
    print(f"\n[1] Fetching {symbols} {start}–{end} ...")

    # Fetch with yfinance via existing project data loader
    prices_df = fetch_data(symbols, start, end, source='yfinance', use_cache=True)

    # Pivot to wide close-price DataFrame
    close_wide = (
        prices_df['close']
        .unstack(level='symbol')
        .sort_index()
        .reindex(columns=symbols)  # preserve requested order
    )
    print(f"    Close prices: {close_wide.shape[0]} trading days, {close_wide.shape[1]} symbols")

    # Log-returns (shift by 1 so no look-ahead on t=0)
    log_returns = np.log(close_wide / close_wide.shift(1)).dropna(how='all')
    print(f"    Log-returns: {log_returns.shape[0]} rows, {log_returns.shape[1]} assets")
    print(f"    NaN fraction: {log_returns.isna().mean().mean():.3%}")

    return log_returns


def label_crisis_periods(dates: pd.DatetimeIndex, crisis_key: str) -> np.ndarray:
    """Return boolean array: True during the crisis window.

    Args:
        dates: DatetimeIndex aligned with the score array.
        crisis_key: Key in ALL_CRISES.

    Returns:
        Boolean array of shape (T,).
    """
    crisis = ALL_CRISES[crisis_key]
    start = pd.Timestamp(crisis['start'])
    end = pd.Timestamp(crisis['end'])
    return (dates >= start) & (dates <= end)


def main():
    print("=" * 60)
    print("Smoke Test: Persistent Homology Detector (TDA)")
    print("=" * 60)

    issues = []

    # ── 1. Fetch and prepare returns ──────────────────────────────────────────
    returns_df = build_returns(SYMBOLS, START_DATE, END_DATE)

    # ── 2. Fit detector ───────────────────────────────────────────────────────
    print("\n[2] Fitting PersistentHomologyDetector ...")
    detector = PersistentHomologyDetector(**DETECTOR_KWARGS)
    detector.fit(returns_df)
    print("    Fit complete (non-parametric; no training computation).")

    # ── 3. Compute regime scores ──────────────────────────────────────────────
    print("\n[3] Computing regime scores ...")
    print("    (Vietoris-Rips filtration at each time step; ~1-3 min)")
    scores, dates = detector.compute_regime_scores()

    valid_mask = ~np.isnan(scores)
    print(f"    Scores: {valid_mask.sum()} valid / {len(scores)} total time steps")
    print(f"    Score range: [{np.nanmin(scores):.4f}, {np.nanmax(scores):.4f}]")

    # ── 4. Evaluate per crisis ────────────────────────────────────────────────
    print("\n[4] Evaluating Cohen's d per crisis ...")
    results = {}

    for crisis_key in TARGET_CRISES:
        if crisis_key not in ALL_CRISES:
            issues.append(f"Crisis '{crisis_key}' not in ALL_CRISES — skipped.")
            continue

        crisis_info = ALL_CRISES[crisis_key]
        is_crisis = label_crisis_periods(dates, crisis_key)
        is_normal = ~is_crisis

        crisis_scores = scores[is_crisis & valid_mask]
        normal_scores = scores[is_normal & valid_mask]

        if len(crisis_scores) < 5:
            msg = (
                f"Crisis '{crisis_key}': only {len(crisis_scores)} valid "
                f"crisis points — skipped."
            )
            issues.append(msg)
            print(f"    WARNING: {msg}")
            continue
        if len(normal_scores) < 20:
            msg = (
                f"Crisis '{crisis_key}': only {len(normal_scores)} valid "
                f"normal points — skipped."
            )
            issues.append(msg)
            print(f"    WARNING: {msg}")
            continue

        d, ci_lo, ci_hi = compute_cohens_d_with_ci(
            crisis_scores, normal_scores, n_bootstrap=N_BOOTSTRAP, seed=SEED
        )
        results[crisis_key] = round(float(d), 4)
        label = crisis_info['label']
        print(
            f"    {label:30s}  d={d:.4f}  95%CI=[{ci_lo:.4f}, {ci_hi:.4f}]"
            f"  (n_crisis={len(crisis_scores)}, n_normal={len(normal_scores)})"
        )

    # ── 5. Summary ────────────────────────────────────────────────────────────
    median_d = float(np.median(list(results.values()))) if results else float('nan')
    passes_threshold = median_d > 0.2   # smoke-test threshold: any signal above noise

    print(f"\n{'─'*60}")
    print(f"  Detector        : {detector.name}")
    print(f"  Score mode      : {DETECTOR_KWARGS['score_mode']}")
    print(f"  Assets          : {SYMBOLS}")
    print(f"  Rolling window  : {DETECTOR_KWARGS['rolling_window']} days")
    print(f"  Median Cohen's d: {median_d:.4f}")
    print(f"  Passes threshold: {passes_threshold}  (threshold > 0.20)")
    if issues:
        print(f"  Issues          : {len(issues)}")
        for iss in issues:
            print(f"    - {iss}")

    # ── 6. Save results ───────────────────────────────────────────────────────
    output = {
        "detector": "PersistentHomologyDetector",
        "tda_method": "Vietoris-Rips via ripser",
        "config": {
            "symbols": SYMBOLS,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "n_bootstrap": N_BOOTSTRAP,
            "rolling_window": DETECTOR_KWARGS['rolling_window'],
            "min_expanding": DETECTOR_KWARGS['min_expanding'],
            "maxdim": DETECTOR_KWARGS['maxdim'],
            "score_mode": DETECTOR_KWARGS['score_mode'],
            "smooth_window": DETECTOR_KWARGS['smooth_window'],
        },
        "cohens_d_per_crisis": results,
        "median_d": round(median_d, 4),
        "passes_threshold": passes_threshold,
        "implementation_issues": issues,
    }

    out_path = Path(__file__).parent / "smoke_results.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    return output


if __name__ == '__main__':
    result = main()
    print("\nDone.")
