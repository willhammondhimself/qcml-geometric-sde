"""
Q16: Hilbert-Space Dimension Sweep for BerryPhaseRateDetector
==============================================================

Does a higher-dimensional Hilbert space (d=3,4,6,8,16) improve regime
detection by avoiding Kramers degeneracy and providing richer spectral
structure?

Uses operator_method='random' throughout to avoid the Kramers degeneracy
that plagues 'pca_inspired' at even dimensions (d=4).

4 smoke crises:  2008 GFC, 2020 COVID, 2011 Euro crisis, 2015 China shock
Fit window: SPY + DIA, 2005-01-01 to 2025-01-01
Metric: Cohen's d (in-crisis vs out-of-crisis z-scores)

Usage:
    python research/ideation/hilbert_dim_sweep/test_sweep.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root or this directory
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from qcml_geometry import BerryPhaseRateDetector
from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HILBERT_DIMS = [3, 4, 6, 8, 16]

SMOKE_CRISES = {
    '2008_gfc':    ALL_CRISES['2008_gfc'],
    '2020_covid':  ALL_CRISES['2020_covid'],
    '2011_euro':   ALL_CRISES['2011_euro'],
    '2015_china':  ALL_CRISES['2015_china'],
}

SYMBOLS    = ['SPY', 'DIA']
START_DATE = '2005-01-01'
END_DATE   = '2025-01-01'

# Match existing pipeline defaults
N_PCA_COMPONENTS = 15
ROLLING_WINDOW   = 20
MIN_EXPANDING    = 60
SEED             = 42

OUTPUT_PATH = Path(__file__).parent / 'smoke_results.json'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cohens_d(in_scores: np.ndarray, out_scores: np.ndarray) -> float:
    """Pooled-SD Cohen's d (always positive; larger = better separation).

    d = (mean_in - mean_out) / pooled_sd
    Returns abs(d) so the sign of the crisis signal does not matter.
    """
    n1, n2 = len(in_scores), len(out_scores)
    if n1 < 2 or n2 < 2:
        return np.nan
    m1, m2 = np.mean(in_scores), np.mean(out_scores)
    var1 = np.var(in_scores,  ddof=1)
    var2 = np.var(out_scores, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_sd < 1e-12:
        return np.nan
    return abs((m1 - m2) / pooled_sd)


def evaluate_detector(detector, scores: np.ndarray, dates: pd.DatetimeIndex,
                      crises: dict) -> dict:
    """Compute Cohen's d for each crisis and the median across all crises.

    Args:
        detector: Fitted detector (unused after scoring, kept for API clarity).
        scores: 1-D array of regime scores aligned with ``dates``.
        dates: DatetimeIndex matching ``scores``.
        crises: Dict mapping crisis_key -> {'start': 'YYYY-MM-DD', 'end': ...}.

    Returns:
        Dict with per_crisis Cohen's d values and the overall median_d.
    """
    # Drop NaN prefix from expanding z-score warm-up
    valid_mask = ~np.isnan(scores)
    scores_valid = scores[valid_mask]
    dates_valid  = dates[valid_mask]

    per_crisis = {}
    for key, meta in crises.items():
        start = pd.Timestamp(meta['start'])
        end   = pd.Timestamp(meta['end'])

        in_mask  = (dates_valid >= start) & (dates_valid <= end)
        out_mask = ~in_mask

        in_scores  = scores_valid[in_mask]
        out_scores = scores_valid[out_mask]

        d = cohens_d(in_scores, out_scores)
        per_crisis[key] = round(float(d), 4) if not np.isnan(d) else None

    valid_ds = [v for v in per_crisis.values() if v is not None]
    median_d = float(np.median(valid_ds)) if valid_ds else None
    return {'median_d': median_d, 'per_crisis': per_crisis}


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep():
    np.random.seed(SEED)

    # ------------------------------------------------------------------
    # 1. Fetch market data once (cached by data_loader)
    # ------------------------------------------------------------------
    print(f"Fetching {SYMBOLS} data {START_DATE} → {END_DATE} ...")
    t0 = time.time()
    prices_df_raw = fetch_data(SYMBOLS, START_DATE, END_DATE, source='yfinance')
    elapsed = time.time() - t0
    print(f"  Data fetched in {elapsed:.1f}s")

    # create_feature_matrix wants a DataFrame indexed by date with symbol cols
    # prices_df_raw has a MultiIndex (symbol, date) — pivot to (date, symbol)
    if isinstance(prices_df_raw.index, pd.MultiIndex):
        close_prices = (
            prices_df_raw['close']
            .unstack(level=0)  # level 0 = symbol
            .sort_index()
        )
    else:
        close_prices = prices_df_raw[['close']].copy()

    print(f"  Close price shape: {close_prices.shape}, "
          f"dates: {close_prices.index[0].date()} → {close_prices.index[-1].date()}")

    X_raw, feature_dates = create_feature_matrix(close_prices)
    feature_dates = pd.DatetimeIndex(feature_dates)
    print(f"  Feature matrix: {X_raw.shape[0]} rows × {X_raw.shape[1]} features")

    # ------------------------------------------------------------------
    # 2. Sweep over Hilbert dimensions
    # ------------------------------------------------------------------
    results_by_dim = {}

    for dim in HILBERT_DIMS:
        print(f"\n[dim={dim}] Fitting BerryPhaseRateDetector ...", flush=True)
        t_start = time.time()

        det = BerryPhaseRateDetector(
            hilbert_dim=dim,
            n_pca_components=N_PCA_COMPONENTS,
            operator_method='random',   # avoids Kramers degeneracy at all dims
            rolling_window=ROLLING_WINDOW,
            min_expanding=MIN_EXPANDING,
            seed=SEED,
            normalization='sphere',
            berry_aggregation='f01',
        )

        det.fit(X_raw)
        scores = det.compute_regime_scores(X_raw)
        elapsed_dim = time.time() - t_start

        result = evaluate_detector(det, scores, feature_dates, SMOKE_CRISES)
        results_by_dim[dim] = result

        per_str = "  ".join(
            f"{k}={v:.3f}" for k, v in result['per_crisis'].items()
            if v is not None
        )
        print(f"  median_d={result['median_d']:.4f}  [{per_str}]  ({elapsed_dim:.1f}s)")

    # ------------------------------------------------------------------
    # 3. Summary
    # ------------------------------------------------------------------
    best_dim = max(
        results_by_dim,
        key=lambda d: results_by_dim[d]['median_d'] or -1
    )
    default_d   = results_by_dim.get(8, {}).get('median_d') or 0.0
    best_d      = results_by_dim[best_dim]['median_d'] or 0.0
    improvement = (best_d - default_d) / max(default_d, 1e-9) * 100

    print(f"\n{'='*60}")
    print(f"Best dim: {best_dim}  (median_d={best_d:.4f})")
    print(f"Default (dim=8): median_d={default_d:.4f}")
    print(f"Improvement over default: {improvement:+.1f}% median d")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 4. Serialize results
    # ------------------------------------------------------------------
    output = {
        'metadata': {
            'symbols':           SYMBOLS,
            'start_date':        START_DATE,
            'end_date':          END_DATE,
            'operator_method':   'random',
            'n_pca_components':  N_PCA_COMPONENTS,
            'rolling_window':    ROLLING_WINDOW,
            'min_expanding':     MIN_EXPANDING,
            'seed':              SEED,
            'normalization':     'sphere',
            'berry_aggregation': 'f01',
            'smoke_crises':      list(SMOKE_CRISES.keys()),
        },
        'results_by_dim': {
            str(d): v for d, v in results_by_dim.items()
        },
        'best_dim':                   best_dim,
        'improvement_over_default':   f"{improvement:+.1f}% median d",
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as fh:
        json.dump(output, fh, indent=2)

    print(f"\nResults saved → {OUTPUT_PATH}")

    # ------------------------------------------------------------------
    # 5. Print YAML-style summary for easy copy-paste
    # ------------------------------------------------------------------
    print("\n--- YAML summary ---")
    print("results_by_dim:")
    for dim, res in results_by_dim.items():
        per = res['per_crisis']
        per_str = ", ".join(f"{k}: {v}" for k, v in per.items())
        print(f"  {dim}: {{median_d: {res['median_d']}, per_crisis: {{{per_str}}}}}")
    print(f"best_dim: {best_dim}")
    print(f'improvement_over_default: "{improvement:+.1f}% median d"')

    return output


if __name__ == '__main__':
    run_sweep()
