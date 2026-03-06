"""
Smoke test: Full-Manifold Chern Number Detector

Tests the FullManifoldChernDetector on 4 selected crises using SPY+DIA data
from 2005-2025, then saves results to smoke_results.json.

Usage:
    cd /Users/willhammond/Will\ x\ Average\ Research/qcml-geometric-sde
    python research/ideation/chern_number/smoke_test.py
"""

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)

from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from experiments.evaluation import compute_cohens_d_with_ci
from research.ideation.chern_number.detector import FullManifoldChernDetector

# ── Configuration ─────────────────────────────────────────────────────────────
SYMBOLS = ['SPY', 'DIA']
START_DATE = '2005-01-01'
END_DATE = '2025-01-01'
TARGET_CRISES = ['2008_gfc', '2020_covid', '2022_rates', '2023_svb']
N_BOOTSTRAP = 1000
SEED = 42

# Detector hyperparameters — conservative for smoke test speed
DETECTOR_KWARGS = dict(
    hilbert_dim=4,
    n_pca_components=6,
    operator_method='random',
    rolling_window=20,
    min_expanding=60,
    seed=SEED,
    normalization='sphere',
    epsilon=1e-4,
    adaptive_epsilon=False,
)


def label_crisis_periods(dates, crisis_key: str) -> np.ndarray:
    """Return boolean array: True during the crisis window."""
    crisis = ALL_CRISES[crisis_key]
    start = np.datetime64(crisis['start'])
    end = np.datetime64(crisis['end'])
    return (dates >= start) & (dates <= end)


def main():
    print("=" * 60)
    print("Smoke Test: Full-Manifold Chern Number Detector")
    print("=" * 60)

    # ── 1. Fetch data ─────────────────────────────────────────────────────────
    print(f"\n[1] Fetching {SYMBOLS} {START_DATE}–{END_DATE} ...")
    prices_df = fetch_data(SYMBOLS, START_DATE, END_DATE, source='yfinance', use_cache=True)

    # Pivot to wide close-price DataFrame
    close_wide = (
        prices_df['close']
        .unstack(level='symbol')
        .sort_index()
    )
    print(f"    Close prices: {close_wide.shape[0]} trading days, {close_wide.shape[1]} symbols")

    # ── 2. Build feature matrix ───────────────────────────────────────────────
    print("\n[2] Building feature matrix ...")
    features, dates = create_feature_matrix(close_wide)
    print(f"    Feature matrix: {features.shape}  (T={features.shape[0]}, d={features.shape[1]})")

    # ── 3. Fit detector ───────────────────────────────────────────────────────
    print("\n[3] Fitting FullManifoldChernDetector ...")
    detector = FullManifoldChernDetector(**DETECTOR_KWARGS)
    detector.fit(features)
    print("    Fit complete.")

    # ── 4. Compute regime scores ──────────────────────────────────────────────
    print("\n[4] Computing regime scores (this will take a few minutes) ...")
    scores = detector.compute_regime_scores(features)
    valid_mask = ~np.isnan(scores)
    print(f"    Scores: {valid_mask.sum()} valid / {len(scores)} total time steps")
    print(f"    Score range: [{np.nanmin(scores):.4f}, {np.nanmax(scores):.4f}]")

    # ── 5. Evaluate per crisis ────────────────────────────────────────────────
    print("\n[5] Evaluating Cohen's d per crisis ...")
    results = {}
    issues = []

    for crisis_key in TARGET_CRISES:
        if crisis_key not in ALL_CRISES:
            issues.append(f"Crisis '{crisis_key}' not in ALL_CRISES — skipped.")
            continue

        crisis_info = ALL_CRISES[crisis_key]
        is_crisis = label_crisis_periods(dates.values, crisis_key)
        is_normal = ~is_crisis

        crisis_scores = scores[is_crisis & valid_mask]
        normal_scores = scores[is_normal & valid_mask]

        if len(crisis_scores) < 5:
            issues.append(
                f"Crisis '{crisis_key}': only {len(crisis_scores)} valid crisis points — skipped."
            )
            continue
        if len(normal_scores) < 20:
            issues.append(
                f"Crisis '{crisis_key}': only {len(normal_scores)} valid normal points — skipped."
            )
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

    # ── 6. Summary ────────────────────────────────────────────────────────────
    median_d = float(np.median(list(results.values()))) if results else float('nan')
    passes_threshold = median_d > 0.2  # smoke-test threshold: any signal above noise

    print(f"\n{'─'*60}")
    print(f"  Median Cohen's d : {median_d:.4f}")
    print(f"  Passes threshold : {passes_threshold}  (threshold > 0.20)")
    if issues:
        print(f"  Issues           : {len(issues)}")
        for iss in issues:
            print(f"    - {iss}")

    # ── 7. Save results ───────────────────────────────────────────────────────
    output = {
        "knight": 2,
        "role": "empirical_test",
        "detector": "FullManifoldChernDetector",
        "config": {
            "symbols": SYMBOLS,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "n_bootstrap": N_BOOTSTRAP,
            "hilbert_dim": DETECTOR_KWARGS['hilbert_dim'],
            "n_pca_components": DETECTOR_KWARGS['n_pca_components'],
            "operator_method": DETECTOR_KWARGS['operator_method'],
            "epsilon": DETECTOR_KWARGS['epsilon'],
            "n_planes": DETECTOR_KWARGS['n_pca_components'] * (DETECTOR_KWARGS['n_pca_components'] - 1) // 2,
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
