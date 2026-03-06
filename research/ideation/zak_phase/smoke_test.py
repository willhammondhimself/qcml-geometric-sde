"""
Smoke test: Zak Phase Detector (Q12)

Tests the ZakPhaseDetector on 4 selected crises using SPY+DIA data from
2005-2025. Evaluates 'connection', 'winding', and 'wilson' variants, then
saves results for the best-performing variant to smoke_results.json.

Usage:
    cd /Users/willhammond/Will\ x\ Average\ Research/qcml-geometric-sde
    python research/ideation/zak_phase/smoke_test.py
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
from research.ideation.zak_phase.detector import ZakPhaseDetector

# ── Configuration ─────────────────────────────────────────────────────────────
SYMBOLS = ['SPY', 'DIA']
START_DATE = '2005-01-01'
END_DATE = '2025-01-01'
TARGET_CRISES = ['2008_gfc', '2020_covid', '2022_rates', '2023_svb']
N_BOOTSTRAP = 1000
SEED = 42
THRESHOLD_D = 0.2  # smoke-test threshold: any signal above noise

# Detector hyperparameters — conservative for smoke test speed
BASE_KWARGS = dict(
    hilbert_dim=4,
    n_pca_components=6,
    operator_method='random',
    rolling_window=20,
    min_expanding=60,
    seed=SEED,
    normalization='sphere',
    epsilon=1e-4,
)

# Methods to evaluate: name -> zak_method kwarg
METHODS = {
    'connection': 'connection',
    'winding': 'winding',
    'wilson': 'wilson',
}


def label_crisis_periods(dates, crisis_key: str) -> np.ndarray:
    """Return boolean array: True during the crisis window.

    Args:
        dates: DatetimeIndex aligned with the feature matrix.
        crisis_key: Key into ALL_CRISES dict.

    Returns:
        Boolean numpy array of length len(dates).
    """
    crisis = ALL_CRISES[crisis_key]
    start = np.datetime64(crisis['start'])
    end = np.datetime64(crisis['end'])
    return (dates.values >= start) & (dates.values <= end)


def evaluate_detector(detector: ZakPhaseDetector, features: np.ndarray,
                      dates, label: str) -> dict:
    """Fit detector, compute scores, and evaluate Cohen's d per crisis.

    Args:
        detector: Initialized but unfitted ZakPhaseDetector.
        features: Feature matrix (T, d).
        dates: DatetimeIndex aligned with features.
        label: Human-readable label for this run.

    Returns:
        dict with cohens_d_per_crisis, median_d, passes_threshold.
    """
    print(f"\n  [{label}] Fitting and scoring ...")
    try:
        detector.fit(features)
        scores = detector.compute_regime_scores(features)
    except Exception as exc:
        print(f"  [{label}] FAILED: {exc}")
        return {
            'cohens_d_per_crisis': {},
            'median_d': float('nan'),
            'passes_threshold': False,
            'issues': [str(exc)],
        }

    valid_mask = ~np.isnan(scores)
    n_valid = valid_mask.sum()
    print(f"  [{label}] {n_valid}/{len(scores)} valid scores, "
          f"range=[{np.nanmin(scores):.4f}, {np.nanmax(scores):.4f}]")

    results = {}
    issues = []

    for crisis_key in TARGET_CRISES:
        if crisis_key not in ALL_CRISES:
            issues.append(f"  Crisis '{crisis_key}' not found — skipped.")
            continue

        is_crisis = label_crisis_periods(dates, crisis_key)
        is_normal = ~is_crisis

        crisis_scores = scores[is_crisis & valid_mask]
        normal_scores = scores[is_normal & valid_mask]

        if len(crisis_scores) < 5:
            issues.append(
                f"  '{crisis_key}': only {len(crisis_scores)} crisis points — skipped."
            )
            continue
        if len(normal_scores) < 20:
            issues.append(
                f"  '{crisis_key}': only {len(normal_scores)} normal points — skipped."
            )
            continue

        d, ci_lo, ci_hi = compute_cohens_d_with_ci(
            crisis_scores, normal_scores, n_bootstrap=N_BOOTSTRAP, seed=SEED
        )
        results[crisis_key] = round(float(d), 4)
        label_str = ALL_CRISES[crisis_key]['label']
        print(
            f"    {label_str:30s}  d={d:.4f}  95%CI=[{ci_lo:.4f}, {ci_hi:.4f}]"
            f"  (n_crisis={len(crisis_scores)}, n_normal={len(normal_scores)})"
        )

    median_d = float(np.median(list(results.values()))) if results else float('nan')
    passes = median_d > THRESHOLD_D

    for iss in issues:
        print(iss)

    return {
        'cohens_d_per_crisis': results,
        'median_d': round(median_d, 4),
        'passes_threshold': passes,
        'issues': issues,
    }


def main():
    print("=" * 65)
    print("Smoke Test: Zak Phase Detector (Q12)")
    print("=" * 65)

    # ── 1. Fetch data ─────────────────────────────────────────────────────────
    print(f"\n[1] Fetching {SYMBOLS} {START_DATE}-{END_DATE} ...")
    prices_df = fetch_data(SYMBOLS, START_DATE, END_DATE, source='yfinance', use_cache=True)

    close_wide = (
        prices_df['close']
        .unstack(level='symbol')
        .sort_index()
    )
    print(f"    Close prices: {close_wide.shape[0]} trading days, "
          f"{close_wide.shape[1]} symbols")

    # ── 2. Build feature matrix ───────────────────────────────────────────────
    print("\n[2] Building feature matrix ...")
    features, dates = create_feature_matrix(close_wide)
    print(f"    Feature matrix: {features.shape}  (T={features.shape[0]}, d={features.shape[1]})")

    # ── 3. Evaluate each method ───────────────────────────────────────────────
    all_method_results = {}
    for method_label, method_key in METHODS.items():
        print(f"\n[{list(METHODS.keys()).index(method_label)+3}] "
              f"Evaluating Zak Phase — {method_label} method")
        detector = ZakPhaseDetector(**BASE_KWARGS, zak_method=method_key)
        method_results = evaluate_detector(detector, features, dates, method_label)
        all_method_results[method_label] = method_results

    # ── 4. Select best method ─────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("Summary:")
    best_method = None
    best_median = -np.inf
    for method_label, res in all_method_results.items():
        median_d = res['median_d']
        passes = res['passes_threshold']
        tag = " PASS" if passes else ""
        print(f"  {method_label:12s}  median d={median_d:.4f}{tag}")
        if median_d > best_median:
            best_median = median_d
            best_method = method_label

    best_results = all_method_results[best_method]
    print(f"\n  Best method      : {best_method}")
    print(f"  Median Cohen's d : {best_results['median_d']:.4f}")
    print(f"  Passes threshold : {best_results['passes_threshold']}  (> {THRESHOLD_D})")

    # Per-crisis breakdown for best method
    print(f"\n  Per-crisis Cohen's d ({best_method}):")
    for ck, d in best_results['cohens_d_per_crisis'].items():
        label = ALL_CRISES[ck]['label']
        tag = " *** LARGE" if d >= 0.8 else (" * medium" if d >= 0.5 else "")
        print(f"    {label:30s}  d={d:.4f}{tag}")

    # ── 5. Save results ───────────────────────────────────────────────────────
    output = {
        "question": "Q12",
        "question_text": (
            "Does the Zak phase (1D Berry phase) for individual sector "
            "time series provide per-sector crisis signals?"
        ),
        "role": "empirical_test",
        "detector": "ZakPhaseDetector",
        "best_method": best_method,
        "config": {
            "symbols": SYMBOLS,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "n_bootstrap": N_BOOTSTRAP,
            "hilbert_dim": BASE_KWARGS['hilbert_dim'],
            "n_pca_components": BASE_KWARGS['n_pca_components'],
            "operator_method": BASE_KWARGS['operator_method'],
            "rolling_window": BASE_KWARGS['rolling_window'],
            "min_expanding": BASE_KWARGS['min_expanding'],
            "normalization": BASE_KWARGS['normalization'],
            "epsilon": BASE_KWARGS['epsilon'],
        },
        "cohens_d_per_crisis": best_results['cohens_d_per_crisis'],
        "median_d": best_results['median_d'],
        "passes_threshold": best_results['passes_threshold'],
        "all_methods": {
            m: {
                "cohens_d_per_crisis": r['cohens_d_per_crisis'],
                "median_d": r['median_d'],
                "passes_threshold": r['passes_threshold'],
            }
            for m, r in all_method_results.items()
        },
        "implementation_issues": best_results.get('issues', []),
    }

    out_path = Path(__file__).parent / "smoke_results.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    return output


if __name__ == '__main__':
    result = main()
    print("\nDone.")
