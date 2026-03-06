"""
Bipartition sensitivity analysis for Reduced State Purity detector.

Tests whether the headline result (Reduced Purity d=0.83, rank 1)
is sensitive to the choice of bipartition. The default is (2,4)
for hilbert_dim=8. We test all valid bipartitions:
  (1,8), (2,4), (4,2), (8,1)

Also tests hilbert_dim=6 with (2,3) and (3,2).

If results are partition-sensitive, we report this as a limitation.

Usage:
    python experiments/purity_sensitivity.py
    python experiments/purity_sensitivity.py --quick  # 4 representative crises
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.data_loader import fetch_data, create_feature_matrix, ALL_CRISES
from experiments.evaluation import compute_cohens_d_with_ci
from qcml_geometry.observables import ReducedPurityDetector, BaseRegimeDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# Valid bipartitions for common hilbert dimensions
PARTITION_CONFIGS = {
    'h8_1x8': {'hilbert_dim': 8, 'partition': (1, 8), 'label': '(1,8)'},
    'h8_2x4': {'hilbert_dim': 8, 'partition': (2, 4), 'label': '(2,4) [default]'},
    'h8_4x2': {'hilbert_dim': 8, 'partition': (4, 2), 'label': '(4,2)'},
    'h8_8x1': {'hilbert_dim': 8, 'partition': (8, 1), 'label': '(8,1)'},
    'h6_2x3': {'hilbert_dim': 6, 'partition': (2, 3), 'label': '(2,3) [h=6]'},
    'h6_3x2': {'hilbert_dim': 6, 'partition': (3, 2), 'label': '(3,2) [h=6]'},
}


def run_sensitivity(quick=False, n_bootstrap=1000):
    """Run bipartition sensitivity analysis.

    Args:
        quick: If True, only test 4 representative crises.
        n_bootstrap: Bootstrap resamples for CIs.

    Returns:
        dict: Results per partition per crisis.
    """
    # Fetch data
    logger.info("Fetching data...")
    symbols = ['SPY', 'DIA']
    raw = fetch_data(symbols, '1995-01-01', '2024-12-31')
    prices_df = raw['close'].unstack('symbol').dropna()
    X, dates = create_feature_matrix(prices_df)

    X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=20)
    dates_enriched = dates[19:]
    logger.info(f"Feature matrix: {X_enriched.shape}")

    # Select crises
    if quick:
        crisis_keys = ['2008_gfc', '2018_q4', '2020_covid', '2022_rates']
    else:
        crisis_keys = [k for k in ALL_CRISES if not k.startswith('1997')]

    # Filter to evaluable crises
    evaluable = {}
    for ck in crisis_keys:
        if ck not in ALL_CRISES:
            continue
        ci = ALL_CRISES[ck]
        start = pd.Timestamp(ci['start'])
        end = pd.Timestamp(ci['end'])
        if start < dates_enriched[0] or end > dates_enriched[-1]:
            continue
        evaluable[ck] = ci

    logger.info(f"Evaluating {len(evaluable)} crises")

    window_size = 10
    results = {}

    for config_name, config in PARTITION_CONFIGS.items():
        label = config['label']
        logger.info(f"\n--- Partition {label} (hilbert_dim={config['hilbert_dim']}) ---")

        results[config_name] = {
            'label': label,
            'hilbert_dim': config['hilbert_dim'],
            'partition': list(config['partition']),
            'per_crisis': {},
        }

        for ck, ci in evaluable.items():
            crisis_start = pd.Timestamp(ci['start'])
            crisis_end = pd.Timestamp(ci['end'])

            cutoff_date = crisis_start - pd.Timedelta(days=window_size)
            fit_end_idx = int(np.searchsorted(dates_enriched, cutoff_date))
            if fit_end_idx < 100:
                continue

            cs = crisis_start - pd.Timedelta(days=window_size)
            ce = crisis_end + pd.Timedelta(days=window_size)
            crisis_mask = (dates_enriched >= cs) & (dates_enriched <= ce)
            normal_mask = ~crisis_mask

            det = ReducedPurityDetector(
                hilbert_dim=config['hilbert_dim'],
                n_pca_components=min(8, config['hilbert_dim']),
                partition=config['partition'],
                causal_fit_length=fit_end_idx,
                seed=42,
            )
            det.fit(X_enriched)
            scores = det.compute_regime_scores(X_enriched)

            d, ci_lo, ci_hi = compute_cohens_d_with_ci(
                scores[crisis_mask], scores[normal_mask], n_bootstrap=n_bootstrap,
            )

            results[config_name]['per_crisis'][ck] = {
                'd': float(d) if not np.isnan(d) else None,
                'ci_lo': float(ci_lo) if not np.isnan(ci_lo) else None,
                'ci_hi': float(ci_hi) if not np.isnan(ci_hi) else None,
            }

            logger.info(f"  {ck:20s}  d = {d:.3f}" if not np.isnan(d) else
                        f"  {ck:20s}  d = N/A")

    # Compute summary statistics
    print("\n" + "=" * 70)
    print("BIPARTITION SENSITIVITY SUMMARY")
    print("=" * 70)

    print(f"\n{'Partition':<20} {'Median d':<12} {'Mean d':<12} {'Std d':<12} {'N crises':<10}")
    print("-" * 66)

    summary = {}
    for config_name, config_results in results.items():
        d_vals = [v['d'] for v in config_results['per_crisis'].values()
                  if v['d'] is not None]
        if not d_vals:
            continue
        d_arr = np.array(d_vals)
        med = float(np.median(d_arr))
        mean = float(np.mean(d_arr))
        std = float(np.std(d_arr))

        summary[config_name] = {
            'median_d': med,
            'mean_d': mean,
            'std_d': std,
            'n_crises': len(d_vals),
        }
        results[config_name]['summary'] = summary[config_name]

        print(f"{config_results['label']:<20} {med:<12.3f} {mean:<12.3f} {std:<12.3f} {len(d_vals):<10}")

    # Assess sensitivity
    if len(summary) >= 2:
        medians = [s['median_d'] for s in summary.values()]
        range_d = max(medians) - min(medians)
        cv = np.std(medians) / np.mean(medians) if np.mean(medians) > 0 else 0

        print(f"\nRange of median d across partitions: {range_d:.3f}")
        print(f"CV of median d: {cv:.3f}")

        if range_d > 0.2 or cv > 0.3:
            print("\nASSESSMENT: PARTITION-SENSITIVE — results vary substantially")
            print("Recommendation: Report partition sensitivity as a limitation")
            assessment = "sensitive"
        elif range_d > 0.1:
            print("\nASSESSMENT: MODERATELY SENSITIVE — some variation")
            print("Recommendation: Report default (2,4) with sensitivity note")
            assessment = "moderate"
        else:
            print("\nASSESSMENT: PARTITION-ROBUST — results are stable")
            print("Recommendation: Keep (2,4) as default, note robustness")
            assessment = "robust"
    else:
        assessment = "insufficient_data"

    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'n_bootstrap': n_bootstrap,
        'quick': quick,
        'results': results,
        'assessment': assessment,
    }

    out_dir = ROOT / 'experiments' / 'outputs' / 'purity_sensitivity'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f'purity_sensitivity_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {out_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description='Bipartition sensitivity analysis')
    parser.add_argument('--quick', action='store_true', help='4 representative crises only')
    parser.add_argument('--n-bootstrap', type=int, default=1000)
    args = parser.parse_args()

    run_sensitivity(quick=args.quick, n_bootstrap=args.n_bootstrap)


if __name__ == '__main__':
    main()
