"""
Fixed-HP ablation: do geometric observables still work with pre-specified
hyperparameters (no crisis-specific tuning)?

Compares fixed defaults (h=8, p=15, pca_inspired, rw=20) against the
LOCO-tuned results from the main comparison.

Usage:
    python experiments/fixed_hp_ablation.py
    python experiments/fixed_hp_ablation.py --quick
"""

import argparse, json, logging, sys, warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / '.env')

from qcml_geometry import BerryPhaseRateDetector, QFIDeterminantDetector, MultiLagFidelityDetector
from qcml_geometry.observables import BaseRegimeDetector
from experiments.data_loader import fetch_polygon_data, create_feature_matrix, ALL_CRISES
from experiments.evaluation import compute_cohens_d_with_ci

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', force=True)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
np.random.seed(42)

FIXED_DEFAULTS = dict(hilbert_dim=8, n_pca_components=15, operator_method='pca_inspired', rolling_window=20, seed=42)
EXTENSION_DAYS = 10


def run_ablation(quick=False):
    """Run the fixed-HP ablation study.

    Args:
        quick: If True, only evaluate on 4 representative crises.

    Returns:
        Dictionary of per-method, per-crisis Cohen's d results.
    """
    # Fetch data
    symbols = ['SPY', 'DIA']
    raw = fetch_polygon_data(symbols, '2005-01-01', '2024-12-31')
    prices_df = raw['close'].unstack('symbol').dropna()
    X_full, dates_full = create_feature_matrix(prices_df)
    X_enriched = BaseRegimeDetector.build_enriched_features(X_full, lookback=20)
    dates_enriched = dates_full[19:]  # lookback-1 offset

    methods = [
        ('Berry Phase Rate', lambda: BerryPhaseRateDetector(**FIXED_DEFAULTS)),
        ('QFI Determinant', lambda: QFIDeterminantDetector(**FIXED_DEFAULTS)),
        ('Multi-Lag Fidelity', lambda: MultiLagFidelityDetector(**FIXED_DEFAULTS)),
    ]

    if quick:
        crisis_keys = ['2008_gfc', '2010_flash', '2020_covid', '2023_svb']
    else:
        crisis_keys = list(ALL_CRISES.keys())

    results = {}
    for method_name, factory in methods:
        logger.info(f"\n--- {method_name} ---")
        det = factory()
        det.fit(X_enriched)
        scores = det.compute_regime_scores(X_enriched)

        for ck in crisis_keys:
            ci = ALL_CRISES[ck]
            cs = pd.Timestamp(ci['start']) - pd.Timedelta(days=EXTENSION_DAYS * 1.5)
            ce = pd.Timestamp(ci['end']) + pd.Timedelta(days=EXTENSION_DAYS * 1.5)

            crisis_mask = (dates_enriched >= cs) & (dates_enriched <= ce)
            normal_mask = ~crisis_mask

            crisis_scores = scores[crisis_mask]
            normal_scores = scores[normal_mask]

            d, ci_lo, ci_hi = compute_cohens_d_with_ci(crisis_scores, normal_scores, n_bootstrap=5000)

            results[f"{method_name}|{ck}"] = {
                'method': method_name, 'crisis': ck,
                'd': round(float(d), 3) if not np.isnan(d) else None,
                'ci_lo': round(float(ci_lo), 3) if not np.isnan(ci_lo) else None,
                'ci_hi': round(float(ci_hi), 3) if not np.isnan(ci_hi) else None,
            }
            logger.info(f"  {ck:20s}: d={d:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("FIXED-HP ABLATION SUMMARY")
    logger.info("=" * 60)
    for m in ['Berry Phase Rate', 'QFI Determinant', 'Multi-Lag Fidelity']:
        ds = [r['d'] for k, r in results.items() if r['method'] == m and r['d'] is not None]
        logger.info(f"  {m:25s}: mean d = {np.mean(ds):.3f}, median d = {np.median(ds):.3f}")

    # Save
    out_dir = ROOT / 'experiments' / 'outputs' / 'regime_detection' / 'fixed_hp_ablation'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {'timestamp': ts, 'config': FIXED_DEFAULTS, 'results': results}
    with open(out_dir / f'fixed_hp_ablation_{ts}.json', 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {out_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Fixed-HP ablation: geometric observables with pre-specified hyperparameters"
    )
    parser.add_argument('--quick', action='store_true',
                        help='Only evaluate on 4 representative crises (GFC, Flash, COVID, SVB)')
    args = parser.parse_args()
    run_ablation(quick=args.quick)


if __name__ == '__main__':
    main()
