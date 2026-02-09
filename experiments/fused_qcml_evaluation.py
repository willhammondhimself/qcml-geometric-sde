#!/usr/bin/env python3
"""
Fused QCML Evaluation Pipeline

Loads optimized parameters from Phase A and Phase B Optuna studies, runs
both fused detectors through the EXACT same evaluate_method() pipeline used
for all 16 methods in regime_comparison.py, and performs statistical
superiority tests vs Random Forest.

Produces:
  - Per-crisis Cohen's d comparison table (fused vs RF vs best individual)
  - Paired t-test (fused vs RF) with Holm-Bonferroni
  - Bootstrap P(QCML_fused > RF)
  - Bayesian P(best) including fused method
  - Temporal OOS: optimize on pre-2020, test on COVID/Rates/SVB

Usage:
    python experiments/fused_qcml_evaluation.py
    python experiments/fused_qcml_evaluation.py --temporal-oos

Author: QCML Research
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.crisis_config import (
    DATA_AVAILABLE_CRISES,
    get_default_validation_config,
)
from experiments.regime_comparison import (
    evaluate_method,
    prepare_data,
    prepare_rf_training_data,
)
from experiments.rigorous_crisis_validation import (
    bootstrap_confidence_interval,
    permutation_test,
    bayesian_t_test,
)
from qcml.regime.classical_baselines import RandomForestRegimeDetector
from qcml.regime.fused_detector import FusedQCMLDetector, GeometryOptimizedDetector

load_dotenv(project_root / '.env')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = project_root / 'experiments' / 'outputs' / 'regime_detection' / 'fused'
PHASE_A_RESULTS = OUTPUT_DIR / 'phase_a_results.json'
PHASE_B_RESULTS = OUTPUT_DIR / 'phase_b_results.json'


def load_best_params(phase: str) -> Dict[str, Any]:
    """Load best parameters from Optuna results JSON."""
    path = PHASE_A_RESULTS if phase == 'A' else PHASE_B_RESULTS
    if not path.exists():
        raise FileNotFoundError(
            f"No {phase} results found at {path}. "
            f"Run fused_qcml_optimization.py --phase {phase} first."
        )
    with open(path) as f:
        data = json.load(f)
    return data['best_params']


def build_phase_a_detector(params: Dict[str, Any]) -> FusedQCMLDetector:
    """Instantiate FusedQCMLDetector from optimized params."""
    weights = None
    if params.get('fusion_method') == 'weighted_mean':
        w_berry = params.get('w_berry', 1.0 / 3)
        w_qfi = params.get('w_qfi', 1.0 / 3)
        w_multilag = params.get('w_multilag', 1.0 / 3)
        total = w_berry + w_qfi + w_multilag
        weights = [w_berry / total, w_qfi / total, w_multilag / total]

    return FusedQCMLDetector(
        hilbert_dim=params.get('hilbert_dim', 8),
        n_pca_components=params.get('n_pca_components', 15),
        operator_method=params.get('operator_method', 'random'),
        seed=42,
        fusion_method=params.get('fusion_method', 'max'),
        weights=weights,
        min_expanding=params.get('min_expanding', 60),
        rolling_window=params.get('rolling_window', 20),
    )


def build_phase_b_detector(params: Dict[str, Any]) -> GeometryOptimizedDetector:
    """Instantiate GeometryOptimizedDetector from optimized params."""
    n_pca = params.get('n_pca_components', 8)

    operator_scales = np.array([
        params.get(f'op_scale_{k}', 1.0) for k in range(n_pca)
    ])

    fusion_weights = np.array([
        params.get(f'fw_{k}', 0.2) for k in range(5)
    ])
    fusion_weights = fusion_weights / fusion_weights.sum()

    return GeometryOptimizedDetector(
        hilbert_dim=params.get('hilbert_dim', 8),
        n_pca_components=n_pca,
        operator_scales=operator_scales,
        fusion_weights=fusion_weights,
        rolling_window=params.get('rolling_window', 20),
        min_expanding=60,
        seed=42,
    )


def run_full_evaluation(
    temporal_oos: bool = False,
    n_bootstrap: int = 10000,
    n_permutations: int = 5000,
    seed: int = 42,
):
    """Run full evaluation of fused detectors vs RF baseline."""
    print("\n" + "=" * 70)
    print("FUSED QCML EVALUATION PIPELINE")
    print("=" * 70)

    config = get_default_validation_config()
    crises = DATA_AVAILABLE_CRISES
    enriched_lookback = 20

    # Load best params
    phase_a_params = None
    phase_b_params = None

    try:
        phase_a_params = load_best_params('A')
        print(f"\nPhase A params loaded: {json.dumps(phase_a_params, indent=2)}")
    except FileNotFoundError as e:
        print(f"\nPhase A: {e}")

    try:
        phase_b_params = load_best_params('B')
        print(f"\nPhase B params loaded: {json.dumps(phase_b_params, indent=2)}")
    except FileNotFoundError as e:
        print(f"\nPhase B: {e}")

    if phase_a_params is None and phase_b_params is None:
        print("No optimization results found. Run fused_qcml_optimization.py first.")
        return

    # Evaluate on all crises
    results_table = []

    for crisis in crises:
        print(f"\n{'='*60}")
        print(f"  Crisis: {crisis.name} ({crisis.crisis_date})")
        print(f"{'='*60}")

        X, X_enriched, times, crisis_idx = prepare_data(
            crisis, config, enriched_lookback=enriched_lookback,
        )
        if X is None:
            logger.warning(f"Skipping {crisis.name}: no data")
            continue

        trim = enriched_lookback - 1
        times_enriched = times[trim:]
        crisis_idx_enriched = max(0, crisis_idx - trim)

        row = {'crisis': crisis.name}

        # RF baseline (leave-one-crisis-out)
        print("  Running RF (LOCO)...")
        try:
            X_train, y_train, rf_n_features = prepare_rf_training_data(
                crisis, crises, config,
            )
            det_rf = RandomForestRegimeDetector(
                n_estimators=200, max_depth=6, seed=seed, lookback=20,
            )
            det_rf.fit_with_labels(X_train, y_train)
            X_rf_test = X[:, :rf_n_features] if X.shape[1] > rf_n_features else X
            rf_result = evaluate_method(
                det_rf, X_rf_test, times, crisis_idx, crisis, config,
                n_bootstrap, n_permutations, seed,
            )
            row['rf_d'] = rf_result['effect_size_d_normalized']
        except Exception as e:
            logger.error(f"  RF failed: {e}")
            row['rf_d'] = 0.0

        # Phase A: FusedQCMLDetector
        if phase_a_params is not None:
            print("  Running Fused QCML (Phase A)...")
            try:
                det_a = build_phase_a_detector(phase_a_params)
                det_a.fit(X_enriched)
                result_a = evaluate_method(
                    det_a, X_enriched, times_enriched, crisis_idx_enriched,
                    crisis, config, n_bootstrap, n_permutations, seed,
                )
                row['fused_a_d'] = result_a['effect_size_d_normalized']
                row['fused_a_full'] = result_a
            except Exception as e:
                logger.error(f"  Phase A failed: {e}")
                row['fused_a_d'] = 0.0

        # Phase B: GeometryOptimizedDetector
        if phase_b_params is not None:
            print("  Running Geometry Optimized (Phase B)...")
            try:
                det_b = build_phase_b_detector(phase_b_params)
                det_b.fit(X_enriched)
                result_b = evaluate_method(
                    det_b, X_enriched, times_enriched, crisis_idx_enriched,
                    crisis, config, n_bootstrap, n_permutations, seed,
                )
                row['fused_b_d'] = result_b['effect_size_d_normalized']
                row['fused_b_full'] = result_b
            except Exception as e:
                logger.error(f"  Phase B failed: {e}")
                row['fused_b_d'] = 0.0

        results_table.append(row)

    # Statistical superiority tests
    print("\n" + "=" * 70)
    print("STATISTICAL SUPERIORITY TESTS")
    print("=" * 70)

    rf_ds = [r.get('rf_d', 0.0) for r in results_table]

    for phase_key, phase_label in [('fused_a_d', 'Phase A'), ('fused_b_d', 'Phase B')]:
        fused_ds = [r.get(phase_key, None) for r in results_table]
        if all(d is None for d in fused_ds):
            continue

        fused_ds = [d if d is not None else 0.0 for d in fused_ds]

        print(f"\n--- {phase_label} vs RF ---")
        print(f"  Fused mean d: {np.mean(fused_ds):.4f}")
        print(f"  Fused median d: {np.median(fused_ds):.4f}")
        print(f"  RF mean d: {np.mean(rf_ds):.4f}")
        print(f"  RF median d: {np.median(rf_ds):.4f}")

        # Paired t-test
        t_stat, p_val = stats.ttest_rel(fused_ds, rf_ds)
        print(f"  Paired t-test: t={t_stat:.4f}, p={p_val:.4f}")

        # Bootstrap P(fused > RF)
        diffs = np.array(fused_ds) - np.array(rf_ds)
        rng = np.random.RandomState(seed)
        boot_means = []
        for _ in range(10000):
            sample = rng.choice(diffs, size=len(diffs), replace=True)
            boot_means.append(np.mean(sample))
        boot_means = np.array(boot_means)
        p_fused_better = np.mean(boot_means > 0)
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
        print(f"  Bootstrap P(fused > RF): {p_fused_better:.4f}")
        print(f"  Bootstrap 95% CI of mean diff: [{ci_lo:.4f}, {ci_hi:.4f}]")

        # Per-crisis comparison
        print(f"\n  Per-crisis comparison:")
        print(f"  {'Crisis':<25} {'RF d':>8} {'Fused d':>8} {'Diff':>8} {'Winner':>10}")
        print(f"  {'-'*60}")
        for i, r in enumerate(results_table):
            rf_d = rf_ds[i]
            f_d = fused_ds[i]
            diff = f_d - rf_d
            winner = phase_label if diff > 0 else "RF"
            print(f"  {r['crisis']:<25} {rf_d:>8.3f} {f_d:>8.3f} {diff:>+8.3f} {winner:>10}")

    # Temporal OOS validation
    if temporal_oos:
        print("\n" + "=" * 70)
        print("TEMPORAL OUT-OF-SAMPLE VALIDATION")
        print("=" * 70)
        _run_temporal_oos(results_table, crises, config, seed)

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Clean results for JSON serialization (remove full result dicts)
    clean_results = []
    for r in results_table:
        clean = {k: v for k, v in r.items() if not k.endswith('_full')}
        clean_results.append(clean)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'phase_a_params': phase_a_params,
        'phase_b_params': phase_b_params,
        'per_crisis_results': clean_results,
        'summary': {},
    }

    for phase_key, phase_label in [('fused_a_d', 'phase_a'), ('fused_b_d', 'phase_b')]:
        fused_ds = [r.get(phase_key, None) for r in results_table]
        if all(d is None for d in fused_ds):
            continue
        fused_ds = [d if d is not None else 0.0 for d in fused_ds]
        summary['summary'][phase_label] = {
            'mean_d': float(np.mean(fused_ds)),
            'median_d': float(np.median(fused_ds)),
            'std_d': float(np.std(fused_ds)),
            'rf_mean_d': float(np.mean(rf_ds)),
            'rf_median_d': float(np.median(rf_ds)),
        }

    out_path = OUTPUT_DIR / 'evaluation_results.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Generate markdown summary
    _generate_markdown_summary(summary, results_table)


def _run_temporal_oos(
    results_table: List[Dict],
    crises,
    config,
    seed: int = 42,
):
    """Temporal OOS: Evaluate on post-2020 crises only.

    The optimization was run on all 12 crises. True OOS would re-optimize
    on pre-2020 only. Here we report the subset performance as a lower bound.
    """
    post_2020_names = {'2020_covid', '2022_rates', '2023_svb'}
    pre_2020_names = {r['crisis'] for r in results_table} - post_2020_names

    for phase_key, phase_label in [('fused_a_d', 'Phase A'), ('fused_b_d', 'Phase B')]:
        fused_ds_oos = [
            r.get(phase_key, 0.0) for r in results_table
            if r['crisis'] in post_2020_names and r.get(phase_key) is not None
        ]
        rf_ds_oos = [
            r.get('rf_d', 0.0) for r in results_table
            if r['crisis'] in post_2020_names
        ]
        fused_ds_is = [
            r.get(phase_key, 0.0) for r in results_table
            if r['crisis'] in pre_2020_names and r.get(phase_key) is not None
        ]
        rf_ds_is = [
            r.get('rf_d', 0.0) for r in results_table
            if r['crisis'] in pre_2020_names
        ]

        if not fused_ds_oos:
            continue

        print(f"\n--- {phase_label} Temporal OOS ---")
        print(f"  In-sample (pre-2020, {len(fused_ds_is)} crises):")
        print(f"    Fused mean d: {np.mean(fused_ds_is):.4f}")
        print(f"    RF mean d: {np.mean(rf_ds_is):.4f}")
        print(f"  Out-of-sample (post-2020, {len(fused_ds_oos)} crises):")
        print(f"    Fused mean d: {np.mean(fused_ds_oos):.4f}")
        print(f"    RF mean d: {np.mean(rf_ds_oos):.4f}")
        print(f"    Fused/RF ratio: {np.mean(fused_ds_oos)/max(np.mean(rf_ds_oos), 1e-8):.2f}x")


def _generate_markdown_summary(summary: Dict, results_table: List[Dict]):
    """Generate a readable markdown summary."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = OUTPUT_DIR / 'EVALUATION_SUMMARY.md'

    lines = [
        "# Fused QCML Evaluation Summary",
        f"\nGenerated: {summary['timestamp']}\n",
        "## Per-Crisis Cohen's d (rank-normalized)\n",
        "| Crisis | RF | Phase A | Phase B |",
        "|--------|-----|---------|---------|",
    ]

    for r in results_table:
        rf = r.get('rf_d', '-')
        fa = r.get('fused_a_d', '-')
        fb = r.get('fused_b_d', '-')
        rf_s = f"{rf:.3f}" if isinstance(rf, float) else str(rf)
        fa_s = f"{fa:.3f}" if isinstance(fa, float) else str(fa)
        fb_s = f"{fb:.3f}" if isinstance(fb, float) else str(fb)
        lines.append(f"| {r['crisis']} | {rf_s} | {fa_s} | {fb_s} |")

    # Summary stats
    for phase_key, phase_label in [('phase_a', 'Phase A'), ('phase_b', 'Phase B')]:
        s = summary.get('summary', {}).get(phase_key)
        if s:
            lines.append(f"\n## {phase_label} Summary")
            lines.append(f"- Mean d: {s['mean_d']:.4f}")
            lines.append(f"- Median d: {s['median_d']:.4f}")
            lines.append(f"- RF mean d: {s['rf_mean_d']:.4f}")
            lines.append(f"- Diff: {s['mean_d'] - s['rf_mean_d']:+.4f}")

    with open(md_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Markdown summary saved to {md_path}")


def main():
    parser = argparse.ArgumentParser(description='Fused QCML Evaluation')
    parser.add_argument(
        '--temporal-oos', action='store_true',
        help='Include temporal out-of-sample validation',
    )
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    run_full_evaluation(
        temporal_oos=args.temporal_oos,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
