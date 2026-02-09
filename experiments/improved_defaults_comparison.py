#!/usr/bin/env python3
"""Improved Defaults Comparison for Top QCML Methods.

Re-runs the top 3 QCML methods (Berry Phase Rate, QFI Determinant,
Multi-Lag Fidelity) with improved hyperparameters from the extended
sensitivity analysis alongside the RF baseline on all 12 crises.

Does NOT modify defaults in classical_baselines.py --- passes improved
configs explicitly at construction time.

Reports the gap vs RF before and after improvement:
  - Original defaults: hilbert_dim=8, operator_method='random', rolling_window=20
  - Improved defaults: from extended_sensitivity_analysis.py RECOMMENDED_CONFIGS.md

Usage:
    python experiments/improved_defaults_comparison.py
    python experiments/improved_defaults_comparison.py --config path/to/stage2.json
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

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from experiments.crisis_config import (
    DATA_AVAILABLE_CRISES,
    get_default_validation_config,
)
from experiments.regime_comparison import (
    evaluate_method,
    prepare_data,
    prepare_rf_training_data,
    seed_everything,
)
from qcml_geometry import (
    BerryPhaseRateDetector,
    MultiLagFidelityDetector,
    QFIDeterminantDetector,
)
from experiments.baselines import RandomForestRegimeDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore", message="Metric tensor has negative eigenvalue")

OUTPUT_DIR = "experiments/outputs/regime_detection/improved_defaults"

# Original defaults from ValidationConfig / regime_comparison.py
ORIGINAL_DEFAULTS = {
    "hilbert_dim": 8,
    "n_pca_components": 15,
    "operator_method": "random",
    "rolling_window": 20,
}

# Improved defaults --- will be overridden by --config if provided.
# These are reasonable starting points based on fused optimization convergence.
IMPROVED_DEFAULTS = {
    "Berry Phase Rate": {
        "hilbert_dim": 12,
        "n_pca_components": 15,
        "operator_method": "pca_inspired",
        "rolling_window": 10,
    },
    "QFI Determinant": {
        "hilbert_dim": 12,
        "n_pca_components": 15,
        "operator_method": "pca_inspired",
        "rolling_window": 10,
    },
    "Multi-Lag Fidelity": {
        "hilbert_dim": 12,
        "n_pca_components": 15,
        "operator_method": "pca_inspired",
        "rolling_window": 10,
    },
}

METHOD_FACTORIES = {
    "Berry Phase Rate": BerryPhaseRateDetector,
    "QFI Determinant": QFIDeterminantDetector,
    "Multi-Lag Fidelity": MultiLagFidelityDetector,
}


def load_improved_configs(config_path: str) -> Dict[str, Dict]:
    """Load improved configs from Stage 2 JSON results.

    Returns {method_name -> {param -> value}} for the best config per method.
    """
    with open(config_path) as f:
        stage2 = json.load(f)

    improved = {}
    for method_name, configs in stage2.items():
        # Find the config with highest mean_d
        best = max(configs.values(), key=lambda c: c.get('mean_d', 0.0))
        improved[method_name] = {
            "hilbert_dim": best['hilbert_dim'],
            "n_pca_components": best['n_pca_components'],
            "operator_method": best['operator_method'],
            "rolling_window": best['rolling_window'],
        }
        logger.info(f"  Loaded improved config for {method_name}: {improved[method_name]}")

    return improved


def run_comparison(
    improved_configs: Dict[str, Dict],
    n_bootstrap: int = 1000,
    n_permutations: int = 500,
    seed: int = 42,
) -> Dict:
    """Run comparison: original defaults vs improved defaults vs RF.

    Returns {crisis_name -> {method_label -> result_dict}}.
    """
    seed_everything(seed)
    config = get_default_validation_config()
    crises = DATA_AVAILABLE_CRISES
    enriched_lookback = 20

    results = {}

    for crisis in crises:
        logger.info(f"\n=== {crisis.name} ({crisis.crisis_date}) ===")

        X, X_enriched, times, crisis_idx = prepare_data(
            crisis, config, enriched_lookback=enriched_lookback,
        )
        if X is None:
            logger.warning(f"Skipping {crisis.name}: no data")
            continue

        trim = enriched_lookback - 1
        times_enriched = times[trim:]
        crisis_idx_enriched = max(0, crisis_idx - trim)

        crisis_results = {}

        # Run each QCML method with original and improved defaults
        for method_name, DetectorClass in METHOD_FACTORIES.items():
            # Original defaults
            det_orig = DetectorClass(
                hilbert_dim=ORIGINAL_DEFAULTS['hilbert_dim'],
                n_pca_components=ORIGINAL_DEFAULTS['n_pca_components'],
                operator_method=ORIGINAL_DEFAULTS['operator_method'],
                rolling_window=ORIGINAL_DEFAULTS['rolling_window'],
                seed=seed,
            )
            det_orig.fit(X_enriched)
            res_orig = evaluate_method(
                det_orig, X_enriched, times_enriched, crisis_idx_enriched,
                crisis, config, n_bootstrap=n_bootstrap,
                n_permutations=n_permutations, seed=seed,
            )
            crisis_results[f"{method_name} (original)"] = res_orig

            # Improved defaults
            imp = improved_configs.get(method_name, ORIGINAL_DEFAULTS)
            det_imp = DetectorClass(
                hilbert_dim=imp['hilbert_dim'],
                n_pca_components=imp['n_pca_components'],
                operator_method=imp['operator_method'],
                rolling_window=imp['rolling_window'],
                seed=seed,
            )
            det_imp.fit(X_enriched)
            res_imp = evaluate_method(
                det_imp, X_enriched, times_enriched, crisis_idx_enriched,
                crisis, config, n_bootstrap=n_bootstrap,
                n_permutations=n_permutations, seed=seed,
            )
            crisis_results[f"{method_name} (improved)"] = res_imp

            logger.info(
                f"  {method_name}: original d={res_orig['effect_size_d']:.3f} "
                f"-> improved d={res_imp['effect_size_d']:.3f}"
            )

        # RF baseline (leave-one-crisis-out)
        try:
            X_train, y_train, rf_n_features = prepare_rf_training_data(
                crisis, crises, config,
            )
            det_rf = RandomForestRegimeDetector(
                n_estimators=200, max_depth=6, seed=seed, lookback=20,
            )
            det_rf.fit_with_labels(X_train, y_train)
            X_rf_test = X[:, :rf_n_features] if X.shape[1] > rf_n_features else X
            res_rf = evaluate_method(
                det_rf, X_rf_test, times, crisis_idx,
                crisis, config, n_bootstrap=n_bootstrap,
                n_permutations=n_permutations, seed=seed,
            )
            crisis_results["Random Forest"] = res_rf
            logger.info(f"  RF: d={res_rf['effect_size_d']:.3f}")
        except Exception as e:
            logger.error(f"  RF failed: {e}")
            from experiments.regime_comparison import _empty_result
            crisis_results["Random Forest"] = _empty_result("Random Forest")

        results[crisis.name] = crisis_results

    return results


def format_results(results: Dict) -> str:
    """Format comparison results as a readable table."""
    lines = [
        "=" * 100,
        "IMPROVED DEFAULTS COMPARISON",
        "=" * 100,
    ]

    # Collect per-method aggregates
    method_labels = set()
    for crisis_results in results.values():
        method_labels.update(crisis_results.keys())
    method_labels = sorted(method_labels)

    # Per-crisis table
    for crisis_name, crisis_results in results.items():
        lines.append(f"\n--- {crisis_name} ---")
        header = f"  {'Method':<35s} {'|d|':>6s}  {'|d|_norm':>8s}  {'p-val':>8s}  {'F1':>5s}"
        lines.append(header)
        lines.append("  " + "-" * 70)

        for label in method_labels:
            if label not in crisis_results:
                continue
            r = crisis_results[label]
            d_norm = r.get('effect_size_d_normalized', 0.0)
            lines.append(
                f"  {label:<35s} {r['effect_size_d']:>6.2f}  {d_norm:>8.2f}  "
                f"{r['p_value']:>8.4f}  {r['f1']:>5.2f}"
            )

    # Aggregate summary
    lines.append(f"\n{'=' * 100}")
    lines.append("AGGREGATE SUMMARY")
    lines.append(f"{'=' * 100}")

    header = f"  {'Method':<35s} {'Avg d':>7s}  {'Med d':>7s}  {'Avg p':>8s}  {'Wins':>5s}"
    lines.append(header)
    lines.append("  " + "-" * 70)

    for label in method_labels:
        ds = []
        ps = []
        wins = 0
        for crisis_results in results.values():
            if label in crisis_results:
                r = crisis_results[label]
                ds.append(r['effect_size_d'])
                ps.append(r['p_value'])
                if r['p_value'] < 0.05 and r['effect_size_d'] > 0.8:
                    wins += 1

        avg_d = np.mean(ds) if ds else 0.0
        med_d = np.median(ds) if ds else 0.0
        avg_p = np.mean(ps) if ps else 1.0

        lines.append(
            f"  {label:<35s} {avg_d:>7.3f}  {med_d:>7.3f}  {avg_p:>8.4f}  {wins:>5d}/{len(results)}"
        )

    # Improvement summary
    lines.append(f"\n{'=' * 100}")
    lines.append("IMPROVEMENT SUMMARY")
    lines.append(f"{'=' * 100}")

    for method_name in METHOD_FACTORIES:
        orig_label = f"{method_name} (original)"
        imp_label = f"{method_name} (improved)"
        rf_label = "Random Forest"

        orig_ds = []
        imp_ds = []
        rf_ds = []
        for crisis_results in results.values():
            if orig_label in crisis_results:
                orig_ds.append(crisis_results[orig_label]['effect_size_d'])
            if imp_label in crisis_results:
                imp_ds.append(crisis_results[imp_label]['effect_size_d'])
            if rf_label in crisis_results:
                rf_ds.append(crisis_results[rf_label]['effect_size_d'])

        orig_mean = np.mean(orig_ds) if orig_ds else 0.0
        imp_mean = np.mean(imp_ds) if imp_ds else 0.0
        rf_mean = np.mean(rf_ds) if rf_ds else 0.0
        delta = imp_mean - orig_mean
        gap_before = rf_mean - orig_mean
        gap_after = rf_mean - imp_mean

        lines.append(f"\n{method_name}:")
        lines.append(f"  Original mean d: {orig_mean:.3f}")
        lines.append(f"  Improved mean d: {imp_mean:.3f} ({delta:+.3f})")
        lines.append(f"  RF mean d:       {rf_mean:.3f}")
        lines.append(f"  Gap to RF: {gap_before:.3f} -> {gap_after:.3f}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare original vs improved QCML defaults against RF"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to Stage 2 JSON from extended_sensitivity_analysis.py",
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load improved configs
    if args.config:
        improved_configs = load_improved_configs(args.config)
    else:
        # Try to find the latest Stage 2 results
        stage2_dir = "experiments/outputs/regime_detection/extended_sensitivity"
        stage2_files = sorted(
            [f for f in os.listdir(stage2_dir) if f.startswith('stage2_') and f.endswith('.json')]
        ) if os.path.isdir(stage2_dir) else []

        if stage2_files:
            config_path = os.path.join(stage2_dir, stage2_files[-1])
            logger.info(f"Loading improved configs from {config_path}")
            improved_configs = load_improved_configs(config_path)
        else:
            logger.info("No Stage 2 results found, using hardcoded improved defaults")
            improved_configs = IMPROVED_DEFAULTS

    print("=" * 100)
    print("IMPROVED DEFAULTS COMPARISON")
    print("=" * 100)
    print(f"Improved configs:")
    for method, cfg in improved_configs.items():
        print(f"  {method}: {cfg}")
    print()

    t0 = time.time()
    results = run_comparison(
        improved_configs,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )
    elapsed = time.time() - t0

    # Format and print
    table = format_results(results)
    print(table)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(OUTPUT_DIR, f"comparison_{timestamp}.json")
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'original_defaults': ORIGINAL_DEFAULTS,
            'improved_configs': improved_configs,
            'results': results,
            'elapsed_seconds': elapsed,
        }, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    # Save formatted table
    table_path = os.path.join(OUTPUT_DIR, f"comparison_table_{timestamp}.txt")
    with open(table_path, 'w') as f:
        f.write(table)

    print(f"\nCompleted in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
