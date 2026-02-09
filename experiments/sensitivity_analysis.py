#!/usr/bin/env python3
"""Hyperparameter sensitivity analysis for top QCML regime detectors.

Varies hilbert_dim and n_pca_components for Berry Phase Rate, QFI Determinant,
and Multi-Lag Fidelity across representative crises.  Produces a heatmap figure
for the paper (Section: Discussion / Threshold Sensitivity).

Usage:
    python -m experiments.sensitivity_analysis [--quick]
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
load_dotenv()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.crisis_config import (
    DATA_AVAILABLE_CRISES,
    ValidationConfig,
    get_default_validation_config,
)
from experiments.regime_comparison import (
    evaluate_method,
    prepare_data,
    seed_everything,
)
from qcml.regime.classical_baselines import (
    BerryPhaseRateDetector,
    MultiLagFidelityDetector,
    QFIDeterminantDetector,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Representative crisis subset — covers different crisis types
SENSITIVITY_CRISES = [
    "2008_crisis",       # Systemic (slow build, large effect)
    "2018_volmageddon",  # Flash (sudden, Berry Phase Rate strongest)
    "2020_covid",        # Exogenous shock
    "2023_svb",          # Sector-specific (QFI Determinant strongest)
]

# Hyperparameter grid
HILBERT_DIMS = [4, 8, 16]
PCA_COMPONENTS = [5, 10, 15, 20]

# Methods to analyze
METHOD_FACTORIES = {
    "Berry Phase Rate": BerryPhaseRateDetector,
    "QFI Determinant": QFIDeterminantDetector,
    "Multi-Lag Fidelity": MultiLagFidelityDetector,
}


def run_sensitivity(
    quick: bool = False,
) -> Dict:
    """Run full sensitivity grid.

    Args:
        quick: If True, use reduced bootstrap (n=100) for speed.

    Returns:
        Nested dict: {method -> {(hilbert_dim, n_pca) -> {crisis -> d}}}
    """
    seed_everything(42)
    config = get_default_validation_config()
    n_bootstrap = 100 if quick else 1000
    n_permutations = 100 if quick else 500

    # Resolve crises
    all_crises = DATA_AVAILABLE_CRISES
    crisis_map = {c.name: c for c in all_crises}
    crises = [crisis_map[name] for name in SENSITIVITY_CRISES if name in crisis_map]

    if len(crises) < len(SENSITIVITY_CRISES):
        missing = set(SENSITIVITY_CRISES) - set(crisis_map.keys())
        logger.warning(f"Missing crises: {missing}")

    logger.info(f"Sensitivity analysis: {len(crises)} crises, "
                f"{len(HILBERT_DIMS)} hilbert_dims x {len(PCA_COMPONENTS)} PCA components, "
                f"{len(METHOD_FACTORIES)} methods")

    # Pre-load data for each crisis (data loading is independent of hyperparams)
    crisis_data = {}
    for crisis in crises:
        X, X_enriched, times, crisis_idx = prepare_data(crisis, config)
        if X is not None:
            crisis_data[crisis.name] = (X, times, crisis_idx, crisis)
            logger.info(f"  Loaded {crisis.name}: T={len(X)}")

    results = {}
    total_runs = len(METHOD_FACTORIES) * len(HILBERT_DIMS) * len(PCA_COMPONENTS) * len(crisis_data)
    run_count = 0

    for method_name, DetectorClass in METHOD_FACTORIES.items():
        results[method_name] = {}

        for hdim, n_pca in product(HILBERT_DIMS, PCA_COMPONENTS):
            key = (hdim, n_pca)
            d_values = {}

            for crisis_name, (X, times, crisis_idx, crisis) in crisis_data.items():
                run_count += 1
                logger.info(
                    f"  [{run_count}/{total_runs}] {method_name} "
                    f"hdim={hdim} pca={n_pca} crisis={crisis_name}"
                )

                try:
                    det = DetectorClass(
                        hilbert_dim=hdim,
                        n_pca_components=n_pca,
                        operator_method='random',
                        seed=42,
                    )
                    det.fit(X)
                    result = evaluate_method(
                        det, X, times, crisis_idx, crisis, config,
                        n_bootstrap=n_bootstrap,
                        n_permutations=n_permutations,
                        seed=42,
                    )
                    d_values[crisis_name] = result.get('effect_size_d', np.nan)
                except Exception as e:
                    logger.warning(f"    Failed: {e}")
                    d_values[crisis_name] = np.nan

            results[method_name][f"h{hdim}_p{n_pca}"] = {
                "hilbert_dim": hdim,
                "n_pca_components": n_pca,
                "per_crisis_d": d_values,
                "mean_d": float(np.nanmean(list(d_values.values()))),
                "std_d": float(np.nanstd(list(d_values.values()))),
            }

    return results


def create_sensitivity_figures(
    results: Dict,
    output_dir: str,
) -> None:
    """Generate publication-quality sensitivity heatmaps."""
    os.makedirs(output_dir, exist_ok=True)

    methods = list(results.keys())
    n_methods = len(methods)

    fig, axes = plt.subplots(1, n_methods, figsize=(5.5 * n_methods + 1.5, 4),
                             sharey=True, constrained_layout=True)
    if n_methods == 1:
        axes = [axes]

    for ax, method_name in zip(axes, methods):
        # Build heatmap matrix: rows = hilbert_dim, cols = n_pca
        data_matrix = np.full((len(HILBERT_DIMS), len(PCA_COMPONENTS)), np.nan)

        for i, hdim in enumerate(HILBERT_DIMS):
            for j, n_pca in enumerate(PCA_COMPONENTS):
                key = f"h{hdim}_p{n_pca}"
                if key in results[method_name]:
                    data_matrix[i, j] = results[method_name][key]["mean_d"]

        im = ax.imshow(
            data_matrix, cmap='RdYlGn', aspect='auto',
            vmin=0, vmax=2.0,
        )

        # Annotate cells
        for i in range(len(HILBERT_DIMS)):
            for j in range(len(PCA_COMPONENTS)):
                val = data_matrix[i, j]
                if not np.isnan(val):
                    color = 'white' if val < 0.4 or val > 1.6 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=10, color=color, fontweight='bold')

        ax.set_xticks(range(len(PCA_COMPONENTS)))
        ax.set_xticklabels(PCA_COMPONENTS)
        ax.set_yticks(range(len(HILBERT_DIMS)))
        ax.set_yticklabels(HILBERT_DIMS)
        ax.set_xlabel('PCA Components', fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel('Hilbert Dimension', fontsize=11)
        ax.set_title(method_name, fontsize=12, fontweight='bold')

    fig.colorbar(im, ax=axes, label="Mean Cohen's d", shrink=0.8, pad=0.02)
    fig.suptitle(
        'Hyperparameter Sensitivity: Mean Effect Size Across 4 Representative Crises',
        fontsize=13, fontweight='bold',
    )

    for ext in ['pdf', 'png']:
        path = os.path.join(output_dir, f'sensitivity_heatmap.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved {path}")
    plt.close(fig)

    # Per-crisis sensitivity for the best method (Berry Phase Rate)
    if "Berry Phase Rate" in results:
        _create_per_crisis_figure(results["Berry Phase Rate"], output_dir)


def _create_per_crisis_figure(method_results: Dict, output_dir: str) -> None:
    """Per-crisis heatmap for Berry Phase Rate across hilbert_dim x n_pca."""
    crisis_names = SENSITIVITY_CRISES

    fig, axes = plt.subplots(1, len(crisis_names),
                             figsize=(4.5 * len(crisis_names) + 1.5, 3.5),
                             sharey=True, constrained_layout=True)
    if len(crisis_names) == 1:
        axes = [axes]

    for ax, crisis_name in zip(axes, crisis_names):
        data_matrix = np.full((len(HILBERT_DIMS), len(PCA_COMPONENTS)), np.nan)
        for i, hdim in enumerate(HILBERT_DIMS):
            for j, n_pca in enumerate(PCA_COMPONENTS):
                key = f"h{hdim}_p{n_pca}"
                if key in method_results:
                    d_val = method_results[key]["per_crisis_d"].get(crisis_name, np.nan)
                    data_matrix[i, j] = d_val

        im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=3.0)
        for i in range(len(HILBERT_DIMS)):
            for j in range(len(PCA_COMPONENTS)):
                val = data_matrix[i, j]
                if not np.isnan(val):
                    color = 'white' if val < 0.4 or val > 2.5 else 'black'
                    ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                            fontsize=9, color=color)

        ax.set_xticks(range(len(PCA_COMPONENTS)))
        ax.set_xticklabels(PCA_COMPONENTS)
        ax.set_yticks(range(len(HILBERT_DIMS)))
        ax.set_yticklabels(HILBERT_DIMS)
        ax.set_xlabel('PCA', fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel('Hilbert Dim', fontsize=10)

        # Pretty crisis name
        pretty = crisis_name.replace('_', ' ').title()
        ax.set_title(pretty, fontsize=10, fontweight='bold')

    fig.colorbar(im, ax=axes, label="Cohen's d", shrink=0.8, pad=0.02)
    fig.suptitle(
        'Berry Phase Rate: Per-Crisis Sensitivity',
        fontsize=12, fontweight='bold',
    )

    for ext in ['pdf', 'png']:
        path = os.path.join(output_dir, f'sensitivity_per_crisis.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="QCML sensitivity analysis")
    parser.add_argument("--quick", action="store_true",
                        help="Use reduced bootstrap/permutation for speed")
    args = parser.parse_args()

    output_dir = "experiments/outputs/regime_detection/sensitivity"
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=== QCML Hyperparameter Sensitivity Analysis ===")
    t0 = time.time()

    results = run_sensitivity(quick=args.quick)

    elapsed = time.time() - t0
    logger.info(f"Sensitivity analysis completed in {elapsed:.1f}s")

    # Save raw results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"sensitivity_{timestamp}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved results to {results_path}")

    # Generate figures
    fig_dir = os.path.join(output_dir, "figures")
    create_sensitivity_figures(results, fig_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 70)
    for method_name, method_results in results.items():
        d_values = [(k, v["mean_d"]) for k, v in method_results.items()]
        d_values.sort(key=lambda x: -x[1])
        best_key, best_d = d_values[0]
        worst_key, worst_d = d_values[-1]
        all_d = [v["mean_d"] for v in method_results.values()]
        print(f"\n{method_name}:")
        print(f"  Best:  {best_key} (mean d = {best_d:.3f})")
        print(f"  Worst: {worst_key} (mean d = {worst_d:.3f})")
        print(f"  Range: {min(all_d):.3f} - {max(all_d):.3f}")
        print(f"  Coefficient of Variation: {np.std(all_d)/np.mean(all_d)*100:.1f}%")


if __name__ == "__main__":
    main()
