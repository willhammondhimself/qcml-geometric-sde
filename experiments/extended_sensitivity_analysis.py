#!/usr/bin/env python3
"""Extended hyperparameter sensitivity analysis for top QCML regime detectors.

Extends the original sensitivity_analysis.py (which varied only hilbert_dim
and n_pca_components) to also sweep operator_method and rolling_window ---
parameters that the fused Optuna optimization identified as important.

Two-stage design:
  Stage 1 (Broad Sweep): 4 representative crises, reduced stats (n_bootstrap=100).
      Grid: 4 hilbert_dims x 4 PCA x 2 operator_methods x 3 rolling_windows
           = 96 configs x 3 methods x 4 crises = 1152 evaluations.
      Runtime: ~20 minutes.

  Stage 2 (Validation): Top 5 configs per method on ALL 12 crises with full
      stats (n_bootstrap=1000).
      15 configs x 12 crises = 180 evaluations.
      Runtime: ~15 minutes.

Usage:
    python experiments/extended_sensitivity_analysis.py
    python experiments/extended_sensitivity_analysis.py --stage1-only
    python experiments/extended_sensitivity_analysis.py --quick
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
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

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
from qcml_geometry import (
    BerryPhaseRateDetector,
    MultiLagFidelityDetector,
    QFIDeterminantDetector,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

# Suppress noisy metric tensor warnings from qcml_geometry
import warnings
warnings.filterwarnings("ignore", message="Metric tensor has negative eigenvalue")

# Representative crisis subset for Stage 1
STAGE1_CRISES = [
    "2008_crisis",
    "2018_volmageddon",
    "2020_covid",
    "2023_svb",
]

# Extended hyperparameter grid
HILBERT_DIMS = [4, 8, 12, 16]
PCA_COMPONENTS = [8, 10, 15, 20]
OPERATOR_METHODS = ['random', 'pca_inspired']
ROLLING_WINDOWS = [10, 20, 30]

# Methods to analyze
METHOD_FACTORIES = {
    "Berry Phase Rate": BerryPhaseRateDetector,
    "QFI Determinant": QFIDeterminantDetector,
    "Multi-Lag Fidelity": MultiLagFidelityDetector,
}

OUTPUT_DIR = "experiments/outputs/regime_detection/extended_sensitivity"


def _config_key(hdim: int, n_pca: int, op_method: str, rw: int) -> str:
    """Generate a unique key for a hyperparameter configuration."""
    return f"h{hdim}_p{n_pca}_{op_method}_rw{rw}"


def _preload_crisis_data(
    crisis_names: List[str],
    config: ValidationConfig,
    enriched_lookback: int = 20,
) -> Dict:
    """Pre-load and cache crisis data for the specified crises."""
    all_crises = DATA_AVAILABLE_CRISES
    crisis_map = {c.name: c for c in all_crises}
    crisis_data = {}

    for name in crisis_names:
        if name not in crisis_map:
            logger.warning(f"Crisis {name} not in DATA_AVAILABLE_CRISES, skipping")
            continue
        crisis = crisis_map[name]
        X, X_enriched, times, crisis_idx = prepare_data(
            crisis, config, enriched_lookback=enriched_lookback,
        )
        if X is not None:
            trim = enriched_lookback - 1
            times_enriched = times[trim:]
            crisis_idx_enriched = max(0, crisis_idx - trim)
            crisis_data[name] = {
                'crisis': crisis,
                'X': X,
                'X_enriched': X_enriched,
                'times': times,
                'crisis_idx': crisis_idx,
                'times_enriched': times_enriched,
                'crisis_idx_enriched': crisis_idx_enriched,
            }
            logger.info(f"  Loaded {name}: T={len(X)}, enriched={X_enriched.shape}")

    return crisis_data


def _evaluate_config(
    DetectorClass,
    hdim: int,
    n_pca: int,
    op_method: str,
    rw: int,
    crisis_data: Dict,
    config: ValidationConfig,
    n_bootstrap: int = 100,
    n_permutations: int = 100,
    seed: int = 42,
) -> Dict[str, float]:
    """Evaluate one detector configuration across the provided crises.

    Returns dict mapping crisis_name -> Cohen's d.
    """
    d_values = {}
    for crisis_name, cd in crisis_data.items():
        try:
            det = DetectorClass(
                hilbert_dim=hdim,
                n_pca_components=n_pca,
                operator_method=op_method,
                rolling_window=rw,
                seed=seed,
            )
            det.fit(cd['X_enriched'])
            result = evaluate_method(
                det,
                cd['X_enriched'],
                cd['times_enriched'],
                cd['crisis_idx_enriched'],
                cd['crisis'],
                config,
                n_bootstrap=n_bootstrap,
                n_permutations=n_permutations,
                seed=seed,
            )
            d_values[crisis_name] = result.get('effect_size_d', np.nan)
        except Exception as e:
            import traceback
            logger.warning(f"    Failed {crisis_name}: {e}")
            logger.warning(traceback.format_exc())
            d_values[crisis_name] = np.nan

    return d_values


def run_stage1(quick: bool = False) -> Dict:
    """Stage 1: Broad sweep on 4 representative crises.

    Returns nested dict: {method -> {config_key -> {crisis -> d, mean_d, ...}}}
    """
    seed_everything(42)
    config = get_default_validation_config()
    n_bootstrap = 50 if quick else 100
    n_permutations = 50 if quick else 100

    logger.info("=== STAGE 1: Broad Sweep ===")
    crisis_data = _preload_crisis_data(STAGE1_CRISES, config)
    logger.info(f"Loaded {len(crisis_data)} crises for Stage 1")

    grid = list(product(HILBERT_DIMS, PCA_COMPONENTS, OPERATOR_METHODS, ROLLING_WINDOWS))
    total_runs = len(METHOD_FACTORIES) * len(grid) * len(crisis_data)
    run_count = 0

    results = {}
    for method_name, DetectorClass in METHOD_FACTORIES.items():
        results[method_name] = {}

        for hdim, n_pca, op_method, rw in grid:
            key = _config_key(hdim, n_pca, op_method, rw)
            run_count += len(crisis_data)
            logger.info(
                f"  [{run_count}/{total_runs}] {method_name} "
                f"hdim={hdim} pca={n_pca} op={op_method} rw={rw}"
            )

            d_values = _evaluate_config(
                DetectorClass, hdim, n_pca, op_method, rw,
                crisis_data, config,
                n_bootstrap=n_bootstrap,
                n_permutations=n_permutations,
            )

            d_list = [v for v in d_values.values() if not np.isnan(v)]
            results[method_name][key] = {
                "hilbert_dim": hdim,
                "n_pca_components": n_pca,
                "operator_method": op_method,
                "rolling_window": rw,
                "per_crisis_d": d_values,
                "mean_d": float(np.nanmean(d_list)) if d_list else 0.0,
                "std_d": float(np.nanstd(d_list)) if d_list else 0.0,
                "median_d": float(np.nanmedian(d_list)) if d_list else 0.0,
            }

    return results


def select_top_configs(stage1_results: Dict, top_n: int = 5) -> Dict[str, List[Dict]]:
    """Select top N configs per method from Stage 1 results.

    Returns {method_name -> list of config dicts sorted by mean_d descending}.
    """
    top_configs = {}
    for method_name, configs in stage1_results.items():
        sorted_configs = sorted(
            configs.values(),
            key=lambda c: c.get('mean_d', 0.0),
            reverse=True,
        )
        top_configs[method_name] = sorted_configs[:top_n]
        logger.info(
            f"  {method_name} top {top_n}: "
            + ", ".join(
                f"{_config_key(c['hilbert_dim'], c['n_pca_components'], c['operator_method'], c['rolling_window'])} "
                f"(mean_d={c['mean_d']:.3f})"
                for c in top_configs[method_name]
            )
        )
    return top_configs


def run_stage2(
    top_configs: Dict[str, List[Dict]],
    quick: bool = False,
) -> Dict:
    """Stage 2: Validate top configs on ALL 12 crises with full stats.

    Returns nested dict: {method -> {config_key -> {crisis -> d, mean_d, ...}}}
    """
    seed_everything(42)
    config = get_default_validation_config()
    n_bootstrap = 100 if quick else 1000
    n_permutations = 100 if quick else 500

    logger.info("=== STAGE 2: Full Validation ===")
    all_crisis_names = [c.name for c in DATA_AVAILABLE_CRISES]
    crisis_data = _preload_crisis_data(all_crisis_names, config)
    logger.info(f"Loaded {len(crisis_data)} crises for Stage 2")

    results = {}
    for method_name, configs in top_configs.items():
        DetectorClass = METHOD_FACTORIES[method_name]
        results[method_name] = {}

        for cfg in configs:
            hdim = cfg['hilbert_dim']
            n_pca = cfg['n_pca_components']
            op_method = cfg['operator_method']
            rw = cfg['rolling_window']
            key = _config_key(hdim, n_pca, op_method, rw)

            logger.info(
                f"  Stage 2: {method_name} {key} on {len(crisis_data)} crises"
            )

            d_values = _evaluate_config(
                DetectorClass, hdim, n_pca, op_method, rw,
                crisis_data, config,
                n_bootstrap=n_bootstrap,
                n_permutations=n_permutations,
            )

            d_list = [v for v in d_values.values() if not np.isnan(v)]
            results[method_name][key] = {
                "hilbert_dim": hdim,
                "n_pca_components": n_pca,
                "operator_method": op_method,
                "rolling_window": rw,
                "per_crisis_d": d_values,
                "mean_d": float(np.nanmean(d_list)) if d_list else 0.0,
                "std_d": float(np.nanstd(d_list)) if d_list else 0.0,
                "median_d": float(np.nanmedian(d_list)) if d_list else 0.0,
                "min_d": float(np.nanmin(d_list)) if d_list else 0.0,
                "cv": float(np.nanstd(d_list) / np.nanmean(d_list) * 100) if d_list and np.nanmean(d_list) > 0 else 0.0,
            }

    return results


def write_recommended_configs(
    stage2_results: Dict,
    stage1_results: Dict,
    output_dir: str,
) -> None:
    """Write RECOMMENDED_CONFIGS.md summarizing the best hyperparameters."""
    lines = [
        "# Extended Sensitivity Analysis: Recommended Configurations",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        "This analysis extends the original sensitivity sweep (hilbert_dim x n_pca)",
        "to also vary `operator_method` (random vs pca_inspired) and `rolling_window`",
        "(10, 20, 30). The fused Optuna optimization consistently found",
        "`pca_inspired` + `hilbert_dim=12` superior to the defaults.",
        "",
    ]

    for method_name in stage2_results:
        configs = stage2_results[method_name]
        sorted_configs = sorted(configs.values(), key=lambda c: c['mean_d'], reverse=True)
        best = sorted_configs[0]

        lines.append(f"## {method_name}")
        lines.append("")
        lines.append(f"**Recommended config:**")
        lines.append(f"- hilbert_dim: {best['hilbert_dim']}")
        lines.append(f"- n_pca_components: {best['n_pca_components']}")
        lines.append(f"- operator_method: {best['operator_method']}")
        lines.append(f"- rolling_window: {best['rolling_window']}")
        lines.append(f"- Mean Cohen's d (12 crises): {best['mean_d']:.3f}")
        lines.append(f"- Median Cohen's d: {best['median_d']:.3f}")
        lines.append(f"- Std: {best['std_d']:.3f}, CV: {best['cv']:.1f}%")
        lines.append("")

        # Compare to original default (h8, p15, random, rw20)
        default_key = "h8_p15_random_rw20"
        if default_key in stage1_results.get(method_name, {}):
            default = stage1_results[method_name][default_key]
            improvement = best['mean_d'] - default['mean_d']
            lines.append(f"**Improvement over default (h8/p15/random/rw20):**")
            lines.append(f"- Default mean d: {default['mean_d']:.3f}")
            lines.append(f"- Improved mean d: {best['mean_d']:.3f}")
            lines.append(f"- Delta: +{improvement:.3f}")
            lines.append("")

        lines.append("**Top 5 configs (Stage 2 validation):**")
        lines.append("")
        lines.append("| Config | Mean d | Median d | Std | CV% |")
        lines.append("|--------|--------|----------|-----|-----|")
        for cfg in sorted_configs[:5]:
            key = _config_key(cfg['hilbert_dim'], cfg['n_pca_components'],
                              cfg['operator_method'], cfg['rolling_window'])
            lines.append(
                f"| {key} | {cfg['mean_d']:.3f} | {cfg['median_d']:.3f} "
                f"| {cfg['std_d']:.3f} | {cfg['cv']:.1f} |"
            )
        lines.append("")

    # Operator method comparison
    lines.append("## Operator Method Comparison")
    lines.append("")
    lines.append("Aggregated across all hilbert_dim, n_pca, rolling_window:")
    lines.append("")
    for method_name in stage1_results:
        random_ds = []
        pca_ds = []
        for key, cfg in stage1_results[method_name].items():
            if cfg['operator_method'] == 'random':
                random_ds.append(cfg['mean_d'])
            else:
                pca_ds.append(cfg['mean_d'])

        lines.append(f"**{method_name}:**")
        if random_ds:
            lines.append(f"- random: mean_d={np.mean(random_ds):.3f} (n={len(random_ds)} configs)")
        if pca_ds:
            lines.append(f"- pca_inspired: mean_d={np.mean(pca_ds):.3f} (n={len(pca_ds)} configs)")
        if random_ds and pca_ds:
            delta = np.mean(pca_ds) - np.mean(random_ds)
            lines.append(f"- pca_inspired advantage: {delta:+.3f}")
        lines.append("")

    path = os.path.join(output_dir, "RECOMMENDED_CONFIGS.md")
    with open(path, 'w') as f:
        f.write("\n".join(lines))
    logger.info(f"Wrote {path}")


def create_extended_figures(
    stage1_results: Dict,
    stage2_results: Dict,
    output_dir: str,
) -> None:
    """Generate publication-quality figures for the extended sensitivity analysis."""
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Figure 1: Operator method comparison (bar chart)
    methods = list(stage1_results.keys())
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4),
                             sharey=True, constrained_layout=True)
    if len(methods) == 1:
        axes = [axes]

    for ax, method_name in zip(axes, methods):
        random_ds = []
        pca_ds = []
        for cfg in stage1_results[method_name].values():
            if cfg['operator_method'] == 'random':
                random_ds.append(cfg['mean_d'])
            else:
                pca_ds.append(cfg['mean_d'])

        positions = [0, 1]
        means = [np.mean(random_ds), np.mean(pca_ds)]
        stds = [np.std(random_ds), np.std(pca_ds)]
        bars = ax.bar(positions, means, yerr=stds, capsize=5,
                      color=['#95a5a6', '#3498db'], alpha=0.8)
        ax.set_xticks(positions)
        ax.set_xticklabels(['random', 'pca_inspired'], fontsize=10)
        ax.set_title(method_name, fontsize=11, fontweight='bold')
        if ax == axes[0]:
            ax.set_ylabel("Mean Cohen's d", fontsize=11)

    fig.suptitle("Operator Method Comparison (Stage 1: 4 Crises)",
                 fontsize=13, fontweight='bold')
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(fig_dir, f'operator_method_comparison.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Figure 2: Extended heatmap (hdim x rolling_window, best operator method)
    fig, axes = plt.subplots(1, len(methods), figsize=(5.5 * len(methods) + 1.5, 4),
                             sharey=True, constrained_layout=True)
    if len(methods) == 1:
        axes = [axes]

    for ax, method_name in zip(axes, methods):
        # Find best operator method for this method
        random_mean = np.mean([c['mean_d'] for c in stage1_results[method_name].values()
                               if c['operator_method'] == 'random'])
        pca_mean = np.mean([c['mean_d'] for c in stage1_results[method_name].values()
                            if c['operator_method'] == 'pca_inspired'])
        best_op = 'pca_inspired' if pca_mean >= random_mean else 'random'

        # Build heatmap: rows = hilbert_dim, cols = rolling_window
        # Aggregate over PCA components (take best PCA for each hdim x rw)
        data_matrix = np.full((len(HILBERT_DIMS), len(ROLLING_WINDOWS)), np.nan)
        for i, hdim in enumerate(HILBERT_DIMS):
            for j, rw in enumerate(ROLLING_WINDOWS):
                best_d = -np.inf
                for n_pca in PCA_COMPONENTS:
                    key = _config_key(hdim, n_pca, best_op, rw)
                    if key in stage1_results[method_name]:
                        d = stage1_results[method_name][key]['mean_d']
                        if d > best_d:
                            best_d = d
                if best_d > -np.inf:
                    data_matrix[i, j] = best_d

        im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto',
                       vmin=0, vmax=2.0)
        for i in range(len(HILBERT_DIMS)):
            for j in range(len(ROLLING_WINDOWS)):
                val = data_matrix[i, j]
                if not np.isnan(val):
                    color = 'white' if val < 0.4 or val > 1.6 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=10, color=color, fontweight='bold')

        ax.set_xticks(range(len(ROLLING_WINDOWS)))
        ax.set_xticklabels(ROLLING_WINDOWS)
        ax.set_yticks(range(len(HILBERT_DIMS)))
        ax.set_yticklabels(HILBERT_DIMS)
        ax.set_xlabel('Rolling Window', fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel('Hilbert Dimension', fontsize=11)
        ax.set_title(f'{method_name}\n(op={best_op})', fontsize=11, fontweight='bold')

    fig.colorbar(im, ax=axes, label="Mean Cohen's d", shrink=0.8, pad=0.02)
    fig.suptitle('Extended Sensitivity: Hilbert Dim x Rolling Window (Best PCA, Best Op)',
                 fontsize=12, fontweight='bold')
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(fig_dir, f'extended_sensitivity_heatmap.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Saved figures to {fig_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extended QCML hyperparameter sensitivity analysis"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Reduced bootstrap/permutation for speed")
    parser.add_argument("--stage1-only", action="store_true",
                        help="Run only Stage 1 (broad sweep)")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Number of top configs per method for Stage 2")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("=" * 70)
    logger.info("EXTENDED HYPERPARAMETER SENSITIVITY ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"Grid: {len(HILBERT_DIMS)} hdims x {len(PCA_COMPONENTS)} PCA "
                f"x {len(OPERATOR_METHODS)} ops x {len(ROLLING_WINDOWS)} rw "
                f"= {len(HILBERT_DIMS) * len(PCA_COMPONENTS) * len(OPERATOR_METHODS) * len(ROLLING_WINDOWS)} configs")

    t0 = time.time()

    # Stage 1
    stage1_results = run_stage1(quick=args.quick)
    stage1_elapsed = time.time() - t0
    logger.info(f"Stage 1 completed in {stage1_elapsed:.0f}s")

    # Save Stage 1 results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stage1_path = os.path.join(OUTPUT_DIR, f"stage1_{timestamp}.json")
    with open(stage1_path, 'w') as f:
        json.dump(stage1_results, f, indent=2, default=str)
    logger.info(f"Saved Stage 1 results to {stage1_path}")

    # Print Stage 1 summary
    print("\n" + "=" * 70)
    print("STAGE 1 SUMMARY (4 representative crises)")
    print("=" * 70)
    for method_name, configs in stage1_results.items():
        sorted_configs = sorted(configs.values(), key=lambda c: c['mean_d'], reverse=True)
        best = sorted_configs[0]
        worst = sorted_configs[-1]
        all_d = [c['mean_d'] for c in configs.values()]
        print(f"\n{method_name}:")
        print(f"  Best:  {_config_key(best['hilbert_dim'], best['n_pca_components'], best['operator_method'], best['rolling_window'])} "
              f"(mean d = {best['mean_d']:.3f})")
        print(f"  Worst: {_config_key(worst['hilbert_dim'], worst['n_pca_components'], worst['operator_method'], worst['rolling_window'])} "
              f"(mean d = {worst['mean_d']:.3f})")
        print(f"  Range: {min(all_d):.3f} - {max(all_d):.3f}")

        # Operator method summary
        random_d = np.mean([c['mean_d'] for c in configs.values() if c['operator_method'] == 'random'])
        pca_d = np.mean([c['mean_d'] for c in configs.values() if c['operator_method'] == 'pca_inspired'])
        print(f"  Operator: random={random_d:.3f}, pca_inspired={pca_d:.3f} (delta={pca_d - random_d:+.3f})")

    if args.stage1_only:
        create_extended_figures(stage1_results, {}, OUTPUT_DIR)
        print(f"\nStage 1 results saved to {stage1_path}")
        return

    # Stage 2
    print("\n" + "=" * 70)
    print("SELECTING TOP CONFIGS FOR STAGE 2")
    print("=" * 70)
    top_configs = select_top_configs(stage1_results, top_n=args.top_n)

    t1 = time.time()
    stage2_results = run_stage2(top_configs, quick=args.quick)
    stage2_elapsed = time.time() - t1
    logger.info(f"Stage 2 completed in {stage2_elapsed:.0f}s")

    # Save Stage 2 results
    stage2_path = os.path.join(OUTPUT_DIR, f"stage2_{timestamp}.json")
    with open(stage2_path, 'w') as f:
        json.dump(stage2_results, f, indent=2, default=str)
    logger.info(f"Saved Stage 2 results to {stage2_path}")

    # Print Stage 2 summary
    print("\n" + "=" * 70)
    print("STAGE 2 SUMMARY (ALL 12 crises, full stats)")
    print("=" * 70)
    for method_name, configs in stage2_results.items():
        sorted_configs = sorted(configs.values(), key=lambda c: c['mean_d'], reverse=True)
        best = sorted_configs[0]
        print(f"\n{method_name}:")
        print(f"  Best:  {_config_key(best['hilbert_dim'], best['n_pca_components'], best['operator_method'], best['rolling_window'])}")
        print(f"  Mean d: {best['mean_d']:.3f}, Median d: {best['median_d']:.3f}")
        print(f"  Std: {best['std_d']:.3f}, CV: {best['cv']:.1f}%")

    # Write recommended configs
    write_recommended_configs(stage2_results, stage1_results, OUTPUT_DIR)

    # Generate figures
    create_extended_figures(stage1_results, stage2_results, OUTPUT_DIR)

    total_elapsed = time.time() - t0
    print(f"\nTotal elapsed: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
