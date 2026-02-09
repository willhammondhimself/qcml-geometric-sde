#!/usr/bin/env python3
"""Causal-optimized evaluation: 5 conditions x 3 methods x 12 crises.

Evaluates QCML detectors under causal constraints with multiple configurations:
  1. Original (non-causal) improved defaults — current baseline
  2. Causal-optimized configs — from causal_sensitivity_analysis.py
  3. Expanding window (interval=20) — with causal-optimal configs
  4. Expanding window (interval=30) — with causal-optimal configs
  5. Random Forest — supervised baseline (leave-one-crisis-out)

Statistical tests: Wilcoxon signed-rank (paired), Friedman ranking, per-crisis.

Usage:
    python experiments/causal_optimized_evaluation.py
    python experiments/causal_optimized_evaluation.py --causal-config path/to/stage2.json
    python experiments/causal_optimized_evaluation.py --quick
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
from scipy import stats

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

OUTPUT_DIR = "experiments/outputs/regime_detection/causal_optimized"

METHOD_FACTORIES = {
    "Berry Phase Rate": BerryPhaseRateDetector,
    "QFI Determinant": QFIDeterminantDetector,
    "Multi-Lag Fidelity": MultiLagFidelityDetector,
}

# Non-causal "improved defaults" from extended_sensitivity_analysis
NONCAUSAL_CONFIGS = {
    "Berry Phase Rate": {
        "hilbert_dim": 8,
        "n_pca_components": 10,
        "operator_method": "pca_inspired",
        "rolling_window": 10,
    },
    "QFI Determinant": {
        "hilbert_dim": 4,
        "n_pca_components": 15,
        "operator_method": "random",
        "rolling_window": 30,
    },
    "Multi-Lag Fidelity": {
        "hilbert_dim": 8,
        "n_pca_components": 15,
        "operator_method": "pca_inspired",
        "rolling_window": 20,
    },
}


def load_causal_configs(json_path: Optional[str]) -> Dict[str, Dict]:
    """Load causal-optimal configs from Stage 2 JSON results (grid search).

    If json_path is None, falls back to non-causal configs (for testing).
    """
    if json_path is None:
        logger.warning("No causal config file specified, using non-causal defaults as fallback")
        return NONCAUSAL_CONFIGS

    with open(json_path) as f:
        stage2 = json.load(f)

    causal_configs = {}
    for method_name in METHOD_FACTORIES:
        if method_name not in stage2:
            logger.warning(f"Method {method_name} not found in causal config, using non-causal")
            causal_configs[method_name] = NONCAUSAL_CONFIGS[method_name]
            continue

        configs = stage2[method_name]
        best_key = max(configs, key=lambda k: configs[k].get('mean_d', 0.0))
        best = configs[best_key]
        causal_configs[method_name] = {
            "hilbert_dim": best['hilbert_dim'],
            "n_pca_components": best['n_pca_components'],
            "operator_method": best['operator_method'],
            "rolling_window": best['rolling_window'],
        }
        logger.info(f"  {method_name} causal-optimal: {best_key} (mean_d={best['mean_d']:.3f})")

    return causal_configs


def load_optuna_configs(optuna_dir: str, phase: str = 'A') -> Dict[str, Dict]:
    """Load Optuna-optimized configs from Phase A or B results.

    Args:
        optuna_dir: Directory containing Optuna Phase A/B JSON results.
        phase: 'A' or 'B' (default: 'A').

    Returns:
        Dict mapping method_name -> best_params.
    """
    optuna_dir_path = Path(optuna_dir)
    if not optuna_dir_path.exists():
        raise FileNotFoundError(f"Optuna results directory not found: {optuna_dir}")

    optuna_configs = {}

    # For Phase B, we need to load Phase A params first as base
    if phase == 'B':
        logger.info(f"Loading Phase A base params for Phase B...")
        phase_a_configs = load_optuna_configs(optuna_dir, phase='A')

    for method_name in METHOD_FACTORIES:
        method_file = optuna_dir_path / f"{method_name.replace(' ', '_').lower()}_phase_{phase.lower()}_results.json"

        if not method_file.exists():
            logger.warning(f"Optuna Phase {phase} results not found for {method_name}, using non-causal")
            optuna_configs[method_name] = NONCAUSAL_CONFIGS[method_name]
            continue

        with open(method_file) as f:
            optuna_data = json.load(f)

        best_params = optuna_data['best_params']

        if phase == 'A':
            # Phase A: Extract all base params
            optuna_configs[method_name] = {
                "hilbert_dim": best_params['hilbert_dim'],
                "n_pca_components": best_params['n_pca_components'],
                "operator_method": best_params['operator_method'],
                "rolling_window": best_params['rolling_window'],
            }

            # Add min_expanding if present
            if 'min_expanding' in best_params:
                optuna_configs[method_name]['min_expanding'] = best_params['min_expanding']
        else:
            # Phase B: Start with Phase A base params, add expanding_refit_interval
            optuna_configs[method_name] = phase_a_configs[method_name].copy()

            # Add expanding_refit_interval from Phase B
            if 'expanding_refit_interval' in best_params:
                optuna_configs[method_name]['expanding_refit_interval'] = best_params['expanding_refit_interval']

        logger.info(
            f"  {method_name} Optuna Phase {phase} (median_d={optuna_data['best_value']:.3f}): "
            f"{optuna_configs[method_name]}"
        )

    return optuna_configs


def evaluate_condition(
    condition_name: str,
    method_name: str,
    DetectorClass,
    det_kwargs: Dict,
    crisis_data: Dict,
    config: ValidationConfig,
    n_bootstrap: int,
    n_permutations: int,
) -> Dict[str, float]:
    """Evaluate one condition on all crises, returning {crisis_name: d}."""
    d_values = {}
    for crisis_name, cd in crisis_data.items():
        try:
            det = DetectorClass(**det_kwargs, seed=42)
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
                seed=42,
            )
            d_values[crisis_name] = result.get('effect_size_d', np.nan)
        except Exception as e:
            logger.warning(f"  {condition_name}/{method_name}/{crisis_name}: {e}")
            d_values[crisis_name] = np.nan

    return d_values


def evaluate_rf(
    crisis_data: Dict,
    config: ValidationConfig,
    n_bootstrap: int,
    n_permutations: int,
) -> Dict[str, float]:
    """Evaluate Random Forest (leave-one-crisis-out) on all crises."""
    crises = DATA_AVAILABLE_CRISES
    d_values = {}

    for crisis_name, cd in crisis_data.items():
        try:
            rf_X, rf_y, rf_n_features = prepare_rf_training_data(
                cd['crisis'], crises, config,
            )
            rf = RandomForestRegimeDetector(
                n_estimators=200, max_depth=6, seed=42, lookback=20,
            )
            rf.fit_with_labels(rf_X, rf_y)

            X_test = cd['X']
            if X_test.shape[1] > rf_n_features:
                X_test = X_test[:, :rf_n_features]

            res = evaluate_method(
                rf, X_test, cd['times'], cd['crisis_idx'],
                cd['crisis'], config,
                n_bootstrap=n_bootstrap, n_permutations=n_permutations, seed=42,
            )
            d_values[crisis_name] = res.get('effect_size_d', np.nan)
        except Exception as e:
            logger.warning(f"  RF/{crisis_name}: {e}")
            d_values[crisis_name] = np.nan

    return d_values


def run_evaluation(
    causal_configs: Dict[str, Dict],
    quick: bool = False,
    optuna_configs_a: Optional[Dict[str, Dict]] = None,
    optuna_configs_b: Optional[Dict[str, Dict]] = None,
) -> Dict:
    """Run full evaluation with Optuna-optimized configs.

    Conditions:
      1. Original (non-causal) improved defaults — current baseline
      2. Causal-optimized configs (grid search or Optuna Phase A)
      2b. Optuna Phase B (if available) — expanding window optimized
      3. Expanding window (interval=20) — with causal-optimal configs
      4. Expanding window (interval=30) — with causal-optimal configs
      5. Random Forest — supervised baseline (leave-one-crisis-out)
    """
    seed_everything(42)
    config = get_default_validation_config()
    n_bootstrap = 100 if quick else 1000
    n_permutations = 100 if quick else 500

    # Load all crisis data
    logger.info("Loading crisis data...")
    crisis_data = {}
    for crisis in DATA_AVAILABLE_CRISES:
        X, X_enriched, times, crisis_idx = prepare_data(
            crisis, config, enriched_lookback=20,
        )
        if X is not None:
            trim = 19
            times_enriched = times[trim:]
            crisis_idx_enriched = max(0, crisis_idx - trim)
            crisis_data[crisis.name] = {
                'crisis': crisis,
                'X': X,
                'X_enriched': X_enriched,
                'times': times,
                'crisis_idx': crisis_idx,
                'times_enriched': times_enriched,
                'crisis_idx_enriched': crisis_idx_enriched,
            }
    logger.info(f"Loaded {len(crisis_data)} crises")

    results = {}

    for method_name, DetectorClass in METHOD_FACTORIES.items():
        logger.info(f"\n=== {method_name} ===")
        method_results = {}

        # Condition 1: Original non-causal improved defaults (causal fit)
        nc_cfg = NONCAUSAL_CONFIGS[method_name].copy()
        nc_cfg_causal = {**nc_cfg}
        # Add causal_fit_length for each crisis
        d_orig = {}
        for crisis_name, cd in crisis_data.items():
            try:
                det = DetectorClass(
                    **nc_cfg, seed=42,
                    causal_fit_length=cd['crisis_idx_enriched'],
                )
                det.fit(cd['X_enriched'])
                res = evaluate_method(
                    det, cd['X_enriched'], cd['times_enriched'],
                    cd['crisis_idx_enriched'], cd['crisis'], config,
                    n_bootstrap=n_bootstrap, n_permutations=n_permutations, seed=42,
                )
                d_orig[crisis_name] = res.get('effect_size_d', np.nan)
            except Exception as e:
                logger.warning(f"  Orig/{crisis_name}: {e}")
                d_orig[crisis_name] = np.nan

        method_results["1_original_causal"] = d_orig
        logger.info(f"  Condition 1 (original, causal): mean d = {np.nanmean(list(d_orig.values())):.3f}")

        # Condition 2: Causal-optimized configs
        c_cfg = causal_configs[method_name].copy()
        d_causal_opt = {}
        for crisis_name, cd in crisis_data.items():
            try:
                det = DetectorClass(
                    **c_cfg, seed=42,
                    causal_fit_length=cd['crisis_idx_enriched'],
                )
                det.fit(cd['X_enriched'])
                res = evaluate_method(
                    det, cd['X_enriched'], cd['times_enriched'],
                    cd['crisis_idx_enriched'], cd['crisis'], config,
                    n_bootstrap=n_bootstrap, n_permutations=n_permutations, seed=42,
                )
                d_causal_opt[crisis_name] = res.get('effect_size_d', np.nan)
            except Exception as e:
                logger.warning(f"  CausalOpt/{crisis_name}: {e}")
                d_causal_opt[crisis_name] = np.nan

        method_results["2_causal_optimized"] = d_causal_opt
        logger.info(f"  Condition 2 (causal-opt): mean d = {np.nanmean(list(d_causal_opt.values())):.3f}")

        # Condition 2b: Optuna Phase B (if available)
        if optuna_configs_b is not None and method_name in optuna_configs_b:
            pb_cfg = optuna_configs_b[method_name].copy()
            d_optuna_b = {}
            for crisis_name, cd in crisis_data.items():
                try:
                    det = DetectorClass(
                        **pb_cfg, seed=42,
                        # Phase B already has expanding_refit_interval optimized
                        # causal_fit_length not needed if expanding window is used
                    )
                    det.fit(cd['X_enriched'])
                    res = evaluate_method(
                        det, cd['X_enriched'], cd['times_enriched'],
                        cd['crisis_idx_enriched'], cd['crisis'], config,
                        n_bootstrap=n_bootstrap, n_permutations=n_permutations, seed=42,
                    )
                    d_optuna_b[crisis_name] = res.get('effect_size_d', np.nan)
                except Exception as e:
                    logger.warning(f"  OptunaB/{crisis_name}: {e}")
                    d_optuna_b[crisis_name] = np.nan

            method_results["2b_optuna_phase_b"] = d_optuna_b
            logger.info(f"  Condition 2b (Optuna Phase B): mean d = {np.nanmean(list(d_optuna_b.values())):.3f}")

        # Condition 3: Expanding window (interval=20) with causal-optimal configs
        d_expand_20 = {}
        for crisis_name, cd in crisis_data.items():
            try:
                det = DetectorClass(
                    **c_cfg, seed=42,
                    expanding_refit_interval=20,
                )
                det.fit(cd['X_enriched'])
                res = evaluate_method(
                    det, cd['X_enriched'], cd['times_enriched'],
                    cd['crisis_idx_enriched'], cd['crisis'], config,
                    n_bootstrap=n_bootstrap, n_permutations=n_permutations, seed=42,
                )
                d_expand_20[crisis_name] = res.get('effect_size_d', np.nan)
            except Exception as e:
                logger.warning(f"  Expand20/{crisis_name}: {e}")
                d_expand_20[crisis_name] = np.nan

        method_results["3_expanding_20"] = d_expand_20
        logger.info(f"  Condition 3 (expand-20): mean d = {np.nanmean(list(d_expand_20.values())):.3f}")

        # Condition 4: Expanding window (interval=30) with causal-optimal configs
        d_expand_30 = {}
        for crisis_name, cd in crisis_data.items():
            try:
                det = DetectorClass(
                    **c_cfg, seed=42,
                    expanding_refit_interval=30,
                )
                det.fit(cd['X_enriched'])
                res = evaluate_method(
                    det, cd['X_enriched'], cd['times_enriched'],
                    cd['crisis_idx_enriched'], cd['crisis'], config,
                    n_bootstrap=n_bootstrap, n_permutations=n_permutations, seed=42,
                )
                d_expand_30[crisis_name] = res.get('effect_size_d', np.nan)
            except Exception as e:
                logger.warning(f"  Expand30/{crisis_name}: {e}")
                d_expand_30[crisis_name] = np.nan

        method_results["4_expanding_30"] = d_expand_30
        logger.info(f"  Condition 4 (expand-30): mean d = {np.nanmean(list(d_expand_30.values())):.3f}")

        results[method_name] = method_results

    # Condition 5: Random Forest baseline
    logger.info("\n=== Random Forest ===")
    rf_d = evaluate_rf(crisis_data, config, n_bootstrap, n_permutations)
    results["Random Forest"] = {"5_rf_baseline": rf_d}
    logger.info(f"  RF: mean d = {np.nanmean(list(rf_d.values())):.3f}")

    return results


def compute_statistical_tests(results: Dict) -> Dict:
    """Compute Wilcoxon signed-rank and Friedman tests."""
    test_results = {}

    # Get crisis names that have data for all conditions
    crisis_names = list(next(iter(
        next(iter(results.values())).values()
    )).keys())

    # RF baseline values
    rf_d = results["Random Forest"]["5_rf_baseline"]

    for method_name in METHOD_FACTORIES:
        method_tests = {}
        for cond_name, cond_d in results[method_name].items():
            # Paired Wilcoxon vs RF
            paired_crises = [c for c in crisis_names
                             if c in cond_d and c in rf_d
                             and not np.isnan(cond_d.get(c, np.nan))
                             and not np.isnan(rf_d.get(c, np.nan))]

            if len(paired_crises) >= 5:
                x = [cond_d[c] for c in paired_crises]
                y = [rf_d[c] for c in paired_crises]
                try:
                    stat, p_val = stats.wilcoxon(x, y, alternative='greater')
                    method_tests[cond_name] = {
                        "wilcoxon_stat": float(stat),
                        "wilcoxon_p": float(p_val),
                        "n_paired": len(paired_crises),
                        "mean_d": float(np.mean(x)),
                        "mean_rf_d": float(np.mean(y)),
                        "wins": sum(1 for a, b in zip(x, y) if a > b),
                        "losses": sum(1 for a, b in zip(x, y) if a < b),
                        "ties": sum(1 for a, b in zip(x, y) if a == b),
                    }
                except Exception as e:
                    method_tests[cond_name] = {"error": str(e)}
            else:
                method_tests[cond_name] = {"error": f"Too few paired crises ({len(paired_crises)})"}

        test_results[method_name] = method_tests

    # Friedman test across all conditions
    condition_names = ["1_original_causal", "2_causal_optimized",
                       "3_expanding_20", "4_expanding_30"]
    for method_name in METHOD_FACTORIES:
        arrays = []
        for cond_name in condition_names:
            cond_d = results[method_name].get(cond_name, {})
            arr = [cond_d.get(c, np.nan) for c in crisis_names]
            arrays.append(arr)

        # Add RF
        arrays.append([rf_d.get(c, np.nan) for c in crisis_names])

        # Filter to crises with all valid
        valid_mask = np.all(~np.isnan(arrays), axis=0)
        if np.sum(valid_mask) >= 5:
            valid_arrays = [np.array(a)[valid_mask] for a in arrays]
            try:
                chi2, p_val = stats.friedmanchisquare(*valid_arrays)
                test_results[method_name]["friedman"] = {
                    "chi2": float(chi2),
                    "p_value": float(p_val),
                    "n_crises": int(np.sum(valid_mask)),
                    "mean_ranks": [
                        float(np.mean(stats.rankdata(
                            [-valid_arrays[i][j] for i in range(len(valid_arrays))]
                        )[i]))
                        for i in range(len(valid_arrays))
                        for j in [0]  # dummy — need actual rank computation
                    ],
                }
                # Proper rank computation
                n_crises = int(np.sum(valid_mask))
                n_conds = len(valid_arrays)
                ranks = np.zeros((n_crises, n_conds))
                for j in range(n_crises):
                    vals = [valid_arrays[i][j] for i in range(n_conds)]
                    ranks[j] = stats.rankdata([-v for v in vals])
                mean_ranks = ranks.mean(axis=0).tolist()
                cond_labels = condition_names + ["5_rf_baseline"]
                test_results[method_name]["friedman"]["mean_ranks"] = {
                    label: float(rank)
                    for label, rank in zip(cond_labels, mean_ranks)
                }
            except Exception as e:
                test_results[method_name]["friedman"] = {"error": str(e)}

    return test_results


def format_summary(results: Dict, test_results: Dict) -> str:
    """Format results into a human-readable summary."""
    lines = [
        "=" * 90,
        "CAUSAL-OPTIMIZED EVALUATION: 5 CONDITIONS x 3 METHODS x 12 CRISES",
        "=" * 90,
        "",
    ]

    crisis_names = list(next(iter(
        next(iter(results.values())).values()
    )).keys())

    rf_d = results["Random Forest"]["5_rf_baseline"]
    rf_mean = np.nanmean(list(rf_d.values()))

    for method_name in METHOD_FACTORIES:
        lines.append(f"\n{'='*70}")
        lines.append(f"  {method_name}")
        lines.append(f"{'='*70}")

        header = f"  {'Condition':<30s} {'Mean d':>7s} {'Med d':>7s} {'vs RF':>7s} {'W/L':>6s} {'Wilcoxon p':>11s}"
        lines.append(header)
        lines.append("  " + "-" * 75)

        for cond_name, cond_d in results[method_name].items():
            d_list = [v for v in cond_d.values() if not np.isnan(v)]
            mean_d = np.mean(d_list) if d_list else 0.0
            med_d = np.median(d_list) if d_list else 0.0
            delta = mean_d - rf_mean

            tests = test_results.get(method_name, {}).get(cond_name, {})
            w = tests.get('wins', '?')
            l = tests.get('losses', '?')
            p = tests.get('wilcoxon_p', None)
            p_str = f"{p:.4f}" if p is not None else "N/A"

            lines.append(
                f"  {cond_name:<30s} {mean_d:>7.3f} {med_d:>7.3f} "
                f"{delta:>+7.3f} {w}/{l}   {p_str:>11s}"
            )

        lines.append(f"\n  RF baseline: mean d = {rf_mean:.3f}")

        # Friedman
        friedman = test_results.get(method_name, {}).get("friedman", {})
        if "chi2" in friedman:
            lines.append(f"\n  Friedman chi2 = {friedman['chi2']:.2f}, p = {friedman['p_value']:.4f}")
            ranks = friedman.get("mean_ranks", {})
            if isinstance(ranks, dict):
                lines.append("  Mean ranks (lower = better):")
                for label, rank in sorted(ranks.items(), key=lambda x: x[1]):
                    lines.append(f"    {label:<30s} {rank:.2f}")

    # Per-crisis breakdown for best QCML condition vs RF
    lines.append(f"\n\n{'='*90}")
    lines.append("PER-CRISIS BREAKDOWN: Best QCML Condition vs RF")
    lines.append(f"{'='*90}")

    header = f"  {'Crisis':<20s}"
    for mn in METHOD_FACTORIES:
        header += f" {mn[:12]:>12s}"
    header += f" {'RF':>8s}"
    lines.append(header)
    lines.append("  " + "-" * 80)

    for cn in crisis_names:
        row = f"  {cn:<20s}"
        for mn in METHOD_FACTORIES:
            # Find best condition for this method
            best_d = -np.inf
            for cond_d in results[mn].values():
                d = cond_d.get(cn, np.nan)
                if not np.isnan(d) and d > best_d:
                    best_d = d
            row += f" {best_d:>12.3f}" if best_d > -np.inf else f" {'N/A':>12s}"
        rf_val = rf_d.get(cn, np.nan)
        row += f" {rf_val:>8.3f}" if not np.isnan(rf_val) else f" {'N/A':>8s}"
        lines.append(row)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Causal-optimized evaluation of QCML regime detectors"
    )
    parser.add_argument("--causal-config", type=str, default=None,
                        help="Path to causal_stage2_*.json from causal_sensitivity_analysis.py (grid search)")
    parser.add_argument("--optuna-dir", type=str, default=None,
                        help="Directory containing Optuna Phase A results (alternative to --causal-config)")
    parser.add_argument("--optuna-dir-phase-b", type=str, default=None,
                        help="Directory containing Optuna Phase B results (optional, adds Condition 2b)")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced bootstrap/permutation for speed")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("=" * 70)
    logger.info("CAUSAL-OPTIMIZED EVALUATION")
    logger.info("=" * 70)

    # Load configs (Optuna takes precedence over grid search)
    optuna_configs_a = None
    optuna_configs_b = None

    if args.optuna_dir:
        logger.info(f"\nLoading Optuna Phase A configs from: {args.optuna_dir}")
        optuna_configs_a = load_optuna_configs(args.optuna_dir, phase='A')
        causal_configs = optuna_configs_a  # Use Optuna Phase A as Condition 2
        for mn, cfg in causal_configs.items():
            logger.info(f"  {mn}: {cfg}")

        if args.optuna_dir_phase_b:
            logger.info(f"\nLoading Optuna Phase B configs from: {args.optuna_dir_phase_b}")
            optuna_configs_b = load_optuna_configs(args.optuna_dir_phase_b, phase='B')
            for mn, cfg in optuna_configs_b.items():
                logger.info(f"  {mn}: {cfg}")
    else:
        logger.info("\nLoading grid search causal configs...")
        causal_configs = load_causal_configs(args.causal_config)
        for mn, cfg in causal_configs.items():
            logger.info(f"  {mn}: {cfg}")

    t0 = time.time()

    # Run evaluation
    results = run_evaluation(
        causal_configs,
        quick=args.quick,
        optuna_configs_a=optuna_configs_a,
        optuna_configs_b=optuna_configs_b,
    )

    # Statistical tests
    test_results = compute_statistical_tests(results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "optuna" if args.optuna_dir else "grid"
    results_path = os.path.join(OUTPUT_DIR, f"causal_eval_{suffix}_{timestamp}.json")
    with open(results_path, 'w') as f:
        json.dump({
            "results": results,
            "statistical_tests": test_results,
            "causal_configs": causal_configs,
            "noncausal_configs": NONCAUSAL_CONFIGS,
            "optuna_configs_a": optuna_configs_a,
            "optuna_configs_b": optuna_configs_b,
            "config_source": "optuna" if args.optuna_dir else "grid_search",
            "timestamp": timestamp,
        }, f, indent=2, default=str)
    logger.info(f"Saved results to {results_path}")

    # Format and print summary
    summary = format_summary(results, test_results)
    print(summary)

    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, f"causal_eval_summary_{timestamp}.txt")
    with open(summary_path, 'w') as f:
        f.write(summary)
    logger.info(f"Saved summary to {summary_path}")

    total_elapsed = time.time() - t0
    print(f"\nTotal elapsed: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
