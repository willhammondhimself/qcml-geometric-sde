#!/usr/bin/env python3
"""
Nested Leave-One-Crisis-Out (LOCO) Validation for Fused QCML

Provides an unbiased performance estimate by ensuring the held-out crisis
is NEVER seen by Optuna during hyperparameter optimization.

Part A — Nested LOCO (12 folds):
  For each of 12 crises as the held-out test:
    1. Remove it from the crisis data
    2. Run Optuna Phase A optimization on remaining 11 crises (50 trials)
    3. Evaluate best params on the held-out crisis
    4. Record: held-out d, best params for this fold
  Report: mean/median d across 12 held-out crises (unbiased estimate).

Part B — True Temporal OOS Re-optimization:
    1. Run Optuna Phase A on pre-2020 crises only (9 crises, 100 trials)
    2. Evaluate on post-2020 crises (COVID, Rates, SVB) with frozen params
    3. Compare to biased temporal OOS

Part C — Parameter Stability Analysis:
    Across the 12 LOCO folds, report categorical agreement rate,
    continuous weight CV, and stability interpretation.

Usage:
    python experiments/nested_loco_validation.py --phase-a
    python experiments/nested_loco_validation.py --temporal-oos
    python experiments/nested_loco_validation.py --all
    python experiments/nested_loco_validation.py --all --n-trials 50
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

load_dotenv(project_root / '.env')

from experiments.crisis_config import (
    DATA_AVAILABLE_CRISES,
    get_default_validation_config,
)
from experiments.regime_comparison import (
    evaluate_method,
    prepare_data,
    seed_everything,
)
# from qcml.regime.fused_detector import FusedQCMLDetector  # archived

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore", message="Metric tensor has negative eigenvalue")

OUTPUT_DIR = project_root / 'experiments' / 'outputs' / 'regime_detection' / 'nested_loco'

# Post-2020 crisis names for temporal OOS split
POST_2020_NAMES = {'2020_covid', '2022_rates', '2023_svb'}


# ---------------------------------------------------------------------------
# Data preparation (mirrors fused_qcml_optimization.py)
# ---------------------------------------------------------------------------

def prepare_all_crisis_data(
    enriched_lookback: int = 20,
) -> List[Dict[str, Any]]:
    """Pre-fetch and prepare data for all 12 crises."""
    config = get_default_validation_config()
    crisis_data = []

    for crisis in DATA_AVAILABLE_CRISES:
        X, X_enriched, times, crisis_idx = prepare_data(
            crisis, config, enriched_lookback=enriched_lookback,
        )
        if X is None:
            logger.warning(f"Skipping {crisis.name}: no data")
            continue

        trim = enriched_lookback - 1
        times_enriched = times[trim:]
        crisis_idx_enriched = max(0, crisis_idx - trim)

        crisis_data.append({
            'crisis': crisis,
            'X': X,
            'X_enriched': X_enriched,
            'times': times,
            'crisis_idx': crisis_idx,
            'times_enriched': times_enriched,
            'crisis_idx_enriched': crisis_idx_enriched,
        })

    logger.info(f"Prepared data for {len(crisis_data)} crises")
    return crisis_data


def evaluate_detector_on_crises(
    detector_factory,
    crisis_data: List[Dict[str, Any]],
    seed: int = 42,
    n_bootstrap: int = 500,
    n_permutations: int = 200,
) -> List[float]:
    """Evaluate a detector across the given crises, return list of d values."""
    config = get_default_validation_config()
    d_values = []

    for cd in crisis_data:
        try:
            detector = detector_factory()
            detector.fit(cd['X_enriched'])
            result = evaluate_method(
                detector,
                cd['X_enriched'],
                cd['times_enriched'],
                cd['crisis_idx_enriched'],
                cd['crisis'],
                config,
                n_bootstrap=n_bootstrap,
                n_permutations=n_permutations,
                seed=seed,
            )
            d = result.get('effect_size_d_normalized', 0.0)
            d_values.append(d)
        except Exception as e:
            logger.warning(f"  {cd['crisis'].name} failed: {e}")
            d_values.append(0.0)

    return d_values


# ---------------------------------------------------------------------------
# Phase A objective (same search space as fused_qcml_optimization.py)
# ---------------------------------------------------------------------------

def phase_a_objective(
    trial: optuna.Trial,
    training_data: List[Dict],
    n_bootstrap: int = 500,
    n_permutations: int = 200,
) -> float:
    """Optuna objective for Phase A FusedQCMLDetector, training on subset."""
    fusion_method = trial.suggest_categorical(
        'fusion_method', ['max', 'weighted_mean', 'rank_mean']
    )
    hilbert_dim = trial.suggest_categorical('hilbert_dim', [4, 8, 12])
    n_pca_components = trial.suggest_categorical('n_pca_components', [8, 10, 15])
    operator_method = trial.suggest_categorical(
        'operator_method', ['random', 'pca_inspired']
    )
    min_expanding = trial.suggest_categorical('min_expanding', [40, 60, 80])
    rolling_window = trial.suggest_int('rolling_window', 10, 40, step=5)

    weights = None
    if fusion_method == 'weighted_mean':
        w_berry = trial.suggest_float('w_berry', 0.0, 1.0)
        w_qfi = trial.suggest_float('w_qfi', 0.0, 1.0)
        w_multilag = trial.suggest_float('w_multilag', 0.0, 1.0)
        total = w_berry + w_qfi + w_multilag
        if total < 1e-8:
            return 0.0
        weights = [w_berry / total, w_qfi / total, w_multilag / total]

    def factory():
        return FusedQCMLDetector(
            hilbert_dim=hilbert_dim,
            n_pca_components=n_pca_components,
            operator_method=operator_method,
            seed=42,
            fusion_method=fusion_method,
            weights=weights,
            min_expanding=min_expanding,
            rolling_window=rolling_window,
        )

    d_values = evaluate_detector_on_crises(
        factory, training_data,
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
    )
    median_d = float(np.median(d_values))

    trial.set_user_attr('d_values', [float(d) for d in d_values])
    trial.set_user_attr('median_d', median_d)

    return median_d


# ---------------------------------------------------------------------------
# Part A: Nested LOCO
# ---------------------------------------------------------------------------

def run_nested_loco(
    n_trials: int = 50,
    n_bootstrap_opt: int = 500,
    n_permutations_opt: int = 200,
    n_bootstrap_eval: int = 1000,
    n_permutations_eval: int = 500,
    seed: int = 42,
) -> Dict:
    """Run nested LOCO: for each crisis held out, optimize on the other 11.

    Returns dict with per-fold results and parameter stability analysis.
    """
    print("\n" + "=" * 70)
    print("PART A: Nested Leave-One-Crisis-Out Validation")
    print(f"  {n_trials} Optuna trials per fold, 12 folds")
    print("=" * 70)

    seed_everything(seed)
    crisis_data = prepare_all_crisis_data()
    n_crises = len(crisis_data)

    fold_results = []
    all_best_params = []

    for fold_idx in range(n_crises):
        held_out = crisis_data[fold_idx]
        held_out_name = held_out['crisis'].name
        training = [cd for i, cd in enumerate(crisis_data) if i != fold_idx]

        print(f"\n--- Fold {fold_idx + 1}/{n_crises}: Held out = {held_out_name} ---")
        logger.info(f"Training on {len(training)} crises, testing on {held_out_name}")

        # Run Optuna optimization on training crises only
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction='maximize',
            study_name=f'nested_loco_fold_{fold_idx}',
            sampler=optuna.samplers.TPESampler(seed=seed + fold_idx),
        )

        t0 = time.time()
        study.optimize(
            lambda trial: phase_a_objective(
                trial, training,
                n_bootstrap=n_bootstrap_opt,
                n_permutations=n_permutations_opt,
            ),
            n_trials=n_trials,
        )
        opt_elapsed = time.time() - t0

        best = study.best_trial
        best_params = best.params
        logger.info(
            f"  Fold {fold_idx + 1} optimization: {opt_elapsed:.0f}s, "
            f"best training median d = {best.value:.3f}"
        )

        # Evaluate best params on held-out crisis
        weights = None
        if best_params.get('fusion_method') == 'weighted_mean':
            w_berry = best_params.get('w_berry', 1/3)
            w_qfi = best_params.get('w_qfi', 1/3)
            w_multilag = best_params.get('w_multilag', 1/3)
            total = w_berry + w_qfi + w_multilag
            if total > 1e-8:
                weights = [w_berry / total, w_qfi / total, w_multilag / total]

        def factory():
            return FusedQCMLDetector(
                hilbert_dim=best_params['hilbert_dim'],
                n_pca_components=best_params['n_pca_components'],
                operator_method=best_params['operator_method'],
                seed=42,
                fusion_method=best_params['fusion_method'],
                weights=weights,
                min_expanding=best_params['min_expanding'],
                rolling_window=best_params['rolling_window'],
            )

        held_out_d = evaluate_detector_on_crises(
            factory, [held_out],
            n_bootstrap=n_bootstrap_eval,
            n_permutations=n_permutations_eval,
        )

        fold_result = {
            'fold': fold_idx,
            'held_out_crisis': held_out_name,
            'best_params': best_params,
            'training_median_d': float(best.value),
            'held_out_d': float(held_out_d[0]) if held_out_d else 0.0,
            'n_trials': n_trials,
            'opt_elapsed_s': opt_elapsed,
        }
        fold_results.append(fold_result)
        all_best_params.append(best_params)

        print(
            f"  Fold {fold_idx + 1}: training d={best.value:.3f}, "
            f"held-out d={fold_result['held_out_d']:.3f} ({held_out_name})"
        )

    # Aggregate
    held_out_ds = [f['held_out_d'] for f in fold_results]
    training_ds = [f['training_median_d'] for f in fold_results]

    summary = {
        'n_folds': n_crises,
        'n_trials_per_fold': n_trials,
        'held_out_d_mean': float(np.mean(held_out_ds)),
        'held_out_d_median': float(np.median(held_out_ds)),
        'held_out_d_std': float(np.std(held_out_ds)),
        'held_out_d_min': float(np.min(held_out_ds)),
        'held_out_d_max': float(np.max(held_out_ds)),
        'training_d_mean': float(np.mean(training_ds)),
        'training_d_median': float(np.median(training_ds)),
        'overfitting_gap': float(np.mean(training_ds) - np.mean(held_out_ds)),
        'per_fold': fold_results,
    }

    # Parameter stability
    stability = compute_parameter_stability(all_best_params)
    summary['parameter_stability'] = stability

    return summary


# ---------------------------------------------------------------------------
# Part B: True Temporal OOS Re-optimization
# ---------------------------------------------------------------------------

def run_temporal_oos(
    n_trials: int = 100,
    n_bootstrap_opt: int = 500,
    n_permutations_opt: int = 200,
    n_bootstrap_eval: int = 1000,
    n_permutations_eval: int = 500,
    seed: int = 42,
) -> Dict:
    """True temporal OOS: optimize on pre-2020, evaluate on post-2020.

    Returns dict with optimization results and post-2020 evaluation.
    """
    print("\n" + "=" * 70)
    print("PART B: True Temporal OOS Re-optimization")
    print(f"  {n_trials} Optuna trials on pre-2020 crises")
    print("=" * 70)

    seed_everything(seed)
    all_crisis_data = prepare_all_crisis_data()

    # Split: pre-2020 (training) vs post-2020 (test)
    pre_2020 = [cd for cd in all_crisis_data
                if cd['crisis'].name not in POST_2020_NAMES]
    post_2020 = [cd for cd in all_crisis_data
                 if cd['crisis'].name in POST_2020_NAMES]

    logger.info(f"Pre-2020 crises: {[cd['crisis'].name for cd in pre_2020]}")
    logger.info(f"Post-2020 crises: {[cd['crisis'].name for cd in post_2020]}")

    # Optimize on pre-2020 only
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='maximize',
        study_name='temporal_oos_pre2020',
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    t0 = time.time()
    study.optimize(
        lambda trial: phase_a_objective(
            trial, pre_2020,
            n_bootstrap=n_bootstrap_opt,
            n_permutations=n_permutations_opt,
        ),
        n_trials=n_trials,
    )
    opt_elapsed = time.time() - t0

    best = study.best_trial
    best_params = best.params
    print(f"\nPre-2020 optimization completed in {opt_elapsed:.0f}s")
    print(f"Best pre-2020 median d: {best.value:.4f}")
    print(f"Best params: {json.dumps(best_params, indent=2)}")

    # Evaluate on post-2020 with frozen params
    weights = None
    if best_params.get('fusion_method') == 'weighted_mean':
        w_berry = best_params.get('w_berry', 1/3)
        w_qfi = best_params.get('w_qfi', 1/3)
        w_multilag = best_params.get('w_multilag', 1/3)
        total = w_berry + w_qfi + w_multilag
        if total > 1e-8:
            weights = [w_berry / total, w_qfi / total, w_multilag / total]

    def factory():
        return FusedQCMLDetector(
            hilbert_dim=best_params['hilbert_dim'],
            n_pca_components=best_params['n_pca_components'],
            operator_method=best_params['operator_method'],
            seed=42,
            fusion_method=best_params['fusion_method'],
            weights=weights,
            min_expanding=best_params['min_expanding'],
            rolling_window=best_params['rolling_window'],
        )

    # Evaluate on pre-2020 (in-sample check)
    pre_2020_d = evaluate_detector_on_crises(
        factory, pre_2020,
        n_bootstrap=n_bootstrap_eval,
        n_permutations=n_permutations_eval,
    )

    # Evaluate on post-2020 (true OOS)
    post_2020_d = evaluate_detector_on_crises(
        factory, post_2020,
        n_bootstrap=n_bootstrap_eval,
        n_permutations=n_permutations_eval,
    )

    post_2020_detail = {}
    for cd, d in zip(post_2020, post_2020_d):
        post_2020_detail[cd['crisis'].name] = float(d)

    results = {
        'best_params': best_params,
        'n_trials': n_trials,
        'opt_elapsed_s': opt_elapsed,
        'pre_2020_crises': [cd['crisis'].name for cd in pre_2020],
        'post_2020_crises': [cd['crisis'].name for cd in post_2020],
        'pre_2020_median_d': float(np.median(pre_2020_d)),
        'pre_2020_mean_d': float(np.mean(pre_2020_d)),
        'post_2020_median_d': float(np.median(post_2020_d)),
        'post_2020_mean_d': float(np.mean(post_2020_d)),
        'post_2020_per_crisis': post_2020_detail,
        'overfitting_gap': float(np.median(pre_2020_d) - np.median(post_2020_d)),
    }

    print(f"\nPre-2020 (in-sample):  median d = {results['pre_2020_median_d']:.3f}")
    print(f"Post-2020 (true OOS): median d = {results['post_2020_median_d']:.3f}")
    print(f"Overfitting gap:      {results['overfitting_gap']:.3f}")
    print(f"Post-2020 per-crisis: {post_2020_detail}")

    return results


# ---------------------------------------------------------------------------
# Part C: Parameter Stability Analysis
# ---------------------------------------------------------------------------

def compute_parameter_stability(all_best_params: List[Dict]) -> Dict:
    """Analyze parameter stability across LOCO folds.

    Returns dict with agreement rates for categorical params
    and CV for continuous params.
    """
    n_folds = len(all_best_params)
    if n_folds == 0:
        return {}

    stability = {}

    # Categorical parameters
    categorical_params = [
        'fusion_method', 'hilbert_dim', 'n_pca_components',
        'operator_method', 'min_expanding',
    ]
    for param in categorical_params:
        values = [p.get(param) for p in all_best_params if param in p]
        if not values:
            continue
        counter = Counter(values)
        mode_value, mode_count = counter.most_common(1)[0]
        agreement_rate = mode_count / len(values)
        stability[param] = {
            'values': [str(v) for v in values],
            'mode': str(mode_value),
            'agreement_rate': round(agreement_rate, 3),
            'distribution': {str(k): v for k, v in counter.items()},
        }

    # Continuous parameters
    continuous_params = ['rolling_window']
    for param in continuous_params:
        values = [p.get(param) for p in all_best_params if param in p]
        if not values:
            continue
        values_arr = np.array(values, dtype=float)
        stability[param] = {
            'values': [float(v) for v in values],
            'mean': float(np.mean(values_arr)),
            'std': float(np.std(values_arr)),
            'cv': float(np.std(values_arr) / np.mean(values_arr) * 100) if np.mean(values_arr) > 0 else 0.0,
            'min': float(np.min(values_arr)),
            'max': float(np.max(values_arr)),
        }

    # Fusion weights (if weighted_mean is used)
    weight_params = ['w_berry', 'w_qfi', 'w_multilag']
    for wp in weight_params:
        values = [p.get(wp) for p in all_best_params if wp in p]
        if not values:
            continue
        values_arr = np.array(values, dtype=float)
        stability[wp] = {
            'values': [float(v) for v in values],
            'mean': float(np.mean(values_arr)),
            'std': float(np.std(values_arr)),
            'cv': float(np.std(values_arr) / np.mean(values_arr) * 100) if np.mean(values_arr) > 0 else 0.0,
        }

    # Overall stability score
    cat_agreements = [
        stability[p]['agreement_rate']
        for p in categorical_params
        if p in stability
    ]
    overall_cat_agreement = np.mean(cat_agreements) if cat_agreements else 0.0

    stability['overall'] = {
        'mean_categorical_agreement': round(float(overall_cat_agreement), 3),
        'interpretation': (
            'robust (parameters are stable across folds)'
            if overall_cat_agreement > 0.8
            else 'moderate (some parameter sensitivity to fold composition)'
            if overall_cat_agreement > 0.5
            else 'unstable (parameters highly dependent on fold composition — overfitting risk)'
        ),
    }

    return stability


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------

def write_overfitting_analysis(
    loco_results: Optional[Dict],
    temporal_results: Optional[Dict],
    output_dir: Path,
) -> None:
    """Write OVERFITTING_ANALYSIS.md summarizing all findings."""
    lines = [
        "# Overfitting Analysis: Fused QCML Optimization",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Context",
        "",
        "The original fused QCML optimization (Phase A median d=1.69, Phase B d=1.79)",
        "optimized hyperparameters on ALL 12 crises simultaneously. This analysis",
        "provides unbiased performance estimates via nested cross-validation.",
        "",
    ]

    if loco_results:
        lines.extend([
            "## Part A: Nested Leave-One-Crisis-Out",
            "",
            f"- **Unbiased held-out median d: {loco_results['held_out_d_median']:.3f}**",
            f"- Unbiased held-out mean d: {loco_results['held_out_d_mean']:.3f}",
            f"- Training median d (mean across folds): {loco_results['training_d_median']:.3f}",
            f"- **Overfitting gap: {loco_results['overfitting_gap']:.3f}**",
            f"- Held-out d range: [{loco_results['held_out_d_min']:.3f}, {loco_results['held_out_d_max']:.3f}]",
            "",
            "### Per-Fold Results",
            "",
            "| Fold | Held-Out Crisis | Training d | Held-Out d |",
            "|------|-----------------|-----------|-----------|",
        ])

        for fold in loco_results['per_fold']:
            lines.append(
                f"| {fold['fold'] + 1} | {fold['held_out_crisis']} "
                f"| {fold['training_median_d']:.3f} | {fold['held_out_d']:.3f} |"
            )

        lines.append("")

        # Comparison to biased estimate
        biased_d = 1.69  # from original Phase A optimization
        unbiased_d = loco_results['held_out_d_median']
        reduction_pct = (biased_d - unbiased_d) / biased_d * 100 if biased_d > 0 else 0

        lines.extend([
            "### Comparison to Biased Estimate",
            "",
            f"- Original (biased) Phase A median d: 1.69",
            f"- Unbiased (nested LOCO) median d: {unbiased_d:.3f}",
            f"- Reduction: {reduction_pct:.1f}%",
            f"- RF baseline median d: 1.15",
            "",
        ])

        if unbiased_d > 1.15:
            lines.append(f"**Conclusion**: Even after correcting for overfitting, "
                         f"fused QCML (d={unbiased_d:.2f}) remains above RF (d=1.15).")
        else:
            lines.append(f"**Conclusion**: After correcting for overfitting, "
                         f"fused QCML (d={unbiased_d:.2f}) is competitive with "
                         f"but does not clearly exceed RF (d=1.15).")

        lines.append("")

        # Parameter stability
        if 'parameter_stability' in loco_results:
            stab = loco_results['parameter_stability']
            lines.extend([
                "### Parameter Stability Across Folds",
                "",
            ])

            for param in ['fusion_method', 'hilbert_dim', 'n_pca_components',
                          'operator_method', 'min_expanding']:
                if param in stab:
                    s = stab[param]
                    lines.append(
                        f"- **{param}**: mode={s['mode']}, "
                        f"agreement={s['agreement_rate']:.0%} "
                        f"({s['distribution']})"
                    )

            if 'rolling_window' in stab:
                s = stab['rolling_window']
                lines.append(
                    f"- **rolling_window**: mean={s['mean']:.1f}, "
                    f"std={s['std']:.1f}, CV={s['cv']:.1f}%"
                )

            for wp in ['w_berry', 'w_qfi', 'w_multilag']:
                if wp in stab:
                    s = stab[wp]
                    lines.append(
                        f"- **{wp}**: mean={s['mean']:.3f}, "
                        f"std={s['std']:.3f}, CV={s['cv']:.1f}%"
                    )

            if 'overall' in stab:
                lines.extend([
                    "",
                    f"**Overall stability**: {stab['overall']['interpretation']}",
                    f"(mean categorical agreement = {stab['overall']['mean_categorical_agreement']:.0%})",
                    "",
                ])

    if temporal_results:
        lines.extend([
            "## Part B: True Temporal Out-of-Sample",
            "",
            f"- Pre-2020 crises (training): {temporal_results['pre_2020_crises']}",
            f"- Post-2020 crises (test): {temporal_results['post_2020_crises']}",
            "",
            f"- Pre-2020 median d (in-sample): {temporal_results['pre_2020_median_d']:.3f}",
            f"- **Post-2020 median d (true OOS): {temporal_results['post_2020_median_d']:.3f}**",
            f"- Overfitting gap: {temporal_results['overfitting_gap']:.3f}",
            "",
            "### Per-Crisis Post-2020 Results",
            "",
        ])
        for crisis_name, d in temporal_results['post_2020_per_crisis'].items():
            lines.append(f"- {crisis_name}: d={d:.3f}")

        # Compare to biased temporal OOS
        biased_temporal_d = 1.72  # from original evaluation
        unbiased_temporal_d = temporal_results['post_2020_median_d']

        lines.extend([
            "",
            "### Comparison to Biased Temporal OOS",
            "",
            f"- Biased (params optimized on all 12): post-2020 median d = 1.72",
            f"- Unbiased (params optimized on pre-2020 only): post-2020 median d = {unbiased_temporal_d:.3f}",
            f"- RF post-2020 median d: 0.88",
            "",
        ])

    lines.extend([
        "## Recommendations for Paper",
        "",
        "1. Replace misleading 'leave-one-crisis-out cross-validation' claim",
        "   with accurate description of the optimization protocol.",
        "2. Report unbiased nested LOCO d alongside the calibration estimate.",
        "3. Discuss the overfitting gap honestly as evidence of the fusion's",
        "   ability to generalize (if gap is modest) or as a limitation.",
        "4. Update the conclusion to reflect unbiased performance numbers.",
        "",
    ])

    path = output_dir / 'OVERFITTING_ANALYSIS.md'
    with open(path, 'w') as f:
        f.write("\n".join(lines))
    logger.info(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Nested LOCO validation for fused QCML"
    )
    parser.add_argument('--phase-a', action='store_true',
                        help='Run Part A: Nested LOCO (12 folds)')
    parser.add_argument('--temporal-oos', action='store_true',
                        help='Run Part B: True temporal OOS re-optimization')
    parser.add_argument('--all', action='store_true',
                        help='Run all parts')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Optuna trials per fold/optimization')
    parser.add_argument('--n-trials-temporal', type=int, default=100,
                        help='Optuna trials for temporal OOS')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if not any([args.phase_a, args.temporal_oos, args.all]):
        parser.error("Specify at least one of: --phase-a, --temporal-oos, --all")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    loco_results = None
    temporal_results = None

    if args.phase_a or args.all:
        t0 = time.time()
        loco_results = run_nested_loco(
            n_trials=args.n_trials,
            seed=args.seed,
        )
        elapsed = time.time() - t0

        # Save
        out_path = OUTPUT_DIR / 'nested_loco_results.json'
        with open(out_path, 'w') as f:
            json.dump(loco_results, f, indent=2, default=str)

        print(f"\n{'=' * 70}")
        print("NESTED LOCO SUMMARY")
        print(f"{'=' * 70}")
        print(f"Unbiased held-out median d: {loco_results['held_out_d_median']:.3f}")
        print(f"Unbiased held-out mean d:   {loco_results['held_out_d_mean']:.3f}")
        print(f"Training median d:          {loco_results['training_d_median']:.3f}")
        print(f"Overfitting gap:            {loco_results['overfitting_gap']:.3f}")
        print(f"Completed in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
        print(f"Results: {out_path}")

    if args.temporal_oos or args.all:
        t0 = time.time()
        temporal_results = run_temporal_oos(
            n_trials=args.n_trials_temporal if not args.all else args.n_trials,
            seed=args.seed,
        )
        elapsed = time.time() - t0

        # Save
        out_path = OUTPUT_DIR / 'temporal_oos_results.json'
        with open(out_path, 'w') as f:
            json.dump(temporal_results, f, indent=2, default=str)

        print(f"\n{'=' * 70}")
        print("TEMPORAL OOS SUMMARY")
        print(f"{'=' * 70}")
        print(f"Pre-2020 median d:  {temporal_results['pre_2020_median_d']:.3f}")
        print(f"Post-2020 median d: {temporal_results['post_2020_median_d']:.3f}")
        print(f"Overfitting gap:    {temporal_results['overfitting_gap']:.3f}")
        print(f"Completed in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
        print(f"Results: {out_path}")

    # Write analysis
    write_overfitting_analysis(loco_results, temporal_results, OUTPUT_DIR)

    print(f"\nAll results saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
