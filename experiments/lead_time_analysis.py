#!/usr/bin/env python3
"""
Lead Time / Early Warning Analysis

For each of the 12 crises and 16 methods, compute how many trading days
*before* the crisis date the method first signals elevated risk (z-score > 2.0
sustained for 2+ consecutive days).

Outputs:
  - lead_time_results.json — 16 methods x 12 crises matrix of lead times
  - figures/signal_trajectory_{crisis}.pdf — score timelines around each crisis
  - figures/lead_time_comparison.pdf — grouped bar chart by method category

Usage:
    python experiments/lead_time_analysis.py --seed 42

Author: QCML Research
"""

import argparse
import json
import logging
import os
import sys
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
    CrisisDefinition,
    DATA_AVAILABLE_CRISES,
    get_default_validation_config,
    config_to_dict,
)
from experiments.regime_comparison import (
    prepare_data,
    seed_everything,
)
from qcml.regime.classical_baselines import (
    BaseRegimeDetector,
    QCMLChernDetector,
    RollingVolatilityDetector,
    CUSUMDetector,
    HMMRegimeDetector,
    RandomForestRegimeDetector,
    MultiScaleChernDetector,
    QuantumEnsembleDetector,
    QFISusceptibilityDetector,
    ScalarCurvatureDetector,
    GeometricConsensusDetector,
    QFIDeterminantDetector,
    BerryPhaseRateDetector,
    MultiLagFidelityDetector,
    MetricConditionNumberDetector,
)
from qcml.regime.adaptive_ensemble import AdaptiveRegimeEnsemble
from experiments.rigorous_crisis_validation import fetch_real_crisis_data
from experiments.regime_comparison import (
    prepare_rf_training_data,
    _align_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("experiments/outputs/regime_detection/lead_time")

# Crisis typology: novel (QCML-favorable) vs conventional (RF-favorable)
NOVEL_CRISES = {
    "2018_volmageddon", "2019_repo_crisis", "2018_fed_selloff", "2023_svb",
}
CONVENTIONAL_CRISES = {
    "2007_quant_meltdown", "2008_crisis", "2010_flash_crash",
    "2011_debt_downgrade", "2015_china", "2016_brexit",
    "2020_covid", "2022_rates",
}

QCML_METHODS = [
    "QCML Chern", "Multi-Scale Chern", "Quantum Ensemble",
    "QFI Susceptibility", "Scalar Curvature", "Geometric Consensus",
    "Adaptive Ensemble", "QFI Determinant", "Berry Phase Rate",
    "Multi-Lag Fidelity", "Metric Condition Number",
]


def compute_principled_lead_time(
    scores: np.ndarray,
    times: pd.DatetimeIndex,
    crisis_date: str,
    z_threshold: float = 2.0,
    min_consecutive: int = 2,
    min_expanding: int = 60,
) -> Optional[int]:
    """Compute lead time using z-score > threshold sustained for min_consecutive days.

    Uses an expanding-window z-score (causal, no look-ahead) to normalize
    each method's scores, then finds the first sustained crossing before
    the crisis date.

    Args:
        scores: 1-D regime score array.
        times: DatetimeIndex aligned with scores.
        crisis_date: Crisis date string (YYYY-MM-DD).
        z_threshold: Z-score threshold for elevated risk.
        min_consecutive: Minimum consecutive days above threshold.
        min_expanding: Minimum observations for z-score to be valid.

    Returns:
        Lead time in trading days, or None if no lead time detected.
    """
    crisis_ts = pd.Timestamp(crisis_date)
    T = len(scores)

    # Compute expanding z-scores (causal)
    z_scores = np.full(T, np.nan)
    for t in range(min_expanding, T):
        past = scores[:t]
        past_valid = past[~np.isnan(past)]
        if len(past_valid) < 10:
            continue
        mu = np.mean(past_valid)
        sigma = np.std(past_valid, ddof=1)
        if sigma > 1e-12 and not np.isnan(scores[t]):
            z_scores[t] = (scores[t] - mu) / sigma

    # Find crisis index
    crisis_mask = times >= crisis_ts
    if not crisis_mask.any():
        return None
    crisis_idx = int(crisis_mask.argmax())

    # Search backward from crisis for first sustained crossing
    # Only look in the pre-crisis window (up to 90 days before)
    search_start = max(min_expanding, crisis_idx - 90)
    search_end = crisis_idx

    for i in range(search_start, search_end - min_consecutive + 1):
        # Check if z_threshold is exceeded for min_consecutive days
        window = z_scores[i:i + min_consecutive]
        if np.all(~np.isnan(window)) and np.all(window > z_threshold):
            lead_time = crisis_idx - i
            return lead_time

    return None


def get_score_trajectory(
    scores: np.ndarray,
    times: pd.DatetimeIndex,
    crisis_date: str,
    window_before: int = 60,
    window_after: int = 30,
    min_expanding: int = 60,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[int]]:
    """Extract z-scored trajectory around a crisis date.

    Returns:
        z_trajectory: Z-scored trajectory in the window.
        time_offsets: Trading day offsets from crisis date.
        crisis_local_idx: Index of crisis date within the trajectory.
    """
    crisis_ts = pd.Timestamp(crisis_date)
    T = len(scores)

    # Compute expanding z-scores
    z_scores = np.full(T, np.nan)
    for t in range(min_expanding, T):
        past = scores[:t]
        past_valid = past[~np.isnan(past)]
        if len(past_valid) < 10:
            continue
        mu = np.mean(past_valid)
        sigma = np.std(past_valid, ddof=1)
        if sigma > 1e-12 and not np.isnan(scores[t]):
            z_scores[t] = (scores[t] - mu) / sigma

    crisis_mask = times >= crisis_ts
    if not crisis_mask.any():
        return None, None, None
    crisis_idx = int(crisis_mask.argmax())

    start = max(0, crisis_idx - window_before)
    end = min(T, crisis_idx + window_after)

    z_trajectory = z_scores[start:end]
    offsets = np.arange(start - crisis_idx, end - crisis_idx)
    crisis_local_idx = crisis_idx - start

    return z_trajectory, offsets, crisis_local_idx


def instantiate_detectors(config, seed, crises):
    """Instantiate all 16 detectors (RF needs separate handling per crisis)."""
    enriched_detectors = {
        "QCML Chern": lambda: QCMLChernDetector(
            hilbert_dim=config.hilbert_dim, window_size=config.window_size,
            n_pca_components=config.n_pca_components,
            operator_method=config.operator_method, seed=seed),
        "Multi-Scale Chern": lambda: MultiScaleChernDetector(
            hilbert_dim=config.hilbert_dim,
            n_pca_components=config.n_pca_components,
            operator_method=config.operator_method, seed=seed),
        "Quantum Ensemble": lambda: QuantumEnsembleDetector(
            hilbert_dim=config.hilbert_dim,
            n_pca_components=config.n_pca_components,
            window_size=config.window_size,
            operator_method=config.operator_method, seed=seed),
        "QFI Susceptibility": lambda: QFISusceptibilityDetector(
            hilbert_dim=config.hilbert_dim,
            n_pca_components=config.n_pca_components,
            operator_method=config.operator_method,
            min_expanding=60, seed=seed),
        "Scalar Curvature": lambda: ScalarCurvatureDetector(
            hilbert_dim=config.hilbert_dim, n_curvature_components=8,
            operator_method=config.operator_method,
            min_expanding=60, seed=seed),
        "Geometric Consensus": lambda: GeometricConsensusDetector(
            hilbert_dim=config.hilbert_dim, n_pca_components=8,
            n_curvature_components=8,
            operator_method=config.operator_method,
            min_persistence=3, min_agreement=2, seed=seed),
        "QFI Determinant": lambda: QFIDeterminantDetector(
            hilbert_dim=config.hilbert_dim,
            n_pca_components=config.n_pca_components,
            operator_method=config.operator_method, seed=seed),
        "Berry Phase Rate": lambda: BerryPhaseRateDetector(
            hilbert_dim=config.hilbert_dim,
            n_pca_components=config.n_pca_components,
            operator_method=config.operator_method, seed=seed),
        "Multi-Lag Fidelity": lambda: MultiLagFidelityDetector(
            hilbert_dim=config.hilbert_dim,
            n_pca_components=config.n_pca_components,
            operator_method=config.operator_method, seed=seed),
        "Metric Condition Number": lambda: MetricConditionNumberDetector(
            hilbert_dim=config.hilbert_dim,
            n_pca_components=config.n_pca_components,
            operator_method=config.operator_method, seed=seed),
    }

    raw_detectors = {
        "Rolling Vol Z": lambda: RollingVolatilityDetector(vol_window=20, min_expanding=60),
        "CUSUM": lambda: CUSUMDetector(burn_in=60),
        "HMM 2-state": lambda: HMMRegimeDetector(n_iter=100, seed=seed),
    }

    return enriched_detectors, raw_detectors


def run_lead_time_analysis(
    crises: List[CrisisDefinition],
    seed: int = 42,
) -> Dict[str, Any]:
    """Run lead time analysis across all methods and crises."""
    seed_everything(seed)
    config = get_default_validation_config()
    enriched_lookback = 20

    enriched_factories, raw_factories = instantiate_detectors(config, seed, crises)

    results = {}  # crisis_name -> {method_name -> lead_time}
    trajectories = {}  # crisis_name -> {method_name -> {z_traj, offsets}}

    for crisis in crises:
        print(f"\n{'='*60}")
        print(f"  Crisis: {crisis.name} ({crisis.crisis_date})")
        print(f"{'='*60}")

        X, X_enriched, times, crisis_idx = prepare_data(
            crisis, config, enriched_lookback=enriched_lookback,
        )
        if X is None:
            logger.warning(f"Skipping {crisis.name} - no data")
            continue

        trim = enriched_lookback - 1
        times_enriched = times[trim:]

        crisis_leads = {}
        crisis_trajs = {}

        # --- Enriched detectors (QCML methods) ---
        for name, factory in enriched_factories.items():
            print(f"  {name}...")
            try:
                det = factory()
                det.fit(X_enriched)
                scores = det.compute_regime_scores(X_enriched)

                lead = compute_principled_lead_time(
                    scores, times_enriched, crisis.crisis_date,
                )
                crisis_leads[name] = lead

                z_traj, offsets, _ = get_score_trajectory(
                    scores, times_enriched, crisis.crisis_date,
                )
                if z_traj is not None:
                    crisis_trajs[name] = {
                        'z_trajectory': z_traj.tolist(),
                        'offsets': offsets.tolist(),
                    }
            except Exception as e:
                logger.error(f"  {name} failed: {e}")
                crisis_leads[name] = None

        # --- Adaptive Ensemble ---
        print(f"  Adaptive Ensemble...")
        try:
            n_pca_adaptive = min(20, X_enriched.shape[1])
            det_adaptive = AdaptiveRegimeEnsemble(
                hilbert_dim=config.hilbert_dim,
                n_pca_components=n_pca_adaptive,
                n_curvature_components=8,
                operator_method=config.operator_method, seed=seed,
            )
            dataset = fetch_real_crisis_data(crisis)
            prices = dataset.prices
            if isinstance(prices, pd.DataFrame):
                prices = prices.iloc[:, 0]
            det_adaptive.fit(X_enriched, prices=prices)
            scores = det_adaptive.compute_regime_scores(X_enriched)
            crisis_leads["Adaptive Ensemble"] = compute_principled_lead_time(
                scores, times_enriched, crisis.crisis_date,
            )
        except Exception as e:
            logger.error(f"  Adaptive Ensemble failed: {e}")
            crisis_leads["Adaptive Ensemble"] = None

        # --- Raw detectors (Vol Z, CUSUM, HMM) ---
        for name, factory in raw_factories.items():
            print(f"  {name}...")
            try:
                det = factory()
                det.fit(X)
                scores = det.compute_regime_scores(X)
                crisis_leads[name] = compute_principled_lead_time(
                    scores, times, crisis.crisis_date,
                )

                z_traj, offsets, _ = get_score_trajectory(
                    scores, times, crisis.crisis_date,
                )
                if z_traj is not None:
                    crisis_trajs[name] = {
                        'z_trajectory': z_traj.tolist(),
                        'offsets': offsets.tolist(),
                    }
            except Exception as e:
                logger.error(f"  {name} failed: {e}")
                crisis_leads[name] = None

        # --- Random Forest (LOCO) ---
        print(f"  Random Forest...")
        try:
            det_rf = RandomForestRegimeDetector(
                n_estimators=200, max_depth=6, seed=seed, lookback=20,
            )
            X_train, y_train, rf_n_features = prepare_rf_training_data(
                crisis, crises, config,
            )
            det_rf.fit_with_labels(X_train, y_train)
            X_rf = X[:, :rf_n_features] if X.shape[1] > rf_n_features else X
            scores = det_rf.compute_regime_scores(X_rf)
            crisis_leads["Random Forest"] = compute_principled_lead_time(
                scores, times, crisis.crisis_date,
            )

            z_traj, offsets, _ = get_score_trajectory(
                scores, times, crisis.crisis_date,
            )
            if z_traj is not None:
                crisis_trajs["Random Forest"] = {
                    'z_trajectory': z_traj.tolist(),
                    'offsets': offsets.tolist(),
                }
        except Exception as e:
            logger.error(f"  RF failed: {e}")
            crisis_leads["Random Forest"] = None

        results[crisis.name] = crisis_leads
        trajectories[crisis.name] = crisis_trajs

        # Print summary for this crisis
        print(f"\n  Lead times for {crisis.name}:")
        for name, lead in sorted(crisis_leads.items(), key=lambda x: x[1] or 999):
            lead_str = f"{lead} days" if lead is not None else "N/A"
            print(f"    {name:<25s}: {lead_str}")

    return results, trajectories


def compute_lead_time_statistics(results: Dict) -> Dict[str, Any]:
    """Compute aggregate statistics and Wilcoxon signed-rank test."""
    method_leads = {}  # method -> list of lead times across crises

    for crisis_name, crisis_results in results.items():
        for method_name, lead in crisis_results.items():
            if method_name not in method_leads:
                method_leads[method_name] = {}
            method_leads[method_name][crisis_name] = lead

    # Aggregate by method category
    stats_by_method = {}
    for method, per_crisis in method_leads.items():
        leads = [v for v in per_crisis.values() if v is not None]
        stats_by_method[method] = {
            'mean_lead': float(np.mean(leads)) if leads else None,
            'median_lead': float(np.median(leads)) if leads else None,
            'n_fires': len(leads),
            'n_crises': len(per_crisis),
            'fire_rate': len(leads) / len(per_crisis) if per_crisis else 0,
            'per_crisis': {k: v for k, v in per_crisis.items()},
        }

    # Category aggregates
    qcml_leads_all = []
    rf_leads_all = []
    for crisis_name in results:
        qcml_best = None
        for m in QCML_METHODS:
            val = results[crisis_name].get(m)
            if val is not None:
                if qcml_best is None or val < qcml_best:
                    qcml_best = val
        rf_val = results[crisis_name].get("Random Forest")

        if qcml_best is not None and rf_val is not None:
            qcml_leads_all.append(qcml_best)
            rf_leads_all.append(rf_val)

    # Paired Wilcoxon signed-rank test on QCML vs RF lead times
    wilcoxon_result = None
    if len(qcml_leads_all) >= 5:
        try:
            stat, p_val = stats.wilcoxon(
                qcml_leads_all, rf_leads_all, alternative='greater',
            )
            wilcoxon_result = {
                'statistic': float(stat),
                'p_value': float(p_val),
                'n_pairs': len(qcml_leads_all),
                'qcml_mean': float(np.mean(qcml_leads_all)),
                'rf_mean': float(np.mean(rf_leads_all)),
                'qcml_leads': qcml_leads_all,
                'rf_leads': rf_leads_all,
            }
        except Exception as e:
            logger.warning(f"Wilcoxon test failed: {e}")

    # Novel vs conventional crisis breakdown
    novel_qcml_leads = []
    novel_rf_leads = []
    conv_qcml_leads = []
    conv_rf_leads = []

    for crisis_name in results:
        best_qcml = None
        for m in QCML_METHODS:
            val = results[crisis_name].get(m)
            if val is not None:
                if best_qcml is None or val < best_qcml:
                    best_qcml = val
        rf_val = results[crisis_name].get("Random Forest")

        if crisis_name in NOVEL_CRISES:
            if best_qcml is not None:
                novel_qcml_leads.append(best_qcml)
            if rf_val is not None:
                novel_rf_leads.append(rf_val)
        elif crisis_name in CONVENTIONAL_CRISES:
            if best_qcml is not None:
                conv_qcml_leads.append(best_qcml)
            if rf_val is not None:
                conv_rf_leads.append(rf_val)

    return {
        'per_method': stats_by_method,
        'wilcoxon_qcml_vs_rf': wilcoxon_result,
        'novel_crises': {
            'qcml_mean_lead': float(np.mean(novel_qcml_leads)) if novel_qcml_leads else None,
            'rf_mean_lead': float(np.mean(novel_rf_leads)) if novel_rf_leads else None,
            'qcml_n': len(novel_qcml_leads),
            'rf_n': len(novel_rf_leads),
        },
        'conventional_crises': {
            'qcml_mean_lead': float(np.mean(conv_qcml_leads)) if conv_qcml_leads else None,
            'rf_mean_lead': float(np.mean(conv_rf_leads)) if conv_rf_leads else None,
            'qcml_n': len(conv_qcml_leads),
            'rf_n': len(conv_rf_leads),
        },
    }


def generate_figures(results: Dict, trajectories: Dict, output_dir: Path):
    """Generate publication-quality figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 150,
    })

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    top_qcml = ["Berry Phase Rate", "QFI Determinant", "Multi-Lag Fidelity"]
    colors = {
        "Berry Phase Rate": "#1f77b4",
        "QFI Determinant": "#2ca02c",
        "Multi-Lag Fidelity": "#ff7f0e",
        "Random Forest": "#d62728",
    }

    # 1. Signal trajectory figures for each crisis
    for crisis_name, trajs in trajectories.items():
        fig, ax = plt.subplots(figsize=(10, 5))

        for method_name in top_qcml + ["Random Forest"]:
            if method_name not in trajs:
                continue
            traj = trajs[method_name]
            offsets = np.array(traj['offsets'])
            z_vals = np.array(traj['z_trajectory'])
            color = colors.get(method_name, '#888888')
            ax.plot(offsets, z_vals, label=method_name, color=color,
                    linewidth=1.5, alpha=0.85)

        ax.axhline(y=2.0, color='gray', linestyle='--', linewidth=0.8,
                    label='z = 2.0 threshold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1.0,
                    alpha=0.5, label='Crisis date')
        ax.set_xlabel('Trading days relative to crisis')
        ax.set_ylabel('Expanding z-score')
        ax.set_title(f'Signal trajectories: {crisis_name.replace("_", " ").title()}')
        ax.legend(loc='upper left', fontsize=8)
        ax.set_xlim(-60, 30)

        fig.tight_layout()
        fig.savefig(fig_dir / f"signal_trajectory_{crisis_name}.pdf",
                    bbox_inches='tight')
        plt.close(fig)

    # 2. Lead time comparison bar chart
    # Compute mean lead time by category
    categories = {'QCML': [], 'Classical': [], 'RF': []}
    for crisis_name, crisis_results in results.items():
        for method_name, lead in crisis_results.items():
            if lead is None:
                continue
            if method_name in QCML_METHODS:
                categories['QCML'].append(lead)
            elif method_name == "Random Forest":
                categories['RF'].append(lead)
            else:
                categories['Classical'].append(lead)

    fig, ax = plt.subplots(figsize=(8, 5))
    cat_names = ['QCML\n(unsupervised)', 'Classical\n(unsupervised)', 'Random Forest\n(supervised)']
    cat_means = [
        np.mean(categories['QCML']) if categories['QCML'] else 0,
        np.mean(categories['Classical']) if categories['Classical'] else 0,
        np.mean(categories['RF']) if categories['RF'] else 0,
    ]
    cat_sems = [
        np.std(categories['QCML']) / np.sqrt(len(categories['QCML'])) if categories['QCML'] else 0,
        np.std(categories['Classical']) / np.sqrt(len(categories['Classical'])) if categories['Classical'] else 0,
        np.std(categories['RF']) / np.sqrt(len(categories['RF'])) if categories['RF'] else 0,
    ]
    bar_colors = ['#1f77b4', '#aec7e8', '#d62728']

    bars = ax.bar(cat_names, cat_means, yerr=cat_sems, capsize=5,
                  color=bar_colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Mean lead time (trading days)')
    ax.set_title('Early warning lead time by method category')

    for bar, mean in zip(bars, cat_means):
        if mean > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{mean:.1f}', ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    fig.savefig(fig_dir / "lead_time_comparison.pdf", bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Figures saved to {fig_dir}")


def main():
    parser = argparse.ArgumentParser(description="Lead Time / Early Warning Analysis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=str,
        default="experiments/outputs/regime_detection/lead_time",
    )
    args = parser.parse_args()

    load_dotenv(project_root / '.env')
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    crises = DATA_AVAILABLE_CRISES

    print("=" * 70)
    print("LEAD TIME / EARLY WARNING ANALYSIS")
    print("=" * 70)
    print(f"Crises: {[c.name for c in crises]}")
    print(f"Seed: {args.seed}")
    print("=" * 70)

    results, trajectories = run_lead_time_analysis(crises, seed=args.seed)

    # Compute statistics
    lead_stats = compute_lead_time_statistics(results)

    # Save results
    output = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'experiment': 'lead_time_analysis',
        'parameters': {'seed': args.seed, 'z_threshold': 2.0, 'min_consecutive': 2},
        'lead_times': results,
        'statistics': lead_stats,
    }

    results_path = output_dir / "lead_time_results.json"
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    # Generate figures
    generate_figures(results, trajectories, output_dir)

    # Print summary
    print(f"\n{'='*70}")
    print("LEAD TIME SUMMARY")
    print(f"{'='*70}")

    if lead_stats.get('wilcoxon_qcml_vs_rf'):
        w = lead_stats['wilcoxon_qcml_vs_rf']
        print(f"QCML mean best lead: {w['qcml_mean']:.1f} days")
        print(f"RF mean lead: {w['rf_mean']:.1f} days")
        print(f"Wilcoxon p-value: {w['p_value']:.4f} (n={w['n_pairs']})")

    novel = lead_stats.get('novel_crises', {})
    conv = lead_stats.get('conventional_crises', {})
    if novel.get('qcml_mean_lead') is not None:
        print(f"\nNovel crises: QCML={novel['qcml_mean_lead']:.1f}d, RF={novel.get('rf_mean_lead', 'N/A')}")
    if conv.get('qcml_mean_lead') is not None:
        print(f"Conventional: QCML={conv['qcml_mean_lead']:.1f}d, RF={conv.get('rf_mean_lead', 'N/A')}")


if __name__ == "__main__":
    main()
