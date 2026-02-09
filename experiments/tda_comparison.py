#!/usr/bin/env python3
"""
TDA / Persistent Homology Comparison

Runs TDA-based regime detection through the same 12-crisis pipeline
and compares to QCML Chern-number-based detection.

Key analyses:
  1. TDA total persistence as regime score across all crises.
  2. Correlation between TDA Betti numbers and QCML Chern numbers.
  3. Example persistence diagrams at crisis vs calm periods.
  4. Cohen's d comparison: TDA vs QCML methods.

Usage:
    python experiments/tda_comparison.py

Author: QCML Research
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from scipy import stats
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qcml.regime.classical_baselines import (
    BaseRegimeDetector,
    QCMLChernDetector,
    BerryPhaseRateDetector,
)
from qcml.regime.tda_baseline import TDARegimeDetector
from experiments.crisis_config import (
    CRISIS_2008,
    CRISIS_2020,
    CRISIS_2022,
    DATA_AVAILABLE_CRISES,
    get_default_validation_config,
)
from experiments.regime_comparison import (
    prepare_data,
    evaluate_method,
    seed_everything,
)

logger = logging.getLogger(__name__)


def compute_betti_chern_correlation(
    X_enriched: np.ndarray,
    config,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute correlation between TDA Betti numbers and QCML Chern numbers.

    Returns:
        Dict with Spearman correlations and p-values.
    """
    # TDA Betti numbers
    tda = TDARegimeDetector(window_size=20, max_dim=1, seed=seed)
    tda.fit(X_enriched)
    betti = tda.compute_betti_numbers(X_enriched)

    # QCML Chern scores
    chern = QCMLChernDetector(
        hilbert_dim=config.hilbert_dim, window_size=config.window_size,
        n_pca_components=config.n_pca_components,
        operator_method=config.operator_method, seed=seed,
    )
    chern.fit(X_enriched)
    chern_scores = chern.compute_regime_scores(X_enriched)

    # Align valid indices
    valid = (
        ~np.isnan(betti["total_persistence"]) &
        ~np.isnan(betti["betti_1"]) &
        ~np.isnan(chern_scores)
    )

    correlations = {}
    if valid.sum() > 10:
        # Total persistence vs Chern
        r, p = stats.spearmanr(
            betti["total_persistence"][valid],
            np.abs(chern_scores[valid]),
        )
        correlations["persistence_vs_chern"] = {"r": r, "p": p}

        # Betti_1 vs |Chern|
        r, p = stats.spearmanr(
            betti["betti_1"][valid],
            np.abs(chern_scores[valid]),
        )
        correlations["betti1_vs_chern"] = {"r": r, "p": p}

    return correlations


def generate_figures(
    crisis_results: List[Dict],
    correlation_results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Generate TDA comparison figures.

    Creates:
      (a) TDA vs QCML detection comparison bar chart.
      (b) Betti vs Chern correlation scatter.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available")
        return

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # (a) Detection comparison: TDA vs QCML Berry vs QCML Chern
    crises = [r["crisis"] for r in crisis_results if "tda_d" in r]
    d_tda = [r["tda_d"] for r in crisis_results if "tda_d" in r]
    d_chern = [r.get("chern_d", np.nan) for r in crisis_results if "tda_d" in r]
    d_berry = [r.get("berry_d", np.nan) for r in crisis_results if "tda_d" in r]

    if crises:
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(crises))
        width = 0.25

        ax.bar(x - width, d_tda, width, label="TDA Persistence", color="mediumpurple")
        ax.bar(x, d_chern, width, label="QCML Chern", color="steelblue")
        ax.bar(x + width, d_berry, width, label="Berry Phase Rate", color="crimson")

        ax.set_xlabel("Crisis")
        ax.set_ylabel("Cohen's d")
        ax.set_title("TDA vs QCML: Crisis Detection Effect Sizes")
        ax.set_xticks(x)
        ax.set_xticklabels(crises, rotation=45, ha="right", fontsize=7)
        ax.axhline(0.8, color="gray", linestyle=":", alpha=0.5)
        ax.legend()

        plt.tight_layout()
        fig.savefig(output_dir / "tda_vs_qcml_comparison.pdf", bbox_inches="tight")
        fig.savefig(output_dir / "tda_vs_qcml_comparison.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # (b) Correlation summary
    if correlation_results:
        fig, ax = plt.subplots(figsize=(6, 4))
        names = list(correlation_results.keys())
        r_vals = [correlation_results[n]["r"] for n in names]
        p_vals = [correlation_results[n]["p"] for n in names]

        colors = ["steelblue" if p < 0.05 else "gray" for p in p_vals]
        bars = ax.barh(names, r_vals, color=colors)
        ax.set_xlabel("Spearman r")
        ax.set_title("TDA-QCML Correlation (blue = p<0.05)")
        ax.axvline(0, color="black", linewidth=0.5)

        for i, (r, p) in enumerate(zip(r_vals, p_vals)):
            ax.text(r + 0.02, i, f"r={r:.2f}, p={p:.3f}", va="center", fontsize=7)

        plt.tight_layout()
        fig.savefig(output_dir / "tda_qcml_correlation.pdf", bbox_inches="tight")
        fig.savefig(output_dir / "tda_qcml_correlation.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"  Figures saved to {output_dir}")


def run_tda_comparison(seed: int = 42) -> Dict[str, Any]:
    """Run TDA comparison experiment."""
    seed_everything(seed)
    config = get_default_validation_config()

    output_dir = Path("experiments/outputs/regime_detection/tda")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TDA / PERSISTENT HOMOLOGY COMPARISON")
    print("=" * 60)

    crises = DATA_AVAILABLE_CRISES
    crisis_results = []
    all_correlations = {}

    for crisis in crises:
        print(f"\n  Processing: {crisis.name}")
        X, X_enriched, times, crisis_idx = prepare_data(crisis, config)
        if X is None:
            print(f"    SKIPPED: no data")
            continue

        trim = 19  # enriched lookback - 1
        times_enriched = times[trim:]
        crisis_idx_enriched = max(0, crisis_idx - trim)

        result = {"crisis": crisis.name}

        # TDA
        try:
            tda = TDARegimeDetector(window_size=20, max_dim=1, seed=seed)
            tda.fit(X_enriched)
            eval_tda = evaluate_method(
                tda, X_enriched, times_enriched, crisis_idx_enriched,
                crisis, config, n_bootstrap=5000, n_permutations=2000, seed=seed,
            )
            result["tda_d"] = eval_tda.get("effect_size_d", float("nan"))
            result["tda_p"] = eval_tda.get("p_value", float("nan"))
        except Exception as e:
            logger.warning(f"TDA failed for {crisis.name}: {e}")
            result["tda_d"] = float("nan")

        # QCML Chern
        try:
            chern = QCMLChernDetector(
                hilbert_dim=config.hilbert_dim, window_size=config.window_size,
                n_pca_components=config.n_pca_components,
                operator_method=config.operator_method, seed=seed,
            )
            chern.fit(X_enriched)
            eval_chern = evaluate_method(
                chern, X_enriched, times_enriched, crisis_idx_enriched,
                crisis, config, n_bootstrap=5000, n_permutations=2000, seed=seed,
            )
            result["chern_d"] = eval_chern.get("effect_size_d", float("nan"))
        except Exception as e:
            result["chern_d"] = float("nan")

        # Berry Phase Rate
        try:
            berry = BerryPhaseRateDetector(
                hilbert_dim=config.hilbert_dim,
                n_pca_components=config.n_pca_components,
                operator_method=config.operator_method, seed=seed,
            )
            berry.fit(X_enriched)
            eval_berry = evaluate_method(
                berry, X_enriched, times_enriched, crisis_idx_enriched,
                crisis, config, n_bootstrap=5000, n_permutations=2000, seed=seed,
            )
            result["berry_d"] = eval_berry.get("effect_size_d", float("nan"))
        except Exception as e:
            result["berry_d"] = float("nan")

        print(f"    TDA d={result.get('tda_d', np.nan):.3f}, "
              f"Chern d={result.get('chern_d', np.nan):.3f}, "
              f"Berry d={result.get('berry_d', np.nan):.3f}")

        crisis_results.append(result)

        # Betti-Chern correlation (on representative crises only)
        if crisis.name in ["2008 Financial Crisis", "2020 COVID Crash", "2022 Rate Hike Crisis"]:
            try:
                corr = compute_betti_chern_correlation(X_enriched, config, seed)
                for k, v in corr.items():
                    key = f"{crisis.name}_{k}"
                    all_correlations[key] = v
                    print(f"    Correlation {k}: r={v['r']:.3f}, p={v['p']:.4f}")
            except Exception as e:
                logger.warning(f"Correlation failed for {crisis.name}: {e}")

    # Summary table
    print("\n" + "=" * 70)
    print("TDA vs QCML — SUMMARY")
    print("=" * 70)
    print(f"{'Crisis':<30} {'TDA d':>10} {'Chern d':>10} {'Berry d':>10}")
    print("-" * 62)

    d_tda_all = []
    d_chern_all = []
    d_berry_all = []

    for r in crisis_results:
        print(f"  {r['crisis']:<30} {r.get('tda_d', np.nan):>8.3f} "
              f"{r.get('chern_d', np.nan):>8.3f} {r.get('berry_d', np.nan):>8.3f}")
        if not np.isnan(r.get("tda_d", np.nan)):
            d_tda_all.append(r["tda_d"])
        if not np.isnan(r.get("chern_d", np.nan)):
            d_chern_all.append(r["chern_d"])
        if not np.isnan(r.get("berry_d", np.nan)):
            d_berry_all.append(r["berry_d"])

    print("-" * 62)
    if d_tda_all:
        print(f"  {'Mean':<30} {np.mean(d_tda_all):>8.3f} "
              f"{np.mean(d_chern_all):>8.3f} {np.mean(d_berry_all):>8.3f}")

    # Generate figures
    print("\nGenerating figures...")
    generate_figures(crisis_results, all_correlations, output_dir)

    # Save results
    serializable = []
    for r in crisis_results:
        sr = {k: float(v) if isinstance(v, (np.floating, float)) else v
              for k, v in r.items()}
        serializable.append(sr)

    corr_serializable = {
        k: {"r": float(v["r"]), "p": float(v["p"])}
        for k, v in all_correlations.items()
    }

    with open(output_dir / "tda_comparison_results.json", "w") as f:
        json.dump({
            "crisis_results": serializable,
            "correlations": corr_serializable,
        }, f, indent=2)

    print(f"\nAll results saved to {output_dir}")
    return {"crisis_results": crisis_results, "correlations": all_correlations}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    load_dotenv(project_root / ".env")
    run_tda_comparison()
