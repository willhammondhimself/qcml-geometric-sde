#!/usr/bin/env python3
"""Causal vs Non-Causal Comparison for QCML Regime Detectors.

Quantifies lookahead bias by comparing detector performance when
PCA/scaler/operators are fit on:
  (a) full data including crisis period (non-causal, current default)
  (b) pre-crisis data only (causal — no future information)

For each crisis and method, reports Cohen's d under both modes and
the inflation factor.

Usage:
    python experiments/causal_comparison.py
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd

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
from qcml.regime.classical_baselines import (
    BerryPhaseRateDetector,
    MultiLagFidelityDetector,
    QFIDeterminantDetector,
    RandomForestRegimeDetector,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore", message="Metric tensor has negative eigenvalue")

OUTPUT_DIR = "experiments/outputs/regime_detection/causal_comparison"

# Improved configs from extended sensitivity analysis
IMPROVED_CONFIGS = {
    "Berry Phase Rate": {
        "DetectorClass": BerryPhaseRateDetector,
        "hilbert_dim": 8,
        "n_pca_components": 10,
        "operator_method": "pca_inspired",
        "rolling_window": 10,
    },
    "QFI Determinant": {
        "DetectorClass": QFIDeterminantDetector,
        "hilbert_dim": 4,
        "n_pca_components": 15,
        "operator_method": "random",
        "rolling_window": 30,
    },
    "Multi-Lag Fidelity": {
        "DetectorClass": MultiLagFidelityDetector,
        "hilbert_dim": 8,
        "n_pca_components": 15,
        "operator_method": "pca_inspired",
        "rolling_window": 20,
    },
}


def run_comparison():
    """Run causal vs non-causal comparison on all 12 crises."""
    seed_everything(42)
    config = get_default_validation_config()
    n_bootstrap = 100
    n_permutations = 100

    crises = DATA_AVAILABLE_CRISES
    results = {}

    logger.info("=" * 70)
    logger.info("CAUSAL vs NON-CAUSAL COMPARISON")
    logger.info("=" * 70)

    for crisis in crises:
        crisis_name = crisis.name
        logger.info(f"\n--- {crisis_name} ---")

        X, X_enriched, times, crisis_idx = prepare_data(
            crisis, config, enriched_lookback=20,
        )
        if X is None:
            logger.warning(f"  Skipping {crisis_name}: no data")
            continue

        trim = 19
        times_enriched = times[trim:]
        crisis_idx_enriched = max(0, crisis_idx - trim)

        logger.info(
            f"  T={len(X)}, enriched={X_enriched.shape}, "
            f"crisis_idx_enriched={crisis_idx_enriched}/{len(X_enriched)}"
        )

        crisis_results = {}

        for method_name, cfg in IMPROVED_CONFIGS.items():
            DetectorClass = cfg["DetectorClass"]
            det_kwargs = {
                k: v for k, v in cfg.items() if k != "DetectorClass"
            }

            # --- Non-causal (default) ---
            det_nc = DetectorClass(**det_kwargs, seed=42, causal_fit_length=None)
            det_nc.fit(X_enriched)
            res_nc = evaluate_method(
                det_nc, X_enriched, times_enriched, crisis_idx_enriched,
                crisis, config,
                n_bootstrap=n_bootstrap, n_permutations=n_permutations, seed=42,
            )
            d_nc = res_nc.get("effect_size_d", np.nan)

            # --- Causal (fit on pre-crisis only) ---
            det_c = DetectorClass(
                **det_kwargs, seed=42,
                causal_fit_length=crisis_idx_enriched,
            )
            det_c.fit(X_enriched)
            res_c = evaluate_method(
                det_c, X_enriched, times_enriched, crisis_idx_enriched,
                crisis, config,
                n_bootstrap=n_bootstrap, n_permutations=n_permutations, seed=42,
            )
            d_c = res_c.get("effect_size_d", np.nan)

            inflation = (d_nc - d_c) / d_c * 100 if d_c > 0.01 else np.nan

            crisis_results[method_name] = {
                "d_noncausal": float(d_nc),
                "d_causal": float(d_c),
                "inflation_pct": float(inflation) if not np.isnan(inflation) else None,
            }

            logger.info(
                f"  {method_name}: non-causal d={d_nc:.3f}, "
                f"causal d={d_c:.3f}, inflation={inflation:+.1f}%"
                if not np.isnan(inflation)
                else f"  {method_name}: non-causal d={d_nc:.3f}, "
                     f"causal d={d_c:.3f}, inflation=N/A"
            )

        # RF baseline (no causal issue — supervised)
        try:
            from experiments.regime_comparison import prepare_rf_training_data
            rf_X, rf_y, rf_n_features = prepare_rf_training_data(
                crisis, crises, config,
            )
            rf = RandomForestRegimeDetector(
                n_estimators=200, max_depth=6, seed=42, lookback=20,
            )
            rf.fit_with_labels(rf_X, rf_y)
            X_rf_test = X[:, :rf_n_features] if X.shape[1] > rf_n_features else X
            res_rf = evaluate_method(
                rf, X_rf_test, times, crisis_idx,
                crisis, config,
                n_bootstrap=n_bootstrap, n_permutations=n_permutations, seed=42,
            )
            crisis_results["Random Forest"] = {
                "d_noncausal": float(res_rf.get("effect_size_d", np.nan)),
                "d_causal": float(res_rf.get("effect_size_d", np.nan)),
                "inflation_pct": 0.0,
            }
        except Exception as e:
            logger.warning(f"  RF failed: {e}")

        results[crisis_name] = crisis_results

    return results


def format_results(results):
    """Print summary table and compute aggregates."""
    lines = [
        "=" * 100,
        "CAUSAL vs NON-CAUSAL COMPARISON RESULTS",
        "=" * 100,
        "",
    ]

    method_names = list(IMPROVED_CONFIGS.keys()) + ["Random Forest"]

    # Per-crisis table
    for crisis_name, crisis_results in results.items():
        lines.append(f"--- {crisis_name} ---")
        header = f"  {'Method':<25s} {'d(nc)':>7s}  {'d(causal)':>9s}  {'inflation':>10s}"
        lines.append(header)
        lines.append("  " + "-" * 60)
        for mn in method_names:
            if mn not in crisis_results:
                continue
            r = crisis_results[mn]
            inf_str = (
                f"{r['inflation_pct']:+.1f}%"
                if r['inflation_pct'] is not None
                else "N/A"
            )
            lines.append(
                f"  {mn:<25s} {r['d_noncausal']:>7.3f}  "
                f"{r['d_causal']:>9.3f}  {inf_str:>10s}"
            )
        lines.append("")

    # Aggregate summary
    lines.append("=" * 100)
    lines.append("AGGREGATE SUMMARY")
    lines.append("=" * 100)

    header = (
        f"  {'Method':<25s} {'Avg d(nc)':>9s}  {'Avg d(c)':>9s}  "
        f"{'Med d(nc)':>9s}  {'Med d(c)':>9s}  {'Avg Infl':>9s}"
    )
    lines.append(header)
    lines.append("  " + "-" * 80)

    for mn in method_names:
        d_nc_list = []
        d_c_list = []
        inf_list = []
        for crisis_results in results.values():
            if mn in crisis_results:
                r = crisis_results[mn]
                d_nc_list.append(r["d_noncausal"])
                d_c_list.append(r["d_causal"])
                if r["inflation_pct"] is not None:
                    inf_list.append(r["inflation_pct"])

        avg_nc = np.mean(d_nc_list) if d_nc_list else 0.0
        avg_c = np.mean(d_c_list) if d_c_list else 0.0
        med_nc = np.median(d_nc_list) if d_nc_list else 0.0
        med_c = np.median(d_c_list) if d_c_list else 0.0
        avg_inf = np.mean(inf_list) if inf_list else 0.0

        lines.append(
            f"  {mn:<25s} {avg_nc:>9.3f}  {avg_c:>9.3f}  "
            f"{med_nc:>9.3f}  {med_c:>9.3f}  {avg_inf:>+8.1f}%"
        )

    lines.append("")
    lines.append("=" * 100)
    lines.append("INTERPRETATION")
    lines.append("=" * 100)

    # Compute overall inflation
    all_infs = []
    for crisis_results in results.values():
        for mn in IMPROVED_CONFIGS:
            if mn in crisis_results:
                r = crisis_results[mn]
                if r["inflation_pct"] is not None:
                    all_infs.append(r["inflation_pct"])

    if all_infs:
        mean_inf = np.mean(all_infs)
        med_inf = np.median(all_infs)
        lines.append(f"  Mean inflation across all methods/crises: {mean_inf:+.1f}%")
        lines.append(f"  Median inflation: {med_inf:+.1f}%")
        if abs(mean_inf) < 10:
            lines.append("  => SMALL lookahead effect (<10%): representation is robust")
        elif abs(mean_inf) < 20:
            lines.append("  => MODERATE lookahead effect (10-20%): results are directionally correct but inflated")
        else:
            lines.append("  => LARGE lookahead effect (>20%): results are materially inflated by lookahead")

    return "\n".join(lines)


def main():
    t0 = time.time()
    results = run_comparison()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save JSON
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(OUTPUT_DIR, f"causal_comparison_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Format and save table
    table = format_results(results)
    table_path = os.path.join(OUTPUT_DIR, f"causal_comparison_table_{ts}.txt")
    with open(table_path, "w") as f:
        f.write(table)

    print(table)
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Results: {json_path}")


if __name__ == "__main__":
    main()
