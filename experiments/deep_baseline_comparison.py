#!/usr/bin/env python3
"""
Deep Learning Baseline Comparison

Trains LSTM and TCN regime detectors using the same leave-one-crisis-out
protocol as the Random Forest baseline, then runs them through the
identical statistical pipeline used in regime_comparison.py.

Usage:
    python experiments/deep_baseline_comparison.py

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
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qcml.regime.deep_baselines import LSTMRegimeDetector, TCNRegimeDetector
from experiments.crisis_config import (
    DATA_AVAILABLE_CRISES,
    get_default_validation_config,
)
from experiments.regime_comparison import (
    prepare_data,
    prepare_rf_training_data,
    evaluate_method,
    seed_everything,
    _align_features,
)

logger = logging.getLogger(__name__)


def train_deep_model_loo(
    model_class,
    test_crisis,
    all_crises,
    config,
    seed: int = 42,
    **model_kwargs,
):
    """Leave-one-crisis-out training for a deep learning model.

    Uses the same data preparation pipeline as prepare_rf_training_data.
    """
    train_crises = [c for c in all_crises if c.name != test_crisis.name]
    all_X = []
    all_y = []

    for crisis in train_crises:
        X, _X_enriched, times, crisis_idx = prepare_data(crisis, config)
        if X is None:
            continue
        y = np.zeros(len(X))
        window = config.analysis_window_days
        start = max(0, crisis_idx - window)
        end = min(len(X), crisis_idx + window)
        y[start:end] = 1
        all_X.append(X)
        all_y.append(y)

    if not all_X:
        return None, 0

    all_X = _align_features(all_X)
    n_features = all_X[0].shape[1]
    X_train = np.vstack(all_X)
    y_train = np.concatenate(all_y)

    detector = model_class(seed=seed, **model_kwargs)
    detector.fit_with_labels(X_train, y_train)
    return detector, n_features


def run_deep_baseline_comparison(seed: int = 42) -> Dict[str, Any]:
    """Run LSTM and TCN through the full comparison pipeline."""
    seed_everything(seed)
    config = get_default_validation_config()

    output_dir = Path("experiments/outputs/regime_detection/deep_baselines")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DEEP LEARNING BASELINE COMPARISON")
    print("=" * 60)

    crises = DATA_AVAILABLE_CRISES
    model_configs = [
        ("LSTM", LSTMRegimeDetector, {"hidden_dim": 64, "seq_len": 20}),
        ("TCN", TCNRegimeDetector, {"hidden_dim": 64, "kernel_size": 3, "seq_len": 20}),
    ]

    all_results = {}

    for model_name, model_class, model_kwargs in model_configs:
        print(f"\n{'='*50}")
        print(f"Model: {model_name}")
        print(f"{'='*50}")

        crisis_results = []

        for crisis in crises:
            print(f"\n  Testing on: {crisis.name}")

            # Prepare test data
            X, X_enriched, times, crisis_idx = prepare_data(crisis, config)
            if X is None:
                print(f"    SKIPPED: no data")
                continue

            # Train leave-one-out
            print(f"    Training {model_name} (leave-one-out)...")
            detector, n_features = train_deep_model_loo(
                model_class, crisis, crises, config, seed, **model_kwargs
            )

            if detector is None:
                print(f"    SKIPPED: no training data")
                continue

            # Truncate test features to match training
            X_test = X[:, :n_features] if X.shape[1] > n_features else X

            # Evaluate
            print(f"    Evaluating...")
            result = evaluate_method(
                detector, X_test, times, crisis_idx, crisis, config,
                n_bootstrap=10000, n_permutations=5000, seed=seed,
            )
            result["crisis"] = crisis.name
            crisis_results.append(result)

            d = result.get("effect_size_d", float("nan"))
            p = result.get("p_value", float("nan"))
            print(f"    d={d:.3f}, p={p:.4f}")

        all_results[model_name] = crisis_results

        # Summary statistics
        if crisis_results:
            d_values = [r["effect_size_d"] for r in crisis_results
                        if not np.isnan(r.get("effect_size_d", float("nan")))]
            if d_values:
                print(f"\n  {model_name} Summary:")
                print(f"    Mean d: {np.mean(d_values):.3f}")
                print(f"    Median d: {np.median(d_values):.3f}")
                print(f"    Crises with d>0.8: {sum(1 for d in d_values if d > 0.8)}/{len(d_values)}")

    # Save results
    results_path = output_dir / "deep_baseline_results.json"

    # Convert to serializable format
    serializable = {}
    for model_name, crisis_results in all_results.items():
        serializable[model_name] = []
        for r in crisis_results:
            sr = {}
            for k, v in r.items():
                if isinstance(v, (np.floating, np.integer)):
                    sr[k] = float(v)
                elif isinstance(v, np.ndarray):
                    sr[k] = v.tolist()
                else:
                    sr[k] = v
            serializable[model_name].append(sr)

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print comparison table
    print("\n" + "=" * 70)
    print("DEEP LEARNING vs. QCML vs. RF COMPARISON")
    print("=" * 70)
    print(f"{'Model':<20} {'Mean d':>10} {'Median d':>10} {'d>0.8':>8}")
    print("-" * 50)

    for model_name, crisis_results in all_results.items():
        d_values = [r["effect_size_d"] for r in crisis_results
                    if not np.isnan(r.get("effect_size_d", float("nan")))]
        if d_values:
            print(f"{model_name:<20} {np.mean(d_values):>10.3f} "
                  f"{np.median(d_values):>10.3f} "
                  f"{sum(1 for d in d_values if d > 0.8):>5}/{len(d_values)}")

    return all_results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    load_dotenv(project_root / ".env")
    run_deep_baseline_comparison()
