#!/usr/bin/env python3
"""
Train Learned Operators Experiment

Leave-one-crisis-out: learn Hermitian operators A_k on 11 crises to
maximize Cohen's d, then test on the 12th held-out crisis.  Compare
d_learned vs d_pca for each held-out crisis.

Usage:
    python experiments/train_learned_operators.py

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

from qcml.qcml_geometry import QCMLGeometry
from qcml.regime.classical_baselines import (
    BaseRegimeDetector,
    BerryPhaseRateDetector,
)
from experiments.crisis_config import (
    DATA_AVAILABLE_CRISES,
    get_default_validation_config,
)
from experiments.regime_comparison import (
    prepare_data,
    evaluate_method,
    seed_everything,
    _align_features,
)

logger = logging.getLogger(__name__)


class LearnedOperatorDetector(BaseRegimeDetector):
    """Detector using learned (not PCA-based) Hermitian operators.

    Wraps the learned operators from LearnedOperatorQCML and uses them
    to compute Berry curvature regime scores.
    """

    def __init__(self, operators, hilbert_dim: int = 8, seed: int = 42):
        self._operators = operators  # list of numpy arrays
        self.hilbert_dim = hilbert_dim
        self.seed = seed
        self._geometry = None

    @property
    def name(self) -> str:
        return "Learned Operators"

    def fit(self, X: np.ndarray, **kwargs) -> 'LearnedOperatorDetector':
        # Create geometry with learned operators
        d = X.shape[1]
        self._geometry = QCMLGeometry(n_features=d, hilbert_dim=self.hilbert_dim)
        # Override PCA operators with learned ones
        self._geometry.operators = [
            self._operators[k][:self.hilbert_dim, :self.hilbert_dim]
            for k in range(min(len(self._operators), d))
        ]
        # Pad if needed
        while len(self._geometry.operators) < d:
            self._geometry.operators.append(
                np.eye(self.hilbert_dim) * 0.01
            )
        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._geometry is None:
            raise RuntimeError("Call fit() first")
        scores = np.empty(len(X))
        for t in range(len(X)):
            try:
                scores[t] = abs(self._geometry.berry_curvature_2d(X[t], indices=(0, 1)))
            except Exception:
                scores[t] = np.nan
        return scores


def run_learned_operators(seed: int = 42) -> Dict[str, Any]:
    """Run leave-one-crisis-out learned operator experiment."""
    seed_everything(seed)
    config = get_default_validation_config()

    output_dir = Path("experiments/outputs/regime_detection/learned_ops")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("END-TO-END LEARNED OPERATORS EXPERIMENT")
    print("=" * 60)

    try:
        import torch
        from qcml.learned_operators import LearnedOperatorQCML, train_learned_operators
    except ImportError:
        print("PyTorch not available. Skipping learned operators experiment.")
        return {}

    crises = DATA_AVAILABLE_CRISES
    results = []

    for test_crisis in crises:
        print(f"\n{'='*50}")
        print(f"Hold-out: {test_crisis.name}")
        print(f"{'='*50}")

        # Prepare test data
        X_test, X_enriched_test, times_test, crisis_idx_test = prepare_data(test_crisis, config)
        if X_test is None:
            print("  SKIPPED: no test data")
            continue

        # Prepare training data (all other crises)
        train_crises = [c for c in crises if c.name != test_crisis.name]
        all_X = []
        all_y = []
        for c in train_crises:
            X, _Xe, t, ci = prepare_data(c, config)
            if X is None:
                continue
            y = np.zeros(len(X))
            w = config.analysis_window_days
            y[max(0, ci - w):min(len(X), ci + w)] = 1
            all_X.append(X)
            all_y.append(y)

        if not all_X:
            print("  SKIPPED: no training data")
            continue

        all_X = _align_features(all_X)
        n_features = all_X[0].shape[1]
        X_train = np.vstack(all_X)
        y_train = np.concatenate(all_y)

        # Train learned operators
        print(f"  Training learned operators (d={n_features}, n={config.hilbert_dim})...")
        model = LearnedOperatorQCML(n_features=n_features, hilbert_dim=config.hilbert_dim)
        history = train_learned_operators(
            model, X_train, y_train,
            n_epochs=300, lr=1e-3, batch_size=128, seed=seed,
        )
        final_d = history["proxy_d"][-1] if history["proxy_d"] else 0.0
        print(f"  Training proxy d: {final_d:.3f}")

        # Extract learned operators
        learned_ops = model.get_numpy_operators()

        # Evaluate learned operator detector
        learned_det = LearnedOperatorDetector(
            learned_ops, hilbert_dim=config.hilbert_dim, seed=seed
        )

        # Build enriched features for test
        X_test_aligned = X_test[:, :n_features] if X_test.shape[1] > n_features else X_test
        X_enriched_aligned = BaseRegimeDetector.build_enriched_features(X_test_aligned, lookback=20)
        times_enriched = times_test[19:]
        crisis_idx_enriched = max(0, crisis_idx_test - 19)

        learned_det.fit(X_enriched_aligned)
        result_learned = evaluate_method(
            learned_det, X_enriched_aligned, times_enriched, crisis_idx_enriched,
            test_crisis, config, n_bootstrap=5000, n_permutations=2000, seed=seed,
        )

        # PCA baseline (same data)
        pca_det = BerryPhaseRateDetector(
            hilbert_dim=config.hilbert_dim,
            n_pca_components=config.n_pca_components,
            operator_method=config.operator_method, seed=seed,
        )
        pca_det.fit(X_enriched_aligned)
        result_pca = evaluate_method(
            pca_det, X_enriched_aligned, times_enriched, crisis_idx_enriched,
            test_crisis, config, n_bootstrap=5000, n_permutations=2000, seed=seed,
        )

        d_learned = result_learned.get("effect_size_d", float("nan"))
        d_pca = result_pca.get("effect_size_d", float("nan"))

        print(f"  d_learned={d_learned:.3f}, d_pca={d_pca:.3f}")

        results.append({
            "crisis": test_crisis.name,
            "d_learned": float(d_learned),
            "d_pca": float(d_pca),
            "training_proxy_d": float(final_d),
            "n_epochs_trained": len(history["loss"]),
        })

        # Save operators for this fold
        ops_path = output_dir / f"operators_{test_crisis.name.replace(' ', '_')}.npz"
        np.savez(ops_path, **{f"A_{k}": op for k, op in enumerate(learned_ops)})

    # Summary
    print("\n" + "=" * 70)
    print("LEARNED vs PCA OPERATORS — SUMMARY")
    print("=" * 70)
    print(f"{'Crisis':<25} {'d_learned':>12} {'d_pca':>12} {'Δd':>10}")
    print("-" * 60)

    d_learned_all = []
    d_pca_all = []
    for r in results:
        delta = r["d_learned"] - r["d_pca"]
        print(f"  {r['crisis']:<25} {r['d_learned']:>10.3f} {r['d_pca']:>10.3f} {delta:>+8.3f}")
        if not np.isnan(r["d_learned"]):
            d_learned_all.append(r["d_learned"])
        if not np.isnan(r["d_pca"]):
            d_pca_all.append(r["d_pca"])

    if d_learned_all and d_pca_all:
        print("-" * 60)
        print(f"  {'Mean':<25} {np.mean(d_learned_all):>10.3f} {np.mean(d_pca_all):>10.3f} "
              f"{np.mean(d_learned_all) - np.mean(d_pca_all):>+8.3f}")

        # Paired t-test
        from scipy.stats import ttest_rel
        pairs = [(r["d_learned"], r["d_pca"]) for r in results
                 if not np.isnan(r["d_learned"]) and not np.isnan(r["d_pca"])]
        if len(pairs) >= 3:
            t_stat, p_val = ttest_rel([p[0] for p in pairs], [p[1] for p in pairs])
            print(f"\n  Paired t-test: t={t_stat:.3f}, p={p_val:.4f}")

    # Save results
    with open(output_dir / "learned_operator_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    return {"results": results}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    load_dotenv(project_root / ".env")
    run_learned_operators()
