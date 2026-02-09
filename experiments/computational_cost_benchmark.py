#!/usr/bin/env python3
"""Computational Cost Benchmark for Regime Detection Methods.

Times each method on standardized SPY data (1 year, ~252 trading days)
to produce a per-method cost comparison for the paper.

Outputs JSON with wall-clock times and relative costs to
experiments/outputs/regime_detection/timing/.

Usage:
    python experiments/computational_cost_benchmark.py
    python experiments/computational_cost_benchmark.py --n-repeats 5
"""

import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from qcml_geometry import (
    BerryPhaseRateDetector,
    QFIDeterminantDetector,
    MultiLagFidelityDetector,
)
from experiments.baselines import (
    RollingVolatilityDetector,
    CUSUMDetector,
    HMMRegimeDetector,
    RandomForestRegimeDetector,
)
from experiments.data import PolygonDataSource, MinimalFeatureEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="Metric tensor has negative eigenvalue")

OUTPUT_DIR = "experiments/outputs/regime_detection/timing"


def fetch_benchmark_data(n_pca: int = 15) -> np.ndarray:
    """Fetch 1 year of multi-symbol data and return PCA-transformed feature matrix.

    Uses ["SPY", "XLF", "QQQ", "TLT"] to match the multi-symbol feature
    engineering used in the actual experiments.
    """
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise ValueError("POLYGON_API_KEY not found in environment")

    symbols = ["SPY", "XLF", "QQQ", "TLT"]
    source = PolygonDataSource(api_key=api_key)
    raw = source.fetch_equities(symbols, start_date="2005-01-01", end_date="2024-12-31")

    prices = raw["close"].unstack(level=0)
    prices = prices.ffill()

    engine = MinimalFeatureEngine(window=20)
    features = engine.create_feature_matrix(prices, benchmark_col="SPY")
    features = features.dropna()
    X_raw = features.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    n_components = min(n_pca, X_raw.shape[1])
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(X_scaled)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    logger.info(f"Benchmark data: {X.shape[0]} time steps, {X.shape[1]} PCA components")
    return X


def time_method(
    name: str,
    fit_fn,
    score_fn,
    n_repeats: int = 3,
) -> Dict:
    """Time a method's fit + score pipeline.

    Returns dict with median/mean/std wall-clock times.
    """
    times = []
    for i in range(n_repeats):
        t0 = time.perf_counter()
        fit_fn()
        score_fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return {
        "name": name,
        "times_s": times,
        "median_s": float(np.median(times)),
        "mean_s": float(np.mean(times)),
        "std_s": float(np.std(times)),
    }


def build_rf_labels(X: np.ndarray, crisis_frac: float = 0.1) -> np.ndarray:
    """Create synthetic labels for RF training: top crisis_frac of vol = crisis."""
    returns = np.diff(X[:, 0])
    vol = np.abs(returns)
    threshold = np.percentile(vol, 100 * (1 - crisis_frac))
    labels = np.zeros(len(X))
    labels[1:] = (vol >= threshold).astype(float)
    return labels


def run_benchmark(n_repeats: int = 3) -> List[Dict]:
    """Run timing benchmark for all methods."""
    X = fetch_benchmark_data(n_pca=15)
    T = X.shape[0]
    results = []

    # --- Classical baselines ---

    # Rolling Vol Z
    det = RollingVolatilityDetector()
    r = time_method(
        "Rolling Vol Z",
        lambda: det.fit(X),
        lambda: det.compute_regime_scores(X),
        n_repeats=n_repeats,
    )
    r["category"] = "Classical"
    r["T"] = T
    results.append(r)

    # CUSUM
    det = CUSUMDetector()
    r = time_method(
        "CUSUM",
        lambda: det.fit(X),
        lambda: det.compute_regime_scores(X),
        n_repeats=n_repeats,
    )
    r["category"] = "Classical"
    r["T"] = T
    results.append(r)

    # HMM
    det = HMMRegimeDetector()
    r = time_method(
        "HMM 2-state",
        lambda: det.fit(X),
        lambda: det.compute_regime_scores(X),
        n_repeats=n_repeats,
    )
    r["category"] = "Classical"
    r["T"] = T
    results.append(r)

    # Random Forest
    labels = build_rf_labels(X)
    det = RandomForestRegimeDetector()
    r = time_method(
        "Random Forest",
        lambda: det.fit_with_labels(X, labels),
        lambda: det.compute_regime_scores(X),
        n_repeats=n_repeats,
    )
    r["category"] = "Classical"
    r["T"] = T
    results.append(r)

    # --- QCML methods at h=8 ---
    for hdim in [8, 12]:
        det = BerryPhaseRateDetector(
            hilbert_dim=hdim,
            n_pca_components=15,
            operator_method="pca_inspired",
            rolling_window=20,
        )
        r = time_method(
            f"Berry Phase Rate (h={hdim})",
            lambda: det.fit(X),
            lambda: det.compute_regime_scores(X),
            n_repeats=n_repeats,
        )
        r["category"] = "QCML"
        r["hilbert_dim"] = hdim
        r["T"] = T
        results.append(r)

        det = QFIDeterminantDetector(
            hilbert_dim=hdim,
            n_pca_components=15,
            operator_method="pca_inspired",
            rolling_window=20,
        )
        r = time_method(
            f"QFI Determinant (h={hdim})",
            lambda: det.fit(X),
            lambda: det.compute_regime_scores(X),
            n_repeats=n_repeats,
        )
        r["category"] = "QCML"
        r["hilbert_dim"] = hdim
        r["T"] = T
        results.append(r)

        det = MultiLagFidelityDetector(
            hilbert_dim=hdim,
            n_pca_components=15,
            operator_method="pca_inspired",
            rolling_window=20,
        )
        r = time_method(
            f"Multi-Lag Fidelity (h={hdim})",
            lambda: det.fit(X),
            lambda: det.compute_regime_scores(X),
            n_repeats=n_repeats,
        )
        r["category"] = "QCML"
        r["hilbert_dim"] = hdim
        r["T"] = T
        results.append(r)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Computational cost benchmark")
    parser.add_argument("--n-repeats", type=int, default=3, help="Timing repeats per method")
    args = parser.parse_args()

    logger.info("=== Computational Cost Benchmark ===")
    results = run_benchmark(n_repeats=args.n_repeats)

    # Compute relative cost vs Rolling Vol Z
    base_time = next(r["median_s"] for r in results if r["name"] == "Rolling Vol Z")
    for r in results:
        r["relative_cost"] = round(r["median_s"] / base_time, 1) if base_time > 0 else None
        r["time_per_point_ms"] = round(1000 * r["median_s"] / r["T"], 3)

    # Print summary
    logger.info("")
    logger.info(f"{'Method':<30} {'Time (s)':>10} {'Rel. Cost':>10} {'ms/point':>10} {'Category'}")
    logger.info("-" * 75)
    for r in results:
        logger.info(
            f"{r['name']:<30} {r['median_s']:>10.3f} {r['relative_cost']:>9.1f}x "
            f"{r['time_per_point_ms']:>9.3f} {r['category']}"
        )

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = os.path.join(OUTPUT_DIR, f"timing_benchmark_{timestamp}.json")
    with open(outpath, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "n_repeats": args.n_repeats,
                "results": results,
            },
            f,
            indent=2,
        )
    logger.info(f"\nResults saved to {outpath}")

    # Also save a latest symlink-style file
    latest_path = os.path.join(OUTPUT_DIR, "timing_benchmark_latest.json")
    with open(latest_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "n_repeats": args.n_repeats,
                "results": results,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
