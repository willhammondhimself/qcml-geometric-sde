#!/usr/bin/env python3
"""
Topological Noise Robustness Test

Tests the central thesis that Chern-number-based detection is
topologically protected against noise, while statistical methods degrade.

For each SNR level {20, 10, 5, 3, 2, 1}:
  1. Add Gaussian noise to the feature matrices at that SNR.
  2. Re-run all detectors (QCML + classical + DL) on noisy data.
  3. Compute degradation: Δd = d_clean - d_noisy per method.
  4. Linear regression of degradation slope per method.

Expected result: QCML Chern/Berry methods degrade <20% at SNR=3;
statistical methods degrade >50%.

Usage:
    python experiments/noise_robustness.py

Author: QCML Research
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qcml.regime.classical_baselines import (
    BaseRegimeDetector,
    QCMLChernDetector,
    RollingVolatilityDetector,
    CUSUMDetector,
    HMMRegimeDetector,
    RandomForestRegimeDetector,
    QFISusceptibilityDetector,
    QFIDeterminantDetector,
    BerryPhaseRateDetector,
    MultiLagFidelityDetector,
    MetricConditionNumberDetector,
    ScalarCurvatureDetector,
)
from experiments.crisis_config import (
    CRISIS_2008,
    CRISIS_2020,
    CRISIS_2022,
    DATA_AVAILABLE_CRISES,
    get_default_validation_config,
)
from experiments.regime_comparison import (
    prepare_data,
    prepare_rf_training_data,
    evaluate_method,
    seed_everything,
)

logger = logging.getLogger(__name__)

SNR_LEVELS = [20, 10, 5, 3, 2, 1]

# Representative crises for noise test (one from each type)
NOISE_TEST_CRISES = [CRISIS_2008, CRISIS_2020, CRISIS_2022]


def add_noise(X: np.ndarray, snr_db: float, seed: int = 42) -> np.ndarray:
    """Add Gaussian noise to feature matrix at a given SNR (in dB).

    SNR_dB = 10 * log10(signal_power / noise_power)
    => noise_power = signal_power / 10^(SNR_dB/10)

    Args:
        X: Feature matrix (T, d).
        snr_db: Signal-to-noise ratio in decibels.
        seed: Random seed.

    Returns:
        X_noisy: Noisy feature matrix (same shape).
    """
    rng = np.random.RandomState(seed)
    signal_power = np.mean(X ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.randn(*X.shape) * np.sqrt(noise_power)
    return X + noise


def build_unsupervised_detectors(config, seed: int = 42) -> List[BaseRegimeDetector]:
    """Build all unsupervised detectors."""
    detectors = [
        QCMLChernDetector(
            hilbert_dim=config.hilbert_dim, window_size=config.window_size,
            n_pca_components=config.n_pca_components,
            operator_method=config.operator_method, seed=seed,
        ),
        RollingVolatilityDetector(vol_window=20, min_expanding=252),
        CUSUMDetector(burn_in=60),
        QFISusceptibilityDetector(
            hilbert_dim=config.hilbert_dim,
            n_pca_components=config.n_pca_components,
            operator_method=config.operator_method, seed=seed,
        ),
        QFIDeterminantDetector(
            hilbert_dim=config.hilbert_dim,
            n_pca_components=config.n_pca_components,
            operator_method=config.operator_method, seed=seed,
        ),
        BerryPhaseRateDetector(
            hilbert_dim=config.hilbert_dim,
            n_pca_components=config.n_pca_components,
            operator_method=config.operator_method, seed=seed,
        ),
        MultiLagFidelityDetector(
            hilbert_dim=config.hilbert_dim,
            n_pca_components=config.n_pca_components,
            operator_method=config.operator_method, seed=seed,
        ),
        MetricConditionNumberDetector(
            hilbert_dim=config.hilbert_dim,
            n_pca_components=config.n_pca_components,
            operator_method=config.operator_method, seed=seed,
        ),
        ScalarCurvatureDetector(
            hilbert_dim=config.hilbert_dim,
            n_curvature_components=min(config.n_pca_components, 8),
            operator_method=config.operator_method, seed=seed,
        ),
    ]

    try:
        detectors.append(HMMRegimeDetector(n_iter=100, seed=seed))
    except Exception:
        pass

    return detectors


def evaluate_at_snr(
    snr_db: float,
    crisis,
    config,
    rf_detector: Optional[RandomForestRegimeDetector],
    rf_n_features: int,
    seed: int = 42,
) -> Dict[str, float]:
    """Evaluate all detectors on noisy data at a given SNR.

    Returns dict mapping method_name -> Cohen's d.
    """
    X, X_enriched, times, crisis_idx = prepare_data(crisis, config)
    if X is None:
        return {}

    # Add noise to the raw PCA features
    if snr_db < 100:  # 100 = clean
        X_noisy = add_noise(X, snr_db, seed=seed + int(snr_db * 100))
    else:
        X_noisy = X.copy()

    # Re-build enriched features from noisy data
    X_enriched_noisy = BaseRegimeDetector.build_enriched_features(X_noisy, lookback=20)

    results = {}

    # Unsupervised detectors use enriched features
    detectors = build_unsupervised_detectors(config, seed)
    for det in detectors:
        try:
            det.fit(X_enriched_noisy)
            result = evaluate_method(
                det, X_enriched_noisy, times[19:], crisis_idx - 19,
                crisis, config, n_bootstrap=1000, n_permutations=500, seed=seed,
            )
            d = result.get("effect_size_d", float("nan"))
            results[det.name] = d
        except Exception as e:
            logger.warning(f"  {det.name} failed at SNR={snr_db}: {e}")
            results[det.name] = float("nan")

    # RF (supervised — retrain is expensive, so we just score on noisy data)
    if rf_detector is not None:
        try:
            X_rf = X_noisy[:, :rf_n_features] if X_noisy.shape[1] > rf_n_features else X_noisy
            result = evaluate_method(
                rf_detector, X_rf, times, crisis_idx,
                crisis, config, n_bootstrap=1000, n_permutations=500, seed=seed,
            )
            results["Random Forest"] = result.get("effect_size_d", float("nan"))
        except Exception as e:
            logger.warning(f"  RF failed at SNR={snr_db}: {e}")

    # DL baselines (if available)
    try:
        from qcml.regime.deep_baselines import LSTMRegimeDetector, TCNRegimeDetector
        from experiments.deep_baseline_comparison import train_deep_model_loo

        for model_name, model_class, model_kwargs in [
            ("LSTM", LSTMRegimeDetector, {"hidden_dim": 64, "seq_len": 20}),
            ("TCN", TCNRegimeDetector, {"hidden_dim": 64, "kernel_size": 3, "seq_len": 20}),
        ]:
            try:
                det, n_feat = train_deep_model_loo(
                    model_class, crisis, DATA_AVAILABLE_CRISES, config, seed, **model_kwargs
                )
                if det is not None:
                    X_test = X_noisy[:, :n_feat] if X_noisy.shape[1] > n_feat else X_noisy
                    result = evaluate_method(
                        det, X_test, times, crisis_idx,
                        crisis, config, n_bootstrap=1000, n_permutations=500, seed=seed,
                    )
                    results[model_name] = result.get("effect_size_d", float("nan"))
            except Exception as e:
                logger.warning(f"  {model_name} failed at SNR={snr_db}: {e}")
    except ImportError:
        logger.info("Deep baselines not available (PyTorch not installed)")

    return results


def generate_figures(
    degradation: Dict[str, Dict[float, float]],
    output_dir: Path,
) -> None:
    """Generate noise robustness figures.

    Creates:
      (a) d vs SNR line plot for all methods
      (b) degradation heatmap (methods × SNR levels)
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

    methods = sorted(degradation.keys())
    snrs = sorted(SNR_LEVELS, reverse=True)

    # Categorize methods
    qcml_methods = [m for m in methods if m in {
        "QCML Chern", "Berry Phase Rate", "QFI Determinant",
        "Multi-Lag Fidelity", "QFI Susceptibility", "Metric Condition Number",
        "Scalar Curvature",
    }]
    stat_methods = [m for m in methods if m in {
        "Rolling Volatility", "CUSUM", "HMM", "Random Forest", "LSTM", "TCN",
    }]

    # (a) d vs SNR line plot
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap_qcml = plt.cm.Blues(np.linspace(0.4, 0.9, max(len(qcml_methods), 1)))
    cmap_stat = plt.cm.Reds(np.linspace(0.4, 0.9, max(len(stat_methods), 1)))

    for i, m in enumerate(qcml_methods):
        d_vals = [degradation[m].get(s, np.nan) for s in snrs]
        ax.plot(snrs, d_vals, "o-", color=cmap_qcml[i], label=m, linewidth=1.5)

    for i, m in enumerate(stat_methods):
        d_vals = [degradation[m].get(s, np.nan) for s in snrs]
        ax.plot(snrs, d_vals, "s--", color=cmap_stat[i], label=m, linewidth=1.5)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Cohen's d (mean across crises)")
    ax.set_title("Noise Robustness: Detection Power vs. Signal-to-Noise Ratio")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    ax.invert_xaxis()
    ax.axhline(0.8, color="gray", linestyle=":", alpha=0.5, label="d=0.8 threshold")

    plt.tight_layout()
    fig.savefig(output_dir / "noise_robustness_curves.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "noise_robustness_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # (b) Degradation heatmap
    # Compute % degradation from clean (SNR=20) baseline
    fig, ax = plt.subplots(figsize=(8, max(4, len(methods) * 0.4)))
    matrix = np.full((len(methods), len(snrs)), np.nan)

    for i, m in enumerate(methods):
        clean_d = degradation[m].get(20, np.nan)
        if np.isnan(clean_d) or clean_d == 0:
            continue
        for j, s in enumerate(snrs):
            noisy_d = degradation[m].get(s, np.nan)
            if not np.isnan(noisy_d):
                matrix[i, j] = (clean_d - noisy_d) / abs(clean_d) * 100

    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=-20, vmax=100)
    ax.set_xticks(range(len(snrs)))
    ax.set_xticklabels([str(s) for s in snrs])
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=7)
    ax.set_xlabel("SNR (dB)")
    ax.set_title("Detection Degradation (% loss from clean)")
    plt.colorbar(im, ax=ax, label="% degradation")

    plt.tight_layout()
    fig.savefig(output_dir / "noise_degradation_heatmap.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "noise_degradation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  Figures saved to {output_dir}")


def run_noise_robustness(seed: int = 42) -> Dict[str, Any]:
    """Run the full noise robustness experiment."""
    seed_everything(seed)
    config = get_default_validation_config()

    output_dir = Path("experiments/outputs/regime_detection/noise_robustness")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TOPOLOGICAL NOISE ROBUSTNESS TEST")
    print("=" * 60)

    # Train RF once on all crises (oracle) for scoring
    print("\nTraining RF baseline...")
    rf = RandomForestRegimeDetector(n_estimators=200, max_depth=6, seed=seed, lookback=20)
    rf_n_features = 0
    try:
        X_all, y_all = [], []
        for c in DATA_AVAILABLE_CRISES:
            X, _Xe, t, ci = prepare_data(c, config)
            if X is None:
                continue
            y = np.zeros(len(X))
            w = config.analysis_window_days
            y[max(0, ci - w):min(len(X), ci + w)] = 1
            X_all.append(X)
            y_all.append(y)
        if X_all:
            min_cols = min(x.shape[1] for x in X_all)
            X_all = [x[:, :min_cols] for x in X_all]
            rf.fit_with_labels(np.vstack(X_all), np.concatenate(y_all))
            rf_n_features = min_cols
            print(f"  RF trained ({rf_n_features} features)")
    except Exception as e:
        logger.error(f"RF training failed: {e}")
        rf = None

    # Run at each SNR level + clean baseline
    all_snr_levels = [100] + SNR_LEVELS  # 100 = clean
    all_results = {}

    for snr in all_snr_levels:
        snr_label = "clean" if snr == 100 else f"SNR={snr}dB"
        print(f"\n{'='*50}")
        print(f"Testing at {snr_label}")
        print(f"{'='*50}")

        snr_results = {}
        for crisis in NOISE_TEST_CRISES:
            print(f"\n  Crisis: {crisis.name}")
            crisis_d = evaluate_at_snr(snr, crisis, config, rf, rf_n_features, seed)

            for method, d_val in crisis_d.items():
                if method not in snr_results:
                    snr_results[method] = []
                snr_results[method].append(d_val)

        # Average across crises
        for method in snr_results:
            vals = [v for v in snr_results[method] if not np.isnan(v)]
            mean_d = np.mean(vals) if vals else float("nan")
            if method not in all_results:
                all_results[method] = {}
            all_results[method][snr] = mean_d
            print(f"    {method}: mean d={mean_d:.3f}")

    # Compute degradation slopes via linear regression
    print("\n" + "=" * 60)
    print("DEGRADATION SLOPES (linear regression d vs SNR)")
    print("=" * 60)

    slopes = {}
    for method, snr_d in all_results.items():
        snr_vals = []
        d_vals = []
        for s in SNR_LEVELS:
            d = snr_d.get(s, np.nan)
            if not np.isnan(d):
                snr_vals.append(s)
                d_vals.append(d)
        if len(snr_vals) >= 3:
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, _ = linregress(snr_vals, d_vals)
            slopes[method] = {
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_value**2,
                "p_value": p_value,
            }
            print(f"  {method}: slope={slope:.4f}, R²={r_value**2:.3f}, p={p_value:.4f}")

    # Clean d at SNR=3 vs clean as robustness metric
    print("\n" + "=" * 60)
    print("ROBUSTNESS: % retention at SNR=3 dB")
    print("=" * 60)

    for method, snr_d in sorted(all_results.items()):
        d_clean = snr_d.get(100, np.nan)
        d_snr3 = snr_d.get(3, np.nan)
        if not np.isnan(d_clean) and d_clean > 0 and not np.isnan(d_snr3):
            retention = d_snr3 / d_clean * 100
            print(f"  {method}: {retention:.1f}% (clean d={d_clean:.3f}, SNR3 d={d_snr3:.3f})")

    # Save results
    serializable = {}
    for method, snr_d in all_results.items():
        serializable[method] = {str(k): float(v) for k, v in snr_d.items()}

    results_out = {
        "d_by_snr": serializable,
        "slopes": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in slopes.items()},
    }

    with open(output_dir / "noise_robustness_results.json", "w") as f:
        json.dump(results_out, f, indent=2)

    # Generate figures
    generate_figures(all_results, output_dir)

    print(f"\nAll results saved to {output_dir}")
    return results_out


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    load_dotenv(project_root / ".env")
    run_noise_robustness()
