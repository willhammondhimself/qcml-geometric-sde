#!/usr/bin/env python3
"""
Crisis Narrative Case Study Figures

Generates 8-panel stacked figures for three representative crises:
  1. 2008 Lehman (fast crash)
  2. 2020 COVID (V-shape)
  3. 2022 Rates (slow regime shift)

Each figure shows: SPY price, VIX (if available), Berry curvature,
QFI determinant, Chern number, multi-lag fidelity, spectral gap, RF probability.
Key events are annotated (bankruptcy dates, Fed announcements, etc.).

Usage:
    python experiments/generate_narrative_figures.py

Author: QCML Research
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.data import PolygonDataSource, MinimalFeatureEngine
from qcml_geometry import QCMLGeometry
from qcml_geometry import (
    BaseRegimeDetector,
    BerryPhaseRateDetector,
    QFIDeterminantDetector,
    MultiLagFidelityDetector,
)
from experiments.baselines import RandomForestRegimeDetector
from experiments.additional_detectors import QCMLChernDetector
from qcml_geometry.indicators import QuantumIndicatorSuite
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
    seed_everything,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Crisis-specific annotations
# ---------------------------------------------------------------------------

CRISIS_ANNOTATIONS = {
    "2008_crisis": {
        "title": "2008 Global Financial Crisis",
        "events": [
            ("2008-03-16", "Bear Stearns\ncollapse"),
            ("2008-09-07", "Fannie/Freddie\nseizure"),
            ("2008-09-15", "Lehman\nbankruptcy"),
            ("2008-10-03", "TARP\nsigned"),
        ],
    },
    "2020_covid": {
        "title": "2020 COVID-19 Pandemic Crash",
        "events": [
            ("2020-01-20", "First US\nCOVID case"),
            ("2020-02-20", "Italy\noutbreak"),
            ("2020-03-09", "Oil price\nwar"),
            ("2020-03-16", "Fed 0% rate\ncircuit breaker"),
            ("2020-03-23", "Fed unlimited\nQE / market bottom"),
        ],
    },
    "2022_rates": {
        "title": "2022 Federal Reserve Rate Hike Regime",
        "events": [
            ("2022-01-26", "Fed hawkish\npivot"),
            ("2022-03-16", "First rate\nhike"),
            ("2022-05-04", "50bp\nhike"),
            ("2022-06-15", "75bp\nhike"),
        ],
    },
}


def fetch_crisis_data_with_prices(
    crisis, config
) -> Optional[Tuple]:
    """Fetch data and return raw prices alongside feature matrices."""
    from experiments.rigorous_crisis_validation import fetch_real_crisis_data

    try:
        dataset = fetch_real_crisis_data(crisis)
    except Exception as e:
        logger.warning(f"Failed to fetch {crisis.name}: {e}")
        return None

    X_raw = dataset.X
    times = dataset.times

    # Get raw prices for price panel
    prices = dataset.prices
    if isinstance(prices, pd.DataFrame):
        spy_prices = prices.iloc[:, 0] if prices.shape[1] > 0 else prices
    else:
        spy_prices = prices

    # Standard pipeline
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    n_components = min(config.n_pca_components, X_raw.shape[1])
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(X_scaled)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    X_enriched = BaseRegimeDetector.build_enriched_features(X, lookback=20)

    crisis_ts = pd.Timestamp(crisis.crisis_date)
    crisis_idx = int((times >= crisis_ts).argmax())

    return X, X_enriched, times, crisis_idx, spy_prices


def compute_narrative_scores(
    X: np.ndarray,
    X_enriched: np.ndarray,
    times: pd.DatetimeIndex,
    config,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Compute all indicator time series for narrative figure."""
    enriched_lookback = 20
    trim = enriched_lookback - 1
    times_enriched = times[trim:]

    scores = {}

    # 1. Berry curvature (raw, not rate-of-change)
    print("    Computing Berry curvature...")
    geometry = QCMLGeometry(n_features=X_enriched.shape[1], hilbert_dim=config.hilbert_dim)
    geometry.fit_operators(X_enriched, method=config.operator_method)

    berry = np.empty(len(X_enriched))
    for t in range(len(X_enriched)):
        berry[t] = geometry.berry_curvature_2d(X_enriched[t], indices=(0, 1))
    scores["Berry Curvature"] = berry

    # 2. QFI Determinant
    print("    Computing QFI determinant...")
    det_qfi = QFIDeterminantDetector(
        hilbert_dim=config.hilbert_dim, n_pca_components=config.n_pca_components,
        operator_method=config.operator_method, seed=seed,
    )
    det_qfi.fit(X_enriched)
    scores["QFI Determinant"] = det_qfi.compute_regime_scores(X_enriched)

    # 3. Chern number (rolling)
    print("    Computing rolling Chern number...")
    det_chern = QCMLChernDetector(
        hilbert_dim=config.hilbert_dim, window_size=config.window_size,
        n_pca_components=config.n_pca_components,
        operator_method=config.operator_method, seed=seed,
    )
    det_chern.fit(X_enriched)
    scores["Chern Number"] = det_chern.compute_regime_scores(X_enriched)

    # 4. Multi-lag fidelity
    print("    Computing multi-lag fidelity...")
    det_fid = MultiLagFidelityDetector(
        hilbert_dim=config.hilbert_dim, n_pca_components=config.n_pca_components,
        operator_method=config.operator_method, seed=seed,
    )
    det_fid.fit(X_enriched)
    scores["Multi-Lag Fidelity"] = det_fid.compute_regime_scores(X_enriched)

    # 5. Spectral gap
    print("    Computing spectral gap...")
    spec_gap = np.empty(len(X_enriched))
    for t in range(len(X_enriched)):
        spec_gap[t] = geometry.spectral_gap(X_enriched[t])
    scores["Spectral Gap"] = spec_gap

    # 6. Berry Phase Rate
    print("    Computing Berry phase rate...")
    det_berry = BerryPhaseRateDetector(
        hilbert_dim=config.hilbert_dim, n_pca_components=config.n_pca_components,
        operator_method=config.operator_method, seed=seed,
    )
    det_berry.fit(X_enriched)
    scores["Berry Phase Rate"] = det_berry.compute_regime_scores(X_enriched)

    scores["times"] = times_enriched

    return scores


def generate_crisis_figure(
    crisis_name: str,
    spy_prices: pd.Series,
    scores: Dict[str, np.ndarray],
    rf_scores: Optional[np.ndarray],
    times: pd.DatetimeIndex,
    times_enriched: pd.DatetimeIndex,
    crisis_date: str,
    output_dir: Path,
) -> None:
    """Generate 8-panel stacked figure for one crisis."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        logger.warning("matplotlib not available")
        return

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    annotations = CRISIS_ANNOTATIONS.get(crisis_name, {})
    title = annotations.get("title", crisis_name)
    events = annotations.get("events", [])

    # Panels: Price, Berry Curvature, QFI Det, Chern, Multi-Lag Fidelity,
    #         Spectral Gap, Berry Phase Rate, RF Probability
    panel_data = [
        ("SPY Price", spy_prices.values, times, "steelblue"),
        ("Berry Curvature", scores["Berry Curvature"], times_enriched, "darkorange"),
        ("QFI Determinant (z-score)", scores["QFI Determinant"], times_enriched, "darkred"),
        ("Rolling Chern Number", scores["Chern Number"], times_enriched, "purple"),
        ("Multi-Lag Fidelity (z-score)", scores["Multi-Lag Fidelity"], times_enriched, "teal"),
        ("Spectral Gap $\\Delta$", scores["Spectral Gap"], times_enriched, "olive"),
        ("Berry Phase Rate (z-score)", scores["Berry Phase Rate"], times_enriched, "crimson"),
    ]
    if rf_scores is not None:
        panel_data.append(("RF P(crisis)", rf_scores, times, "forestgreen"))

    n_panels = len(panel_data)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 2.2 * n_panels), sharex=True)

    crisis_ts = pd.Timestamp(crisis_date)

    for idx, (label, data, t, color) in enumerate(panel_data):
        ax = axes[idx]

        # Ensure times and data are aligned
        plot_len = min(len(t), len(data))
        ax.plot(t[:plot_len], data[:plot_len], color=color, linewidth=0.8, alpha=0.9)

        # Crisis date vertical line
        ax.axvline(crisis_ts, color="red", linestyle="--", alpha=0.6, linewidth=0.8)

        # Fill crisis window
        window = 10  # days
        ax.axvspan(
            crisis_ts - pd.Timedelta(days=window * 1.5),
            crisis_ts + pd.Timedelta(days=window * 1.5),
            color="red",
            alpha=0.05,
        )

        ax.set_ylabel(label, fontsize=8)
        ax.tick_params(labelsize=7)

        # Add event annotations to first panel only
        if idx == 0:
            for event_date, event_label in events:
                ev_ts = pd.Timestamp(event_date)
                if t[0] <= ev_ts <= t[-1]:
                    ax.axvline(ev_ts, color="gray", linestyle=":", alpha=0.5, linewidth=0.5)
                    ymin, ymax = ax.get_ylim()
                    ax.text(
                        ev_ts, ymax * 0.95, event_label,
                        fontsize=6, ha="center", va="top",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
                    )

    axes[0].set_title(title, fontsize=13, fontweight="bold")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    # Save
    safe_name = crisis_name.replace(" ", "_")
    fig.savefig(output_dir / f"narrative_{safe_name}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"narrative_{safe_name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {safe_name} figure")


def run_narrative_figures(seed: int = 42) -> None:
    """Generate narrative figures for 3 representative crises."""
    seed_everything(seed)
    config = get_default_validation_config()

    output_dir = Path("experiments/outputs/regime_detection/narratives")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CRISIS NARRATIVE CASE STUDY FIGURES")
    print("=" * 60)

    # Train RF for RF probability panel
    print("\nTraining Random Forest for RF panel...")
    crises_for_rf = DATA_AVAILABLE_CRISES
    rf = RandomForestRegimeDetector(n_estimators=200, max_depth=6, seed=seed, lookback=20)
    rf_n_features = None
    try:
        X_all, y_all = [], []
        for c in crises_for_rf:
            X, X_e, t, ci = prepare_data(c, config)
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

    # Generate figures for each crisis
    target_crises = [
        (CRISIS_2008, "2008_crisis"),
        (CRISIS_2020, "2020_covid"),
        (CRISIS_2022, "2022_rates"),
    ]

    for crisis, crisis_key in target_crises:
        print(f"\nProcessing {crisis.name}...")
        result = fetch_crisis_data_with_prices(crisis, config)
        if result is None:
            print(f"  SKIPPED: no data")
            continue

        X, X_enriched, times, crisis_idx, spy_prices = result
        trim = 19  # enriched lookback - 1
        times_enriched = times[trim:]

        # Compute QCML scores
        scores = compute_narrative_scores(X, X_enriched, times, config, seed)

        # RF scores
        rf_scores = None
        if rf is not None and rf_n_features is not None:
            try:
                X_rf = X[:, :rf_n_features] if X.shape[1] > rf_n_features else X
                rf_scores = rf.compute_regime_scores(X_rf)
            except Exception as e:
                logger.warning(f"RF scoring failed for {crisis.name}: {e}")

        generate_crisis_figure(
            crisis_key,
            spy_prices,
            scores,
            rf_scores,
            times,
            times_enriched,
            crisis.crisis_date,
            output_dir,
        )

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    load_dotenv(project_root / ".env")
    run_narrative_figures()
