#!/usr/bin/env python3
"""
Visualize Chern Number Evolution for All Crises

Creates 3-panel plots (Price, Chern, ΔChern) for 2008, 2020, and 2022 crises.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qcml.data import PolygonDataSource, MinimalFeatureEngine, QCMLDataset
from qcml.qcml_geometry import QCMLGeometry
from qcml.topological_regime import TopologicalRegimeDetector
from experiments.crisis_config import ALL_CRISES, CrisisDefinition

load_dotenv(project_root / '.env')


def fetch_crisis_data(crisis: CrisisDefinition) -> tuple:
    """Fetch real data and compute Chern series for a crisis."""
    api_key = os.getenv('POLYGON_API_KEY')

    crisis_date = pd.Timestamp(crisis.crisis_date)
    start_date = crisis_date - pd.DateOffset(months=crisis.lookback_months)
    end_date = crisis_date + pd.DateOffset(months=crisis.lookahead_months)

    print(f"Fetching {crisis.name}: {start_date.date()} to {end_date.date()}")

    # Fetch data
    source = PolygonDataSource(api_key=api_key)
    raw_data = source.fetch_equities(
        crisis.universe,
        str(start_date.date()),
        str(end_date.date()),
        timeframe="1d"
    )

    # Get prices
    prices = raw_data['close'].unstack(level=0).ffill()
    benchmark = 'SPY' if 'SPY' in prices.columns else prices.columns[0]

    # Compute features
    engine = MinimalFeatureEngine(window=20)
    features = engine.create_feature_matrix(prices, benchmark_col=benchmark)
    features = features.dropna()

    # Prepare for QCML
    X_raw = features.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    pca = PCA(n_components=min(15, X_raw.shape[1]))
    X = pca.fit_transform(X_scaled)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    # Fit geometry
    geometry = QCMLGeometry(n_features=X.shape[1], hilbert_dim=8)
    geometry.fit_operators(X, method='random')

    # Compute Chern series
    detector = TopologicalRegimeDetector(geometry, window_size=20, chern_threshold=0.1)
    chern_series = detector.rolling_chern_number(X, window=20)

    # Align times
    chern_times = features.index[19:]  # Account for window
    if len(chern_times) > len(chern_series):
        chern_times = chern_times[:len(chern_series)]

    # Get SPY prices aligned with features
    spy_prices = prices[benchmark].loc[features.index]

    return chern_times, chern_series, spy_prices, crisis_date


def create_crisis_plot(crisis: CrisisDefinition, output_dir: Path):
    """Create 3-panel plot for a single crisis."""

    chern_times, chern_series, spy_prices, crisis_date = fetch_crisis_data(crisis)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Title mapping
    titles = {
        '2008_crisis': '2008 Financial Crisis: Chern Number Evolution',
        '2020_covid': '2020 COVID Crash: Chern Number Evolution',
        '2022_rates': '2022 Rate Hike Regime: Chern Number Evolution'
    }

    labels = {
        '2008_crisis': 'Lehman Collapse',
        '2020_covid': 'COVID Crash',
        '2022_rates': 'First Rate Hike'
    }

    # Plot 1: SPY Price
    ax1 = axes[0]
    ax1.plot(spy_prices.index, spy_prices.values, 'b-', linewidth=1.2)
    ax1.axvline(crisis_date, color='r', linestyle='--', linewidth=2, label=labels[crisis.name])
    ax1.set_ylabel('SPY Price', fontsize=11)
    ax1.set_title(titles[crisis.name], fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Chern Number
    ax2 = axes[1]
    ax2.plot(chern_times, chern_series, 'g-', linewidth=1.5)
    ax2.axvline(crisis_date, color='r', linestyle='--', linewidth=2)
    ax2.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax2.fill_between(chern_times, chern_series, 0, alpha=0.3, color='green')
    ax2.set_ylabel('Chern Number', fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Add mean lines before/after crisis
    crisis_idx = (chern_times >= crisis_date).argmax()
    mean_before = np.mean(chern_series[:crisis_idx])
    mean_after = np.mean(chern_series[crisis_idx:])
    ax2.axhline(mean_before, color='blue', linestyle=':', alpha=0.7, label=f'Mean before: {mean_before:.4f}')
    ax2.axhline(mean_after, color='orange', linestyle=':', alpha=0.7, label=f'Mean after: {mean_after:.4f}')
    ax2.legend(loc='upper right', fontsize=9)

    # Plot 3: ΔChern
    ax3 = axes[2]
    delta_chern = np.diff(chern_series, prepend=chern_series[0])

    # Color bars by sign
    colors = ['red' if d < 0 else 'green' for d in delta_chern]
    ax3.bar(chern_times, delta_chern, width=1.5, color=colors, alpha=0.7)
    ax3.axvline(crisis_date, color='r', linestyle='--', linewidth=2)
    ax3.axhline(0, color='k', linestyle='-', alpha=0.5)

    # Dynamic threshold based on actual data
    threshold = np.std(delta_chern) * 2
    ax3.axhline(threshold, color='orange', linestyle=':', alpha=0.7, label=f'±2σ threshold')
    ax3.axhline(-threshold, color='orange', linestyle=':', alpha=0.7)

    ax3.set_ylabel('ΔChern', fontsize=11)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Format x-axis
    ax3.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    plt.tight_layout()

    # Save
    output_path = output_dir / f'chern_{crisis.name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    return {
        'crisis': crisis.name,
        'chern_mean': np.mean(chern_series),
        'chern_std': np.std(chern_series),
        'chern_min': np.min(chern_series),
        'chern_max': np.max(chern_series),
        'mean_before': mean_before,
        'mean_after': mean_after,
        'delta': mean_after - mean_before
    }


def create_combined_plot(output_dir: Path):
    """Create a combined figure with all 3 crises side by side."""

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))

    crisis_data = []

    for col, crisis in enumerate(ALL_CRISES):
        chern_times, chern_series, spy_prices, crisis_date = fetch_crisis_data(crisis)

        titles = {
            '2008_crisis': '2008 Financial Crisis',
            '2020_covid': '2020 COVID Crash',
            '2022_rates': '2022 Rate Hikes'
        }

        # Row 0: SPY Price
        ax = axes[0, col]
        ax.plot(spy_prices.index, spy_prices.values, 'b-', linewidth=1)
        ax.axvline(crisis_date, color='r', linestyle='--', linewidth=1.5)
        if col == 0:
            ax.set_ylabel('SPY Price', fontsize=10)
        ax.set_title(titles[crisis.name], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

        # Row 1: Chern Number
        ax = axes[1, col]
        ax.plot(chern_times, chern_series, 'g-', linewidth=1.2)
        ax.axvline(crisis_date, color='r', linestyle='--', linewidth=1.5)
        ax.fill_between(chern_times, chern_series, 0, alpha=0.3, color='green')
        if col == 0:
            ax.set_ylabel('Chern Number', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

        # Add stats
        crisis_idx = (chern_times >= crisis_date).argmax()
        mean_before = np.mean(chern_series[:crisis_idx])
        mean_after = np.mean(chern_series[crisis_idx:])
        ax.axhline(mean_before, color='blue', linestyle=':', alpha=0.6)
        ax.axhline(mean_after, color='orange', linestyle=':', alpha=0.6)

        # Row 2: ΔChern
        ax = axes[2, col]
        delta_chern = np.diff(chern_series, prepend=chern_series[0])
        colors = ['red' if d < 0 else 'green' for d in delta_chern]
        ax.bar(chern_times, delta_chern, width=1, color=colors, alpha=0.7)
        ax.axvline(crisis_date, color='r', linestyle='--', linewidth=1.5)
        ax.axhline(0, color='k', linestyle='-', alpha=0.5)
        if col == 0:
            ax.set_ylabel('ΔChern', fontsize=10)
        ax.set_xlabel('Date', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

        crisis_data.append({
            'name': crisis.name,
            'mean_before': mean_before,
            'mean_after': mean_after,
            'delta': mean_after - mean_before
        })

    plt.suptitle('Chern Number Evolution Across Market Crises', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = output_dir / 'chern_all_crises_combined.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved combined plot: {output_path}")
    plt.close()

    return crisis_data


if __name__ == "__main__":
    output_dir = project_root / 'experiments' / 'outputs'
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Generating Crisis Visualizations")
    print("=" * 60)

    # Generate individual plots
    stats = []
    for crisis in ALL_CRISES:
        print(f"\n--- {crisis.name} ---")
        stat = create_crisis_plot(crisis, output_dir)
        stats.append(stat)

    # Generate combined plot
    print("\n--- Combined Plot ---")
    combined_stats = create_combined_plot(output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("CHERN NUMBER SUMMARY")
    print("=" * 60)
    print(f"{'Crisis':<15} {'Mean Before':>12} {'Mean After':>12} {'ΔChern':>10} {'Direction':<10}")
    print("-" * 60)
    for s in stats:
        direction = "↑ UP" if s['delta'] > 0 else "↓ DOWN"
        print(f"{s['crisis']:<15} {s['mean_before']:>12.5f} {s['mean_after']:>12.5f} {s['delta']:>10.5f} {direction:<10}")

    print("\n✓ All visualizations saved to experiments/outputs/")
