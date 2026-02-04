#!/usr/bin/env python3
"""
Full History Visualization with All Spikes Labeled

Creates a comprehensive visualization showing:
1. Full 20-year SPY price history
2. Chern number evolution
3. All detected spikes labeled with what actually happened

Author: QCML Research
Date: 2024
"""

import os
import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, YearLocator
from matplotlib.patches import Rectangle
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qcml.data import PolygonDataSource, MinimalFeatureEngine
from qcml.qcml_geometry import QCMLGeometry
from qcml.topological_regime import TopologicalRegimeDetector

load_dotenv(project_root / '.env')
warnings.filterwarnings('ignore')

# Known events for labeling
KNOWN_EVENTS = {
    '2008-09-15': 'Lehman',
    '2008-10-06': 'Crash',
    '2010-05-06': 'Flash Crash',
    '2011-08-08': 'Downgrade',
    '2015-08-24': 'China',
    '2018-02-05': 'Volmageddon',
    '2018-12-24': 'Xmas',
    '2020-02-24': 'COVID Start',
    '2020-03-16': 'COVID Bottom',
    '2022-01-24': 'Fed Fear',
    '2022-03-16': 'Rate Hike',
    '2022-06-13': 'Bear',
    '2023-03-10': 'SVB',
}


def fetch_data():
    """Fetch full history."""
    api_key = os.getenv('POLYGON_API_KEY')
    source = PolygonDataSource(api_key=api_key)

    print("Fetching SPY data...")
    spy_data = source.fetch_equities(['SPY'], '2006-01-01', '2024-06-30', timeframe='1d')
    spy_prices = spy_data['close'].unstack(level=0)['SPY'].ffill()

    print(f"Fetched {len(spy_prices)} days")

    # Features
    engine = MinimalFeatureEngine(window=20)
    features = engine.create_feature_matrix(spy_prices.to_frame('SPY'), benchmark_col='SPY')
    features = features.dropna()

    return features, spy_prices


def compute_chern(features):
    """Compute Chern series."""
    X_raw = features.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    pca = PCA(n_components=min(15, X_raw.shape[1]))
    X = pca.fit_transform(X_scaled)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    geometry = QCMLGeometry(n_features=X.shape[1], hilbert_dim=8)
    geometry.fit_operators(X, method='random')

    detector = TopologicalRegimeDetector(geometry, window_size=20, chern_threshold=0.1)
    chern_values = detector.rolling_chern_number(X, window=20)

    chern_times = features.index[19:]
    if len(chern_times) > len(chern_values):
        chern_times = chern_times[:len(chern_values)]

    return pd.Series(chern_values, index=chern_times, name='chern')


def create_labeled_visualization():
    """Create the comprehensive labeled visualization."""
    features, spy_prices = fetch_data()
    chern_series = compute_chern(features)

    # Compute delta and detect spikes
    delta_chern = chern_series.diff()
    rolling_std = delta_chern.rolling(window=60, min_periods=20).std()
    threshold = 2.0 * rolling_std
    spikes = delta_chern.abs() > threshold

    # Get spike dates
    spike_dates = chern_series.index[spikes]

    # Classify spikes
    spike_classifications = []
    for spike_date in spike_dates:
        # Find closest known event
        closest_event = None
        closest_dist = float('inf')
        for event_date, event_name in KNOWN_EVENTS.items():
            dist = abs((spike_date - pd.Timestamp(event_date)).days)
            if dist < closest_dist:
                closest_dist = dist
                closest_event = event_name

        if closest_dist <= 15:
            spike_classifications.append((spike_date, closest_event, 'TP'))
        else:
            spike_classifications.append((spike_date, 'Unknown', 'FP'))

    print(f"\nSpike Classification Summary:")
    print(f"  Total Spikes: {len(spike_classifications)}")
    tp = sum(1 for _, _, c in spike_classifications if c == 'TP')
    fp = sum(1 for _, _, c in spike_classifications if c == 'FP')
    print(f"  True Positives: {tp}")
    print(f"  False Positives: {fp}")
    print(f"  Precision: {tp/(tp+fp):.1%}")

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(20, 14), sharex=True)

    # Common dates
    common_dates = chern_series.index.intersection(spy_prices.index)
    spy_aligned = spy_prices.loc[common_dates]

    # Panel 1: SPY Price
    ax1 = axes[0]
    ax1.plot(spy_aligned.index, spy_aligned.values, 'b-', linewidth=0.8)
    ax1.set_ylabel('SPY Price ($)', fontsize=12)
    ax1.set_title('20-Year SPY History with All Chern Spikes Labeled', fontsize=14, fontweight='bold')

    # Shade major crisis periods
    crisis_periods = [
        ('2008-09-01', '2009-03-31', '2008 Crisis', 'red'),
        ('2020-02-15', '2020-04-15', 'COVID', 'orange'),
        ('2022-01-01', '2022-10-31', '2022 Bear', 'purple'),
    ]

    for start, end, name, color in crisis_periods:
        start_date = pd.Timestamp(start)
        end_date = pd.Timestamp(end)
        ax1.axvspan(start_date, end_date, alpha=0.15, color=color, label=name)

    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Chern Number
    ax2 = axes[1]
    ax2.plot(chern_series.index, chern_series.values, 'g-', linewidth=0.6, alpha=0.8)
    ax2.fill_between(chern_series.index, chern_series.values, 0, alpha=0.15, color='green')
    ax2.set_ylabel('Chern Number', fontsize=12)
    ax2.set_title('Chern Number Evolution', fontsize=12)

    # Add crisis shading
    for start, end, name, color in crisis_periods:
        ax2.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.15, color=color)

    ax2.grid(True, alpha=0.3)

    # Panel 3: Delta Chern with ALL spikes labeled
    ax3 = axes[2]

    # Plot delta Chern
    colors = ['red' if d < 0 else 'green' for d in delta_chern]
    ax3.bar(chern_series.index, delta_chern, width=1, color=colors, alpha=0.5)

    # Plot threshold
    ax3.plot(chern_series.index, threshold, 'k--', linewidth=0.8, alpha=0.5)
    ax3.plot(chern_series.index, -threshold, 'k--', linewidth=0.8, alpha=0.5)
    ax3.axhline(0, color='k', linewidth=0.5)

    # Label spikes
    # We'll only label the top spikes by year to avoid clutter
    labeled_this_year = {}

    for spike_date, event_name, classification in sorted(spike_classifications, key=lambda x: -abs(delta_chern.get(x[0], 0))):
        year = spike_date.year

        # Only label a few per year to avoid clutter
        if year not in labeled_this_year:
            labeled_this_year[year] = 0

        if labeled_this_year[year] >= 2:  # Max 2 labels per year
            continue

        labeled_this_year[year] += 1

        delta_val = delta_chern.loc[spike_date]

        # Color based on classification
        if classification == 'TP':
            color = 'green'
        else:
            color = 'red'

        # Add marker
        ax3.scatter(spike_date, delta_val, c=color, s=50, zorder=5, marker='v' if delta_val < 0 else '^')

        # Add label
        va = 'bottom' if delta_val > 0 else 'top'
        label_text = event_name if event_name != 'Unknown' else f'?{spike_date.strftime("%m/%y")}'
        ax3.annotate(label_text, (spike_date, delta_val),
                    textcoords='offset points',
                    xytext=(0, 10 if delta_val > 0 else -10),
                    ha='center', va=va,
                    fontsize=7, rotation=45,
                    color=color)

    # Add crisis shading
    for start, end, name, color in crisis_periods:
        ax3.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.15, color=color)

    ax3.set_ylabel('ΔChern', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_title('Daily Change in Chern Number (All 2σ+ Spikes Labeled: Green=True Positive, Red=False Positive)',
                  fontsize=12)
    ax3.grid(True, alpha=0.3)

    # Format x-axis
    ax3.xaxis.set_major_locator(YearLocator())
    ax3.xaxis.set_major_formatter(DateFormatter('%Y'))
    plt.xticks(rotation=45)

    # Add summary box
    summary_text = (
        f"Summary Statistics:\n"
        f"Total Spikes (>2σ): {len(spike_classifications)}\n"
        f"True Positives: {tp} ({tp/len(spike_classifications)*100:.1f}%)\n"
        f"False Positives: {fp} ({fp/len(spike_classifications)*100:.1f}%)\n"
        f"Precision: {tp/(tp+fp)*100:.1f}%\n"
        f"\nVerdict: Too noisy for standalone trading"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax3.text(0.02, 0.98, summary_text, transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', bbox=props)

    plt.tight_layout()

    output_path = project_root / 'experiments' / 'outputs' / 'chern_full_history_labeled.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {output_path}")
    plt.close()

    # Also create a zoomed version for recent years
    create_zoomed_plot(chern_series, spy_prices, delta_chern, threshold, spike_classifications)


def create_zoomed_plot(chern_series, spy_prices, delta_chern, threshold, spike_classifications):
    """Create a zoomed plot for 2020-2024."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Filter to recent years
    start_zoom = '2019-01-01'
    end_zoom = '2024-06-30'

    mask = (chern_series.index >= start_zoom) & (chern_series.index <= end_zoom)
    chern_zoom = chern_series[mask]
    delta_zoom = delta_chern[mask]
    thresh_zoom = threshold[mask]

    price_mask = (spy_prices.index >= start_zoom) & (spy_prices.index <= end_zoom)
    spy_zoom = spy_prices[price_mask]

    # Panel 1: SPY with key events
    ax1 = axes[0]
    ax1.plot(spy_zoom.index, spy_zoom.values, 'b-', linewidth=1)
    ax1.set_ylabel('SPY Price ($)', fontsize=12)
    ax1.set_title('Recent History (2019-2024): Can You See the Crises from Chern Alone?', fontsize=14, fontweight='bold')

    # Mark major events
    events_to_mark = [
        ('2020-02-24', 'COVID Start'),
        ('2020-03-16', 'COVID Bottom'),
        ('2022-01-24', 'Fed Fears'),
        ('2022-03-16', 'Rate Hike'),
        ('2022-06-13', 'Bear Market'),
        ('2023-03-10', 'SVB'),
    ]

    for date, name in events_to_mark:
        ax1.axvline(pd.Timestamp(date), color='red', linestyle='--', alpha=0.7)
        ax1.annotate(name, xy=(pd.Timestamp(date), spy_zoom.max()), fontsize=8, rotation=45)

    ax1.grid(True, alpha=0.3)

    # Panel 2: Delta Chern
    ax2 = axes[1]
    colors = ['red' if d < 0 else 'green' for d in delta_zoom]
    ax2.bar(chern_zoom.index, delta_zoom, width=1, color=colors, alpha=0.6)
    ax2.plot(chern_zoom.index, thresh_zoom, 'k--', linewidth=1, label='±2σ threshold')
    ax2.plot(chern_zoom.index, -thresh_zoom, 'k--', linewidth=1)
    ax2.axhline(0, color='k', linewidth=0.5)

    # Mark the same events
    for date, name in events_to_mark:
        ax2.axvline(pd.Timestamp(date), color='red', linestyle='--', alpha=0.7)

    ax2.set_ylabel('ΔChern', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_title('ΔChern with ±2σ Threshold (Red lines = actual crisis dates)', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Add the key question
    question_text = (
        "THE BLIND TEST QUESTION:\n"
        "Looking at the ΔChern panel alone,\n"
        "could you have identified the COVID crash\n"
        "or 2022 bear market?\n"
        "\n"
        "Answer: Maybe COVID, but with many\n"
        "false alarms in between."
    )
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
    ax2.text(0.02, 0.98, question_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.tight_layout()

    output_path = project_root / 'experiments' / 'outputs' / 'chern_zoomed_2019_2024.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved zoomed plot to: {output_path}")
    plt.close()


if __name__ == "__main__":
    np.random.seed(42)
    create_labeled_visualization()
