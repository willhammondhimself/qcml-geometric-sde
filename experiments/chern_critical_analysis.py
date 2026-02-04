#!/usr/bin/env python3
"""
Critical Analysis: Is the Chern Number Signal Actually Useful?

This script provides an honest, rigorous evaluation of whether the Chern number
is a useful trading signal or just academically interesting noise.

Key Questions Addressed:
1. False Positive Rate: How many Chern spikes occur that AREN'T crises?
2. Dec 2021/Jan 2022 Investigation: What caused that specific spike?
3. Blind Detection Test: Can we detect crises without knowing when they are?
4. Chern vs VIX Comparison: Is Chern any better than just watching VIX?
5. Honest Assessment: Is this signal tradeable?

Author: QCML Research
Date: 2024
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.patches import Rectangle
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qcml.data import PolygonDataSource, MinimalFeatureEngine
from qcml.qcml_geometry import QCMLGeometry
from qcml.topological_regime import TopologicalRegimeDetector

load_dotenv(project_root / '.env')

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Major market events for ground truth
KNOWN_MARKET_EVENTS = {
    # Major Crises (Should Detect)
    '2008-09-15': ('Lehman Collapse', 'MAJOR_CRISIS'),
    '2008-10-06': ('Market Crash Accelerates', 'MAJOR_CRISIS'),
    '2010-05-06': ('Flash Crash', 'FLASH_CRASH'),
    '2011-08-08': ('US Debt Downgrade', 'MAJOR_CRISIS'),
    '2015-08-24': ('China Black Monday', 'MAJOR_CRISIS'),
    '2018-02-05': ('Volmageddon', 'FLASH_CRASH'),
    '2018-12-24': ('Christmas Eve Crash', 'CORRECTION'),
    '2020-02-24': ('COVID Begins', 'MAJOR_CRISIS'),
    '2020-03-16': ('COVID Bottom', 'MAJOR_CRISIS'),
    '2022-01-24': ('Fed Pivot Fears Start', 'REGIME_SHIFT'),
    '2022-03-16': ('First Rate Hike', 'REGIME_SHIFT'),
    '2022-06-13': ('Bear Market Official', 'CORRECTION'),

    # Minor Events (Borderline - May or May Not Detect)
    '2011-03-16': ('Japan Earthquake', 'MINOR'),
    '2014-10-15': ('Flash Rally', 'FLASH_CRASH'),
    '2016-06-24': ('Brexit Vote', 'MINOR'),
    '2018-10-10': ('October Selloff', 'CORRECTION'),
    '2019-08-05': ('Trade War Spike', 'MINOR'),
    '2021-01-27': ('Meme Stock Chaos', 'MINOR'),
    '2021-12-01': ('Omicron Variant', 'MINOR'),
    '2023-03-10': ('SVB Collapse', 'MINOR'),
}

# Events that we want to investigate specifically
INVESTIGATE_EVENTS = [
    ('2021-12-01', '2022-01-15', 'Dec 2021/Jan 2022 Spike Investigation'),
]


def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    import random
    random.seed(seed)


# ==============================================================================
# DATA FETCHING
# ==============================================================================

def fetch_full_history(start_date: str = '2006-01-01', end_date: str = '2024-12-31') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Fetch full SPY history and compute features.

    Returns:
        features: Feature matrix with dates as index
        spy_prices: SPY price series
    """
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        raise ValueError("POLYGON_API_KEY not found")

    print(f"Fetching SPY data from {start_date} to {end_date}...")

    # Fetch SPY and VIX (we'll compare against VIX later)
    source = PolygonDataSource(api_key=api_key)

    # Get SPY
    spy_data = source.fetch_equities(['SPY'], start_date, end_date, timeframe='1d')
    if spy_data.empty:
        raise ValueError("No SPY data returned")

    spy_prices = spy_data['close'].unstack(level=0)['SPY'].ffill()

    print(f"Fetched {len(spy_prices)} days of SPY data")

    # Create features
    engine = MinimalFeatureEngine(window=20)
    features = engine.create_feature_matrix(spy_prices.to_frame('SPY'), benchmark_col='SPY')
    features = features.dropna()

    print(f"Created {len(features)} feature rows with {features.shape[1]} features")

    return features, spy_prices


def fetch_vix_data(start_date: str, end_date: str) -> pd.Series:
    """Fetch VIX data for comparison."""
    api_key = os.getenv('POLYGON_API_KEY')
    source = PolygonDataSource(api_key=api_key)

    # Try VIXY as a proxy (VIX futures ETF)
    try:
        vix_data = source.fetch_equities(['VIXY'], start_date, end_date, timeframe='1d')
        if not vix_data.empty:
            return vix_data['close'].unstack(level=0)['VIXY'].ffill()
    except:
        pass

    # Fallback: compute realized volatility as VIX proxy
    print("Note: Using realized volatility as VIX proxy")
    return None


def compute_chern_series(features: pd.DataFrame,
                         hilbert_dim: int = 8,
                         window_size: int = 20,
                         n_pca: int = 15) -> Tuple[pd.Series, np.ndarray]:
    """
    Compute Chern number series from features.

    Returns:
        chern_series: Chern numbers with datetime index
        raw_chern: Raw numpy array of Chern values
    """
    X_raw = features.values

    # Standardize and PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    pca = PCA(n_components=min(n_pca, X_raw.shape[1]))
    X = pca.fit_transform(X_scaled)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")

    # Fit geometry
    geometry = QCMLGeometry(n_features=X.shape[1], hilbert_dim=hilbert_dim)
    geometry.fit_operators(X, method='random')

    # Compute Chern series
    detector = TopologicalRegimeDetector(geometry, window_size=window_size, chern_threshold=0.1)
    chern_values = detector.rolling_chern_number(X, window=window_size)

    # Align times
    chern_times = features.index[window_size - 1:]
    if len(chern_times) > len(chern_values):
        chern_times = chern_times[:len(chern_values)]

    chern_series = pd.Series(chern_values, index=chern_times, name='chern')

    return chern_series, chern_values


# ==============================================================================
# ANALYSIS FUNCTIONS
# ==============================================================================

def detect_spikes(chern_series: pd.Series,
                  threshold_sigma: float = 2.0,
                  min_separation_days: int = 5) -> pd.DataFrame:
    """
    Detect all spikes above threshold in Chern series.

    Returns DataFrame with spike dates, magnitudes, and directions.
    """
    delta_chern = chern_series.diff()

    # Calculate rolling statistics for adaptive threshold
    rolling_std = delta_chern.rolling(window=60, min_periods=20).std()

    # Detect spikes where |delta| > threshold * rolling_std
    threshold = threshold_sigma * rolling_std
    spikes = delta_chern.abs() > threshold

    # Find spike dates
    spike_dates = chern_series.index[spikes]

    # Filter for minimum separation
    filtered_dates = []
    last_spike = None
    for date in spike_dates:
        if last_spike is None or (date - last_spike).days >= min_separation_days:
            filtered_dates.append(date)
            last_spike = date

    # Build spike dataframe
    spike_data = []
    for date in filtered_dates:
        spike_data.append({
            'date': date,
            'delta_chern': delta_chern.loc[date],
            'chern_value': chern_series.loc[date],
            'threshold': threshold.loc[date],
            'sigma_magnitude': abs(delta_chern.loc[date]) / rolling_std.loc[date] if rolling_std.loc[date] > 0 else 0
        })

    return pd.DataFrame(spike_data)


def classify_spikes(spikes_df: pd.DataFrame,
                    known_events: Dict,
                    tolerance_days: int = 10) -> pd.DataFrame:
    """
    Classify each spike as True Positive, False Positive, etc.
    """
    classifications = []

    for _, spike in spikes_df.iterrows():
        spike_date = spike['date']

        # Find closest known event
        closest_event = None
        closest_distance = float('inf')

        for event_date_str, (event_name, event_type) in known_events.items():
            event_date = pd.Timestamp(event_date_str)
            distance = abs((spike_date - event_date).days)

            if distance < closest_distance:
                closest_distance = distance
                closest_event = (event_date_str, event_name, event_type)

        # Classify
        if closest_distance <= tolerance_days:
            if closest_event[2] in ['MAJOR_CRISIS', 'FLASH_CRASH']:
                classification = 'TRUE_POSITIVE'
            elif closest_event[2] == 'REGIME_SHIFT':
                classification = 'TRUE_POSITIVE_REGIME'
            elif closest_event[2] == 'CORRECTION':
                classification = 'TRUE_POSITIVE_CORRECTION'
            else:
                classification = 'TRUE_POSITIVE_MINOR'
            matched_event = closest_event[1]
        else:
            classification = 'FALSE_POSITIVE'
            matched_event = f"No known event (closest: {closest_event[1]} at {closest_distance} days)" if closest_event else "None"

        classifications.append({
            **spike.to_dict(),
            'classification': classification,
            'matched_event': matched_event,
            'distance_days': closest_distance if closest_distance != float('inf') else None
        })

    return pd.DataFrame(classifications)


def compute_detection_metrics(classified_spikes: pd.DataFrame,
                              known_events: Dict,
                              tolerance_days: int = 10) -> Dict:
    """
    Compute precision, recall, F1 for crisis detection.
    """
    # True Positives: spikes that matched real events
    tp = len(classified_spikes[classified_spikes['classification'].str.startswith('TRUE_POSITIVE')])

    # False Positives: spikes with no matching event
    fp = len(classified_spikes[classified_spikes['classification'] == 'FALSE_POSITIVE'])

    # False Negatives: events we missed
    detected_events = set()
    for _, row in classified_spikes.iterrows():
        if row['classification'].startswith('TRUE_POSITIVE') and row['distance_days'] is not None:
            detected_events.add(row['matched_event'])

    major_events = [name for date, (name, etype) in known_events.items()
                    if etype in ['MAJOR_CRISIS', 'FLASH_CRASH', 'REGIME_SHIFT']]
    fn = len([e for e in major_events if e not in detected_events])

    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'total_spikes': len(classified_spikes),
        'major_events_in_data': len(major_events),
        'events_detected': list(detected_events)
    }


def compare_chern_vs_volatility(chern_series: pd.Series,
                                spy_prices: pd.Series) -> Dict:
    """
    Compare Chern signal to simple realized volatility.

    Key question: Is Chern number just a fancy volatility measure?
    """
    # Compute realized volatility (20-day rolling std of returns)
    returns = spy_prices.pct_change()
    realized_vol = returns.rolling(window=20).std() * np.sqrt(252)  # Annualized

    # Align series
    common_dates = chern_series.index.intersection(realized_vol.index)
    chern_aligned = chern_series.loc[common_dates]
    vol_aligned = realized_vol.loc[common_dates]

    # Correlation analysis
    correlation = chern_aligned.corr(vol_aligned)

    # Compute delta series
    delta_chern = chern_aligned.diff()
    delta_vol = vol_aligned.diff()

    # Lead-lag analysis: does Chern lead volatility?
    lead_lag_corrs = {}
    for lag in range(-10, 11):
        if lag < 0:
            # Chern leads
            shifted_vol = delta_vol.shift(-lag)
        else:
            # Volatility leads
            shifted_vol = delta_vol.shift(-lag)

        corr = delta_chern.corr(shifted_vol.dropna())
        lead_lag_corrs[lag] = corr

    # Check if Chern provides early warning
    best_lag = max(lead_lag_corrs, key=lambda k: abs(lead_lag_corrs[k]))

    # Information ratio: unique information in Chern not in vol
    # Regress Chern on volatility, look at residuals
    from sklearn.linear_model import LinearRegression

    mask = ~(chern_aligned.isna() | vol_aligned.isna())
    X = vol_aligned[mask].values.reshape(-1, 1)
    y = chern_aligned[mask].values

    reg = LinearRegression().fit(X, y)
    residuals = y - reg.predict(X)

    # R² tells us how much variance is explained by volatility
    r_squared = reg.score(X, y)
    unique_variance = 1 - r_squared  # Variance NOT explained by vol

    return {
        'correlation': correlation,
        'r_squared': r_squared,
        'unique_variance': unique_variance,
        'best_lead_lag': best_lag,
        'lead_lag_corrs': lead_lag_corrs,
        'vol_series': vol_aligned,
        'residuals': pd.Series(residuals, index=chern_aligned[mask].index)
    }


def investigate_specific_period(chern_series: pd.Series,
                                spy_prices: pd.Series,
                                start_date: str,
                                end_date: str,
                                period_name: str) -> Dict:
    """
    Deep dive into a specific period to understand what happened.
    """
    mask = (chern_series.index >= start_date) & (chern_series.index <= end_date)
    chern_period = chern_series[mask]

    price_mask = (spy_prices.index >= start_date) & (spy_prices.index <= end_date)
    price_period = spy_prices[price_mask]

    # Calculate various metrics
    price_return = (price_period.iloc[-1] - price_period.iloc[0]) / price_period.iloc[0]
    max_drawdown = (price_period / price_period.cummax() - 1).min()

    chern_change = chern_period.iloc[-1] - chern_period.iloc[0]
    chern_max_spike = chern_period.diff().abs().max()

    # What was happening in markets during this period?
    # (We'll add context based on date range)
    context = []
    for event_date_str, (event_name, event_type) in KNOWN_MARKET_EVENTS.items():
        event_date = pd.Timestamp(event_date_str)
        if pd.Timestamp(start_date) <= event_date <= pd.Timestamp(end_date):
            context.append(f"{event_date_str}: {event_name} ({event_type})")

    return {
        'period_name': period_name,
        'start_date': start_date,
        'end_date': end_date,
        'price_return': price_return,
        'max_drawdown': max_drawdown,
        'chern_change': chern_change,
        'chern_max_spike': chern_max_spike,
        'chern_mean': chern_period.mean(),
        'chern_std': chern_period.std(),
        'known_events_in_period': context,
        'chern_period': chern_period,
        'price_period': price_period
    }


def blind_detection_test(chern_series: pd.Series,
                         spy_prices: pd.Series,
                         threshold_sigma: float = 2.5) -> Dict:
    """
    Run a blind detection test: predict crisis periods from Chern alone.

    This simulates what a trader would actually experience:
    - No hindsight about when crises occur
    - Only the Chern signal to work with
    - Need to make real-time decisions
    """
    # Detect all anomalies using only the signal
    delta_chern = chern_series.diff()
    rolling_std = delta_chern.rolling(window=60, min_periods=20).std()

    # Flag periods where Chern is anomalous
    anomaly_score = delta_chern.abs() / rolling_std
    anomalies = anomaly_score > threshold_sigma

    # Get returns after each anomaly
    returns = spy_prices.pct_change()

    # Forward returns analysis
    forward_returns = {}
    for days in [5, 10, 20, 60]:
        fwd_ret = returns.rolling(window=days).sum().shift(-days)
        forward_returns[f'{days}d'] = fwd_ret

    # Compare returns after anomalies vs normal periods
    results = {}
    for period, fwd_ret in forward_returns.items():
        # Align
        common = anomaly_score.index.intersection(fwd_ret.index)
        anom_aligned = anomalies.loc[common]
        ret_aligned = fwd_ret.loc[common]

        # Stats
        anomaly_returns = ret_aligned[anom_aligned].dropna()
        normal_returns = ret_aligned[~anom_aligned].dropna()

        if len(anomaly_returns) > 0 and len(normal_returns) > 0:
            t_stat, p_val = stats.ttest_ind(anomaly_returns, normal_returns)
        else:
            t_stat, p_val = 0, 1

        results[period] = {
            'anomaly_mean_return': anomaly_returns.mean() if len(anomaly_returns) > 0 else None,
            'normal_mean_return': normal_returns.mean() if len(normal_returns) > 0 else None,
            'anomaly_std': anomaly_returns.std() if len(anomaly_returns) > 0 else None,
            'normal_std': normal_returns.std() if len(normal_returns) > 0 else None,
            'n_anomalies': len(anomaly_returns),
            'n_normal': len(normal_returns),
            't_statistic': t_stat,
            'p_value': p_val
        }

    return {
        'threshold_used': threshold_sigma,
        'total_anomalies_detected': anomalies.sum(),
        'forward_return_analysis': results,
        'anomaly_dates': chern_series.index[anomalies]
    }


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def create_comprehensive_analysis_plot(chern_series: pd.Series,
                                       spy_prices: pd.Series,
                                       classified_spikes: pd.DataFrame,
                                       vol_comparison: Dict,
                                       output_path: Path):
    """
    Create a comprehensive 4-panel analysis figure.
    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 18), sharex=True)

    # Align prices to Chern dates
    common_dates = chern_series.index.intersection(spy_prices.index)
    spy_aligned = spy_prices.loc[common_dates]

    # Panel 1: SPY with crisis markers
    ax1 = axes[0]
    ax1.plot(spy_aligned.index, spy_aligned.values, 'b-', linewidth=0.8, alpha=0.8)
    ax1.set_ylabel('SPY Price', fontsize=11)
    ax1.set_title('SPY Price with Known Market Events', fontsize=12, fontweight='bold')

    # Mark known events
    for event_date_str, (event_name, event_type) in KNOWN_MARKET_EVENTS.items():
        event_date = pd.Timestamp(event_date_str)
        if event_date in spy_aligned.index or (spy_aligned.index.min() <= event_date <= spy_aligned.index.max()):
            if event_type == 'MAJOR_CRISIS':
                color, alpha = 'red', 0.8
            elif event_type == 'FLASH_CRASH':
                color, alpha = 'orange', 0.7
            elif event_type == 'REGIME_SHIFT':
                color, alpha = 'purple', 0.6
            else:
                color, alpha = 'gray', 0.4
            ax1.axvline(event_date, color=color, linestyle='--', alpha=alpha, linewidth=1)

    ax1.grid(True, alpha=0.3)
    ax1.legend(['SPY', 'Major Crisis', 'Flash Crash', 'Regime Shift'],
               loc='upper left', fontsize=9)

    # Panel 2: Chern Number with spike classifications
    ax2 = axes[1]
    ax2.plot(chern_series.index, chern_series.values, 'g-', linewidth=0.8)
    ax2.fill_between(chern_series.index, chern_series.values, 0, alpha=0.2, color='green')
    ax2.set_ylabel('Chern Number', fontsize=11)
    ax2.set_title('Chern Number with Detected Spikes (Classified)', fontsize=12, fontweight='bold')

    # Mark spikes with colors based on classification
    for _, spike in classified_spikes.iterrows():
        if spike['classification'] == 'TRUE_POSITIVE':
            color = 'green'
            marker = '^'
        elif spike['classification'] == 'FALSE_POSITIVE':
            color = 'red'
            marker = 'x'
        else:
            color = 'blue'
            marker = 'o'

        if spike['date'] in chern_series.index:
            ax2.scatter(spike['date'], chern_series.loc[spike['date']],
                       c=color, marker=marker, s=60, zorder=5)

    ax2.grid(True, alpha=0.3)
    ax2.legend(['Chern', 'True Positive', 'False Positive'], loc='upper left', fontsize=9)

    # Panel 3: Chern vs Volatility comparison
    ax3 = axes[2]
    vol_series = vol_comparison['vol_series']

    ax3_twin = ax3.twinx()
    l1 = ax3.plot(chern_series.index, chern_series.values, 'g-', linewidth=0.8, label='Chern')
    l2 = ax3_twin.plot(vol_series.index, vol_series.values, 'r-', linewidth=0.8, alpha=0.7, label='Realized Vol')

    ax3.set_ylabel('Chern Number', fontsize=11, color='green')
    ax3_twin.set_ylabel('Realized Volatility (ann.)', fontsize=11, color='red')
    ax3.set_title(f'Chern vs Realized Volatility (Correlation: {vol_comparison["correlation"]:.3f})',
                  fontsize=12, fontweight='bold')

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Delta Chern with 2σ threshold
    ax4 = axes[3]
    delta_chern = chern_series.diff()
    rolling_std = delta_chern.rolling(window=60, min_periods=20).std()
    threshold_upper = 2 * rolling_std
    threshold_lower = -2 * rolling_std

    colors = ['red' if d < 0 else 'green' for d in delta_chern]
    ax4.bar(chern_series.index, delta_chern, width=1, color=colors, alpha=0.6)
    ax4.plot(chern_series.index, threshold_upper, 'k--', linewidth=1, alpha=0.7, label='±2σ')
    ax4.plot(chern_series.index, threshold_lower, 'k--', linewidth=1, alpha=0.7)
    ax4.axhline(0, color='k', linewidth=0.5)

    ax4.set_ylabel('ΔChern', fontsize=11)
    ax4.set_xlabel('Date', fontsize=11)
    ax4.set_title('Daily Change in Chern Number with Adaptive Threshold', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Format x-axis
    ax4.xaxis.set_major_formatter(DateFormatter('%Y'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved analysis plot to: {output_path}")
    plt.close()


def create_chern_vs_vix_plot(chern_series: pd.Series,
                              vol_comparison: Dict,
                              output_path: Path):
    """
    Create detailed Chern vs VIX comparison plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Scatter plot
    ax1 = axes[0, 0]
    vol_series = vol_comparison['vol_series']
    common = chern_series.index.intersection(vol_series.index)

    ax1.scatter(vol_series.loc[common], chern_series.loc[common], alpha=0.3, s=10)
    ax1.set_xlabel('Realized Volatility')
    ax1.set_ylabel('Chern Number')
    ax1.set_title(f'Chern vs Volatility (R² = {vol_comparison["r_squared"]:.3f})', fontweight='bold')

    # Add trend line
    z = np.polyfit(vol_series.loc[common].dropna(), chern_series.loc[common].dropna(), 1)
    p = np.poly1d(z)
    x_line = np.linspace(vol_series.min(), vol_series.max(), 100)
    ax1.plot(x_line, p(x_line), 'r-', linewidth=2, label='Linear fit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Lead-lag correlation
    ax2 = axes[0, 1]
    lags = list(vol_comparison['lead_lag_corrs'].keys())
    corrs = list(vol_comparison['lead_lag_corrs'].values())

    ax2.bar(lags, corrs, color=['green' if c > 0 else 'red' for c in corrs])
    ax2.axhline(0, color='k', linewidth=0.5)
    ax2.axvline(0, color='k', linewidth=0.5, linestyle='--')
    ax2.set_xlabel('Lag (negative = Chern leads)')
    ax2.set_ylabel('Correlation')
    ax2.set_title('Lead-Lag Correlation: ΔChern vs ΔVolatility', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Residuals (unique Chern information)
    ax3 = axes[1, 0]
    residuals = vol_comparison['residuals']
    ax3.plot(residuals.index, residuals.values, 'b-', linewidth=0.5, alpha=0.7)
    ax3.fill_between(residuals.index, residuals.values, 0, alpha=0.2)
    ax3.axhline(0, color='k', linewidth=0.5)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Residual')
    ax3.set_title(f'Chern Residuals (Unique Info: {vol_comparison["unique_variance"]:.1%})', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Distribution of residuals
    ax4 = axes[1, 1]
    ax4.hist(residuals.dropna(), bins=50, density=True, alpha=0.7, color='blue')
    ax4.set_xlabel('Residual Value')
    ax4.set_ylabel('Density')
    ax4.set_title('Distribution of Chern Residuals', fontweight='bold')

    # Add normal distribution overlay
    mu, std = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax4.plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2, label='Normal fit')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved Chern vs VIX comparison to: {output_path}")
    plt.close()


def create_blind_test_plot(blind_results: Dict, output_path: Path):
    """
    Visualize results of blind detection test.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Forward returns comparison
    ax1 = axes[0]
    periods = list(blind_results['forward_return_analysis'].keys())
    anomaly_returns = [blind_results['forward_return_analysis'][p]['anomaly_mean_return'] or 0
                       for p in periods]
    normal_returns = [blind_results['forward_return_analysis'][p]['normal_mean_return'] or 0
                      for p in periods]

    x = np.arange(len(periods))
    width = 0.35

    bars1 = ax1.bar(x - width/2, [r * 100 for r in anomaly_returns], width,
                    label='After Chern Spike', color='red', alpha=0.7)
    bars2 = ax1.bar(x + width/2, [r * 100 for r in normal_returns], width,
                    label='Normal Periods', color='blue', alpha=0.7)

    ax1.set_xlabel('Forward Return Period')
    ax1.set_ylabel('Mean Return (%)')
    ax1.set_title('Forward Returns: After Chern Spikes vs Normal', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(periods)
    ax1.legend()
    ax1.axhline(0, color='k', linewidth=0.5)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Statistical significance
    ax2 = axes[1]
    p_values = [blind_results['forward_return_analysis'][p]['p_value'] for p in periods]
    colors = ['green' if p < 0.05 else 'red' for p in p_values]

    ax2.bar(periods, [-np.log10(p) if p > 0 else 0 for p in p_values], color=colors, alpha=0.7)
    ax2.axhline(-np.log10(0.05), color='k', linestyle='--', label='p=0.05')
    ax2.axhline(-np.log10(0.01), color='gray', linestyle='--', label='p=0.01')
    ax2.set_xlabel('Forward Return Period')
    ax2.set_ylabel('-log10(p-value)')
    ax2.set_title('Statistical Significance of Return Difference', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved blind test results to: {output_path}")
    plt.close()


# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================

def generate_honest_assessment(metrics: Dict,
                               vol_comparison: Dict,
                               blind_results: Dict,
                               classified_spikes: pd.DataFrame) -> str:
    """
    Generate an honest written assessment of whether the signal is useful.
    """
    assessment = []
    assessment.append("=" * 70)
    assessment.append("HONEST ASSESSMENT: Is the Chern Number Signal Actually Useful?")
    assessment.append("=" * 70)
    assessment.append("")

    # Detection Metrics
    assessment.append("1. DETECTION PERFORMANCE")
    assessment.append("-" * 40)
    assessment.append(f"   Precision: {metrics['precision']:.1%}")
    assessment.append(f"   Recall: {metrics['recall']:.1%}")
    assessment.append(f"   F1 Score: {metrics['f1_score']:.1%}")
    assessment.append(f"   Total Spikes Detected: {metrics['total_spikes']}")
    assessment.append(f"   True Positives: {metrics['true_positives']}")
    assessment.append(f"   False Positives: {metrics['false_positives']}")
    assessment.append(f"   Major Events Detected: {len(metrics['events_detected'])}/{metrics['major_events_in_data']}")
    assessment.append("")

    if metrics['precision'] < 0.3:
        assessment.append("   ⚠️ LOW PRECISION: Too many false alarms for practical trading")
    elif metrics['precision'] < 0.5:
        assessment.append("   ⚠️ MODERATE PRECISION: Needs additional filters to be tradeable")
    else:
        assessment.append("   ✓ GOOD PRECISION: Most signals correspond to real events")

    if metrics['recall'] < 0.5:
        assessment.append("   ⚠️ LOW RECALL: Misses too many major crises")
    else:
        assessment.append("   ✓ GOOD RECALL: Catches most major events")
    assessment.append("")

    # Comparison to Volatility
    assessment.append("2. CHERN VS SIMPLE VOLATILITY")
    assessment.append("-" * 40)
    assessment.append(f"   Correlation with Realized Vol: {vol_comparison['correlation']:.3f}")
    assessment.append(f"   R² (variance explained by vol): {vol_comparison['r_squared']:.1%}")
    assessment.append(f"   Unique Information in Chern: {vol_comparison['unique_variance']:.1%}")
    assessment.append("")

    if vol_comparison['r_squared'] > 0.7:
        assessment.append("   ⚠️ HIGH CORRELATION: Chern is mostly just fancy volatility")
        assessment.append("      The signal provides little beyond what VIX already shows")
    elif vol_comparison['r_squared'] > 0.4:
        assessment.append("   ~ MODERATE CORRELATION: Some overlap with volatility")
        assessment.append("      Chern captures some unique information")
    else:
        assessment.append("   ✓ LOW CORRELATION: Chern is distinct from volatility")
        assessment.append("      The signal captures something genuinely different")
    assessment.append("")

    # Blind Test Results
    assessment.append("3. BLIND DETECTION TEST (Would this work in practice?)")
    assessment.append("-" * 40)

    # Check 20-day forward returns
    fwd_20d = blind_results['forward_return_analysis'].get('20d', {})
    if fwd_20d.get('p_value', 1) < 0.05:
        anom_ret = fwd_20d.get('anomaly_mean_return', 0) * 100
        norm_ret = fwd_20d.get('normal_mean_return', 0) * 100
        assessment.append(f"   ✓ SIGNIFICANT 20-day signal (p={fwd_20d['p_value']:.4f})")
        assessment.append(f"      After spike: {anom_ret:.2f}%  Normal: {norm_ret:.2f}%")
    else:
        assessment.append(f"   ⚠️ NOT SIGNIFICANT at 20-day horizon (p={fwd_20d.get('p_value', 1):.4f})")

    # Check if returns are actually worse after spikes
    for period in ['5d', '10d', '20d', '60d']:
        data = blind_results['forward_return_analysis'].get(period, {})
        anom = data.get('anomaly_mean_return', 0)
        norm = data.get('normal_mean_return', 0)
        if anom is not None and norm is not None and anom < norm:
            assessment.append(f"   ✓ {period}: Returns worse after spikes (as expected for risk signal)")
    assessment.append("")

    # Overall Verdict
    assessment.append("4. OVERALL VERDICT")
    assessment.append("-" * 40)

    # Scoring
    score = 0
    if metrics['precision'] > 0.3: score += 1
    if metrics['recall'] > 0.5: score += 1
    if vol_comparison['unique_variance'] > 0.3: score += 1
    if fwd_20d.get('p_value', 1) < 0.05: score += 1

    if score >= 3:
        assessment.append("   ✓ POTENTIALLY USEFUL")
        assessment.append("   The signal has merit and could be part of a trading system")
        assessment.append("   Recommendation: Use as ONE input among many, not standalone")
    elif score >= 2:
        assessment.append("   ~ ACADEMICALLY INTERESTING, MARGINALLY USEFUL")
        assessment.append("   The signal has some value but isn't reliable enough alone")
        assessment.append("   Recommendation: Combine with other signals, use as confirmation")
    else:
        assessment.append("   ⚠️ NOT CLEARLY USEFUL")
        assessment.append("   The signal doesn't provide enough edge over simple alternatives")
        assessment.append("   Recommendation: More research needed before trading")

    assessment.append("")
    assessment.append("5. FALSE POSITIVE ANALYSIS")
    assessment.append("-" * 40)
    fps = classified_spikes[classified_spikes['classification'] == 'FALSE_POSITIVE']
    assessment.append(f"   False Positives: {len(fps)}")
    if len(fps) > 0:
        assessment.append("   Dates with unexplained spikes:")
        for _, fp in fps.head(10).iterrows():
            assessment.append(f"      {fp['date'].strftime('%Y-%m-%d')}: σ={fp['sigma_magnitude']:.1f}, {fp['matched_event']}")

    assessment.append("")
    assessment.append("=" * 70)

    return "\n".join(assessment)


def main():
    """Run the complete critical analysis."""
    seed_everything(42)

    output_dir = project_root / 'experiments' / 'outputs'
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("CRITICAL ANALYSIS: Is the Chern Number Signal Actually Useful?")
    print("=" * 70)
    print()

    # Step 1: Fetch data
    print("Step 1: Fetching full market history...")
    try:
        features, spy_prices = fetch_full_history('2006-01-01', '2024-06-30')
    except Exception as e:
        print(f"Error fetching data: {e}")
        print("Attempting with shorter history...")
        features, spy_prices = fetch_full_history('2018-01-01', '2024-06-30')

    # Step 2: Compute Chern series
    print("\nStep 2: Computing Chern number series...")
    chern_series, raw_chern = compute_chern_series(features)
    print(f"Chern series: {len(chern_series)} values from {chern_series.index.min().date()} to {chern_series.index.max().date()}")

    # Step 3: Detect and classify all spikes
    print("\nStep 3: Detecting and classifying spikes...")
    spikes_df = detect_spikes(chern_series, threshold_sigma=2.0)
    print(f"Detected {len(spikes_df)} spikes above 2σ threshold")

    classified_spikes = classify_spikes(spikes_df, KNOWN_MARKET_EVENTS, tolerance_days=10)

    # Step 4: Compute detection metrics
    print("\nStep 4: Computing detection metrics...")
    metrics = compute_detection_metrics(classified_spikes, KNOWN_MARKET_EVENTS)

    print(f"\n   Precision: {metrics['precision']:.1%}")
    print(f"   Recall: {metrics['recall']:.1%}")
    print(f"   F1 Score: {metrics['f1_score']:.1%}")
    print(f"   True Positives: {metrics['true_positives']}")
    print(f"   False Positives: {metrics['false_positives']}")

    # Step 5: Compare to volatility
    print("\nStep 5: Comparing Chern to realized volatility...")
    vol_comparison = compare_chern_vs_volatility(chern_series, spy_prices)
    print(f"   Correlation: {vol_comparison['correlation']:.3f}")
    print(f"   R² (overlap): {vol_comparison['r_squared']:.1%}")
    print(f"   Unique variance: {vol_comparison['unique_variance']:.1%}")

    # Step 6: Investigate Dec 2021/Jan 2022
    print("\nStep 6: Investigating Dec 2021/Jan 2022 spike...")
    for start, end, name in INVESTIGATE_EVENTS:
        investigation = investigate_specific_period(chern_series, spy_prices, start, end, name)
        print(f"\n   {name}:")
        print(f"   SPY Return: {investigation['price_return']:.1%}")
        print(f"   Max Drawdown: {investigation['max_drawdown']:.1%}")
        print(f"   Chern Change: {investigation['chern_change']:.6f}")
        print(f"   Known events in period: {investigation['known_events_in_period']}")

    # Step 7: Blind detection test
    print("\nStep 7: Running blind detection test...")
    blind_results = blind_detection_test(chern_series, spy_prices, threshold_sigma=2.5)
    print(f"   Total anomalies detected: {blind_results['total_anomalies_detected']}")

    for period, data in blind_results['forward_return_analysis'].items():
        if data['anomaly_mean_return'] is not None:
            print(f"   {period} forward returns:")
            print(f"      After spike: {data['anomaly_mean_return']*100:.2f}%")
            print(f"      Normal: {data['normal_mean_return']*100:.2f}%")
            print(f"      p-value: {data['p_value']:.4f}")

    # Step 8: Generate visualizations
    print("\nStep 8: Generating visualizations...")
    create_comprehensive_analysis_plot(
        chern_series, spy_prices, classified_spikes, vol_comparison,
        output_dir / 'chern_critical_analysis.png'
    )

    create_chern_vs_vix_plot(
        chern_series, vol_comparison,
        output_dir / 'chern_vs_volatility.png'
    )

    create_blind_test_plot(
        blind_results,
        output_dir / 'chern_blind_test.png'
    )

    # Step 9: Generate honest assessment
    print("\nStep 9: Generating honest assessment...")
    assessment = generate_honest_assessment(metrics, vol_comparison, blind_results, classified_spikes)

    # Print assessment
    print("\n")
    print(assessment)

    # Save assessment to file
    assessment_path = output_dir / 'chern_honest_assessment.txt'
    with open(assessment_path, 'w') as f:
        f.write(assessment)
        f.write("\n\n")
        f.write("CLASSIFIED SPIKES:\n")
        f.write("-" * 70 + "\n")
        f.write(classified_spikes.to_string())

    print(f"\nFull assessment saved to: {assessment_path}")

    # Save spike classification to CSV
    csv_path = output_dir / 'chern_spike_classifications.csv'
    classified_spikes.to_csv(csv_path, index=False)
    print(f"Spike classifications saved to: {csv_path}")

    return {
        'chern_series': chern_series,
        'classified_spikes': classified_spikes,
        'metrics': metrics,
        'vol_comparison': vol_comparison,
        'blind_results': blind_results
    }


if __name__ == "__main__":
    results = main()
