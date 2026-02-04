#!/usr/bin/env python3
"""
Track A: Chern Interpretation Test

Validates what the Chern number ACTUALLY measures by testing the hypothesis:
"Chern detects correlation structure transitions, not volatility"

Three experiments:
1. Correlation Eigenstructure Comparison
   - Compute rolling correlation matrix eigenvalues
   - Track eigenvalue ratio (largest/smallest) as "correlation stability"
   - Test correlation between Chern spikes and eigenvalue shifts

2. Information Geometry Connection
   - Compute Fisher Information Matrix from return distributions
   - Compare FIM eigenvalues to Berry curvature
   - Validates "twisted probability manifold" interpretation

3. Lead/Lag Analysis
   - Cross-correlation between Chern and correlation breakdown
   - Determines if Chern leads, lags, or coincides with structural shifts

Author: QCML Research
Date: 2024
"""

import os
import sys
from pathlib import Path
import warnings
from typing import Tuple, Dict, List
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import correlate
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
warnings.filterwarnings('ignore')


@dataclass
class InterpretationResults:
    """Results from interpretation tests."""
    # Correlation eigenstructure
    eigenvalue_chern_correlation: float
    eigenvalue_shift_spike_alignment: float

    # Information geometry
    fisher_berry_correlation: float
    fisher_metric_stability: float

    # Lead/lag analysis
    optimal_lag_days: int
    lead_lag_correlation: float
    chern_leads_correlation: bool

    # Validation
    explained_false_positives: int
    total_false_positives: int
    fp_explanation_rate: float


def fetch_multi_asset_data(symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch multi-asset data for correlation analysis."""
    api_key = os.getenv('POLYGON_API_KEY')
    source = PolygonDataSource(api_key=api_key)

    print(f"Fetching data for {len(symbols)} symbols...")
    data = source.fetch_equities(symbols, start_date, end_date, timeframe='1d')
    prices = data['close'].unstack(level=0)

    # Forward fill missing data
    prices = prices.ffill().dropna(how='all')

    print(f"Fetched {len(prices)} days of data")
    return prices


def compute_rolling_correlation_eigenvalues(
    prices: pd.DataFrame,
    window: int = 60
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Compute rolling correlation matrix eigenvalues.

    Returns:
        eigenvalues: DataFrame with all eigenvalues over time
        eigenvalue_ratio: Series of largest/smallest eigenvalue (correlation stability)
        condition_number: Series of condition number (matrix stability)
    """
    returns = prices.pct_change().dropna()
    n_assets = returns.shape[1]

    dates = []
    all_eigenvalues = []
    ratios = []
    condition_numbers = []

    for i in range(window, len(returns)):
        window_returns = returns.iloc[i-window:i]

        # Compute correlation matrix
        corr_matrix = window_returns.corr().values

        # Handle any NaN/inf
        if not np.isfinite(corr_matrix).all():
            continue

        # Eigenvalues
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Descending

        # Correlation stability metrics
        # High ratio = one dominant eigenvalue = concentrated correlation
        # Low ratio = spread eigenvalues = diverse/stable correlation
        ratio = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
        cond_num = np.linalg.cond(corr_matrix)

        dates.append(returns.index[i])
        all_eigenvalues.append(eigenvalues)
        ratios.append(ratio)
        condition_numbers.append(cond_num)

    eigenvalues_df = pd.DataFrame(
        all_eigenvalues,
        index=dates,
        columns=[f'eig_{i}' for i in range(n_assets)]
    )

    return (
        eigenvalues_df,
        pd.Series(ratios, index=dates, name='eigenvalue_ratio'),
        pd.Series(condition_numbers, index=dates, name='condition_number')
    )


def compute_fisher_information_series(
    returns: pd.Series,
    window: int = 60
) -> pd.Series:
    """
    Compute Fisher Information for rolling windows.

    For a normal distribution, Fisher Information I = 1/σ²
    We compute a more general version using score function variance.
    """
    fisher_values = []
    dates = []

    for i in range(window, len(returns)):
        window_data = returns.iloc[i-window:i].values

        # Estimate mean and std
        mu = np.mean(window_data)
        sigma = np.std(window_data) + 1e-10

        # Score function for normal distribution: (x - μ)/σ²
        scores = (window_data - mu) / (sigma ** 2)

        # Fisher Information ≈ Var(score function)
        fisher = np.var(scores) * sigma ** 2  # Normalized

        dates.append(returns.index[i])
        fisher_values.append(fisher)

    return pd.Series(fisher_values, index=dates, name='fisher_info')


def compute_correlation_breakdown_indicator(
    eigenvalue_ratio: pd.Series,
    lookback: int = 20
) -> pd.Series:
    """
    Compute a correlation breakdown indicator.

    A spike in eigenvalue ratio indicates sudden concentration of correlation
    (e.g., during crisis when everything correlates to 1).
    """
    # Rate of change in eigenvalue ratio
    ratio_change = eigenvalue_ratio.diff()

    # Z-score of rate of change
    rolling_mean = ratio_change.rolling(window=lookback*3, min_periods=lookback).mean()
    rolling_std = ratio_change.rolling(window=lookback*3, min_periods=lookback).std()

    z_score = (ratio_change - rolling_mean) / (rolling_std + 1e-10)

    return z_score


def compute_lead_lag_correlation(
    series1: pd.Series,
    series2: pd.Series,
    max_lag: int = 30
) -> Tuple[int, float, np.ndarray]:
    """
    Compute lead/lag relationship between two series.

    Returns:
        optimal_lag: Positive if series1 leads series2
        correlation: Correlation at optimal lag
        correlations: Array of correlations at all lags
    """
    # Align series
    common_idx = series1.index.intersection(series2.index)
    s1 = series1.loc[common_idx].values
    s2 = series2.loc[common_idx].values

    # Remove NaN
    mask = np.isfinite(s1) & np.isfinite(s2)
    s1, s2 = s1[mask], s2[mask]

    if len(s1) < 2 * max_lag:
        return 0, 0.0, np.array([])

    # Standardize
    s1 = (s1 - s1.mean()) / (s1.std() + 1e-10)
    s2 = (s2 - s2.mean()) / (s2.std() + 1e-10)

    # Cross-correlation at different lags
    correlations = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            corr = np.corrcoef(s1[-lag:], s2[:lag])[0, 1]
        elif lag > 0:
            corr = np.corrcoef(s1[:-lag], s2[lag:])[0, 1]
        else:
            corr = np.corrcoef(s1, s2)[0, 1]
        correlations.append(corr)

    correlations = np.array(correlations)

    # Find optimal lag
    optimal_idx = np.argmax(np.abs(correlations))
    optimal_lag = optimal_idx - max_lag
    optimal_corr = correlations[optimal_idx]

    return optimal_lag, optimal_corr, correlations


def explain_false_positives(
    spike_dates: pd.DatetimeIndex,
    eigenvalue_ratio: pd.Series,
    known_events: Dict[str, str],
    event_window: int = 15
) -> Tuple[int, List[Dict]]:
    """
    Attempt to explain "false positive" Chern spikes using correlation data.

    A spike is "explained" if:
    1. It corresponds to a significant correlation structure shift, OR
    2. It occurs shortly before a known event (early warning)
    """
    explained = []
    ratio_zscore = (eigenvalue_ratio - eigenvalue_ratio.mean()) / eigenvalue_ratio.std()
    ratio_change = eigenvalue_ratio.diff()
    ratio_change_zscore = (ratio_change - ratio_change.mean()) / (ratio_change.std() + 1e-10)

    for spike_date in spike_dates:
        explanation = {
            'date': spike_date,
            'explained': False,
            'reason': '',
            'details': {}
        }

        # Check if near known event
        for event_date_str, event_name in known_events.items():
            event_date = pd.Timestamp(event_date_str)
            days_before = (event_date - spike_date).days

            if 0 <= days_before <= event_window:
                explanation['explained'] = True
                explanation['reason'] = f'Early warning for {event_name}'
                explanation['details']['days_before'] = days_before
                break

        if not explanation['explained']:
            # Check correlation structure shift
            try:
                idx = eigenvalue_ratio.index.get_indexer([spike_date], method='nearest')[0]
                if idx > 0 and idx < len(eigenvalue_ratio):
                    nearby_dates = eigenvalue_ratio.index[max(0, idx-5):min(len(eigenvalue_ratio), idx+5)]
                    nearby_zscore = ratio_change_zscore.loc[nearby_dates].abs().max()

                    if nearby_zscore > 2.0:
                        explanation['explained'] = True
                        explanation['reason'] = 'Correlation structure shift'
                        explanation['details']['zscore'] = float(nearby_zscore)
            except Exception:
                pass

        explained.append(explanation)

    n_explained = sum(1 for e in explained if e['explained'])
    return n_explained, explained


def run_interpretation_tests(
    spy_only: bool = False,
    start_date: str = '2006-01-01',
    end_date: str = '2024-06-30'
) -> InterpretationResults:
    """
    Run all interpretation tests.

    Args:
        spy_only: If True, use only SPY (faster but less correlation info)
        start_date: Start date for data
        end_date: End date for data
    """
    print("=" * 60)
    print("TRACK A: CHERN INTERPRETATION TEST")
    print("=" * 60)

    # Symbols for correlation analysis
    if spy_only:
        symbols = ['SPY']
    else:
        # Diverse set for correlation analysis
        symbols = ['SPY', 'QQQ', 'IWM', 'XLF', 'XLE', 'XLK', 'XLV', 'TLT', 'GLD', 'UUP']

    # Fetch data
    prices = fetch_multi_asset_data(symbols, start_date, end_date)
    spy_prices = prices['SPY'] if 'SPY' in prices.columns else prices.iloc[:, 0]

    # Create features and compute Chern
    print("\nComputing Chern number series...")
    engine = MinimalFeatureEngine(window=20)
    features = engine.create_feature_matrix(spy_prices.to_frame('SPY'), benchmark_col='SPY')
    features = features.dropna()

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

    chern_series = pd.Series(chern_values, index=chern_times, name='chern')
    delta_chern = chern_series.diff().fillna(0)

    # =========================================
    # TEST 1: Correlation Eigenstructure
    # =========================================
    print("\n" + "-" * 40)
    print("TEST 1: Correlation Eigenstructure")
    print("-" * 40)

    if len(symbols) > 1:
        eigenvalues_df, eigenvalue_ratio, condition_number = compute_rolling_correlation_eigenvalues(
            prices, window=60
        )

        # Align with Chern series
        common_idx = chern_series.index.intersection(eigenvalue_ratio.index)
        chern_aligned = chern_series.loc[common_idx]
        delta_chern_aligned = delta_chern.loc[common_idx]
        ratio_aligned = eigenvalue_ratio.loc[common_idx]

        # Correlation between |ΔChern| and eigenvalue ratio changes
        ratio_change = ratio_aligned.diff().fillna(0)
        correlation = np.corrcoef(delta_chern_aligned.abs(), ratio_change.abs())[0, 1]

        print(f"Correlation between |ΔChern| and |Δ eigenvalue_ratio|: {correlation:.3f}")

        # Check if Chern spikes align with eigenvalue shifts
        rolling_std = delta_chern_aligned.rolling(window=60, min_periods=20).std()
        threshold = 2.0 * rolling_std
        spike_mask = delta_chern_aligned.abs() > threshold
        spike_dates = delta_chern_aligned[spike_mask].index

        # For each spike, check if there's a corresponding eigenvalue shift
        aligned_count = 0
        for spike_date in spike_dates:
            try:
                idx = ratio_change.index.get_indexer([spike_date], method='nearest')[0]
                nearby = ratio_change.iloc[max(0, idx-5):min(len(ratio_change), idx+5)]
                if nearby.abs().max() > ratio_change.abs().quantile(0.9):
                    aligned_count += 1
            except Exception:
                pass

        alignment_rate = aligned_count / len(spike_dates) if len(spike_dates) > 0 else 0
        print(f"Chern spikes aligned with eigenvalue shifts: {alignment_rate:.1%}")

        eigenvalue_chern_correlation = correlation
        eigenvalue_shift_spike_alignment = alignment_rate
    else:
        print("Single asset mode - skipping correlation eigenstructure test")
        eigenvalue_chern_correlation = 0.0
        eigenvalue_shift_spike_alignment = 0.0
        ratio_aligned = pd.Series(dtype=float)

    # =========================================
    # TEST 2: Information Geometry Connection
    # =========================================
    print("\n" + "-" * 40)
    print("TEST 2: Information Geometry Connection")
    print("-" * 40)

    returns = spy_prices.pct_change().dropna()
    fisher_series = compute_fisher_information_series(returns, window=60)

    # Compute Berry curvature magnitude series
    berry_curvature_mag = []
    for i in range(len(X)):
        F = geometry.berry_curvature(X[i])
        berry_curvature_mag.append(np.linalg.norm(F))

    berry_series = pd.Series(
        berry_curvature_mag,
        index=features.index,
        name='berry_curvature'
    )

    # Align and correlate
    common_idx = fisher_series.index.intersection(berry_series.index)
    fisher_aligned = fisher_series.loc[common_idx]
    berry_aligned = berry_series.loc[common_idx]

    fisher_berry_corr = np.corrcoef(fisher_aligned, berry_aligned)[0, 1]
    print(f"Correlation between Fisher Information and Berry Curvature: {fisher_berry_corr:.3f}")

    # Fisher metric stability (lower = more stable)
    fisher_stability = fisher_aligned.rolling(window=20).std().mean() / fisher_aligned.mean()
    print(f"Fisher metric relative stability: {fisher_stability:.3f}")

    fisher_berry_correlation = fisher_berry_corr if np.isfinite(fisher_berry_corr) else 0.0
    fisher_metric_stability = fisher_stability if np.isfinite(fisher_stability) else 0.0

    # =========================================
    # TEST 3: Lead/Lag Analysis
    # =========================================
    print("\n" + "-" * 40)
    print("TEST 3: Lead/Lag Analysis")
    print("-" * 40)

    if len(symbols) > 1 and len(ratio_aligned) > 0:
        # Compute correlation breakdown indicator
        corr_breakdown = compute_correlation_breakdown_indicator(ratio_aligned, lookback=20)

        # Lead/lag between |ΔChern| and correlation breakdown
        optimal_lag, lead_lag_corr, all_correlations = compute_lead_lag_correlation(
            delta_chern_aligned.abs(),
            corr_breakdown.abs(),
            max_lag=30
        )

        print(f"Optimal lag: {optimal_lag} days (positive = Chern leads)")
        print(f"Correlation at optimal lag: {lead_lag_corr:.3f}")

        chern_leads = optimal_lag > 0
        print(f"Chern leads correlation breakdown: {chern_leads}")

        optimal_lag_days = optimal_lag
        lead_lag_correlation = lead_lag_corr
        chern_leads_correlation = chern_leads
    else:
        print("Single asset mode - skipping lead/lag test")
        optimal_lag_days = 0
        lead_lag_correlation = 0.0
        chern_leads_correlation = False

    # =========================================
    # TEST 4: Explain False Positives
    # =========================================
    print("\n" + "-" * 40)
    print("TEST 4: False Positive Explanation")
    print("-" * 40)

    known_events = {
        '2008-09-15': 'Lehman',
        '2008-10-06': 'October Crash',
        '2010-05-06': 'Flash Crash',
        '2011-08-08': 'Downgrade',
        '2015-08-24': 'China',
        '2018-02-05': 'Volmageddon',
        '2018-12-24': 'Xmas Selloff',
        '2020-02-24': 'COVID Start',
        '2020-03-16': 'COVID Bottom',
        '2022-01-24': 'Fed Fear',
        '2022-03-16': 'Rate Hike',
        '2022-06-13': 'Bear Market',
        '2023-03-10': 'SVB',
    }

    # Find all Chern spikes
    rolling_std = delta_chern.rolling(window=60, min_periods=20).std()
    threshold = 2.0 * rolling_std
    spike_mask = delta_chern.abs() > threshold
    all_spike_dates = delta_chern[spike_mask].index

    # Classify as TP or FP
    fp_dates = []
    for spike_date in all_spike_dates:
        is_near_event = False
        for event_date_str in known_events:
            event_date = pd.Timestamp(event_date_str)
            if abs((spike_date - event_date).days) <= 15:
                is_near_event = True
                break
        if not is_near_event:
            fp_dates.append(spike_date)

    print(f"Total spikes: {len(all_spike_dates)}")
    print(f"False positives (not near known events): {len(fp_dates)}")

    if len(symbols) > 1 and len(ratio_aligned) > 0:
        n_explained, explanations = explain_false_positives(
            pd.DatetimeIndex(fp_dates),
            ratio_aligned,
            known_events,
            event_window=30  # 30 days is "early warning"
        )

        print(f"False positives explained by correlation shifts: {n_explained}")
        print(f"Explanation rate: {n_explained / len(fp_dates) if len(fp_dates) > 0 else 0:.1%}")

        # Show some examples
        print("\nSample explanations:")
        for exp in explanations[:5]:
            if exp['explained']:
                print(f"  {exp['date'].strftime('%Y-%m-%d')}: {exp['reason']}")

        explained_fps = n_explained
        fp_explanation_rate = n_explained / len(fp_dates) if len(fp_dates) > 0 else 0.0
    else:
        explained_fps = 0
        fp_explanation_rate = 0.0

    # =========================================
    # SUMMARY
    # =========================================
    print("\n" + "=" * 60)
    print("INTERPRETATION TEST SUMMARY")
    print("=" * 60)

    results = InterpretationResults(
        eigenvalue_chern_correlation=eigenvalue_chern_correlation,
        eigenvalue_shift_spike_alignment=eigenvalue_shift_spike_alignment,
        fisher_berry_correlation=fisher_berry_correlation,
        fisher_metric_stability=fisher_metric_stability,
        optimal_lag_days=optimal_lag_days,
        lead_lag_correlation=lead_lag_correlation,
        chern_leads_correlation=chern_leads_correlation,
        explained_false_positives=explained_fps,
        total_false_positives=len(fp_dates),
        fp_explanation_rate=fp_explanation_rate
    )

    print(f"\n1. CORRELATION EIGENSTRUCTURE:")
    print(f"   |ΔChern| ↔ |Δ eigenvalue_ratio| correlation: {results.eigenvalue_chern_correlation:.3f}")
    print(f"   Spike-shift alignment rate: {results.eigenvalue_shift_spike_alignment:.1%}")

    print(f"\n2. INFORMATION GEOMETRY:")
    print(f"   Fisher ↔ Berry correlation: {results.fisher_berry_correlation:.3f}")
    print(f"   Fisher stability: {results.fisher_metric_stability:.3f}")

    print(f"\n3. LEAD/LAG ANALYSIS:")
    print(f"   Optimal lag: {results.optimal_lag_days} days")
    print(f"   Lead/lag correlation: {results.lead_lag_correlation:.3f}")
    print(f"   Chern leads correlation breakdown: {results.chern_leads_correlation}")

    print(f"\n4. FALSE POSITIVE EXPLANATION:")
    print(f"   Explained: {results.explained_false_positives}/{results.total_false_positives}")
    print(f"   Explanation rate: {results.fp_explanation_rate:.1%}")

    # Verdict
    print("\n" + "-" * 40)
    print("VERDICT:")
    if results.eigenvalue_chern_correlation > 0.3 or results.eigenvalue_shift_spike_alignment > 0.5:
        print("✓ Chern IS measuring correlation structure changes")
    else:
        print("✗ No clear link between Chern and correlation structure")

    if results.chern_leads_correlation and results.optimal_lag_days > 5:
        print(f"✓ Chern LEADS correlation breakdown by ~{results.optimal_lag_days} days")
    elif results.optimal_lag_days < -5:
        print(f"✗ Chern LAGS correlation breakdown by ~{-results.optimal_lag_days} days")
    else:
        print("○ Chern is roughly contemporaneous with correlation changes")

    if results.fp_explanation_rate > 0.5:
        print("✓ Most 'false positives' are actually correlation structure events")
    else:
        print("✗ False positives remain unexplained")

    return results


def create_visualization(
    chern_series: pd.Series,
    eigenvalue_ratio: pd.Series,
    fisher_series: pd.Series,
    berry_series: pd.Series,
    output_path: Path
):
    """Create visualization of interpretation results."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)

    # Panel 1: Chern number
    ax1 = axes[0]
    ax1.plot(chern_series.index, chern_series.values, 'b-', linewidth=0.8)
    ax1.set_ylabel('Chern Number')
    ax1.set_title('Chern Number Evolution', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Eigenvalue ratio
    ax2 = axes[1]
    ax2.plot(eigenvalue_ratio.index, eigenvalue_ratio.values, 'r-', linewidth=0.8)
    ax2.set_ylabel('Eigenvalue Ratio')
    ax2.set_title('Correlation Matrix Eigenvalue Ratio (Larger = More Concentrated)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Fisher Information
    ax3 = axes[2]
    ax3.plot(fisher_series.index, fisher_series.values, 'g-', linewidth=0.8)
    ax3.set_ylabel('Fisher Information')
    ax3.set_title('Fisher Information (Information Geometry)', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Berry Curvature
    ax4 = axes[3]
    ax4.plot(berry_series.index, berry_series.values, 'm-', linewidth=0.8)
    ax4.set_ylabel('Berry Curvature ||F||')
    ax4.set_title('Berry Curvature Magnitude', fontsize=12)
    ax4.set_xlabel('Date')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")
    plt.close()


if __name__ == "__main__":
    np.random.seed(42)

    # Run full test with multi-asset correlation analysis
    results = run_interpretation_tests(spy_only=False)

    # Save results
    output_dir = project_root / 'experiments' / 'outputs'
    output_dir.mkdir(exist_ok=True)

    results_dict = {
        'eigenvalue_chern_correlation': float(results.eigenvalue_chern_correlation),
        'eigenvalue_shift_spike_alignment': float(results.eigenvalue_shift_spike_alignment),
        'fisher_berry_correlation': float(results.fisher_berry_correlation),
        'fisher_metric_stability': float(results.fisher_metric_stability),
        'optimal_lag_days': int(results.optimal_lag_days),
        'lead_lag_correlation': float(results.lead_lag_correlation),
        'chern_leads_correlation': bool(results.chern_leads_correlation),
        'explained_false_positives': int(results.explained_false_positives),
        'total_false_positives': int(results.total_false_positives),
        'fp_explanation_rate': float(results.fp_explanation_rate)
    }

    with open(output_dir / 'track_a_interpretation_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'track_a_interpretation_results.json'}")
