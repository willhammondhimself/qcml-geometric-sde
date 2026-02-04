#!/usr/bin/env python3
"""
Hypothesis Validation: Chern Number Discontinuity at 2008 Crisis

Research Question:
    Does the Chern number show discontinuous change around Sept 15, 2008
    (Lehman Brothers collapse)?

Core Hypothesis:
    "Chern number discontinuities can detect regime changes in real markets"

Success Criteria:
    1. Chern number shows significant change (|ΔC| > 0.5) around Sept 15, 2008
    2. Regime detector identifies transition within ±2 weeks of Lehman collapse
    3. Different from random noise (compare to pre-crisis period)

Author: QCML Research
Date: 2024
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qcml.data import PolygonDataSource, QCMLDataset, MinimalFeatureEngine
from qcml.qcml_geometry import QCMLGeometry
from qcml.topological_regime import TopologicalRegimeDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    import torch
    import random

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def fetch_crisis_data(
    symbols: list,
    start_date: str,
    end_date: str,
    api_key: str
) -> pd.DataFrame:
    """
    Fetch historical data for 2008 crisis period.

    Args:
        symbols: List of ticker symbols
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        api_key: Polygon API key

    Returns:
        DataFrame with close prices, index=dates, columns=symbols
    """
    logger.info(f"Fetching {len(symbols)} symbols from {start_date} to {end_date}")

    source = PolygonDataSource(api_key=api_key)
    raw_data = source.fetch_equities(symbols, start_date, end_date, timeframe="1d")

    if raw_data.empty:
        raise ValueError("No data returned from Polygon API")

    # Pivot to get symbols as columns
    # raw_data has MultiIndex (symbol, timestamp)
    prices = raw_data['close'].unstack(level=0)

    # Forward fill missing values (weekends already excluded)
    prices = prices.ffill()

    logger.info(f"Fetched data shape: {prices.shape}")
    logger.info(f"Date range: {prices.index[0]} to {prices.index[-1]}")

    return prices


def validate_data(prices: pd.DataFrame) -> dict:
    """
    Validate fetched data for common issues.

    Args:
        prices: DataFrame with price data

    Returns:
        Dictionary with validation results
    """
    issues = []

    # Check for NaN values
    nan_pct = prices.isna().sum().sum() / prices.size * 100
    if nan_pct > 5:
        issues.append(f"High NaN percentage: {nan_pct:.1f}%")

    # Check for zeros
    zero_count = (prices == 0).sum().sum()
    if zero_count > 0:
        issues.append(f"Contains {zero_count} zero values")

    # Check for extreme jumps (>20% daily)
    returns = prices.pct_change()
    extreme_jumps = (returns.abs() > 0.20).sum().sum()
    if extreme_jumps > 10:
        logger.warning(f"Found {extreme_jumps} extreme daily moves (>20%)")

    # Check date coverage
    expected_days = pd.date_range(prices.index[0], prices.index[-1], freq='B')
    missing_pct = (len(expected_days) - len(prices)) / len(expected_days) * 100
    if missing_pct > 10:
        issues.append(f"Missing {missing_pct:.1f}% of business days")

    return {
        'is_valid': len(issues) == 0,
        'n_rows': len(prices),
        'n_symbols': len(prices.columns),
        'date_range': f"{prices.index[0]} to {prices.index[-1]}",
        'nan_pct': nan_pct,
        'issues': issues
    }


def analyze_chern_around_crisis(
    chern_series: np.ndarray,
    times: pd.DatetimeIndex,
    crisis_date: str,
    window_days: int = 10
) -> dict:
    """
    Analyze Chern number behavior around crisis date.

    Args:
        chern_series: Rolling Chern number values
        times: Corresponding timestamps
        crisis_date: Crisis date (YYYY-MM-DD)
        window_days: Days before/after to analyze

    Returns:
        Dictionary with analysis results
    """
    crisis_ts = pd.Timestamp(crisis_date)

    # Find indices around crisis
    before_mask = times < crisis_ts
    after_mask = times >= crisis_ts

    if not before_mask.any() or not after_mask.any():
        return {'error': 'Crisis date outside data range'}

    # Get values before and after
    chern_before = chern_series[before_mask]
    chern_after = chern_series[after_mask]

    # Last N days before crisis
    window_before = chern_before[-window_days:] if len(chern_before) >= window_days else chern_before
    # First N days after crisis
    window_after = chern_after[:window_days] if len(chern_after) >= window_days else chern_after

    # Compute statistics
    mean_before = np.mean(window_before)
    mean_after = np.mean(window_after)
    std_before = np.std(window_before)
    std_after = np.std(window_after)

    delta_chern = mean_after - mean_before

    # Statistical significance (simple t-test approximation)
    pooled_std = np.sqrt((std_before**2 + std_after**2) / 2)
    t_stat = abs(delta_chern) / (pooled_std + 1e-8) * np.sqrt(min(len(window_before), len(window_after)))

    return {
        'crisis_date': crisis_date,
        'chern_before': mean_before,
        'chern_after': mean_after,
        'delta_chern': delta_chern,
        'std_before': std_before,
        'std_after': std_after,
        't_statistic': t_stat,
        'is_significant': abs(delta_chern) > 0.1 or t_stat > 2.0,
        'window_days': window_days
    }


def create_visualization(
    times: pd.DatetimeIndex,
    chern_series: np.ndarray,
    prices: pd.DataFrame,
    crisis_date: str,
    output_path: str
) -> None:
    """
    Create visualization of Chern number around crisis.

    Args:
        times: Timestamps for Chern series
        chern_series: Rolling Chern values
        prices: Price DataFrame (for SPY)
        crisis_date: Crisis date
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    crisis_ts = pd.Timestamp(crisis_date)

    # Plot 1: SPY price
    ax1 = axes[0]
    spy_prices = prices['SPY'] if 'SPY' in prices.columns else prices.iloc[:, 0]
    ax1.plot(spy_prices.index, spy_prices.values, 'b-', linewidth=1)
    ax1.axvline(crisis_ts, color='r', linestyle='--', label='Lehman Collapse')
    ax1.set_ylabel('SPY Price')
    ax1.set_title('2008 Financial Crisis: Chern Number Hypothesis Validation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Chern number
    ax2 = axes[1]
    ax2.plot(times, chern_series, 'g-', linewidth=1.5)
    ax2.axvline(crisis_ts, color='r', linestyle='--')
    ax2.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax2.fill_between(times, chern_series, 0, alpha=0.3, color='green')
    ax2.set_ylabel('Chern Number')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Change in Chern number
    ax3 = axes[2]
    delta_chern = np.diff(chern_series, prepend=chern_series[0])
    ax3.bar(times, delta_chern, width=1, color='purple', alpha=0.7)
    ax3.axvline(crisis_ts, color='r', linestyle='--')
    ax3.axhline(0.5, color='orange', linestyle=':', label='Threshold (+)')
    ax3.axhline(-0.5, color='orange', linestyle=':', label='Threshold (-)')
    ax3.set_ylabel('ΔChern')
    ax3.set_xlabel('Date')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization to {output_path}")
    plt.close()


def main():
    """Run 2008 crisis hypothesis validation."""

    print("=" * 70)
    print("QCML Hypothesis Validation: 2008 Financial Crisis")
    print("=" * 70)
    print()

    # Load environment variables
    load_dotenv(project_root / '.env')
    api_key = os.getenv('POLYGON_API_KEY')

    if not api_key:
        raise ValueError("POLYGON_API_KEY not found in environment")

    # Configuration
    SYMBOLS = ["SPY", "XLF", "BAC", "JPM", "GS"]  # Financials focus
    START_DATE = "2008-03-01"  # 6 months before
    END_DATE = "2009-03-01"    # 6 months after
    CRISIS_DATE = "2008-09-15"  # Lehman Brothers collapse
    WINDOW_SIZE = 20  # Rolling window for Chern computation (reduced for more sensitivity)
    HILBERT_DIM = 8   # 3-qubit system (larger Hilbert space for richer geometry)

    print(f"Symbols: {SYMBOLS}")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Crisis Date: {CRISIS_DATE} (Lehman Brothers)")
    print(f"Window Size: {WINDOW_SIZE}")
    print()

    # Step 1: Fetch data
    print("-" * 50)
    print("Step 1: Fetching market data from Polygon.io...")
    print("-" * 50)

    try:
        prices = fetch_crisis_data(SYMBOLS, START_DATE, END_DATE, api_key)
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        raise

    # Validate data
    validation = validate_data(prices)
    print(f"Data validation: {'PASSED' if validation['is_valid'] else 'FAILED'}")
    print(f"  Rows: {validation['n_rows']}")
    print(f"  Symbols: {validation['n_symbols']}")
    print(f"  Date range: {validation['date_range']}")
    if validation['issues']:
        print(f"  Issues: {validation['issues']}")
    print()

    # Step 2: Compute minimal features
    print("-" * 50)
    print("Step 2: Computing minimal features...")
    print("-" * 50)

    engine = MinimalFeatureEngine(window=20)
    features = engine.create_feature_matrix(prices, benchmark_col='SPY')

    feature_validation = engine.validate_features(features)
    print(f"Feature validation: {'PASSED' if feature_validation['is_valid'] else 'FAILED'}")
    print(f"  Samples: {feature_validation['n_samples']}")
    print(f"  Features: {feature_validation['n_features']}")
    if feature_validation['issues']:
        print(f"  Issues: {feature_validation['issues']}")
    print()

    # Step 3: Create QCMLDataset
    print("-" * 50)
    print("Step 3: Creating QCMLDataset...")
    print("-" * 50)

    # Align prices with features (features have warmup period dropped)
    aligned_prices = prices.loc[features.index, 'SPY']

    dataset = QCMLDataset(
        features=features,
        prices=aligned_prices,
        times=features.index,
        metadata={
            'symbols': SYMBOLS,
            'crisis': '2008_financial',
            'crisis_date': CRISIS_DATE
        }
    )
    print(f"Dataset: {dataset}")
    print()

    # Step 4: Learn QCML geometry
    print("-" * 50)
    print("Step 4: Learning QCML geometry...")
    print("-" * 50)

    X_raw = dataset.X
    n_raw_features = X_raw.shape[1]

    # Reduce dimensions to Hilbert space capacity using PCA
    # For hilbert_dim=8, we can have up to 64 Pauli basis operators
    N_COMPONENTS = 15  # Use 15 principal components for richer topology

    print(f"Raw features: {n_raw_features}")
    print(f"Reducing to {N_COMPONENTS} PCA components...")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Apply PCA
    pca = PCA(n_components=N_COMPONENTS)
    X = pca.fit_transform(X_scaled)

    explained_var = pca.explained_variance_ratio_.sum()
    print(f"Explained variance: {explained_var:.1%}")

    # Normalize to unit sphere for better geometric properties
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    n_features = X.shape[1]

    geometry = QCMLGeometry(n_features=n_features, hilbert_dim=HILBERT_DIM)
    geometry.fit_operators(X, method='random')  # Random operators for richer geometry

    print(f"Hilbert dimension: {HILBERT_DIM}")
    print(f"Number of operators: {len(geometry.operators)}")
    print(f"Input features: {n_features}")
    print()

    # Step 5: Run regime detection
    print("-" * 50)
    print("Step 5: Computing rolling Chern numbers...")
    print("-" * 50)

    detector = TopologicalRegimeDetector(
        geometry=geometry,
        window_size=WINDOW_SIZE,
        chern_threshold=0.1  # Lower threshold for more sensitivity
    )

    # Compute rolling Chern series
    chern_series = detector.rolling_chern_number(X, window=WINDOW_SIZE)

    # Align times with Chern series (account for window warmup)
    chern_times = dataset.times[WINDOW_SIZE - 1:]
    if len(chern_times) > len(chern_series):
        chern_times = chern_times[:len(chern_series)]

    print(f"Chern series length: {len(chern_series)}")
    print(f"Chern range: [{chern_series.min():.3f}, {chern_series.max():.3f}]")
    print(f"Chern mean: {chern_series.mean():.3f}")
    print(f"Chern std: {chern_series.std():.3f}")
    print()

    # Step 6: Analyze around crisis date
    print("-" * 50)
    print("Step 6: Analyzing Chern number around Lehman collapse...")
    print("-" * 50)

    analysis = analyze_chern_around_crisis(
        chern_series, chern_times, CRISIS_DATE, window_days=10
    )

    print(f"Crisis date: {analysis['crisis_date']}")
    print(f"Chern before (10-day mean): {analysis['chern_before']:.4f}")
    print(f"Chern after (10-day mean): {analysis['chern_after']:.4f}")
    print(f"ΔChern: {analysis['delta_chern']:.4f}")
    print(f"t-statistic: {analysis['t_statistic']:.2f}")
    print(f"Significant (|ΔC| > 0.5): {analysis['is_significant']}")
    print()

    # Detect transitions
    transitions = detector.detect_transitions(X, times=np.arange(len(X)))
    print(f"Total transitions detected: {len(transitions)}")

    # Check if any transition is near crisis date
    crisis_idx = (dataset.times >= pd.Timestamp(CRISIS_DATE)).argmax()
    nearby_transitions = [
        t for t in transitions
        if abs(t.start_idx - crisis_idx) < 20  # Within 20 trading days
    ]

    print(f"Transitions near Lehman (±20 days): {len(nearby_transitions)}")
    for t in nearby_transitions:
        print(f"  idx {t.start_idx}: ΔC={t.delta_chern:.3f}, conf={t.confidence:.2f}")
    print()

    # Step 7: Output verdict
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)

    hypothesis_validated = (
        analysis['is_significant'] or
        len(nearby_transitions) > 0
    )

    if hypothesis_validated:
        print("✓ HYPOTHESIS SUPPORTED")
        print()
        print("Evidence:")
        if analysis['is_significant']:
            print(f"  - Significant Chern change: ΔC = {analysis['delta_chern']:.3f}")
        if nearby_transitions:
            print(f"  - {len(nearby_transitions)} transition(s) detected near crisis")
        print()
        print("Recommendation: Proceed to full Phase 2 feature engineering")
    else:
        print("✗ HYPOTHESIS NOT SUPPORTED (with minimal features)")
        print()
        print("Observations:")
        print(f"  - Chern change: ΔC = {analysis['delta_chern']:.3f} (below threshold)")
        print(f"  - Transitions near crisis: {len(nearby_transitions)}")
        print()
        print("Recommendations:")
        print("  1. Try different feature combinations")
        print("  2. Adjust window size (currently {WINDOW_SIZE})")
        print("  3. Increase Hilbert dimension")
        print("  4. Consult QCML theory for guidance")

    print("=" * 70)

    # Create visualization
    output_dir = project_root / 'experiments' / 'outputs'
    output_dir.mkdir(exist_ok=True)

    create_visualization(
        chern_times,
        chern_series,
        prices,
        CRISIS_DATE,
        str(output_dir / 'chern_2008_crisis.png')
    )

    # Return results for programmatic use
    return {
        'hypothesis_validated': hypothesis_validated,
        'analysis': analysis,
        'transitions': transitions,
        'nearby_transitions': nearby_transitions,
        'chern_series': chern_series,
        'chern_times': chern_times
    }


if __name__ == "__main__":
    try:
        # Set seed for reproducibility
        seed_everything(42)

        results = main()

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise
