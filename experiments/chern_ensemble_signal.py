#!/usr/bin/env python3
"""
Track C: Chern Ensemble Signal

Combines Chern number with other market indicators to create a more
robust regime detection signal.

Key components:
1. Chern number (topological component)
2. Realized volatility (20-day)
3. Correlation stability metric
4. VIX if available

Tests:
- Simple classifier: "high risk" when multiple signals agree
- Ablation test: Does Chern contribute unique information?
- Precision/recall comparison: Ensemble vs individual signals

Author: QCML Research
Date: 2024
"""

import os
import sys
from pathlib import Path
import warnings
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
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
class EnsembleResults:
    """Results from ensemble signal analysis."""
    # Individual signal metrics
    chern_precision: float
    chern_recall: float
    chern_f1: float

    volatility_precision: float
    volatility_recall: float
    volatility_f1: float

    correlation_precision: float
    correlation_recall: float
    correlation_f1: float

    # Ensemble metrics
    ensemble_precision: float
    ensemble_recall: float
    ensemble_f1: float

    # Ablation (ensemble without Chern)
    ablation_precision: float
    ablation_recall: float
    ablation_f1: float

    # Chern contribution
    chern_lift: float  # F1 improvement from adding Chern
    chern_unique_detections: int
    total_ensemble_detections: int


def compute_realized_volatility(
    prices: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    Compute annualized realized volatility.

    Args:
        prices: Price series
        window: Rolling window in days

    Returns:
        volatility: Annualized realized volatility
    """
    returns = prices.pct_change()
    vol = returns.rolling(window=window).std() * np.sqrt(252)
    return vol


def compute_correlation_stability(
    prices: pd.DataFrame,
    window: int = 60
) -> pd.Series:
    """
    Compute correlation stability metric from multi-asset prices.

    Uses eigenvalue ratio of correlation matrix - higher values indicate
    more concentrated (less stable) correlation structure.

    Args:
        prices: Multi-asset price DataFrame
        window: Rolling window

    Returns:
        stability: Correlation stability metric (inverted - higher = less stable)
    """
    if prices.shape[1] < 2:
        # Single asset - use return autocorrelation as proxy
        returns = prices.iloc[:, 0].pct_change()
        autocorr = returns.rolling(window=window).apply(
            lambda x: x.autocorr() if len(x) > 1 else 0,
            raw=False
        )
        return autocorr.abs()

    returns = prices.pct_change()
    eigenvalue_ratios = []
    indices = []

    for i in range(window, len(returns)):
        window_returns = returns.iloc[i-window:i]
        corr_matrix = window_returns.corr().values

        if not np.isfinite(corr_matrix).all():
            continue

        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]

        # Ratio of largest to smallest eigenvalue
        ratio = eigenvalues[0] / (eigenvalues[-1] + 1e-10)

        eigenvalue_ratios.append(ratio)
        indices.append(returns.index[i])

    return pd.Series(eigenvalue_ratios, index=indices, name='corr_stability')


def fetch_vix(start_date: str, end_date: str) -> Optional[pd.Series]:
    """
    Attempt to fetch VIX data.

    Args:
        start_date: Start date
        end_date: End date

    Returns:
        vix: VIX series or None if unavailable
    """
    try:
        api_key = os.getenv('POLYGON_API_KEY')
        source = PolygonDataSource(api_key=api_key)

        # Try to fetch VIX
        vix_data = source.fetch_equities(['VIXY'], start_date, end_date, timeframe='1d')
        vix = vix_data['close'].unstack(level=0)['VIXY'].ffill()
        return vix
    except Exception as e:
        print(f"VIX not available: {e}")
        return None


def compute_signal_spikes(
    signal: pd.Series,
    threshold_std: float = 2.0,
    lookback: int = 60
) -> pd.Series:
    """
    Detect spikes in a signal using rolling z-score.

    Args:
        signal: Signal series
        threshold_std: Number of standard deviations for spike
        lookback: Rolling window for statistics

    Returns:
        spikes: Boolean series indicating spikes
    """
    rolling_mean = signal.rolling(window=lookback, min_periods=lookback//2).mean()
    rolling_std = signal.rolling(window=lookback, min_periods=lookback//2).std()

    z_score = (signal - rolling_mean) / (rolling_std + 1e-10)
    spikes = z_score.abs() > threshold_std

    return spikes


def create_crisis_labels(
    dates: pd.DatetimeIndex,
    known_events: Dict[str, str],
    window_before: int = 10,
    window_after: int = 10
) -> pd.Series:
    """
    Create binary labels for crisis periods.

    A period is labeled as "crisis" if it's within window of a known event.

    Args:
        dates: DatetimeIndex of observations
        known_events: Dict mapping date strings to event names
        window_before: Days before event to include
        window_after: Days after event to include

    Returns:
        labels: Binary series (1 = crisis period)
    """
    labels = pd.Series(0, index=dates)

    for event_date_str in known_events.keys():
        event_date = pd.Timestamp(event_date_str)

        for date in dates:
            days_diff = (date - event_date).days
            if -window_before <= days_diff <= window_after:
                labels.loc[date] = 1

    return labels


def compute_signal_metrics(
    signal_spikes: pd.Series,
    labels: pd.Series
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, F1 for a signal.

    Args:
        signal_spikes: Boolean spike indicators
        labels: Ground truth crisis labels

    Returns:
        precision, recall, f1
    """
    common_idx = signal_spikes.index.intersection(labels.index)
    spikes = signal_spikes.loc[common_idx].astype(int)
    truth = labels.loc[common_idx].astype(int)

    if spikes.sum() == 0 or truth.sum() == 0:
        return 0.0, 0.0, 0.0

    precision = precision_score(truth, spikes, zero_division=0)
    recall = recall_score(truth, spikes, zero_division=0)
    f1 = f1_score(truth, spikes, zero_division=0)

    return precision, recall, f1


def build_ensemble_signal(
    chern_spikes: pd.Series,
    vol_spikes: pd.Series,
    corr_spikes: pd.Series,
    vix_spikes: Optional[pd.Series] = None,
    min_signals: int = 2
) -> pd.Series:
    """
    Build ensemble signal that fires when multiple indicators agree.

    Args:
        chern_spikes: Chern number spike indicators
        vol_spikes: Volatility spike indicators
        corr_spikes: Correlation stability spike indicators
        vix_spikes: Optional VIX spike indicators
        min_signals: Minimum number of signals that must agree

    Returns:
        ensemble: Boolean series indicating ensemble signal
    """
    # Align all signals
    common_idx = chern_spikes.index.intersection(vol_spikes.index).intersection(corr_spikes.index)

    if vix_spikes is not None:
        common_idx = common_idx.intersection(vix_spikes.index)

    signal_sum = (
        chern_spikes.reindex(common_idx, fill_value=False).astype(int) +
        vol_spikes.reindex(common_idx, fill_value=False).astype(int) +
        corr_spikes.reindex(common_idx, fill_value=False).astype(int)
    )

    if vix_spikes is not None:
        signal_sum += vix_spikes.reindex(common_idx, fill_value=False).astype(int)

    ensemble = signal_sum >= min_signals

    return ensemble


def run_ablation_test(
    vol_spikes: pd.Series,
    corr_spikes: pd.Series,
    labels: pd.Series,
    vix_spikes: Optional[pd.Series] = None
) -> Tuple[float, float, float]:
    """
    Run ablation test: ensemble WITHOUT Chern number.

    This tests whether Chern provides unique information.

    Args:
        vol_spikes: Volatility spikes
        corr_spikes: Correlation spikes
        labels: Ground truth labels
        vix_spikes: Optional VIX spikes

    Returns:
        precision, recall, f1 for ablated ensemble
    """
    common_idx = vol_spikes.index.intersection(corr_spikes.index)

    if vix_spikes is not None:
        common_idx = common_idx.intersection(vix_spikes.index)

    signal_sum = (
        vol_spikes.reindex(common_idx, fill_value=False).astype(int) +
        corr_spikes.reindex(common_idx, fill_value=False).astype(int)
    )

    if vix_spikes is not None:
        signal_sum += vix_spikes.reindex(common_idx, fill_value=False).astype(int)
        min_signals = 2
    else:
        min_signals = 2

    ablated_ensemble = signal_sum >= min_signals

    return compute_signal_metrics(ablated_ensemble, labels)


def train_ml_ensemble(
    features_df: pd.DataFrame,
    labels: pd.Series,
    test_ratio: float = 0.3
) -> Dict:
    """
    Train ML-based ensemble for comparison.

    Args:
        features_df: DataFrame with all signal features
        labels: Ground truth labels
        test_ratio: Fraction of data for testing

    Returns:
        results: Dict with model performance
    """
    common_idx = features_df.index.intersection(labels.index)
    X = features_df.loc[common_idx].values
    y = labels.loc[common_idx].values

    # Handle NaN
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    y = y[mask]

    if len(X) < 100:
        return {'error': 'Insufficient data'}

    # Time series split
    split_idx = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Logistic regression
    lr = LogisticRegression(class_weight='balanced', max_iter=1000)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    # Random forest
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    return {
        'logistic_regression': {
            'precision': precision_score(y_test, lr_pred, zero_division=0),
            'recall': recall_score(y_test, lr_pred, zero_division=0),
            'f1': f1_score(y_test, lr_pred, zero_division=0),
            'feature_importance': dict(zip(features_df.columns, lr.coef_[0]))
        },
        'random_forest': {
            'precision': precision_score(y_test, rf_pred, zero_division=0),
            'recall': recall_score(y_test, rf_pred, zero_division=0),
            'f1': f1_score(y_test, rf_pred, zero_division=0),
            'feature_importance': dict(zip(features_df.columns, rf.feature_importances_))
        }
    }


def run_ensemble_test(
    start_date: str = '2006-01-01',
    end_date: str = '2024-06-30'
) -> EnsembleResults:
    """
    Run the full ensemble signal test.

    Args:
        start_date: Start date for data
        end_date: End date for data

    Returns:
        results: EnsembleResults with all metrics
    """
    print("=" * 60)
    print("TRACK C: CHERN ENSEMBLE SIGNAL")
    print("=" * 60)

    # Fetch multi-asset data
    api_key = os.getenv('POLYGON_API_KEY')
    source = PolygonDataSource(api_key=api_key)

    symbols = ['SPY', 'QQQ', 'IWM', 'XLF', 'XLE', 'XLK', 'XLV', 'TLT', 'GLD']

    print("\nFetching data...")
    data = source.fetch_equities(symbols, start_date, end_date, timeframe='1d')
    prices = data['close'].unstack(level=0)
    prices = prices.ffill().dropna(how='all')

    spy_prices = prices['SPY'] if 'SPY' in prices.columns else prices.iloc[:, 0]

    print(f"Fetched {len(prices)} days of data for {len(prices.columns)} assets")

    # Known events for labeling
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

    # =========================================
    # COMPUTE INDIVIDUAL SIGNALS
    # =========================================
    print("\n" + "-" * 40)
    print("Computing individual signals...")
    print("-" * 40)

    # 1. Chern number
    print("1. Computing Chern number...")
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
    chern_spikes = compute_signal_spikes(delta_chern.abs(), threshold_std=2.0)

    print(f"   Chern spikes: {chern_spikes.sum()}")

    # 2. Realized volatility
    print("2. Computing realized volatility...")
    realized_vol = compute_realized_volatility(spy_prices, window=20)
    vol_change = realized_vol.diff()
    vol_spikes = compute_signal_spikes(vol_change.abs(), threshold_std=2.0)

    print(f"   Volatility spikes: {vol_spikes.sum()}")

    # 3. Correlation stability
    print("3. Computing correlation stability...")
    corr_stability = compute_correlation_stability(prices, window=60)
    corr_change = corr_stability.diff()
    corr_spikes = compute_signal_spikes(corr_change.abs(), threshold_std=2.0)

    print(f"   Correlation spikes: {corr_spikes.sum()}")

    # 4. VIX (optional)
    print("4. Attempting to fetch VIX...")
    vix_series = fetch_vix(start_date, end_date)
    vix_spikes = None

    if vix_series is not None:
        vix_change = vix_series.diff()
        vix_spikes = compute_signal_spikes(vix_change.abs(), threshold_std=2.0)
        print(f"   VIX spikes: {vix_spikes.sum()}")
    else:
        print("   VIX not available")

    # Create labels
    labels = create_crisis_labels(spy_prices.index, known_events)

    # =========================================
    # EVALUATE INDIVIDUAL SIGNALS
    # =========================================
    print("\n" + "-" * 40)
    print("Evaluating individual signals...")
    print("-" * 40)

    chern_prec, chern_rec, chern_f1 = compute_signal_metrics(chern_spikes, labels)
    print(f"Chern: P={chern_prec:.3f}, R={chern_rec:.3f}, F1={chern_f1:.3f}")

    vol_prec, vol_rec, vol_f1 = compute_signal_metrics(vol_spikes, labels)
    print(f"Volatility: P={vol_prec:.3f}, R={vol_rec:.3f}, F1={vol_f1:.3f}")

    corr_prec, corr_rec, corr_f1 = compute_signal_metrics(corr_spikes, labels)
    print(f"Correlation: P={corr_prec:.3f}, R={corr_rec:.3f}, F1={corr_f1:.3f}")

    # =========================================
    # ENSEMBLE SIGNAL
    # =========================================
    print("\n" + "-" * 40)
    print("Building ensemble signal...")
    print("-" * 40)

    ensemble = build_ensemble_signal(chern_spikes, vol_spikes, corr_spikes, vix_spikes, min_signals=2)
    ens_prec, ens_rec, ens_f1 = compute_signal_metrics(ensemble, labels)

    print(f"Ensemble (≥2 signals): P={ens_prec:.3f}, R={ens_rec:.3f}, F1={ens_f1:.3f}")
    print(f"Total ensemble detections: {ensemble.sum()}")

    # =========================================
    # ABLATION TEST
    # =========================================
    print("\n" + "-" * 40)
    print("Running ablation test (ensemble WITHOUT Chern)...")
    print("-" * 40)

    abl_prec, abl_rec, abl_f1 = run_ablation_test(vol_spikes, corr_spikes, labels, vix_spikes)
    print(f"Ablated ensemble: P={abl_prec:.3f}, R={abl_rec:.3f}, F1={abl_f1:.3f}")

    chern_lift = ens_f1 - abl_f1
    print(f"Chern lift (F1 improvement): {chern_lift:+.3f}")

    # Count unique Chern detections
    common_idx = ensemble.index.intersection(chern_spikes.index)
    chern_unique = (
        chern_spikes.reindex(common_idx, fill_value=False) &
        ~vol_spikes.reindex(common_idx, fill_value=False) &
        ~corr_spikes.reindex(common_idx, fill_value=False)
    ).sum()

    print(f"Unique Chern detections: {chern_unique}")

    # =========================================
    # ML ENSEMBLE (OPTIONAL)
    # =========================================
    print("\n" + "-" * 40)
    print("Training ML ensemble (for reference)...")
    print("-" * 40)

    # Build feature DataFrame
    common_idx = chern_series.index.intersection(realized_vol.index).intersection(corr_stability.index)

    features_df = pd.DataFrame({
        'chern': chern_series.reindex(common_idx),
        'delta_chern': delta_chern.reindex(common_idx),
        'realized_vol': realized_vol.reindex(common_idx),
        'vol_change': vol_change.reindex(common_idx),
        'corr_stability': corr_stability.reindex(common_idx),
        'corr_change': corr_change.reindex(common_idx)
    }).dropna()

    ml_results = train_ml_ensemble(features_df, labels)

    if 'error' not in ml_results:
        print(f"Logistic Regression: F1={ml_results['logistic_regression']['f1']:.3f}")
        print(f"Random Forest: F1={ml_results['random_forest']['f1']:.3f}")

        print("\nFeature importance (Random Forest):")
        for feat, imp in sorted(ml_results['random_forest']['feature_importance'].items(),
                               key=lambda x: -x[1])[:5]:
            print(f"   {feat}: {imp:.3f}")
    else:
        print(f"ML training failed: {ml_results['error']}")

    # =========================================
    # SUMMARY
    # =========================================
    print("\n" + "=" * 60)
    print("ENSEMBLE SIGNAL SUMMARY")
    print("=" * 60)

    results = EnsembleResults(
        chern_precision=chern_prec,
        chern_recall=chern_rec,
        chern_f1=chern_f1,
        volatility_precision=vol_prec,
        volatility_recall=vol_rec,
        volatility_f1=vol_f1,
        correlation_precision=corr_prec,
        correlation_recall=corr_rec,
        correlation_f1=corr_f1,
        ensemble_precision=ens_prec,
        ensemble_recall=ens_rec,
        ensemble_f1=ens_f1,
        ablation_precision=abl_prec,
        ablation_recall=abl_rec,
        ablation_f1=abl_f1,
        chern_lift=chern_lift,
        chern_unique_detections=int(chern_unique),
        total_ensemble_detections=int(ensemble.sum())
    )

    print(f"\nIndividual Signal F1 Scores:")
    print(f"  Chern: {results.chern_f1:.3f}")
    print(f"  Volatility: {results.volatility_f1:.3f}")
    print(f"  Correlation: {results.correlation_f1:.3f}")

    print(f"\nEnsemble F1: {results.ensemble_f1:.3f}")
    print(f"Ablated F1: {results.ablation_f1:.3f}")
    print(f"Chern Lift: {results.chern_lift:+.3f}")

    print(f"\nFalse Positive Rate: {1 - results.ensemble_precision:.1%}")

    # Verdict
    print("\n" + "-" * 40)
    print("VERDICT:")

    if results.ensemble_f1 > max(results.chern_f1, results.volatility_f1, results.correlation_f1):
        print("✓ Ensemble OUTPERFORMS individual signals")
    else:
        best_individual = max(
            [('Chern', results.chern_f1), ('Volatility', results.volatility_f1),
             ('Correlation', results.correlation_f1)],
            key=lambda x: x[1]
        )
        print(f"✗ {best_individual[0]} alone performs best")

    if results.chern_lift > 0.02:
        print(f"✓ Chern CONTRIBUTES unique information (+{results.chern_lift:.1%} F1)")
    elif results.chern_lift > 0:
        print(f"○ Chern provides marginal improvement (+{results.chern_lift:.1%} F1)")
    else:
        print(f"✗ Chern does NOT improve ensemble ({results.chern_lift:+.1%} F1)")

    if results.ensemble_precision > 0.5:
        print(f"✓ Tradeable signal: {results.ensemble_precision:.1%} precision")
    else:
        print(f"✗ High false positive rate: {1-results.ensemble_precision:.1%}")

    return results


def create_ensemble_visualization(
    chern_series: pd.Series,
    realized_vol: pd.Series,
    corr_stability: pd.Series,
    ensemble: pd.Series,
    output_path: Path
):
    """Create visualization of ensemble components."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)

    # Panel 1: Chern
    ax1 = axes[0]
    ax1.plot(chern_series.index, chern_series.values, 'b-', linewidth=0.8)
    ax1.set_ylabel('Chern Number')
    ax1.set_title('Component 1: Chern Number (Topological)', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Realized Vol
    ax2 = axes[1]
    common_idx = realized_vol.index.intersection(chern_series.index)
    ax2.plot(realized_vol.loc[common_idx], 'r-', linewidth=0.8)
    ax2.set_ylabel('Realized Vol')
    ax2.set_title('Component 2: Realized Volatility (20-day)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Correlation Stability
    ax3 = axes[2]
    common_idx = corr_stability.index.intersection(chern_series.index)
    ax3.plot(corr_stability.loc[common_idx], 'g-', linewidth=0.8)
    ax3.set_ylabel('Eigenvalue Ratio')
    ax3.set_title('Component 3: Correlation Stability', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Ensemble Signal
    ax4 = axes[3]
    common_idx = ensemble.index.intersection(chern_series.index)
    ensemble_plot = ensemble.reindex(common_idx).fillna(False)
    ax4.fill_between(ensemble_plot.index, 0, ensemble_plot.astype(int),
                     where=ensemble_plot, alpha=0.5, color='red')
    ax4.set_ylabel('Ensemble Signal')
    ax4.set_title('Ensemble Signal (≥2 components agree)', fontsize=12)
    ax4.set_xlabel('Date')
    ax4.set_ylim(-0.1, 1.1)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")
    plt.close()


if __name__ == "__main__":
    np.random.seed(42)

    # Run ensemble test
    results = run_ensemble_test()

    # Save results
    output_dir = project_root / 'experiments' / 'outputs'
    output_dir.mkdir(exist_ok=True)

    results_dict = {
        'chern_precision': results.chern_precision,
        'chern_recall': results.chern_recall,
        'chern_f1': results.chern_f1,
        'volatility_precision': results.volatility_precision,
        'volatility_recall': results.volatility_recall,
        'volatility_f1': results.volatility_f1,
        'correlation_precision': results.correlation_precision,
        'correlation_recall': results.correlation_recall,
        'correlation_f1': results.correlation_f1,
        'ensemble_precision': results.ensemble_precision,
        'ensemble_recall': results.ensemble_recall,
        'ensemble_f1': results.ensemble_f1,
        'ablation_precision': results.ablation_precision,
        'ablation_recall': results.ablation_recall,
        'ablation_f1': results.ablation_f1,
        'chern_lift': results.chern_lift,
        'chern_unique_detections': results.chern_unique_detections,
        'total_ensemble_detections': results.total_ensemble_detections
    }

    with open(output_dir / 'track_c_ensemble_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'track_c_ensemble_results.json'}")
