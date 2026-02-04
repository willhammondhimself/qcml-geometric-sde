#!/usr/bin/env python3
"""
Track B: Persistent Homology Baseline

Implements Topological Data Analysis (TDA) using persistent homology as a baseline
comparison to the Chern number approach.

Key techniques:
1. Takens embedding: Reconstruct attractor from time series
2. Persistent homology: Detect topological features across scales
3. Persistence diagrams: Visualize and quantify topology
4. Feature extraction: Convert persistence to regime indicators

Reference: "Topological Data Analysis for Financial Time Series" (Gidea & Katz, 2018)

Author: QCML Research
Date: 2024
"""

import os
import sys
from pathlib import Path
import warnings
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
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

# Try to import ripser, fall back to custom implementation
try:
    from ripser import ripser
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False
    print("Warning: ripser not available, using simplified persistent homology")


@dataclass
class PersistenceResults:
    """Results from persistent homology analysis."""
    birth_times: np.ndarray
    death_times: np.ndarray
    persistence: np.ndarray  # death - birth
    dimension: int


@dataclass
class TDAComparisonResults:
    """Comparison between TDA and Chern approaches."""
    # TDA metrics
    tda_spikes_detected: int
    tda_precision: float
    tda_recall: float
    tda_f1: float

    # Chern metrics
    chern_spikes_detected: int
    chern_precision: float
    chern_recall: float
    chern_f1: float

    # Agreement
    agreement_rate: float
    unique_tda_detections: int
    unique_chern_detections: int


def takens_embedding(
    time_series: np.ndarray,
    delay: int = 1,
    dimension: int = 3
) -> np.ndarray:
    """
    Create Takens embedding of a time series.

    Converts 1D time series into higher-dimensional point cloud for
    topological analysis.

    Args:
        time_series: 1D array of observations
        delay: Time delay between coordinates (τ)
        dimension: Embedding dimension (m)

    Returns:
        embedded: (N - (m-1)*τ, m) array of embedded points
    """
    N = len(time_series)
    n_points = N - (dimension - 1) * delay

    if n_points <= 0:
        raise ValueError(f"Time series too short for delay={delay}, dim={dimension}")

    embedded = np.zeros((n_points, dimension))

    for i in range(n_points):
        for d in range(dimension):
            embedded[i, d] = time_series[i + d * delay]

    return embedded


def compute_vietoris_rips_complex(
    points: np.ndarray,
    max_epsilon: float,
    n_steps: int = 50
) -> List[Tuple[float, int, int]]:
    """
    Simplified Vietoris-Rips complex computation for 0-dimensional homology.

    Returns connected components as epsilon increases.

    Args:
        points: (N, D) point cloud
        max_epsilon: Maximum scale to consider
        n_steps: Number of epsilon steps

    Returns:
        events: List of (epsilon, birth_or_death, component_id)
    """
    n_points = len(points)
    distances = squareform(pdist(points))

    epsilons = np.linspace(0, max_epsilon, n_steps)
    events = []

    # Track connected components using union-find
    parent = list(range(n_points))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            return True
        return False

    # Process edges in order of distance
    edges = []
    for i in range(n_points):
        for j in range(i + 1, n_points):
            edges.append((distances[i, j], i, j))

    edges.sort()

    # Track births (all points born at epsilon=0)
    births = {i: 0.0 for i in range(n_points)}
    deaths = {}

    edge_idx = 0
    for epsilon in epsilons:
        while edge_idx < len(edges) and edges[edge_idx][0] <= epsilon:
            dist, i, j = edges[edge_idx]
            pi, pj = find(i), find(j)

            if pi != pj:
                # Components merge - the younger one dies
                younger = max(pi, pj, key=lambda x: births.get(x, 0))
                deaths[younger] = dist
                union(pi, pj)

            edge_idx += 1

    # Components that never die persist to infinity
    for i in range(n_points):
        if find(i) == i and i not in deaths:
            deaths[i] = max_epsilon

    return births, deaths


def compute_persistence_diagram(
    points: np.ndarray,
    max_dim: int = 1,
    max_epsilon: Optional[float] = None
) -> Dict[int, PersistenceResults]:
    """
    Compute persistence diagram for a point cloud.

    Args:
        points: (N, D) point cloud
        max_dim: Maximum homology dimension to compute
        max_epsilon: Maximum scale (default: diameter of point cloud)

    Returns:
        diagrams: Dict mapping dimension to PersistenceResults
    """
    if max_epsilon is None:
        distances = pdist(points)
        max_epsilon = np.max(distances) if len(distances) > 0 else 1.0

    if RIPSER_AVAILABLE:
        # Use ripser for full persistence
        result = ripser(points, maxdim=max_dim, thresh=max_epsilon)
        diagrams = {}

        for dim in range(max_dim + 1):
            dgm = result['dgms'][dim]
            if len(dgm) > 0:
                births = dgm[:, 0]
                deaths = dgm[:, 1]
                # Handle infinity
                deaths = np.where(np.isinf(deaths), max_epsilon, deaths)
                persistence = deaths - births

                diagrams[dim] = PersistenceResults(
                    birth_times=births,
                    death_times=deaths,
                    persistence=persistence,
                    dimension=dim
                )
            else:
                diagrams[dim] = PersistenceResults(
                    birth_times=np.array([]),
                    death_times=np.array([]),
                    persistence=np.array([]),
                    dimension=dim
                )

        return diagrams

    else:
        # Simplified: only H0 (connected components)
        births, deaths = compute_vietoris_rips_complex(points, max_epsilon)

        birth_arr = np.array(list(births.values()))
        death_arr = np.array([deaths.get(k, max_epsilon) for k in births.keys()])
        persistence = death_arr - birth_arr

        return {
            0: PersistenceResults(
                birth_times=birth_arr,
                death_times=death_arr,
                persistence=persistence,
                dimension=0
            )
        }


def persistence_landscape(
    persistence_results: PersistenceResults,
    n_points: int = 100,
    n_landscapes: int = 5
) -> np.ndarray:
    """
    Compute persistence landscape from persistence diagram.

    The persistence landscape is a stable, vectorized representation
    of the persistence diagram.

    Args:
        persistence_results: Persistence diagram
        n_points: Number of discretization points
        n_landscapes: Number of landscape functions to compute

    Returns:
        landscape: (n_landscapes, n_points) array
    """
    births = persistence_results.birth_times
    deaths = persistence_results.death_times

    if len(births) == 0:
        return np.zeros((n_landscapes, n_points))

    # Create tent functions for each point
    t_min = births.min()
    t_max = deaths.max()
    t_range = np.linspace(t_min, t_max, n_points)

    # Compute tent function values
    tent_values = np.zeros((len(births), n_points))

    for i, (b, d) in enumerate(zip(births, deaths)):
        mid = (b + d) / 2
        height = (d - b) / 2

        for j, t in enumerate(t_range):
            if b <= t <= mid:
                tent_values[i, j] = t - b
            elif mid < t <= d:
                tent_values[i, j] = d - t
            else:
                tent_values[i, j] = 0

    # Take top-k envelopes
    landscape = np.zeros((n_landscapes, n_points))

    for j in range(n_points):
        sorted_vals = np.sort(tent_values[:, j])[::-1]
        for k in range(min(n_landscapes, len(sorted_vals))):
            landscape[k, j] = sorted_vals[k]

    return landscape


def persistence_entropy(persistence_results: PersistenceResults) -> float:
    """
    Compute persistence entropy as a summary statistic.

    Higher entropy indicates more complex topological structure.
    """
    persistence = persistence_results.persistence

    if len(persistence) == 0 or persistence.sum() == 0:
        return 0.0

    # Normalize to probability distribution
    p = persistence / persistence.sum()
    p = p[p > 0]  # Remove zeros

    # Shannon entropy
    return -np.sum(p * np.log(p))


def persistence_norm(persistence_results: PersistenceResults, p: float = 2) -> float:
    """
    Compute p-norm of persistence values.

    This is related to Wasserstein distance from the diagonal.
    """
    persistence = persistence_results.persistence

    if len(persistence) == 0:
        return 0.0

    return np.power(np.sum(np.power(persistence, p)), 1/p)


def rolling_tda_features(
    time_series: np.ndarray,
    window: int = 50,
    delay: int = 1,
    embed_dim: int = 3,
    stride: int = 1
) -> pd.DataFrame:
    """
    Compute rolling TDA features over a time series.

    Args:
        time_series: 1D array of returns
        window: Rolling window size
        delay: Takens embedding delay
        embed_dim: Takens embedding dimension
        stride: Step between windows

    Returns:
        features: DataFrame with TDA features for each window
    """
    n_windows = (len(time_series) - window) // stride + 1

    features = {
        'persistence_entropy_h0': [],
        'persistence_norm_h0': [],
        'max_persistence_h0': [],
        'n_features_h0': [],
    }

    if RIPSER_AVAILABLE:
        features.update({
            'persistence_entropy_h1': [],
            'persistence_norm_h1': [],
            'max_persistence_h1': [],
            'n_features_h1': [],
        })

    indices = []

    for i in range(0, len(time_series) - window + 1, stride):
        window_data = time_series[i:i + window]

        # Takens embedding
        try:
            embedded = takens_embedding(window_data, delay=delay, dimension=embed_dim)

            # Compute persistence
            diagrams = compute_persistence_diagram(embedded, max_dim=1 if RIPSER_AVAILABLE else 0)

            # H0 features
            h0 = diagrams[0]
            features['persistence_entropy_h0'].append(persistence_entropy(h0))
            features['persistence_norm_h0'].append(persistence_norm(h0))
            features['max_persistence_h0'].append(h0.persistence.max() if len(h0.persistence) > 0 else 0)
            features['n_features_h0'].append(len(h0.persistence))

            # H1 features (if available)
            if RIPSER_AVAILABLE and 1 in diagrams:
                h1 = diagrams[1]
                features['persistence_entropy_h1'].append(persistence_entropy(h1))
                features['persistence_norm_h1'].append(persistence_norm(h1))
                features['max_persistence_h1'].append(h1.persistence.max() if len(h1.persistence) > 0 else 0)
                features['n_features_h1'].append(len(h1.persistence))

            indices.append(i + window - 1)

        except Exception as e:
            # Skip problematic windows
            continue

    return pd.DataFrame(features, index=indices)


def detect_tda_regime_changes(
    tda_features: pd.DataFrame,
    threshold_std: float = 2.0
) -> pd.Series:
    """
    Detect regime changes using TDA features.

    A regime change is detected when TDA features show significant deviation.

    Args:
        tda_features: DataFrame of TDA features
        threshold_std: Number of standard deviations for spike detection

    Returns:
        spikes: Series of spike indicators (True where spike detected)
    """
    # Use persistence entropy as primary indicator
    entropy = tda_features['persistence_entropy_h0']

    # Compute rolling statistics
    rolling_mean = entropy.rolling(window=60, min_periods=20).mean()
    rolling_std = entropy.rolling(window=60, min_periods=20).std()

    # Detect spikes
    z_score = (entropy - rolling_mean) / (rolling_std + 1e-10)
    spikes = z_score.abs() > threshold_std

    return spikes


def compare_tda_to_chern(
    tda_spikes: pd.Series,
    chern_spikes: pd.Series,
    known_events: Dict[str, str],
    event_window: int = 15
) -> TDAComparisonResults:
    """
    Compare TDA and Chern detection performance.

    Args:
        tda_spikes: Series of TDA spike indicators
        chern_spikes: Series of Chern spike indicators
        known_events: Dict of event dates to names
        event_window: Days around event to count as detection

    Returns:
        comparison: TDAComparisonResults with metrics
    """
    def compute_metrics(spikes: pd.Series, event_dates: List[pd.Timestamp]) -> Tuple[int, float, float, float]:
        """Compute precision, recall, F1 for spike detection."""
        spike_dates = spikes[spikes].index

        # True positives: spikes near events
        tp = 0
        detected_events = set()

        for spike_date in spike_dates:
            for event_date in event_dates:
                if abs((spike_date - event_date).days) <= event_window:
                    if event_date not in detected_events:
                        tp += 1
                        detected_events.add(event_date)
                    break

        # False positives: spikes not near any event
        fp = len(spike_dates) - tp

        # False negatives: events not detected
        fn = len(event_dates) - len(detected_events)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return len(spike_dates), precision, recall, f1

    # Parse event dates
    event_dates = [pd.Timestamp(d) for d in known_events.keys()]

    # Compute metrics for both
    tda_n, tda_prec, tda_rec, tda_f1 = compute_metrics(tda_spikes, event_dates)
    chern_n, chern_prec, chern_rec, chern_f1 = compute_metrics(chern_spikes, event_dates)

    # Compute agreement
    common_idx = tda_spikes.index.intersection(chern_spikes.index)
    tda_aligned = tda_spikes.reindex(common_idx, fill_value=False)
    chern_aligned = chern_spikes.reindex(common_idx, fill_value=False)

    both = tda_aligned & chern_aligned
    either = tda_aligned | chern_aligned

    agreement = both.sum() / either.sum() if either.sum() > 0 else 0

    unique_tda = (tda_aligned & ~chern_aligned).sum()
    unique_chern = (chern_aligned & ~tda_aligned).sum()

    return TDAComparisonResults(
        tda_spikes_detected=tda_n,
        tda_precision=tda_prec,
        tda_recall=tda_rec,
        tda_f1=tda_f1,
        chern_spikes_detected=chern_n,
        chern_precision=chern_prec,
        chern_recall=chern_rec,
        chern_f1=chern_f1,
        agreement_rate=agreement,
        unique_tda_detections=unique_tda,
        unique_chern_detections=unique_chern
    )


def run_tda_baseline_test(
    start_date: str = '2006-01-01',
    end_date: str = '2024-06-30'
) -> TDAComparisonResults:
    """
    Run the full TDA baseline comparison.

    Args:
        start_date: Start date for data
        end_date: End date for data

    Returns:
        results: Comparison results
    """
    print("=" * 60)
    print("TRACK B: PERSISTENT HOMOLOGY BASELINE")
    print("=" * 60)

    if RIPSER_AVAILABLE:
        print("Using ripser for persistent homology (H0 + H1)")
    else:
        print("Using simplified implementation (H0 only)")
        print("Install ripser for full analysis: pip install ripser")

    # Fetch data
    api_key = os.getenv('POLYGON_API_KEY')
    source = PolygonDataSource(api_key=api_key)

    print("\nFetching SPY data...")
    spy_data = source.fetch_equities(['SPY'], start_date, end_date, timeframe='1d')
    spy_prices = spy_data['close'].unstack(level=0)['SPY'].ffill()
    returns = spy_prices.pct_change().dropna()

    print(f"Fetched {len(returns)} days of returns")

    # =========================================
    # TDA ANALYSIS
    # =========================================
    print("\n" + "-" * 40)
    print("Computing TDA features...")
    print("-" * 40)

    tda_features = rolling_tda_features(
        returns.values,
        window=50,
        delay=1,
        embed_dim=3,
        stride=1
    )

    # Align indices with dates
    tda_features.index = returns.index[tda_features.index]

    print(f"Computed TDA features for {len(tda_features)} windows")
    print(f"Features: {list(tda_features.columns)}")

    # Detect spikes
    tda_spikes = detect_tda_regime_changes(tda_features, threshold_std=2.0)

    print(f"TDA spikes detected: {tda_spikes.sum()}")

    # =========================================
    # CHERN ANALYSIS (for comparison)
    # =========================================
    print("\n" + "-" * 40)
    print("Computing Chern number (for comparison)...")
    print("-" * 40)

    # Create features and compute Chern
    engine = MinimalFeatureEngine(window=20)
    features = engine.create_feature_matrix(spy_prices.to_frame('SPY'), benchmark_col='SPY')
    features = features.dropna()

    X_raw = features.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    from sklearn.decomposition import PCA
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

    rolling_std = delta_chern.rolling(window=60, min_periods=20).std()
    threshold = 2.0 * rolling_std
    chern_spikes = delta_chern.abs() > threshold

    print(f"Chern spikes detected: {chern_spikes.sum()}")

    # =========================================
    # COMPARISON
    # =========================================
    print("\n" + "-" * 40)
    print("Comparing TDA and Chern approaches...")
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

    results = compare_tda_to_chern(tda_spikes, chern_spikes, known_events)

    # =========================================
    # SUMMARY
    # =========================================
    print("\n" + "=" * 60)
    print("TDA BASELINE COMPARISON RESULTS")
    print("=" * 60)

    print(f"\nTDA (Persistent Homology):")
    print(f"  Spikes detected: {results.tda_spikes_detected}")
    print(f"  Precision: {results.tda_precision:.1%}")
    print(f"  Recall: {results.tda_recall:.1%}")
    print(f"  F1 Score: {results.tda_f1:.3f}")

    print(f"\nChern Number:")
    print(f"  Spikes detected: {results.chern_spikes_detected}")
    print(f"  Precision: {results.chern_precision:.1%}")
    print(f"  Recall: {results.chern_recall:.1%}")
    print(f"  F1 Score: {results.chern_f1:.3f}")

    print(f"\nAgreement Analysis:")
    print(f"  Agreement rate: {results.agreement_rate:.1%}")
    print(f"  Unique TDA detections: {results.unique_tda_detections}")
    print(f"  Unique Chern detections: {results.unique_chern_detections}")

    # Verdict
    print("\n" + "-" * 40)
    print("VERDICT:")

    if results.tda_f1 > results.chern_f1 + 0.05:
        print("✓ TDA performs BETTER than Chern number")
        print("  Consider using persistent homology as primary signal")
    elif results.chern_f1 > results.tda_f1 + 0.05:
        print("✓ Chern number performs BETTER than TDA")
        print("  Quantum geometry approach is justified")
    else:
        print("○ TDA and Chern perform SIMILARLY")
        print("  They may capture different aspects of topology")

    if results.agreement_rate < 0.5:
        print("○ Low agreement suggests complementary signals")
        print("  Consider ensemble approach combining both")
    else:
        print("○ High agreement suggests similar underlying detection")

    return results


def create_tda_visualization(
    tda_features: pd.DataFrame,
    chern_series: pd.Series,
    output_path: Path
):
    """Create visualization comparing TDA and Chern."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    # Panel 1: Persistence entropy
    ax1 = axes[0]
    ax1.plot(tda_features.index, tda_features['persistence_entropy_h0'], 'b-', linewidth=0.8)
    ax1.set_ylabel('Persistence Entropy (H0)')
    ax1.set_title('TDA: Persistence Entropy from Takens Embedding', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Max persistence
    ax2 = axes[1]
    ax2.plot(tda_features.index, tda_features['max_persistence_h0'], 'r-', linewidth=0.8)
    ax2.set_ylabel('Max Persistence (H0)')
    ax2.set_title('TDA: Maximum Persistence Value', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Chern number for comparison
    ax3 = axes[2]
    common_idx = chern_series.index.intersection(tda_features.index)
    ax3.plot(chern_series.loc[common_idx], 'g-', linewidth=0.8)
    ax3.set_ylabel('Chern Number')
    ax3.set_title('Chern Number (for comparison)', fontsize=12)
    ax3.set_xlabel('Date')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")
    plt.close()


if __name__ == "__main__":
    np.random.seed(42)

    # Run baseline test
    results = run_tda_baseline_test()

    # Save results
    output_dir = project_root / 'experiments' / 'outputs'
    output_dir.mkdir(exist_ok=True)

    results_dict = {
        'tda_spikes_detected': int(results.tda_spikes_detected),
        'tda_precision': float(results.tda_precision),
        'tda_recall': float(results.tda_recall),
        'tda_f1': float(results.tda_f1),
        'chern_spikes_detected': int(results.chern_spikes_detected),
        'chern_precision': float(results.chern_precision),
        'chern_recall': float(results.chern_recall),
        'chern_f1': float(results.chern_f1),
        'agreement_rate': float(results.agreement_rate),
        'unique_tda_detections': int(results.unique_tda_detections),
        'unique_chern_detections': int(results.unique_chern_detections),
        'ripser_available': RIPSER_AVAILABLE
    }

    with open(output_dir / 'track_b_tda_baseline_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'track_b_tda_baseline_results.json'}")
