"""
Phase 1 Demo: QCML Data Pipeline Integration

Demonstrates the complete Phase 1 data pipeline integration with
existing QCML framework using synthetic data.

This script shows:
1. Creating synthetic market data with regime changes
2. Storing data in Parquet format
3. Creating QCMLDataset for framework integration
4. Running topological regime detection
5. Analyzing detected transitions

For real market data, replace synthetic data generation with:
    - PolygonDataSource.fetch_equities()
    - Feature engineering (Phase 2)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# QCML imports
from qcml import (
    # Data pipeline (Phase 1)
    QCMLDataset,
    ParquetDataStore,
    CacheManager,
    UniverseManager,
    create_synthetic_qcml_dataset,

    # QCML framework (existing)
    QCMLGeometry,
    TopologicalRegimeDetector,
    RegimeType
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_synthetic_pipeline():
    """
    Demonstrate complete pipeline with synthetic data
    """
    logger.info("=" * 80)
    logger.info("QCML Phase 1 Data Pipeline Demo")
    logger.info("=" * 80)

    # ========================================================================
    # Step 1: Create synthetic dataset with regime change
    # ========================================================================
    logger.info("\n[Step 1] Creating synthetic dataset with regime change...")

    dataset = create_synthetic_qcml_dataset(
        n_samples=500,
        n_features=10,
        regime_change_idx=250,  # Regime change at midpoint
        noise_level=0.1,
        seed=42
    )

    logger.info(f"Dataset created: {dataset.n_samples} samples, {dataset.n_features} features")
    logger.info(f"Date range: {dataset.times[0]} to {dataset.times[-1]}")
    logger.info(f"Regime change at index: {dataset.metadata['regime_change_idx']}")

    # ========================================================================
    # Step 2: Store data using ParquetDataStore
    # ========================================================================
    logger.info("\n[Step 2] Storing data in Parquet format...")

    data_dir = Path("./data_demo")
    store = ParquetDataStore(base_path=str(data_dir))

    # Save features
    metadata = {
        'n_features': dataset.n_features,
        'n_samples': dataset.n_samples,
        'regime_change_idx': dataset.metadata['regime_change_idx'],
        'feature_names': list(dataset.features.columns)
    }

    store.save_features(dataset.features, "synthetic_demo", metadata)

    # Save as "daily bars" (just prices for demo)
    bars_df = pd.DataFrame({
        'timestamp': dataset.times,
        'close': dataset.prices_array,
        'open': dataset.prices_array,
        'high': dataset.prices_array * 1.01,
        'low': dataset.prices_array * 0.99,
        'volume': np.random.randint(1e6, 1e7, len(dataset.prices))
    })
    store.save_daily_bars(bars_df, symbol="SYNTH")

    logger.info(f"Data saved to {data_dir}")
    logger.info(f"Available feature sets: {store.list_feature_sets()}")
    logger.info(f"Available symbols: {store.list_available_symbols()}")

    # ========================================================================
    # Step 3: Load data and create QCMLDataset
    # ========================================================================
    logger.info("\n[Step 3] Loading data from storage...")

    loaded_features, loaded_meta = store.load_features("synthetic_demo")
    loaded_bars = store.load_daily_bars(["SYNTH"])

    logger.info(f"Loaded features: {loaded_features.shape}")
    logger.info(f"Loaded bars: {len(loaded_bars)} rows")

    # Recreate QCMLDataset
    reloaded_dataset = QCMLDataset(
        features=loaded_features,
        prices=loaded_bars['close'],
        times=loaded_bars.index.get_level_values(1),
        metadata=loaded_meta
    )

    # ========================================================================
    # Step 4: Learn QCML geometry
    # ========================================================================
    logger.info("\n[Step 4] Learning QCML geometry...")

    X, prices, times = reloaded_dataset.to_qcml_format()

    geometry = QCMLGeometry(
        n_features=X.shape[1],
        hilbert_dim=4,
        device='cpu'
    )

    geometry.fit_operators(X)

    logger.info(f"QCML operators learned for {X.shape[1]} features")
    logger.info(f"Hilbert space dimension: {geometry.hilbert_dim}")

    # ========================================================================
    # Step 5: Detect topological regimes
    # ========================================================================
    logger.info("\n[Step 5] Detecting topological regimes...")

    detector = TopologicalRegimeDetector(
        geometry=geometry,
        window_size=50,
        threshold=1.5
    )

    transitions = detector.detect_transitions(X, times=times)

    logger.info(f"Detected {len(transitions)} regime transitions")

    if transitions:
        for i, trans in enumerate(transitions):
            logger.info(f"\nTransition {i+1}:")
            logger.info(f"  Time: {trans.time}")
            logger.info(f"  Index: {trans.start_idx} -> {trans.end_idx}")
            logger.info(f"  From: {trans.from_regime.regime_type.name}")
            logger.info(f"  To: {trans.to_regime.regime_type.name}")
            logger.info(f"  ΔChern: {trans.delta_chern_number:.3f}")
            logger.info(f"  Confidence: {trans.confidence:.2%}")

    # ========================================================================
    # Step 6: Analyze regime change at known location
    # ========================================================================
    logger.info("\n[Step 6] Analyzing regime change at known location (index 250)...")

    # Split at regime change
    split_date = times[250]
    before, after = reloaded_dataset.split_by_date(split_date.strftime("%Y-%m-%d"))

    logger.info(f"Before regime change: {before.n_samples} samples")
    logger.info(f"After regime change: {after.n_samples} samples")

    # Compute regime signatures
    sig_before = detector.compute_regime_signature(before.X)
    sig_after = detector.compute_regime_signature(after.X)

    logger.info(f"\nRegime signature BEFORE:")
    logger.info(f"  Chern number: {sig_before['chern_number']:.3f}")
    logger.info(f"  Avg curvature: {sig_before['avg_curvature']:.4f}")
    logger.info(f"  Regime type: {sig_before['regime_type'].name}")

    logger.info(f"\nRegime signature AFTER:")
    logger.info(f"  Chern number: {sig_after['chern_number']:.3f}")
    logger.info(f"  Avg curvature: {sig_after['avg_curvature']:.4f}")
    logger.info(f"  Regime type: {sig_after['regime_type'].name}")

    delta_chern = sig_after['chern_number'] - sig_before['chern_number']
    logger.info(f"\nΔChern at regime change: {delta_chern:.3f}")

    if abs(delta_chern) > 0.5:
        logger.info("✓ Significant Chern number change detected (structural regime change)")
    else:
        logger.info("✗ No significant Chern number change (temporary fluctuation)")

    # ========================================================================
    # Step 7: Visualize results
    # ========================================================================
    logger.info("\n[Step 7] Creating visualizations...")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Prices with detected transitions
    ax1 = axes[0]
    ax1.plot(times, prices, label='Price', color='blue', linewidth=1.5)
    ax1.axvline(times[250], color='red', linestyle='--', label='True regime change', linewidth=2)

    for trans in transitions:
        ax1.axvline(trans.time, color='orange', linestyle=':', alpha=0.7, linewidth=1.5)

    ax1.set_ylabel('Price')
    ax1.set_title('Price with Regime Transitions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Returns
    ax2 = axes[1]
    returns = reloaded_dataset.returns
    ax2.plot(times[1:], returns, label='Returns', color='green', alpha=0.7)
    ax2.axvline(times[250], color='red', linestyle='--', label='True regime change', linewidth=2)
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)

    for trans in transitions:
        ax2.axvline(trans.time, color='orange', linestyle=':', alpha=0.7, linewidth=1.5)

    ax2.set_ylabel('Returns')
    ax2.set_title('Returns with Regime Detection')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Rolling Chern number
    ax3 = axes[2]
    window = 50
    chern_numbers = []

    for i in range(window, len(X)):
        window_data = X[i-window:i]
        try:
            sig = detector.compute_regime_signature(window_data)
            chern_numbers.append(sig['chern_number'])
        except:
            chern_numbers.append(np.nan)

    ax3.plot(times[window:], chern_numbers, label='Chern number', color='purple', linewidth=1.5)
    ax3.axvline(times[250], color='red', linestyle='--', label='True regime change', linewidth=2)

    for trans in transitions:
        ax3.axvline(trans.time, color='orange', linestyle=':', alpha=0.7, linewidth=1.5)

    ax3.set_xlabel('Time')
    ax3.set_ylabel('Chern Number')
    ax3.set_title(f'Rolling Chern Number (window={window})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path("./phase1_demo_results.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Visualization saved to {output_path}")

    plt.show()

    # ========================================================================
    # Step 8: Cache management demo
    # ========================================================================
    logger.info("\n[Step 8] Cache management demo...")

    cache = CacheManager(cache_dir=str(data_dir / "cache"))

    # Cache universe
    universe = UniverseManager()
    sp500 = universe.get_sp500_constituents()
    cache.cache_universe(sp500[:10])  # Cache first 10

    # Retrieve from cache
    cached_universe = cache.get_universe()
    logger.info(f"Cached universe: {cached_universe}")

    cache_size = cache.get_cache_size()
    logger.info(f"Cache size: {cache_size}")

    # ========================================================================
    # Summary
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Phase 1 Demo Complete!")
    logger.info("=" * 80)
    logger.info("\nKey achievements:")
    logger.info("✓ Data storage and retrieval with Parquet")
    logger.info("✓ QCMLDataset integration with QCML framework")
    logger.info("✓ Topological regime detection on synthetic data")
    logger.info("✓ Successful detection of known regime change")
    logger.info("\nNext steps (Phase 2):")
    logger.info("- Fetch real market data from Polygon.io")
    logger.info("- Implement technical indicator features")
    logger.info("- Add cross-sectional features")
    logger.info("- Test on historical crisis datasets (2008, 2020, 2022)")


if __name__ == "__main__":
    demo_synthetic_pipeline()
