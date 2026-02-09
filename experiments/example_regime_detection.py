"""
Example: Topological Regime Detection on Synthetic Data

Demonstrates the complete pipeline:
1. Generate synthetic market data with regime changes
2. Learn QCML geometry
3. Detect regime transitions using Chern numbers
4. Generate trading signals
5. Backtest strategy

Run with:
    python experiments/example_regime_detection.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from qcml_geometry import QCMLGeometry
from qcml_geometry.topology import TopologicalRegimeDetector
# TopologicalTradingStrategy, generate_synthetic_market_data, backtest_topological_strategy archived


def main():
    print("="*70)
    print("QCML Topological Regime Detection - Example")
    print("="*70)

    # 1. Generate synthetic market data with regime changes
    print("\n[1/5] Generating synthetic market data...")
    X, prices, true_regimes = generate_synthetic_market_data(
        n_samples=500,
        n_features=5,
        n_regimes=3,
        regime_persistence=0.98,
        seed=42
    )

    print(f"  Data shape: {X.shape}")
    print(f"  Number of regime transitions: {np.sum(np.diff(true_regimes) != 0)}")

    # 2. Learn QCML geometry
    print("\n[2/5] Learning QCML geometry...")
    qcml = QCMLGeometry(n_features=5, hilbert_dim=4)
    qcml.fit_operators(X, method='pca_inspired')

    print(f"  Number of operators: {len(qcml.operators)}")
    print(f"  Hilbert dimension: {qcml.hilbert_dim}")

    # 3. Detect regime transitions
    print("\n[3/5] Detecting regime transitions...")
    detector = TopologicalRegimeDetector(
        geometry=qcml,
        window_size=30,
        chern_threshold=0.3
    )

    # Compute rolling Chern numbers
    chern_series = detector.rolling_chern_number(X, indices=(0, 1))
    transitions = detector.detect_transitions(X, indices=(0, 1))

    print(f"  Chern series shape: {chern_series.shape}")
    print(f"  Chern range: [{chern_series.min():.3f}, {chern_series.max():.3f}]")
    print(f"  Detected {len(transitions)} regime transitions")

    for i, t in enumerate(transitions):
        print(f"    Transition {i+1}: idx {t.start_idx}-{t.end_idx}, "
              f"ΔC={t.delta_chern:.3f}, confidence={t.confidence:.2f}")

    # 4. Generate trading signals
    print("\n[4/5] Generating trading signals...")
    strategy = TopologicalTradingStrategy(
        geometry=qcml,
        lookback=30,
        curvature_threshold=2.0,
        chern_threshold=0.3,
        position_limit=1.0
    )

    # 5. Backtest
    print("\n[5/5] Running backtest...")
    results = backtest_topological_strategy(
        strategy, X, prices,
        indices=(0, 1),
        transaction_cost=0.001
    )

    # Print results
    print("\n" + "="*70)
    print("BACKTEST RESULTS")
    print("="*70)
    metrics = results['metrics']

    print(f"Total Return:        {metrics['total_return']*100:7.2f}%")
    print(f"Sharpe Ratio:        {metrics['sharpe']:7.2f}")
    print(f"Sortino Ratio:       {metrics['sortino']:7.2f}")
    print(f"Max Drawdown:        {metrics['max_drawdown']*100:7.2f}%")
    print(f"Calmar Ratio:        {metrics['calmar']:7.2f}")
    print(f"Number of Trades:    {metrics['n_trades']:7d}")
    print(f"Topology Signals:    {metrics['n_topology_signals']:7d}")
    print(f"Avg Position Size:   {metrics['avg_position']:7.2f}")

    # Visualize
    try:
        print("\nGenerating visualizations...")
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

        # Price and equity
        ax = axes[0]
        ax.plot(prices / prices[0], 'b-', alpha=0.5, label='Buy & Hold', linewidth=1)
        ax.plot(results['equity'], 'g-', linewidth=2, label='Strategy')
        ax.set_ylabel('Cumulative Return')
        ax.set_title('Strategy vs Buy & Hold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Positions
        ax = axes[1]
        ax.fill_between(range(len(results['positions'])), results['positions'],
                        alpha=0.5, color='blue')
        ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
        ax.set_ylabel('Position')
        ax.set_title('Position Over Time')
        ax.grid(True, alpha=0.3)

        # Chern number
        chern_times = np.arange(len(chern_series))
        ax = axes[2]
        ax.plot(chern_times, chern_series, 'b-', linewidth=1)
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        for t in transitions:
            ax.axvspan(t.start_idx, t.end_idx, alpha=0.3, color='red')
        ax.set_ylabel('Chern Number')
        ax.set_title('Rolling Chern Number (Topological Invariant)')
        ax.grid(True, alpha=0.3)

        # True regimes
        ax = axes[3]
        ax.plot(true_regimes, 'k-', linewidth=2)
        ax.set_ylabel('Regime')
        ax.set_xlabel('Time')
        ax.set_title('True Regime Labels')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('regime_detection_example.png', dpi=150, bbox_inches='tight')
        print("  Saved: regime_detection_example.png")
        plt.close()

    except Exception as e:
        print(f"  Visualization failed: {e}")

    print("\n" + "="*70)
    print("Example completed!")
    print("="*70)


if __name__ == "__main__":
    main()
