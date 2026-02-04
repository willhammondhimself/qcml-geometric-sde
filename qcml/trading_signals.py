"""
Trading Signals Module - Strategy Generation from Topological Invariants

Generates trading signals based on:
1. Berry curvature anomalies → market stress indicators
2. Chern number transitions → regime change signals
3. Quantum metric divergence → volatility expansion signals
4. Spectral gap compression → instability warnings

Core Hypothesis: Topological transitions PRECEDE major market moves.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Union, Callable
from dataclasses import dataclass
from enum import Enum
import warnings

from .qcml_geometry import QCMLGeometry
from .topological_regime import TopologicalRegimeDetector, RegimeTransition


class SignalType(Enum):
    """Types of trading signals from topological analysis."""
    TOPOLOGY_TRANSITION = "topology_transition"  # Chern number change
    CURVATURE_SPIKE = "curvature_spike"  # Extreme Berry curvature
    METRIC_EXPANSION = "metric_expansion"  # Volatility regime shift
    SPECTRAL_WARNING = "spectral_warning"  # Instability indicator
    COMBINED = "combined"  # Multiple signals


@dataclass
class TradingSignal:
    """A single trading signal from topological analysis."""
    timestamp: int
    signal_type: SignalType
    direction: int  # -1 (risk-off), 0 (neutral), +1 (risk-on)
    strength: float  # Signal strength in [0, 1]
    confidence: float  # Confidence in [0, 1]
    metadata: Dict  # Additional signal information


@dataclass
class Position:
    """Current position state."""
    size: float  # Positive = long, negative = short
    entry_price: float
    entry_time: int
    signal: Optional[TradingSignal] = None


class TopologicalTradingStrategy:
    """
    Trading strategy based on topological market analysis.

    Uses QCML geometry to generate signals:
    1. Monitor Berry curvature for stress detection
    2. Track Chern numbers for regime classification
    3. Watch spectral gap for instability
    4. Combine signals for position management

    Hypothesis: Topological transitions precede major market moves,
    providing early warning for position adjustment.
    """

    def __init__(self,
                 geometry: QCMLGeometry,
                 lookback: int = 50,
                 curvature_threshold: float = 2.0,
                 chern_threshold: float = 0.5,
                 gap_threshold: float = 0.1,
                 position_limit: float = 1.0):
        """
        Initialize trading strategy.

        Args:
            geometry: Fitted QCMLGeometry instance
            lookback: Lookback window for signal computation
            curvature_threshold: Std multiplier for curvature signals
            chern_threshold: Threshold for Chern number transitions
            gap_threshold: Threshold for spectral gap warnings
            position_limit: Maximum position size
        """
        self.geometry = geometry
        self.lookback = lookback
        self.curvature_threshold = curvature_threshold
        self.chern_threshold = chern_threshold
        self.gap_threshold = gap_threshold
        self.position_limit = position_limit

        self.regime_detector = TopologicalRegimeDetector(
            geometry=geometry,
            window_size=lookback,
            chern_threshold=chern_threshold
        )

        # State tracking
        self._curvature_history: List[float] = []
        self._chern_history: List[float] = []
        self._gap_history: List[float] = []
        self._metric_trace_history: List[float] = []
        self._signals: List[TradingSignal] = []
        self._position: Optional[Position] = None

    def compute_signal(self, x: np.ndarray, t: int,
                      price: Optional[float] = None,
                      indices: Tuple[int, int] = (0, 1)) -> TradingSignal:
        """
        Compute trading signal from new data point.

        Args:
            x: Feature vector at time t
            t: Current timestamp
            price: Optional current price
            indices: 2D plane for curvature

        Returns:
            signal: TradingSignal with direction and strength
        """
        x = np.asarray(x).flatten()

        # Compute topological indicators
        F = self.geometry.berry_curvature(x)
        curvature = F[indices[0], indices[1]]
        self._curvature_history.append(curvature)

        gap = self.geometry.spectral_gap(x)
        self._gap_history.append(gap)

        g = self.geometry.quantum_metric(x)
        metric_trace = np.trace(g)
        self._metric_trace_history.append(metric_trace)

        # Check for regime transition
        transition = self.regime_detector.online_update(x, indices)
        if transition is not None:
            self._chern_history.append(transition.chern_after)
        elif len(self._chern_history) > 0:
            self._chern_history.append(self._chern_history[-1])
        else:
            self._chern_history.append(0.0)

        # Generate signals from each indicator
        signals = []

        # 1. Curvature spike signal
        if len(self._curvature_history) >= self.lookback:
            recent = self._curvature_history[-self.lookback:]
            mean_curv = np.mean(recent)
            std_curv = np.std(recent) + 1e-8
            z_score = (curvature - mean_curv) / std_curv

            if abs(z_score) > self.curvature_threshold:
                # High curvature = stress = risk-off
                signals.append({
                    'type': SignalType.CURVATURE_SPIKE,
                    'direction': -np.sign(z_score),  # Opposite to curvature direction
                    'strength': min(1.0, abs(z_score) / (2 * self.curvature_threshold)),
                    'z_score': z_score
                })

        # 2. Topology transition signal
        if transition is not None:
            # Regime change detected
            direction = 1 if transition.delta_chern > 0 else -1
            signals.append({
                'type': SignalType.TOPOLOGY_TRANSITION,
                'direction': direction,
                'strength': min(1.0, abs(transition.delta_chern)),
                'delta_chern': transition.delta_chern
            })

        # 3. Metric expansion signal (volatility regime)
        if len(self._metric_trace_history) >= self.lookback:
            recent_metric = self._metric_trace_history[-self.lookback:]
            metric_ratio = metric_trace / (np.mean(recent_metric) + 1e-8)

            if metric_ratio > 1.5:
                # Metric expanding = volatility increasing = reduce risk
                signals.append({
                    'type': SignalType.METRIC_EXPANSION,
                    'direction': -1,
                    'strength': min(1.0, (metric_ratio - 1) / 2),
                    'metric_ratio': metric_ratio
                })
            elif metric_ratio < 0.67:
                # Metric contracting = volatility decreasing = add risk
                signals.append({
                    'type': SignalType.METRIC_EXPANSION,
                    'direction': 1,
                    'strength': min(1.0, (1 - metric_ratio) / 0.5),
                    'metric_ratio': metric_ratio
                })

        # 4. Spectral gap warning
        if len(self._gap_history) >= self.lookback:
            recent_gap = self._gap_history[-self.lookback:]
            gap_ratio = gap / (np.mean(recent_gap) + 1e-8)

            if gap_ratio < self.gap_threshold / np.mean(recent_gap):
                # Small spectral gap = instability = reduce risk
                signals.append({
                    'type': SignalType.SPECTRAL_WARNING,
                    'direction': -1,
                    'strength': min(1.0, 1 - gap_ratio),
                    'gap_ratio': gap_ratio
                })

        # Combine signals
        if len(signals) == 0:
            return TradingSignal(
                timestamp=t,
                signal_type=SignalType.COMBINED,
                direction=0,
                strength=0.0,
                confidence=0.0,
                metadata={'reason': 'no_signal'}
            )

        # Weight and combine signals
        weights = {
            SignalType.TOPOLOGY_TRANSITION: 2.0,  # Strongest signal
            SignalType.CURVATURE_SPIKE: 1.5,
            SignalType.METRIC_EXPANSION: 1.0,
            SignalType.SPECTRAL_WARNING: 1.0
        }

        total_weight = 0.0
        weighted_direction = 0.0
        total_strength = 0.0

        for sig in signals:
            w = weights[sig['type']]
            total_weight += w
            weighted_direction += w * sig['direction'] * sig['strength']
            total_strength += w * sig['strength']

        combined_direction = np.sign(weighted_direction) if abs(weighted_direction) > 0.1 else 0
        combined_strength = total_strength / total_weight
        confidence = min(1.0, len(signals) / 2)  # More agreeing signals = more confident

        combined_signal = TradingSignal(
            timestamp=t,
            signal_type=SignalType.COMBINED,
            direction=int(combined_direction),
            strength=combined_strength,
            confidence=confidence,
            metadata={
                'component_signals': signals,
                'weighted_direction': weighted_direction,
                'curvature': curvature,
                'spectral_gap': gap,
                'metric_trace': metric_trace
            }
        )

        self._signals.append(combined_signal)
        return combined_signal

    def update_position(self, signal: TradingSignal,
                       current_price: float, t: int) -> Dict:
        """
        Update position based on signal.

        Args:
            signal: TradingSignal from compute_signal
            current_price: Current market price
            t: Current timestamp

        Returns:
            action: Dict describing trade action
        """
        target_size = signal.direction * signal.strength * self.position_limit

        if self._position is None:
            if abs(target_size) > 0.1:
                self._position = Position(
                    size=target_size,
                    entry_price=current_price,
                    entry_time=t,
                    signal=signal
                )
                return {
                    'action': 'open',
                    'size': target_size,
                    'price': current_price,
                    'signal': signal
                }
            return {'action': 'hold', 'size': 0.0}

        # Existing position
        size_diff = target_size - self._position.size

        if abs(size_diff) < 0.1:
            return {'action': 'hold', 'size': self._position.size}

        if np.sign(target_size) != np.sign(self._position.size) and abs(target_size) > 0.1:
            # Flip position
            old_size = self._position.size
            self._position = Position(
                size=target_size,
                entry_price=current_price,
                entry_time=t,
                signal=signal
            )
            return {
                'action': 'flip',
                'old_size': old_size,
                'new_size': target_size,
                'price': current_price,
                'signal': signal
            }

        # Adjust position size
        self._position.size = target_size
        return {
            'action': 'adjust',
            'size': target_size,
            'size_change': size_diff,
            'price': current_price
        }

    def close_position(self, current_price: float, t: int) -> Dict:
        """Close current position."""
        if self._position is None:
            return {'action': 'no_position'}

        pnl = self._position.size * (current_price - self._position.entry_price)
        result = {
            'action': 'close',
            'size': self._position.size,
            'entry_price': self._position.entry_price,
            'exit_price': current_price,
            'pnl': pnl,
            'duration': t - self._position.entry_time
        }

        self._position = None
        return result

    def reset(self):
        """Reset strategy state."""
        self._curvature_history = []
        self._chern_history = []
        self._gap_history = []
        self._metric_trace_history = []
        self._signals = []
        self._position = None
        self.regime_detector.reset()


def backtest_topological_strategy(strategy: TopologicalTradingStrategy,
                                 X: np.ndarray,
                                 prices: np.ndarray,
                                 indices: Tuple[int, int] = (0, 1),
                                 transaction_cost: float = 0.001) -> Dict:
    """
    Backtest topological trading strategy.

    Args:
        strategy: TopologicalTradingStrategy instance
        X: Feature data of shape (T, n_features)
        prices: Price series of shape (T,)
        indices: 2D plane for curvature
        transaction_cost: Transaction cost as fraction

    Returns:
        results: Backtest results with metrics
    """
    T = len(X)
    assert len(prices) == T, "X and prices must have same length"

    strategy.reset()

    # Track performance
    equity = [1.0]
    positions = []
    signals = []
    trades = []

    current_position = 0.0
    pnl = 0.0

    for t in range(T):
        # Compute signal
        signal = strategy.compute_signal(X[t], t, prices[t], indices)
        signals.append(signal)

        # Determine target position
        target = signal.direction * signal.strength * strategy.position_limit

        # Track position change
        position_change = target - current_position
        if abs(position_change) > 0.01:
            # Transaction cost
            cost = abs(position_change) * prices[t] * transaction_cost
            pnl -= cost

            trades.append({
                'time': t,
                'price': prices[t],
                'old_position': current_position,
                'new_position': target,
                'change': position_change,
                'cost': cost
            })

        # Update position
        current_position = target
        positions.append(current_position)

        # Mark-to-market PnL
        if t > 0:
            price_return = (prices[t] - prices[t-1]) / prices[t-1]
            position_pnl = positions[t-1] * price_return
            pnl += position_pnl

        equity.append(equity[-1] * (1 + pnl / equity[-1]) if equity[-1] > 0 else equity[-1])
        pnl = 0  # Reset daily pnl after equity update

    equity = np.array(equity[1:])  # Remove initial 1.0
    positions = np.array(positions)
    returns = np.diff(np.log(equity + 1e-10))

    # Calculate metrics
    sharpe = np.sqrt(252) * np.mean(returns) / (np.std(returns) + 1e-10)
    sortino_denom = np.std(returns[returns < 0]) if np.any(returns < 0) else 1e-10
    sortino = np.sqrt(252) * np.mean(returns) / sortino_denom

    cumulative_max = np.maximum.accumulate(equity)
    drawdown = (cumulative_max - equity) / (cumulative_max + 1e-10)
    max_drawdown = np.max(drawdown)

    calmar = (equity[-1] / equity[0] - 1) / (max_drawdown + 1e-10) if max_drawdown > 0 else 0

    # Signal statistics
    transition_signals = [s for s in signals if
                         any(cs['type'] == SignalType.TOPOLOGY_TRANSITION
                             for cs in s.metadata.get('component_signals', []))]

    return {
        'equity': equity,
        'positions': positions,
        'returns': returns,
        'trades': trades,
        'signals': signals,
        'metrics': {
            'total_return': equity[-1] / equity[0] - 1,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_drawdown,
            'calmar': calmar,
            'n_trades': len(trades),
            'n_topology_signals': len(transition_signals),
            'avg_position': np.mean(np.abs(positions)),
            'position_std': np.std(positions)
        }
    }


class EnsembleTopologicalStrategy:
    """
    Ensemble strategy combining multiple QCML geometries.

    Uses multiple learned geometries (different Hilbert space dimensions,
    operator methods, etc.) and combines their signals for more robust
    regime detection.
    """

    def __init__(self, strategies: List[TopologicalTradingStrategy],
                 weights: Optional[List[float]] = None):
        """
        Initialize ensemble.

        Args:
            strategies: List of TopologicalTradingStrategy instances
            weights: Optional weights for each strategy
        """
        self.strategies = strategies
        self.weights = weights or [1.0 / len(strategies)] * len(strategies)

        assert len(self.weights) == len(strategies)
        self.weights = np.array(self.weights) / np.sum(self.weights)

    def compute_signal(self, x: np.ndarray, t: int,
                      price: Optional[float] = None,
                      indices: Tuple[int, int] = (0, 1)) -> TradingSignal:
        """Compute ensemble signal."""
        signals = [s.compute_signal(x, t, price, indices) for s in self.strategies]

        # Weighted combination
        direction = 0.0
        strength = 0.0
        confidence = 0.0

        for sig, w in zip(signals, self.weights):
            direction += w * sig.direction * sig.strength
            strength += w * sig.strength
            confidence += w * sig.confidence

        return TradingSignal(
            timestamp=t,
            signal_type=SignalType.COMBINED,
            direction=int(np.sign(direction)) if abs(direction) > 0.1 else 0,
            strength=strength,
            confidence=confidence,
            metadata={
                'component_signals': signals,
                'weights': self.weights.tolist()
            }
        )

    def reset(self):
        """Reset all strategies."""
        for s in self.strategies:
            s.reset()


def generate_synthetic_market_data(n_samples: int = 1000,
                                  n_features: int = 5,
                                  n_regimes: int = 3,
                                  regime_persistence: float = 0.98,
                                  seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic market data with regime changes.

    Creates data with known regime transitions for testing.

    Args:
        n_samples: Number of time steps
        n_features: Feature dimension
        n_regimes: Number of distinct regimes
        regime_persistence: Probability of staying in current regime
        seed: Random seed

    Returns:
        X: Feature data of shape (n_samples, n_features)
        prices: Price series of shape (n_samples,)
        regimes: True regime labels of shape (n_samples,)
    """
    rng = np.random.default_rng(seed)

    # Generate regime sequence
    regimes = np.zeros(n_samples, dtype=int)
    regimes[0] = 0

    for t in range(1, n_samples):
        if rng.random() < regime_persistence:
            regimes[t] = regimes[t-1]
        else:
            # Transition to different regime
            other_regimes = [r for r in range(n_regimes) if r != regimes[t-1]]
            regimes[t] = rng.choice(other_regimes)

    # Generate features based on regime
    regime_means = rng.uniform(-1, 1, (n_regimes, n_features))
    regime_stds = rng.uniform(0.5, 2.0, (n_regimes, n_features))

    X = np.zeros((n_samples, n_features))
    for t in range(n_samples):
        r = regimes[t]
        X[t] = regime_means[r] + regime_stds[r] * rng.standard_normal(n_features)

    # Generate prices (GBM with regime-dependent drift and vol)
    regime_drift = rng.uniform(-0.001, 0.002, n_regimes)
    regime_vol = rng.uniform(0.01, 0.03, n_regimes)

    log_prices = np.zeros(n_samples)
    log_prices[0] = np.log(100)

    for t in range(1, n_samples):
        r = regimes[t]
        dt = 1.0 / 252
        log_prices[t] = log_prices[t-1] + (regime_drift[r] - 0.5 * regime_vol[r]**2) * dt + \
                       regime_vol[r] * np.sqrt(dt) * rng.standard_normal()

    prices = np.exp(log_prices)

    return X, prices, regimes


if __name__ == "__main__":
    print("Testing Trading Signals Module...")

    # Generate synthetic data
    X, prices, true_regimes = generate_synthetic_market_data(
        n_samples=500, n_features=3, n_regimes=3, seed=42
    )

    print(f"Data shape: X={X.shape}, prices={prices.shape}")
    print(f"Regime transitions: {np.sum(np.diff(true_regimes) != 0)}")

    # Fit QCML geometry
    from .qcml_geometry import QCMLGeometry

    qcml = QCMLGeometry(n_features=3, hilbert_dim=4)
    qcml.fit_operators(X, method='pca_inspired')

    # Create strategy
    strategy = TopologicalTradingStrategy(
        geometry=qcml,
        lookback=30,
        curvature_threshold=2.0,
        chern_threshold=0.3,
        position_limit=1.0
    )

    # Backtest
    results = backtest_topological_strategy(
        strategy, X, prices,
        indices=(0, 1),
        transaction_cost=0.001
    )

    print("\nBacktest Results:")
    print(f"  Total Return: {results['metrics']['total_return']*100:.2f}%")
    print(f"  Sharpe Ratio: {results['metrics']['sharpe']:.2f}")
    print(f"  Sortino Ratio: {results['metrics']['sortino']:.2f}")
    print(f"  Max Drawdown: {results['metrics']['max_drawdown']*100:.2f}%")
    print(f"  Calmar Ratio: {results['metrics']['calmar']:.2f}")
    print(f"  Number of Trades: {results['metrics']['n_trades']}")
    print(f"  Topology Signals: {results['metrics']['n_topology_signals']}")

    # Test ensemble
    print("\nTesting Ensemble Strategy...")

    # Create multiple geometries
    geometries = [
        QCMLGeometry(n_features=3, hilbert_dim=2),
        QCMLGeometry(n_features=3, hilbert_dim=4),
        QCMLGeometry(n_features=3, hilbert_dim=8)
    ]

    for g in geometries:
        g.fit_operators(X, method='pca_inspired')

    strategies = [
        TopologicalTradingStrategy(g, lookback=30)
        for g in geometries
    ]

    ensemble = EnsembleTopologicalStrategy(strategies)

    # Quick ensemble test
    signal = ensemble.compute_signal(X[100], 100, prices[100])
    print(f"Ensemble signal: direction={signal.direction}, strength={signal.strength:.3f}")

    print("\nTrading Signals Module tests passed!")
