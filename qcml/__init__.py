"""
QCML Package - Quantum Cognition-inspired Metric Learning

A groundbreaking framework combining quantum-inspired geometry learning with
stochastic dynamics and topological regime detection for quantitative finance.

Modules:
    geometry: QCML operators, quantum metric tensor, Berry curvature
    dynamics: SDEs on learned manifolds, neural SDE learning
    regime: Topological regime detection via Chern numbers
    trading: Trading signal generation from geometric features
"""

from .qcml_geometry import (
    QCMLGeometry,
    create_test_data_sphere,
    create_test_data_torus
)

from .geometric_sde import (
    GeometricSDE,
    NeuralGeometricSDE,
    SDETrajectoryDataset,
    train_neural_sde,
    simulate_from_neural_sde,
    gaussian_nll_loss
)

from .topological_regime import (
    TopologicalRegimeDetector,
    MultiScaleRegimeDetector,
    RegimeTransition,
    RegimeState,
    RegimeType,
    analyze_historical_crises
)

from .trading_signals import (
    TopologicalTradingStrategy,
    EnsembleTopologicalStrategy,
    TradingSignal,
    SignalType,
    backtest_topological_strategy,
    generate_synthetic_market_data
)

__version__ = "0.1.0"
__author__ = "Will Hammond"

__all__ = [
    # Geometry
    "QCMLGeometry",
    "create_test_data_sphere",
    "create_test_data_torus",

    # Dynamics
    "GeometricSDE",
    "NeuralGeometricSDE",
    "SDETrajectoryDataset",
    "train_neural_sde",
    "simulate_from_neural_sde",
    "gaussian_nll_loss",

    # Regime detection
    "TopologicalRegimeDetector",
    "MultiScaleRegimeDetector",
    "RegimeTransition",
    "RegimeState",
    "RegimeType",
    "analyze_historical_crises",

    # Trading
    "TopologicalTradingStrategy",
    "EnsembleTopologicalStrategy",
    "TradingSignal",
    "SignalType",
    "backtest_topological_strategy",
    "generate_synthetic_market_data",
]
