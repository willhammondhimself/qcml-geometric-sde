"""
QCML Package - Quantum Cognition-inspired Metric Learning

A groundbreaking framework combining quantum-inspired geometry learning with
stochastic dynamics and topological regime detection for quantitative finance.

Modules:
    geometry: QCML operators, quantum metric tensor, Berry curvature
    dynamics: SDEs on learned manifolds, neural SDE learning
    regime: Topological regime detection via Chern numbers
    trading: Trading signal generation from geometric features
    data: Professional data pipeline for market data acquisition and processing
    supervised_qcml: Supervised QCML for volatility forecasting
    qcml_similarity: QCML similarity analysis for regime detection
"""

# Supervised QCML (new)
from .supervised_qcml import (
    SupervisedQCML,
    HermitianParameter,
    QCMLTrainer,
    QCMLTrainingConfig,
    train_supervised_qcml,
    seed_everything,
)

# QCML Similarity Analysis (new)
from .qcml_similarity import (
    QCMLSimilarityAnalyzer,
    FidelityAnalysis,
    analyze_vol_regimes,
)

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

from .improved_chern import (
    ImprovedChernDetector,
    ImprovedChernResult,
    ThresholdMethod,
    compare_detection_methods
)

from .trading_signals import (
    TopologicalTradingStrategy,
    EnsembleTopologicalStrategy,
    TradingSignal,
    SignalType,
    backtest_topological_strategy,
    generate_synthetic_market_data
)

# Data pipeline (Phase 1)
from .data import (
    PolygonDataSource,
    AlpacaDataSource,
    UniverseManager,
    ParquetDataStore,
    CacheManager,
    QCMLDataset,
    load_crisis_dataset,
    create_multi_timeframe_dataset
)

__version__ = "0.2.0"
__author__ = "Will Hammond"

__all__ = [
    # Supervised QCML (new)
    "SupervisedQCML",
    "HermitianParameter",
    "QCMLTrainer",
    "QCMLTrainingConfig",
    "train_supervised_qcml",
    "seed_everything",

    # QCML Similarity (new)
    "QCMLSimilarityAnalyzer",
    "FidelityAnalysis",
    "analyze_vol_regimes",

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

    # Improved Chern
    "ImprovedChernDetector",
    "ImprovedChernResult",
    "ThresholdMethod",
    "compare_detection_methods",

    # Trading
    "TopologicalTradingStrategy",
    "EnsembleTopologicalStrategy",
    "TradingSignal",
    "SignalType",
    "backtest_topological_strategy",
    "generate_synthetic_market_data",

    # Data pipeline
    "PolygonDataSource",
    "AlpacaDataSource",
    "UniverseManager",
    "ParquetDataStore",
    "CacheManager",
    "QCMLDataset",
    "load_crisis_dataset",
    "create_multi_timeframe_dataset",
]
