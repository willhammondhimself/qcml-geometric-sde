"""
QCML Data Pipeline Package

Professional-grade data pipeline for feeding real financial market data into the
QCML-Geometric SDE framework. Enables research on topological regime detection for
quantitative trading applications.

Modules:
    acquisition: Data fetching from Polygon.io and Alpaca APIs
    storage: Efficient Parquet-based storage and caching
    qcml_data: Dataset interface for QCML framework integration
    features: Full technical indicators and cross-sectional feature engineering
    features_minimal: Minimal features for hypothesis validation
    preprocessing: Data cleaning, normalization, and walk-forward validation
"""

from .acquisition import (
    PolygonDataSource,
    AlpacaDataSource,
    UniverseManager
)

from .storage import (
    ParquetDataStore,
    CacheManager
)

from .qcml_data import (
    QCMLDataset,
    load_crisis_dataset,
    create_multi_timeframe_dataset,
    create_synthetic_qcml_dataset
)

from .features_minimal import MinimalFeatureEngine

from .features import (
    FeatureEngine,
    FeatureConfig,
    FeatureCategory
)

from .preprocessing import (
    DataPreprocessor,
    PreprocessingConfig,
    NormalizationMethod,
    OutlierMethod,
    MissingDataMethod,
    WalkForwardFold,
    create_preprocessor
)

__version__ = "0.2.0"

__all__ = [
    # Acquisition
    "PolygonDataSource",
    "AlpacaDataSource",
    "UniverseManager",

    # Storage
    "ParquetDataStore",
    "CacheManager",

    # QCML Integration
    "QCMLDataset",
    "load_crisis_dataset",
    "create_multi_timeframe_dataset",
    "create_synthetic_qcml_dataset",

    # Minimal Features (Hypothesis Validation)
    "MinimalFeatureEngine",

    # Full Feature Engineering (Phase 2)
    "FeatureEngine",
    "FeatureConfig",
    "FeatureCategory",

    # Preprocessing (Phase 2)
    "DataPreprocessor",
    "PreprocessingConfig",
    "NormalizationMethod",
    "OutlierMethod",
    "MissingDataMethod",
    "WalkForwardFold",
    "create_preprocessor",
]
