"""
Data acquisition and feature engineering for Paper 1 experiments.

Provides PolygonDataSource for market data and MinimalFeatureEngine
for creating the 5-feature matrix used in regime detection experiments.
"""

from .acquisition import PolygonDataSource
from .features_minimal import MinimalFeatureEngine

__all__ = [
    "PolygonDataSource",
    "MinimalFeatureEngine",
]
