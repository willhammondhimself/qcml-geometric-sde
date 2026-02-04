"""
QCML Volatility Module - Quantum Volatility Forecasting

This module implements Pillar 1 of the QCML research:
- Quantum uncertainty principle between IV and RV
- Commutator-based volatility forecasting
- Straddle strategy backtesting

Key hypothesis: Implied volatility and realized volatility behave as
noncommutative quantum observables. Their commutator [A_IV, A_RV] != 0
encodes an uncertainty principle that improves volatility forecasting.
"""

from .vol_data import VolatilityDataPipeline, VolatilityDataset
from .qcml_vol_forecaster import QCMLVolForecaster
from .commutator_analysis import CommutatorAnalyzer
from .benchmarks import LinearVolForecaster, GARCHForecaster, NNVolForecaster

__all__ = [
    'VolatilityDataPipeline',
    'VolatilityDataset',
    'QCMLVolForecaster',
    'CommutatorAnalyzer',
    'LinearVolForecaster',
    'GARCHForecaster',
    'NNVolForecaster',
]
