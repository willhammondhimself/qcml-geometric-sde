"""
QCML Regime Detection Subpackage

Provides:
- Novel quantum-inspired indicators (spectral gap, energy, fidelity, Chern consensus)
- Classical baseline detectors for head-to-head comparison
- Common BaseRegimeDetector interface for all methods

Reference: Quantum Phase Transitions (Sachdev, 2011)
"""

from .quantum_indicators import (
    SpectralGapIndicator,
    EnergyEvolutionIndicator,
    FidelityDecayIndicator,
    MultiScaleChernConsensus,
    QuantumIndicatorSuite,
    IndicatorResult,
)

from .classical_baselines import (
    BaseRegimeDetector,
    QCMLChernDetector,
    RollingVolatilityDetector,
    CUSUMDetector,
    HMMRegimeDetector,
    RandomForestRegimeDetector,
    MultiScaleChernDetector,
    QuantumEnsembleDetector,
    QFISusceptibilityDetector,
    ScalarCurvatureDetector,
    GeometricConsensusDetector,
    FastGeometricConsensusDetector,
    SlowGeometricConsensusDetector,
    ShockMagnitudeDetector,
)

from .crisis_type_classifier import CrisisTypeClassifier
from .adaptive_ensemble import AdaptiveRegimeEnsemble

__all__ = [
    # Quantum indicators
    "SpectralGapIndicator",
    "EnergyEvolutionIndicator",
    "FidelityDecayIndicator",
    "MultiScaleChernConsensus",
    "QuantumIndicatorSuite",
    "IndicatorResult",
    # Classical baselines & common interface
    "BaseRegimeDetector",
    "QCMLChernDetector",
    "RollingVolatilityDetector",
    "CUSUMDetector",
    "HMMRegimeDetector",
    "RandomForestRegimeDetector",
    "MultiScaleChernDetector",
    "QuantumEnsembleDetector",
    "QFISusceptibilityDetector",
    "ScalarCurvatureDetector",
    "GeometricConsensusDetector",
    # Adaptive ensemble components
    "FastGeometricConsensusDetector",
    "SlowGeometricConsensusDetector",
    "ShockMagnitudeDetector",
    "CrisisTypeClassifier",
    "AdaptiveRegimeEnsemble",
]
