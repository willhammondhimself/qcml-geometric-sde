"""
qcml_geometry - Spectral Metric Learning for Geometric Structure Discovery

Core library for learned market geometry: error Hamiltonian, Fubini-Study metric tensor,
Berry curvature, Chern numbers, Fisher information susceptibility, and spectral gap
computations.  Uses mathematical tools from spectral theory (Hilbert spaces, Hermitian
operators) as a structured nonlinear embedding---no quantum physics involved.

Paper 1: "Geometric Observables for Financial Regime Detection"
"""

__version__ = "1.0.0"

from .adaptive_threshold import (
    CombinedAdaptiveThreshold,
    RollingQuantileThreshold,
    ScoreVelocityThreshold,
)
from .core import QCMLGeometry, create_test_data_sphere, create_test_data_torus
from .fusion import (
    ACTIVE_CHANNELS,
    DEAD_CHANNELS,
    OBSERVABLE_FAMILIES,
    BayesianEvidenceAccumulator,
    DynamicSwitchingDetector,
    HierarchicalFusionDetector,
    RankFusionDetector,
    RegimeAdaptiveFusionDetector,
    StackingFusionDetector,
)
from .indicators import (
    EnergyEvolutionIndicator,
    FidelityDecayIndicator,
    GeometricIndicatorSuite,
    IndicatorResult,
    MultiScaleChernConsensus,
    SpectralGapIndicator,
)
from .info_geometry import (
    FisherRaoDetector,
    KLDivergenceDetector,
    SinkhornDetector,
    WassersteinDetector,
)
from .observables import (
    BaseRegimeDetector,
    BerryPhaseRateDetector,
    BerryVelocityCouplingDetector,
    CommutatorNormDetector,
    CurvatureRateDetector,
    DimensionalityCollapseDetector,
    EffectiveStateDimensionDetector,
    ExpandingWindowMixin,
    GeodesicCurvatureDetector,
    GeodesicVelocityDetector,
    GeometricEnsembleDetector,
    GeometricPhaseRateDetector,
    HamiltonianSensitivityDetector,
    LevelSpacingRatioDetector,
    MetricConditionDetector,
    MultiLagFidelityDetector,
    MultiScaleBerryDetector,
    QFIDeterminantDetector,
    QGTPhaseRigidityDetector,
    QuantumRelativeEntropyDetector,
    ReducedPurityDetector,
    RicciScalarDetector,
    SectionalCurvatureDetector,
    SpectralComplexityDetector,
    SpectralEntropyDetector,
    SpectralFlowDetector,
    SpectralGapDetector,
    SpeedLimitRatioDetector,
)
from .topology import (
    MultiScaleRegimeDetector,
    RegimeState,
    RegimeTransition,
    RegimeType,
    TopologicalRegimeDetector,
)

# Deprecated alias for backward compatibility
QuantumIndicatorSuite = GeometricIndicatorSuite

__all__ = [
    "QCMLGeometry",
    "create_test_data_sphere",
    "create_test_data_torus",
    "BaseRegimeDetector",
    "ExpandingWindowMixin",
    "BerryPhaseRateDetector",
    "MultiScaleBerryDetector",
    "QFIDeterminantDetector",
    "MultiLagFidelityDetector",
    "SpectralGapDetector",
    "MetricConditionDetector",
    "GeometricEnsembleDetector",
    "RicciScalarDetector",
    "SectionalCurvatureDetector",
    "GeodesicVelocityDetector",
    "SpeedLimitRatioDetector",
    "DimensionalityCollapseDetector",
    "SpectralFlowDetector",
    "CommutatorNormDetector",
    "SpectralEntropyDetector",
    "GeometricPhaseRateDetector",
    "HamiltonianSensitivityDetector",
    "GeodesicCurvatureDetector",
    "EffectiveStateDimensionDetector",
    "QGTPhaseRigidityDetector",
    "ReducedPurityDetector",
    "SpectralComplexityDetector",
    "BerryVelocityCouplingDetector",
    "CurvatureRateDetector",
    "LevelSpacingRatioDetector",
    "QuantumRelativeEntropyDetector",
    "FisherRaoDetector",
    "WassersteinDetector",
    "KLDivergenceDetector",
    "SinkhornDetector",
    "RankFusionDetector",
    "StackingFusionDetector",
    "DynamicSwitchingDetector",
    "HierarchicalFusionDetector",
    "RegimeAdaptiveFusionDetector",
    "BayesianEvidenceAccumulator",
    "OBSERVABLE_FAMILIES",
    "ACTIVE_CHANNELS",
    "DEAD_CHANNELS",
    "TopologicalRegimeDetector",
    "MultiScaleRegimeDetector",
    "RegimeTransition",
    "RegimeType",
    "RegimeState",
    "SpectralGapIndicator",
    "EnergyEvolutionIndicator",
    "FidelityDecayIndicator",
    "MultiScaleChernConsensus",
    "GeometricIndicatorSuite",
    "QuantumIndicatorSuite",  # deprecated alias
    "IndicatorResult",
    "RollingQuantileThreshold",
    "ScoreVelocityThreshold",
    "CombinedAdaptiveThreshold",
]
