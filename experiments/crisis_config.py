"""
Crisis Validation Configuration Module

Configuration dataclasses for crisis validation experiments.
Defines crisis parameters, validation settings, and result structures.

Author: QCML Research
Date: 2024
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class CrisisType(Enum):
    """Classification of historical crisis types."""
    FINANCIAL = "financial"
    PANDEMIC = "pandemic"
    MONETARY = "monetary"
    FLASH_CRASH = "flash_crash"
    GEOPOLITICAL = "geopolitical"


@dataclass
class CrisisDefinition:
    """
    Definition of a historical crisis event.

    Attributes:
        name: Unique identifier for the crisis (e.g., "2008_crisis")
        crisis_date: Date of crisis peak/trigger (YYYY-MM-DD)
        description: Human-readable description of the event
        universe: List of ticker symbols relevant to this crisis
        crisis_type: Type classification of the crisis
        expected_lead_days: Expected days before crisis that Chern should change
        lookback_months: Months of data before crisis to analyze
        lookahead_months: Months of data after crisis to analyze

    Example:
        >>> crisis = CrisisDefinition(
        ...     name="2008_crisis",
        ...     crisis_date="2008-09-15",
        ...     description="Lehman Brothers collapse",
        ...     universe=["SPY", "XLF", "BAC", "JPM"],
        ...     crisis_type=CrisisType.FINANCIAL,
        ...     expected_lead_days=10
        ... )
    """
    name: str
    crisis_date: str
    description: str
    universe: List[str]
    crisis_type: CrisisType = CrisisType.FINANCIAL
    expected_lead_days: int = 10
    lookback_months: int = 6
    lookahead_months: int = 6


@dataclass
class ValidationConfig:
    """
    Configuration for Chern number validation experiments.

    Attributes:
        hilbert_dim: Dimension of Hilbert space (2^k for k qubits)
        window_size: Rolling window size for Chern computation
        chern_threshold: Threshold for detecting topological transitions
        n_pca_components: Number of PCA components for dimensionality reduction
        use_full_features: Use full 30+ feature engine (True) or minimal 5 (False)
        operator_method: Method for fitting operators ('random', 'pca_inspired', 'pauli')
        smoothing_window: Window for smoothing Chern series
        normalize_features: Whether to apply z-score normalization
        analysis_window_days: Days before/after crisis for statistical analysis

    Example:
        >>> config = ValidationConfig(
        ...     hilbert_dim=8,
        ...     window_size=20,
        ...     chern_threshold=0.1
        ... )
    """
    hilbert_dim: int = 8
    window_size: int = 20
    chern_threshold: float = 0.1
    n_pca_components: int = 15
    use_full_features: bool = True
    operator_method: str = 'random'
    smoothing_window: int = 5
    normalize_features: bool = True
    analysis_window_days: int = 10


@dataclass
class CrisisValidationResult:
    """
    Results from validating a single crisis.

    Attributes:
        crisis_name: Name of the crisis validated
        crisis_date: Date of the crisis
        delta_chern: Change in Chern number around crisis
        chern_before: Mean Chern number before crisis
        chern_after: Mean Chern number after crisis
        t_statistic: t-statistic for significance test
        p_value: p-value for significance test
        effect_size: Cohen's d effect size
        lead_time_days: Days before crisis that Chern change was detected
        precision: Precision of detection (TP / (TP + FP))
        recall: Recall of detection (TP / (TP + FN))
        f1_score: F1 score (harmonic mean of precision and recall)
        n_transitions_detected: Number of transitions detected near crisis
        is_significant: Whether the result is statistically significant
        hypothesis_supported: Whether the hypothesis is supported for this crisis
        raw_chern_series: Full Chern number time series
        transition_indices: Indices where transitions were detected

    Example:
        >>> result = CrisisValidationResult(
        ...     crisis_name="2008_crisis",
        ...     crisis_date="2008-09-15",
        ...     delta_chern=0.234,
        ...     chern_before=0.15,
        ...     chern_after=0.384,
        ...     t_statistic=3.45,
        ...     p_value=0.001,
        ...     effect_size=1.2,
        ...     lead_time_days=8,
        ...     precision=0.67,
        ...     recall=1.0,
        ...     f1_score=0.80,
        ...     n_transitions_detected=2,
        ...     is_significant=True,
        ...     hypothesis_supported=True
        ... )
    """
    crisis_name: str
    crisis_date: str
    delta_chern: float
    chern_before: float
    chern_after: float
    t_statistic: float
    p_value: float
    effect_size: float
    lead_time_days: Optional[int]
    precision: float
    recall: float
    f1_score: float
    n_transitions_detected: int
    is_significant: bool
    hypothesis_supported: bool
    raw_chern_series: Optional[List[float]] = None
    transition_indices: Optional[List[int]] = None


@dataclass
class AggregateMetrics:
    """
    Aggregate metrics across all crisis validations.

    Attributes:
        n_crises_total: Total number of crises tested
        n_crises_validated: Number of crises where hypothesis was supported
        success_rate: Fraction of crises validated successfully
        avg_delta_chern: Average absolute Chern change across crises
        avg_t_statistic: Average t-statistic across crises
        avg_p_value: Average p-value across crises
        avg_lead_time_days: Average lead time across crises
        avg_precision: Average precision across crises
        avg_recall: Average recall across crises
        avg_f1_score: Average F1 score across crises
        median_effect_size: Median effect size across crises

    Example:
        >>> metrics = AggregateMetrics(
        ...     n_crises_total=3,
        ...     n_crises_validated=3,
        ...     success_rate=1.0,
        ...     avg_t_statistic=2.89
        ... )
    """
    n_crises_total: int
    n_crises_validated: int
    success_rate: float
    avg_delta_chern: float
    avg_t_statistic: float
    avg_p_value: float
    avg_lead_time_days: Optional[float]
    avg_precision: float
    avg_recall: float
    avg_f1_score: float
    median_effect_size: float


@dataclass
class OptunaConfig:
    """
    Configuration for Optuna hyperparameter optimization.

    Attributes:
        n_trials: Number of optimization trials
        timeout: Maximum time in seconds for optimization
        study_name: Name for the Optuna study
        storage: Optional database storage for study persistence
        sampler: Sampler type ('tpe', 'random', 'cmaes')
        pruner: Pruner type ('median', 'hyperband', 'none')
        objective_metric: Metric to optimize ('composite', 't_stat', 'recall', 'f1')

    Search Space (defined in optimizer):
        hilbert_dim: [4, 8, 16]
        window_size: [10, 20, 30, 50]
        chern_threshold: [0.05, 0.5]
        n_pca_components: [8, 30]
        operator_method: ['random', 'pca_inspired', 'pauli']
    """
    n_trials: int = 50
    timeout: Optional[int] = None
    study_name: str = "qcml_crisis_validation"
    storage: Optional[str] = None
    sampler: str = 'tpe'
    pruner: str = 'median'
    objective_metric: str = 'composite'


# Pre-defined crisis configurations
CRISIS_2008 = CrisisDefinition(
    name="2008_crisis",
    crisis_date="2008-09-15",
    description="Lehman Brothers collapse - Global Financial Crisis",
    universe=["SPY", "XLF", "BAC", "JPM", "C", "GS", "MS", "WFC"],
    crisis_type=CrisisType.FINANCIAL,
    expected_lead_days=10,
    lookback_months=6,
    lookahead_months=6
)

CRISIS_2020 = CrisisDefinition(
    name="2020_covid",
    crisis_date="2020-03-16",
    description="COVID-19 pandemic crash",
    universe=["SPY", "QQQ", "XLF", "XLE", "XLK", "XLV", "XLY", "IWM"],
    crisis_type=CrisisType.PANDEMIC,
    expected_lead_days=10,
    lookback_months=6,
    lookahead_months=6
)

CRISIS_2022 = CrisisDefinition(
    name="2022_rates",
    crisis_date="2022-03-16",
    description="Federal Reserve rate hike regime shift",
    universe=["SPY", "XLF", "XLRE", "XLU", "TLT", "IEF", "SHY"],
    crisis_type=CrisisType.MONETARY,
    expected_lead_days=10,
    lookback_months=6,
    lookahead_months=6
)

# All crises for validation
ALL_CRISES = [CRISIS_2008, CRISIS_2020, CRISIS_2022]


def get_crisis_by_name(name: str) -> CrisisDefinition:
    """
    Get crisis definition by name.

    Args:
        name: Crisis name (e.g., "2008_crisis", "2020_covid", "2022_rates")

    Returns:
        CrisisDefinition for the specified crisis

    Raises:
        ValueError: If crisis name is not found
    """
    crisis_map = {c.name: c for c in ALL_CRISES}
    if name not in crisis_map:
        raise ValueError(f"Unknown crisis: {name}. Available: {list(crisis_map.keys())}")
    return crisis_map[name]


def get_default_validation_config() -> ValidationConfig:
    """
    Get default validation configuration.

    Returns:
        ValidationConfig with recommended default settings
    """
    return ValidationConfig(
        hilbert_dim=8,
        window_size=20,
        chern_threshold=0.1,
        n_pca_components=15,
        use_full_features=True,
        operator_method='random',
        smoothing_window=5,
        normalize_features=True,
        analysis_window_days=10
    )


def config_to_dict(config: ValidationConfig) -> Dict[str, Any]:
    """Convert ValidationConfig to dictionary for serialization."""
    return {
        'hilbert_dim': config.hilbert_dim,
        'window_size': config.window_size,
        'chern_threshold': config.chern_threshold,
        'n_pca_components': config.n_pca_components,
        'use_full_features': config.use_full_features,
        'operator_method': config.operator_method,
        'smoothing_window': config.smoothing_window,
        'normalize_features': config.normalize_features,
        'analysis_window_days': config.analysis_window_days
    }


def result_to_dict(result: CrisisValidationResult) -> Dict[str, Any]:
    """Convert CrisisValidationResult to dictionary for serialization."""
    return {
        'crisis_name': str(result.crisis_name),
        'crisis_date': str(result.crisis_date),
        'delta_chern': float(result.delta_chern),
        'chern_before': float(result.chern_before),
        'chern_after': float(result.chern_after),
        't_statistic': float(result.t_statistic),
        'p_value': float(result.p_value),
        'effect_size': float(result.effect_size),
        'lead_time_days': int(result.lead_time_days) if result.lead_time_days is not None else None,
        'precision': float(result.precision),
        'recall': float(result.recall),
        'f1_score': float(result.f1_score),
        'n_transitions_detected': int(result.n_transitions_detected),
        'is_significant': bool(result.is_significant),
        'hypothesis_supported': bool(result.hypothesis_supported)
    }


def aggregate_to_dict(metrics: AggregateMetrics) -> Dict[str, Any]:
    """Convert AggregateMetrics to dictionary for serialization."""
    return {
        'n_crises_total': int(metrics.n_crises_total),
        'n_crises_validated': int(metrics.n_crises_validated),
        'success_rate': float(metrics.success_rate),
        'avg_delta_chern': float(metrics.avg_delta_chern),
        'avg_t_statistic': float(metrics.avg_t_statistic),
        'avg_p_value': float(metrics.avg_p_value),
        'avg_lead_time_days': float(metrics.avg_lead_time_days) if metrics.avg_lead_time_days is not None else None,
        'avg_precision': float(metrics.avg_precision),
        'avg_recall': float(metrics.avg_recall),
        'avg_f1_score': float(metrics.avg_f1_score),
        'median_effect_size': float(metrics.median_effect_size)
    }
