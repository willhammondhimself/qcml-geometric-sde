"""
QCML Experiments Module

Provides tools for validating the QCML hypothesis on historical crisis data
and optimizing hyperparameters.

Main Components:
    - crisis_config: Configuration dataclasses for crisis validation
    - crisis_metrics: Statistical metrics for hypothesis testing
    - validate_crisis_detection: Main validation script and CrisisValidator class

Example:
    >>> from experiments.crisis_config import CRISIS_2008, get_default_validation_config
    >>> from experiments.validate_crisis_detection import CrisisValidator
    >>>
    >>> config = get_default_validation_config()
    >>> validator = CrisisValidator(config)
    >>> result = validator.validate_single_crisis(CRISIS_2008, use_synthetic=True)
    >>> print(f"Hypothesis supported: {result.hypothesis_supported}")
"""

from .crisis_config import (
    CrisisDefinition,
    CrisisType,
    ValidationConfig,
    CrisisValidationResult,
    AggregateMetrics,
    OptunaConfig,
    CRISIS_2008,
    CRISIS_2020,
    CRISIS_2022,
    ALL_CRISES,
    get_crisis_by_name,
    get_default_validation_config,
    config_to_dict,
    result_to_dict,
    aggregate_to_dict
)

from .crisis_metrics import (
    compute_statistical_significance,
    compute_precision_recall,
    compute_lead_time,
    compute_rolling_statistics,
    aggregate_cross_crisis_metrics,
    evaluate_hypothesis,
    compute_effect_size_interpretation,
    format_results_summary
)

from .validate_crisis_detection import (
    CrisisValidator,
    run_optuna_optimization,
    seed_everything
)

# New interpretation experiments (Track A-D)
# These are standalone scripts, import functions selectively
# Run them via: python experiments/chern_interpretation_test.py
# Run them via: python experiments/persistent_homology_baseline.py
# Run them via: python experiments/chern_ensemble_signal.py

__all__ = [
    # Config classes
    'CrisisDefinition',
    'CrisisType',
    'ValidationConfig',
    'CrisisValidationResult',
    'AggregateMetrics',
    'OptunaConfig',
    # Pre-defined crises
    'CRISIS_2008',
    'CRISIS_2020',
    'CRISIS_2022',
    'ALL_CRISES',
    # Config functions
    'get_crisis_by_name',
    'get_default_validation_config',
    'config_to_dict',
    'result_to_dict',
    'aggregate_to_dict',
    # Metrics functions
    'compute_statistical_significance',
    'compute_precision_recall',
    'compute_lead_time',
    'compute_rolling_statistics',
    'aggregate_cross_crisis_metrics',
    'evaluate_hypothesis',
    'compute_effect_size_interpretation',
    'format_results_summary',
    # Validation classes
    'CrisisValidator',
    'run_optuna_optimization',
    'seed_everything'
]
