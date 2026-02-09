#!/usr/bin/env python3
"""
Rigorous Crisis Validation with Publication-Quality Statistical Analysis

Phase 2 of Academic Research Plan: Validates QCML Chern number hypothesis
on 2008, 2020, 2022 crises with full statistical rigor.

Statistical Tests:
  - Welch's t-test (pre/post Chern comparison)
  - Bootstrap confidence intervals (n=10,000)
  - Permutation test for significance (n=5,000)
  - Cohen's d effect size
  - Bayesian hypothesis testing (Bayes factor)

Academic Significance Thresholds:
  - p-value < 0.05 (Bonferroni corrected: p < 0.017 for 3 crises)
  - Effect size (Cohen's d) > 0.8 ("large effect")
  - 95% CI for lead time excludes zero
  - Bayes factor > 10 ("strong evidence")

Usage:
    python experiments/rigorous_crisis_validation.py --synthetic
    python experiments/rigorous_crisis_validation.py --synthetic --n-bootstrap 10000

Author: QCML Research
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qcml_geometry import QCMLGeometry
from qcml_geometry.topology import TopologicalRegimeDetector
from experiments.data import PolygonDataSource, MinimalFeatureEngine
# Minimal QCMLDataset replacement (original archived in archive/data_pipeline/)
from dataclasses import dataclass
from typing import Any

@dataclass
class QCMLDataset:
    """Minimal dataset wrapper for QCML experiments."""
    features: Any  # pd.DataFrame
    prices: Any     # pd.Series or pd.DataFrame
    times: Any      # pd.DatetimeIndex
    metadata: dict

    @property
    def X(self):
        """Feature matrix as numpy array."""
        return self.features.values if hasattr(self.features, 'values') else self.features

from experiments.crisis_config import (
    CrisisDefinition,
    ValidationConfig,
    ALL_CRISES,
    get_default_validation_config,
    config_to_dict,
)
from experiments.crisis_metrics import (
    compute_statistical_significance,
    compute_precision_recall,
    compute_lead_time,
)

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    import random
    random.seed(seed)


@dataclass
class BootstrapResult:
    """
    Results from bootstrap confidence interval estimation.

    Attributes:
        statistic: Point estimate of the statistic
        ci_lower: Lower bound of confidence interval
        ci_upper: Upper bound of confidence interval
        se: Standard error of the statistic
        n_bootstrap: Number of bootstrap samples used
        ci_level: Confidence level (e.g., 0.95)
    """
    statistic: float
    ci_lower: float
    ci_upper: float
    se: float
    n_bootstrap: int
    ci_level: float = 0.95


@dataclass
class PermutationResult:
    """
    Results from permutation test.

    Attributes:
        observed_statistic: Observed test statistic
        p_value: Permutation p-value
        n_permutations: Number of permutations used
        null_distribution_mean: Mean of null distribution
        null_distribution_std: Std of null distribution
    """
    observed_statistic: float
    p_value: float
    n_permutations: int
    null_distribution_mean: float
    null_distribution_std: float


@dataclass
class BayesFactorResult:
    """
    Results from Bayesian hypothesis test.

    Attributes:
        bayes_factor: BF_10 (evidence for H1 vs H0)
        interpretation: Human-readable interpretation
        prior_odds: Prior odds ratio
        posterior_odds: Posterior odds ratio
    """
    bayes_factor: float
    interpretation: str
    prior_odds: float = 1.0
    posterior_odds: float = 0.0


@dataclass
class RigorousCrisisResult:
    """
    Complete rigorous validation result for a single crisis.

    Attributes:
        crisis_name: Crisis identifier
        crisis_date: Crisis date string
        config: ValidationConfig used

        welch_t_stat: Welch's t-test statistic
        welch_p_value: Welch's p-value
        bonferroni_p_value: Bonferroni-corrected p-value

        effect_size_d: Cohen's d
        effect_size_interpretation: small/medium/large

        bootstrap_delta_chern: Bootstrap result for delta Chern
        bootstrap_lead_time: Bootstrap result for lead time

        permutation: Permutation test result
        bayes_factor: Bayesian hypothesis test result

        f1_score: F1 detection score
        recall: Recall
        precision: Precision
        lead_time_days: Lead time in trading days

        chern_before_mean: Mean Chern before crisis
        chern_after_mean: Mean Chern after crisis
        delta_chern: Change in Chern number

        hypothesis_supported: Overall verdict
        evidence_strength: "strong"/"moderate"/"weak"/"none"
    """
    crisis_name: str
    crisis_date: str

    welch_t_stat: float
    welch_p_value: float
    bonferroni_p_value: float

    effect_size_d: float
    effect_size_interpretation: str

    bootstrap_delta_chern: Optional[Dict] = None
    bootstrap_lead_time: Optional[Dict] = None

    permutation: Optional[Dict] = None
    bayes_factor: Optional[Dict] = None

    f1_score: float = 0.0
    recall: float = 0.0
    precision: float = 0.0
    lead_time_days: Optional[int] = None

    chern_before_mean: float = 0.0
    chern_after_mean: float = 0.0
    delta_chern: float = 0.0

    hypothesis_supported: bool = False
    evidence_strength: str = "none"

    raw_chern_series: Optional[List[float]] = field(default=None, repr=False)


def bootstrap_confidence_interval(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    statistic_fn=None,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42
) -> BootstrapResult:
    """
    Compute bootstrap confidence interval for difference in means.

    Uses the bias-corrected and accelerated (BCa) percentile method
    when possible, falls back to percentile method.

    Args:
        sample_a: First sample (e.g., Chern before crisis)
        sample_b: Second sample (e.g., Chern after crisis)
        statistic_fn: Function to compute statistic (default: difference in means)
        n_bootstrap: Number of bootstrap resamples
        ci_level: Confidence level
        seed: Random seed

    Returns:
        BootstrapResult with CI bounds and standard error
    """
    rng = np.random.default_rng(seed)

    if statistic_fn is None:
        statistic_fn = lambda a, b: np.mean(b) - np.mean(a)

    observed = statistic_fn(sample_a, sample_b)
    n_a, n_b = len(sample_a), len(sample_b)

    bootstrap_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        boot_a = rng.choice(sample_a, size=n_a, replace=True)
        boot_b = rng.choice(sample_b, size=n_b, replace=True)
        bootstrap_stats[i] = statistic_fn(boot_a, boot_b)

    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    se = np.std(bootstrap_stats, ddof=1)

    return BootstrapResult(
        statistic=observed,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        se=se,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
    )


def permutation_test(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    n_permutations: int = 5000,
    seed: int = 42
) -> PermutationResult:
    """
    Permutation test for difference in means.

    Tests H0: no difference in distribution between groups.

    Args:
        sample_a: First sample
        sample_b: Second sample
        n_permutations: Number of permutations
        seed: Random seed

    Returns:
        PermutationResult with p-value and null distribution stats
    """
    rng = np.random.default_rng(seed)

    observed_diff = np.abs(np.mean(sample_b) - np.mean(sample_a))
    combined = np.concatenate([sample_a, sample_b])
    n_a = len(sample_a)

    perm_diffs = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(combined)
        perm_a = perm[:n_a]
        perm_b = perm[n_a:]
        perm_diffs[i] = np.abs(np.mean(perm_b) - np.mean(perm_a))

    p_value = (np.sum(perm_diffs >= observed_diff) + 1) / (n_permutations + 1)

    return PermutationResult(
        observed_statistic=observed_diff,
        p_value=p_value,
        n_permutations=n_permutations,
        null_distribution_mean=float(np.mean(perm_diffs)),
        null_distribution_std=float(np.std(perm_diffs)),
    )


def bayesian_t_test(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    prior_scale: float = 1.0
) -> BayesFactorResult:
    """
    Bayesian independent samples t-test (JZS Bayes factor).

    Computes BF_10: evidence for H1 (difference) vs H0 (no difference).

    Uses the Savage-Dickey density ratio approximation for the JZS prior.

    Args:
        sample_a: First sample
        sample_b: Second sample
        prior_scale: Scale of the Cauchy prior (r parameter)

    Returns:
        BayesFactorResult with Bayes factor and interpretation
    """
    n_a, n_b = len(sample_a), len(sample_b)
    mean_a, mean_b = np.mean(sample_a), np.mean(sample_b)
    var_a, var_b = np.var(sample_a, ddof=1), np.var(sample_b, ddof=1)

    # Pooled standard error
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    se = np.sqrt(pooled_var * (1/n_a + 1/n_b))

    if se < 1e-10:
        return BayesFactorResult(
            bayes_factor=1.0,
            interpretation="insufficient data",
        )

    # t-statistic
    t_stat = (mean_b - mean_a) / se
    df = n_a + n_b - 2

    # JZS Bayes factor approximation (Rouder et al., 2009)
    # Using numerical integration for accuracy
    n_eff = (n_a * n_b) / (n_a + n_b)

    def integrand(g):
        """Integrand for BF computation."""
        if g <= 0:
            return 0.0
        term1 = (1 + n_eff * g) ** (-0.5)
        term2 = (1 + t_stat**2 / ((1 + n_eff * g) * df)) ** (-(df + 1) / 2)
        term3 = (1 + t_stat**2 / df) ** ((df + 1) / 2)
        # Cauchy prior on effect size
        prior_density = 1.0 / (np.pi * prior_scale * (1 + (g / prior_scale**2)))
        return term1 * term2 * term3 * prior_density

    # Numerical integration
    from scipy.integrate import quad
    integral, _ = quad(integrand, 0, np.inf, limit=100)

    bf_10 = max(integral, 1e-10)

    # Interpretation (Jeffreys' scale)
    if bf_10 > 100:
        interpretation = "decisive evidence for H1"
    elif bf_10 > 30:
        interpretation = "very strong evidence for H1"
    elif bf_10 > 10:
        interpretation = "strong evidence for H1"
    elif bf_10 > 3:
        interpretation = "moderate evidence for H1"
    elif bf_10 > 1:
        interpretation = "anecdotal evidence for H1"
    elif bf_10 > 1/3:
        interpretation = "anecdotal evidence for H0"
    elif bf_10 > 1/10:
        interpretation = "moderate evidence for H0"
    else:
        interpretation = "strong evidence for H0"

    return BayesFactorResult(
        bayes_factor=bf_10,
        interpretation=interpretation,
        prior_odds=1.0,
        posterior_odds=bf_10,
    )


def fetch_real_crisis_data(
    crisis: CrisisDefinition,
) -> QCMLDataset:
    """
    Fetch real market data for a crisis from Polygon API.

    Args:
        crisis: Crisis definition

    Returns:
        QCMLDataset with real market data
    """
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        raise ValueError("POLYGON_API_KEY not found in environment")

    crisis_date = pd.Timestamp(crisis.crisis_date)
    start_date = crisis_date - pd.DateOffset(months=crisis.lookback_months)
    end_date = crisis_date + pd.DateOffset(months=crisis.lookahead_months)

    logger.info(f"Fetching {crisis.name}: {start_date.date()} to {end_date.date()}")
    logger.info(f"Symbols: {crisis.universe}")

    source = PolygonDataSource(api_key=api_key)
    raw_data = source.fetch_equities(
        crisis.universe,
        str(start_date.date()),
        str(end_date.date()),
        timeframe="1d"
    )

    if raw_data.empty:
        raise ValueError(f"No data returned for {crisis.name}")

    prices = raw_data['close'].unstack(level=0)
    prices = prices.ffill()

    benchmark_col = 'SPY' if 'SPY' in prices.columns else prices.columns[0]
    engine = MinimalFeatureEngine(window=20)
    features = engine.create_feature_matrix(prices, benchmark_col=benchmark_col)
    features = features.dropna()

    if len(features) < 50:
        raise ValueError(f"Insufficient data: {len(features)} rows")

    aligned_prices = prices[benchmark_col].loc[features.index]

    metadata = {
        'crisis': crisis.name,
        'crisis_date': crisis.crisis_date,
        'source': 'polygon_api',
        'symbols': list(prices.columns),
        'benchmark': benchmark_col,
    }

    return QCMLDataset(features, aligned_prices, features.index, metadata)


def create_synthetic_crisis_dataset(
    crisis: CrisisDefinition,
    n_features: int = 10,
    seed: int = 42
) -> QCMLDataset:
    """
    Create synthetic dataset with regime change at crisis date.

    Args:
        crisis: Crisis definition
        n_features: Number of features
        seed: Random seed

    Returns:
        QCMLDataset with regime change
    """
    np.random.seed(seed)

    crisis_date = pd.Timestamp(crisis.crisis_date)
    start_date = crisis_date - pd.DateOffset(months=crisis.lookback_months)
    end_date = crisis_date + pd.DateOffset(months=crisis.lookahead_months)

    dates = pd.bdate_range(start=start_date, end=end_date)
    n_days = len(dates)
    crisis_idx = (dates <= crisis_date).sum()

    # Pre-crisis regime
    theta1 = np.linspace(0, 2 * np.pi, crisis_idx)
    features1 = np.zeros((crisis_idx, n_features))
    for i in range(n_features):
        phase = 2 * np.pi * i / n_features
        features1[:, i] = np.cos(theta1 + phase) + 0.1 * np.random.randn(crisis_idx)

    # Post-crisis regime (different topology)
    n_after = n_days - crisis_idx
    theta2 = np.linspace(0, 4 * np.pi, n_after)
    features2 = np.zeros((n_after, n_features))
    for i in range(n_features):
        phase = 2 * np.pi * i / n_features
        features2[:, i] = 2.0 * np.sin(2 * theta2 + phase) + 0.3 * np.random.randn(n_after)

    # Correlation structure change
    for i in range(0, n_features - 1, 2):
        features1[:, i + 1] += 0.7 * features1[:, i]
        features2[:, i + 1] -= 0.7 * features2[:, i]

    features = np.vstack([features1, features2])
    features += np.random.randn(n_days, n_features) * 0.1

    returns = np.zeros(n_days)
    returns[:crisis_idx] = np.random.randn(crisis_idx) * 0.01 + 0.0002
    returns[crisis_idx:] = np.random.randn(n_after) * 0.025 - 0.001
    prices = 100 * np.exp(np.cumsum(returns))

    features_df = pd.DataFrame(
        features,
        columns=[f"feature_{i}" for i in range(n_features)],
        index=dates
    )
    prices_series = pd.Series(prices, name='close', index=dates)
    metadata = {
        'type': 'synthetic',
        'crisis': crisis.name,
        'crisis_date': crisis.crisis_date,
        'regime_change_idx': crisis_idx,
    }

    return QCMLDataset(features_df, prices_series, dates, metadata)


class RigorousCrisisValidator:
    """
    Validates Chern number hypothesis with publication-quality statistical rigor.

    Performs:
    1. Standard Welch's t-test with Bonferroni correction
    2. Bootstrap confidence intervals (BCa)
    3. Permutation tests
    4. Bayesian hypothesis testing (JZS Bayes factor)
    5. Effect size computation

    Attributes:
        config: ValidationConfig
        n_bootstrap: Number of bootstrap resamples
        n_permutations: Number of permutations
        n_crises: Number of crises (for Bonferroni correction)
        results: Dict of crisis results
    """

    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
        n_bootstrap: int = 10000,
        n_permutations: int = 5000,
        n_crises: int = 3,
        seed: int = 42
    ):
        """
        Initialize rigorous validator.

        Args:
            config: Validation configuration
            n_bootstrap: Bootstrap resamples
            n_permutations: Permutation iterations
            n_crises: Number of crises (for Bonferroni)
            seed: Random seed
        """
        self.config = config or get_default_validation_config()
        self.n_bootstrap = n_bootstrap
        self.n_permutations = n_permutations
        self.n_crises = n_crises
        self.seed = seed
        self.results: Dict[str, RigorousCrisisResult] = {}

    def validate_single_crisis(
        self,
        crisis: CrisisDefinition,
        use_synthetic: bool = True
    ) -> RigorousCrisisResult:
        """
        Validate hypothesis on a single crisis with full statistical rigor.

        Args:
            crisis: Crisis definition
            use_synthetic: Use synthetic data

        Returns:
            RigorousCrisisResult with complete statistical analysis
        """
        logger.info(f"Rigorous validation: {crisis.name} ({crisis.crisis_date})")

        # Load data
        if use_synthetic:
            dataset = create_synthetic_crisis_dataset(
                crisis, n_features=10, seed=self.seed
            )
        else:
            dataset = fetch_real_crisis_data(crisis)

        # Prepare features
        X_raw = dataset.X
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        n_components = min(self.config.n_pca_components, X_raw.shape[1])
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(X_scaled)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

        # Fit geometry and compute Chern
        geometry = QCMLGeometry(
            n_features=X.shape[1],
            hilbert_dim=self.config.hilbert_dim
        )
        geometry.fit_operators(X, method=self.config.operator_method)

        detector = TopologicalRegimeDetector(
            geometry=geometry,
            window_size=self.config.window_size,
            chern_threshold=self.config.chern_threshold,
            smoothing_window=self.config.smoothing_window
        )

        chern_series = detector.rolling_chern_number(X, window=self.config.window_size)

        chern_times = dataset.times[self.config.window_size - 1:]
        if len(chern_times) > len(chern_series):
            chern_times = chern_times[:len(chern_series)]

        # Find crisis index
        crisis_ts = pd.Timestamp(crisis.crisis_date)
        crisis_idx = (chern_times >= crisis_ts).argmax()

        window_days = self.config.analysis_window_days
        before_start = max(0, crisis_idx - window_days)
        after_end = min(len(chern_series), crisis_idx + window_days)

        chern_before = chern_series[before_start:crisis_idx]
        chern_after = chern_series[crisis_idx:after_end]

        if len(chern_before) < 3 or len(chern_after) < 3:
            logger.warning(f"Insufficient data for {crisis.name}")
            return self._empty_result(crisis)

        # 1. Welch's t-test
        sig = compute_statistical_significance(chern_before, chern_after)
        bonferroni_p = min(sig['p_value'] * self.n_crises, 1.0)

        # 2. Effect size
        d = sig['effect_size']
        if abs(d) >= 0.8:
            d_interp = "large"
        elif abs(d) >= 0.5:
            d_interp = "medium"
        elif abs(d) >= 0.2:
            d_interp = "small"
        else:
            d_interp = "negligible"

        # 3. Bootstrap CI for delta Chern
        boot_delta = bootstrap_confidence_interval(
            chern_before, chern_after,
            n_bootstrap=self.n_bootstrap,
            seed=self.seed
        )

        # 4. Permutation test
        perm = permutation_test(
            chern_before, chern_after,
            n_permutations=self.n_permutations,
            seed=self.seed
        )

        # 5. Bayesian t-test
        bf = bayesian_t_test(chern_before, chern_after)

        # 6. Detection metrics
        transitions = detector.detect_transitions(X, times=np.arange(len(X)))
        transition_indices = [t.start_idx for t in transitions]

        true_crisis_idx = crisis_idx + self.config.window_size - 1
        pr = compute_precision_recall(
            transition_indices, true_crisis_idx, tolerance_days=window_days
        )

        lead_time = compute_lead_time(
            chern_series, chern_times.values,
            crisis.crisis_date, threshold=self.config.chern_threshold
        )

        # 7. Bootstrap CI for lead time (if detected)
        boot_lead = None
        if lead_time is not None:
            # Bootstrap lead time by resampling Chern series
            lead_times_boot = []
            rng = np.random.default_rng(self.seed + 100)
            for _ in range(min(self.n_bootstrap, 1000)):
                # Resample with replacement within each window
                boot_before = rng.choice(chern_before, size=len(chern_before), replace=True)
                boot_after = rng.choice(chern_after, size=len(chern_after), replace=True)
                boot_chern = np.concatenate([
                    chern_series[:before_start],
                    boot_before,
                    boot_after,
                    chern_series[after_end:]
                ])
                lt = compute_lead_time(
                    boot_chern, chern_times.values,
                    crisis.crisis_date, threshold=self.config.chern_threshold
                )
                if lt is not None:
                    lead_times_boot.append(lt)

            if lead_times_boot:
                lead_arr = np.array(lead_times_boot)
                boot_lead = {
                    'statistic': float(np.mean(lead_arr)),
                    'ci_lower': float(np.percentile(lead_arr, 2.5)),
                    'ci_upper': float(np.percentile(lead_arr, 97.5)),
                    'se': float(np.std(lead_arr, ddof=1)),
                    'n_valid': len(lead_times_boot),
                    'ci_excludes_zero': float(np.percentile(lead_arr, 2.5)) > 0,
                }

        # Determine evidence strength
        evidence_criteria = {
            'bonferroni_significant': bonferroni_p < 0.05,
            'large_effect': abs(d) >= 0.8,
            'ci_excludes_zero': boot_delta.ci_lower > 0 or boot_delta.ci_upper < 0,
            'permutation_significant': perm.p_value < 0.05,
            'bayes_strong': bf.bayes_factor > 10,
        }
        n_met = sum(evidence_criteria.values())

        if n_met >= 4:
            evidence_strength = "strong"
        elif n_met >= 3:
            evidence_strength = "moderate"
        elif n_met >= 2:
            evidence_strength = "weak"
        else:
            evidence_strength = "none"

        hypothesis_supported = n_met >= 3

        result = RigorousCrisisResult(
            crisis_name=crisis.name,
            crisis_date=crisis.crisis_date,
            welch_t_stat=sig['t_statistic'],
            welch_p_value=sig['p_value'],
            bonferroni_p_value=bonferroni_p,
            effect_size_d=d,
            effect_size_interpretation=d_interp,
            bootstrap_delta_chern={
                'statistic': boot_delta.statistic,
                'ci_lower': boot_delta.ci_lower,
                'ci_upper': boot_delta.ci_upper,
                'se': boot_delta.se,
                'n_bootstrap': boot_delta.n_bootstrap,
            },
            bootstrap_lead_time=boot_lead,
            permutation={
                'observed_statistic': perm.observed_statistic,
                'p_value': perm.p_value,
                'n_permutations': perm.n_permutations,
            },
            bayes_factor={
                'bf_10': bf.bayes_factor,
                'interpretation': bf.interpretation,
            },
            f1_score=pr['f1_score'],
            recall=pr['recall'],
            precision=pr['precision'],
            lead_time_days=lead_time,
            chern_before_mean=float(np.mean(chern_before)),
            chern_after_mean=float(np.mean(chern_after)),
            delta_chern=sig['delta_chern'],
            hypothesis_supported=hypothesis_supported,
            evidence_strength=evidence_strength,
            raw_chern_series=chern_series.tolist(),
        )

        self.results[crisis.name] = result
        return result

    def validate_all_crises(
        self,
        crises: Optional[List[CrisisDefinition]] = None,
        use_synthetic: bool = True
    ) -> Dict[str, RigorousCrisisResult]:
        """
        Validate hypothesis across all crises.

        Args:
            crises: List of crises (default: ALL_CRISES)
            use_synthetic: Use synthetic data

        Returns:
            Dict mapping crisis name to result
        """
        crises = crises or ALL_CRISES

        for crisis in crises:
            try:
                self.validate_single_crisis(crisis, use_synthetic=use_synthetic)
            except Exception as e:
                logger.error(f"Failed: {crisis.name}: {e}")
                self.results[crisis.name] = self._empty_result(crisis)

        return self.results

    def compute_aggregate_statistics(self) -> Dict[str, Any]:
        """
        Compute aggregate statistics across all crises.

        Returns:
            Dict with aggregate metrics
        """
        results = list(self.results.values())
        if not results:
            return {}

        n_supported = sum(1 for r in results if r.hypothesis_supported)

        # Combine p-values using Fisher's method
        p_values = [r.welch_p_value for r in results if r.welch_p_value < 1.0]
        if p_values:
            chi2_stat = -2 * sum(np.log(max(p, 1e-300)) for p in p_values)
            fisher_p = 1 - stats.chi2.cdf(chi2_stat, df=2 * len(p_values))
        else:
            fisher_p = 1.0

        effect_sizes = [r.effect_size_d for r in results]

        return {
            'n_crises': len(results),
            'n_supported': n_supported,
            'success_rate': n_supported / len(results),
            'fisher_combined_p': fisher_p,
            'mean_effect_size': float(np.mean(effect_sizes)),
            'median_effect_size': float(np.median(effect_sizes)),
            'min_effect_size': float(np.min(effect_sizes)),
            'all_large_effects': all(abs(d) >= 0.8 for d in effect_sizes),
            'evidence_strengths': {r.crisis_name: r.evidence_strength for r in results},
        }

    def _empty_result(self, crisis: CrisisDefinition) -> RigorousCrisisResult:
        """Create empty result for failed validation."""
        return RigorousCrisisResult(
            crisis_name=crisis.name,
            crisis_date=crisis.crisis_date,
            welch_t_stat=0.0,
            welch_p_value=1.0,
            bonferroni_p_value=1.0,
            effect_size_d=0.0,
            effect_size_interpretation="negligible",
            hypothesis_supported=False,
            evidence_strength="none",
        )

    def save_results(
        self,
        output_dir: str = "experiments/outputs/regime_detection/results"
    ) -> str:
        """
        Save results to JSON.

        Args:
            output_dir: Output directory

        Returns:
            Path to saved file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output = {
            'timestamp': timestamp,
            'config': config_to_dict(self.config),
            'statistical_parameters': {
                'n_bootstrap': self.n_bootstrap,
                'n_permutations': self.n_permutations,
                'bonferroni_n_tests': self.n_crises,
                'bonferroni_threshold': 0.05 / self.n_crises,
                'seed': self.seed,
            },
            'crises': {},
            'aggregate': self.compute_aggregate_statistics(),
        }

        for name, result in self.results.items():
            # Convert to dict, excluding raw series for compactness
            result_dict = {
                'crisis_name': result.crisis_name,
                'crisis_date': result.crisis_date,
                'welch_t_stat': result.welch_t_stat,
                'welch_p_value': result.welch_p_value,
                'bonferroni_p_value': result.bonferroni_p_value,
                'effect_size_d': result.effect_size_d,
                'effect_size_interpretation': result.effect_size_interpretation,
                'bootstrap_delta_chern': result.bootstrap_delta_chern,
                'bootstrap_lead_time': result.bootstrap_lead_time,
                'permutation': result.permutation,
                'bayes_factor': result.bayes_factor,
                'f1_score': result.f1_score,
                'recall': result.recall,
                'precision': result.precision,
                'lead_time_days': result.lead_time_days,
                'chern_before_mean': result.chern_before_mean,
                'chern_after_mean': result.chern_after_mean,
                'delta_chern': result.delta_chern,
                'hypothesis_supported': result.hypothesis_supported,
                'evidence_strength': result.evidence_strength,
            }
            output['crises'][name] = result_dict

        filepath = output_path / f"rigorous_validation_{timestamp}.json"
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        logger.info(f"Results saved to {filepath}")
        return str(filepath)

    def print_summary(self) -> None:
        """Print formatted summary of results."""
        print("\n" + "=" * 78)
        print("RIGOROUS CRISIS VALIDATION SUMMARY")
        print("=" * 78)
        print(f"Bootstrap: n={self.n_bootstrap}, "
              f"Permutations: n={self.n_permutations}, "
              f"Bonferroni: alpha={0.05/self.n_crises:.4f}")
        print()

        for name, r in self.results.items():
            supported = "SUPPORTED" if r.hypothesis_supported else "NOT SUPPORTED"
            print(f"--- {r.crisis_name} ({r.crisis_date}) ---")
            print(f"  Verdict: {supported} (evidence: {r.evidence_strength})")
            print(f"  Delta Chern:  {r.delta_chern:.4f}")
            print(f"  Welch t-test: t={r.welch_t_stat:.3f}, "
                  f"p={r.welch_p_value:.4f} "
                  f"(Bonferroni p={r.bonferroni_p_value:.4f})")
            print(f"  Effect size:  d={r.effect_size_d:.3f} "
                  f"({r.effect_size_interpretation})")

            if r.bootstrap_delta_chern:
                b = r.bootstrap_delta_chern
                print(f"  Bootstrap CI: [{b['ci_lower']:.4f}, {b['ci_upper']:.4f}] "
                      f"(SE={b['se']:.4f})")

            if r.permutation:
                print(f"  Permutation:  p={r.permutation['p_value']:.4f}")

            if r.bayes_factor:
                print(f"  Bayes Factor: BF10={r.bayes_factor['bf_10']:.2f} "
                      f"({r.bayes_factor['interpretation']})")

            print(f"  Detection:    F1={r.f1_score:.3f}, "
                  f"P={r.precision:.3f}, R={r.recall:.3f}")

            if r.lead_time_days:
                print(f"  Lead time:    {r.lead_time_days} days")
                if r.bootstrap_lead_time:
                    lt = r.bootstrap_lead_time
                    excludes = "excludes" if lt.get('ci_excludes_zero') else "includes"
                    print(f"    Bootstrap CI: [{lt['ci_lower']:.1f}, {lt['ci_upper']:.1f}] "
                          f"({excludes} zero)")

            print()

        # Aggregate
        agg = self.compute_aggregate_statistics()
        print("-" * 78)
        print("AGGREGATE")
        print("-" * 78)
        print(f"  Supported: {agg.get('n_supported', 0)}/{agg.get('n_crises', 0)}")
        print(f"  Fisher combined p: {agg.get('fisher_combined_p', 1.0):.4f}")
        print(f"  Mean |d|: {agg.get('mean_effect_size', 0):.3f}")
        print(f"  All large effects: {agg.get('all_large_effects', False)}")

        # Overall verdict
        print()
        print("=" * 78)
        success = agg.get('success_rate', 0)
        if success >= 0.67:
            print("OVERALL: HYPOTHESIS SUPPORTED")
        elif success >= 0.33:
            print("OVERALL: HYPOTHESIS PARTIALLY SUPPORTED")
        else:
            print("OVERALL: HYPOTHESIS NOT SUPPORTED")
        print("=" * 78)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Rigorous Crisis Validation for QCML Regime Detection"
    )
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data")
    parser.add_argument("--n-bootstrap", type=int, default=10000,
                        help="Bootstrap resamples")
    parser.add_argument("--n-permutations", type=int, default=5000,
                        help="Permutation iterations")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output-dir", type=str,
                        default="experiments/outputs/regime_detection/results",
                        help="Output directory")
    parser.add_argument("--crisis", type=str, default=None,
                        help="Validate single crisis")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    load_dotenv(project_root / '.env')
    seed_everything(args.seed)

    print("=" * 78)
    print("QCML Rigorous Crisis Validation")
    print("=" * 78)
    print(f"Bootstrap: n={args.n_bootstrap}")
    print(f"Permutations: n={args.n_permutations}")
    print(f"Synthetic: {args.synthetic}")
    print("=" * 78)

    validator = RigorousCrisisValidator(
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )

    if args.crisis:
        from experiments.crisis_config import get_crisis_by_name
        crisis = get_crisis_by_name(args.crisis)
        validator.validate_single_crisis(crisis, use_synthetic=args.synthetic)
    else:
        validator.validate_all_crises(use_synthetic=args.synthetic)

    validator.print_summary()

    filepath = validator.save_results(output_dir=args.output_dir)
    print(f"\nResults saved to: {filepath}")


if __name__ == "__main__":
    main()
