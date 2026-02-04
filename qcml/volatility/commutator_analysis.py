"""
Commutator Analysis Module for QCML Volatility Research

Provides tools for analyzing the commutator [A_IV, A_RV] between
implied volatility and realized volatility operators.

Key analyses:
1. Rolling commutator norm to track non-commutativity over time
2. Correlation between commutator magnitude and forecast errors
3. Statistical tests for the uncertainty principle hypothesis
4. Permutation tests for significance

Reference: QCML Pillar 1 - Quantum Uncertainty Principle for IV/RV
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class CommutatorResult:
    """Results from commutator analysis."""
    commutator: np.ndarray
    frobenius_norm: float
    spectral_norm: float
    trace_norm: float
    is_anti_hermitian: bool
    max_eigenvalue_abs: float


@dataclass
class RollingCommutatorResult:
    """Results from rolling commutator analysis."""
    dates: np.ndarray
    commutator_norms: np.ndarray
    spectral_norms: np.ndarray
    uncertainty_bounds: np.ndarray
    mean_norm: float
    std_norm: float
    pct_above_threshold: float


@dataclass
class UncertaintyValidationResult:
    """Results from uncertainty principle validation."""
    correlation: float
    correlation_pvalue: float
    granger_f_stat: float
    granger_pvalue: float
    permutation_mean: float
    permutation_std: float
    actual_correlation: float
    permutation_pvalue: float
    is_significant: bool


class CommutatorAnalyzer:
    """
    Analyzer for commutator [A_IV, A_RV] and its relationship to forecasting.

    Provides:
    - Basic commutator computation and statistics
    - Rolling analysis over time windows
    - Correlation with forecast errors
    - Statistical significance tests

    Example:
        >>> from qcml.volatility import QCMLVolForecaster, CommutatorAnalyzer
        >>> forecaster = QCMLVolForecaster(n_features=5)
        >>> forecaster.fit(X_train, y_train)
        >>> analyzer = CommutatorAnalyzer()
        >>> result = analyzer.analyze_commutator(forecaster)
        >>> print(f"Commutator norm: {result.frobenius_norm}")
    """

    def __init__(self, threshold: float = 0.01):
        """
        Initialize analyzer.

        Args:
            threshold: Threshold for considering commutator "significant"
        """
        self.threshold = threshold

    @staticmethod
    def compute_commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Compute commutator [A, B] = AB - BA.

        Args:
            A, B: Square matrices of same dimension

        Returns:
            Commutator matrix
        """
        return A @ B - B @ A

    def analyze_commutator(self, forecaster) -> CommutatorResult:
        """
        Analyze the commutator from a fitted forecaster.

        Args:
            forecaster: Fitted QCMLVolForecaster instance

        Returns:
            CommutatorResult with statistics
        """
        A_IV = forecaster.get_iv_operator()
        A_RV = forecaster.get_rv_operator()

        C = self.compute_commutator(A_IV, A_RV)

        # Compute various norms
        frobenius_norm = np.linalg.norm(C, 'fro')
        spectral_norm = np.linalg.norm(C, 2)

        # Trace norm (nuclear norm) = sum of singular values
        singular_values = np.linalg.svd(C, compute_uv=False)
        trace_norm = np.sum(singular_values)

        # Check if anti-Hermitian: C = -C^dagger
        is_anti_hermitian = np.allclose(C, -C.conj().T, atol=1e-10)

        # Maximum absolute eigenvalue
        eigenvalues = np.linalg.eigvals(C)
        max_eigenvalue_abs = np.max(np.abs(eigenvalues))

        return CommutatorResult(
            commutator=C,
            frobenius_norm=frobenius_norm,
            spectral_norm=spectral_norm,
            trace_norm=trace_norm,
            is_anti_hermitian=is_anti_hermitian,
            max_eigenvalue_abs=max_eigenvalue_abs
        )

    def rolling_analysis(
        self,
        X: np.ndarray,
        y: np.ndarray,
        forecaster_class,
        window: int = 60,
        step: int = 1,
        forecaster_kwargs: Optional[Dict] = None
    ) -> RollingCommutatorResult:
        """
        Perform rolling window analysis of commutator.

        Re-fits the forecaster on each window and computes commutator statistics.

        Args:
            X: Feature data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            forecaster_class: Class to instantiate (e.g., QCMLVolForecaster)
            window: Rolling window size
            step: Step size between windows
            forecaster_kwargs: Kwargs for forecaster initialization

        Returns:
            RollingCommutatorResult with time series of commutator norms
        """
        forecaster_kwargs = forecaster_kwargs or {}
        n_samples = len(X)

        dates = np.arange(n_samples)
        commutator_norms = []
        spectral_norms = []
        uncertainty_bounds = []
        window_dates = []

        for start in range(0, n_samples - window, step):
            end = start + window

            # Fit forecaster on window
            forecaster = forecaster_class(**forecaster_kwargs)

            try:
                forecaster.fit(X[start:end], y[start:end], verbose=False)

                # Compute commutator
                result = self.analyze_commutator(forecaster)
                commutator_norms.append(result.frobenius_norm)
                spectral_norms.append(result.spectral_norm)

                # Compute uncertainty bound at window center
                center_idx = start + window // 2
                bound = forecaster.compute_uncertainty_bound(X[center_idx])
                uncertainty_bounds.append(bound)

                window_dates.append(dates[end - 1])

            except Exception as e:
                logger.warning(f"Failed to fit window {start}-{end}: {e}")
                continue

        commutator_norms = np.array(commutator_norms)
        spectral_norms = np.array(spectral_norms)
        uncertainty_bounds = np.array(uncertainty_bounds)
        window_dates = np.array(window_dates)

        pct_above = np.mean(commutator_norms > self.threshold) * 100

        return RollingCommutatorResult(
            dates=window_dates,
            commutator_norms=commutator_norms,
            spectral_norms=spectral_norms,
            uncertainty_bounds=uncertainty_bounds,
            mean_norm=np.mean(commutator_norms),
            std_norm=np.std(commutator_norms),
            pct_above_threshold=pct_above
        )

    def correlation_with_errors(
        self,
        commutator_series: np.ndarray,
        errors: np.ndarray,
        lag: int = 0
    ) -> Tuple[float, float]:
        """
        Compute correlation between commutator norms and forecast errors.

        Args:
            commutator_series: Time series of commutator norms
            errors: Time series of absolute forecast errors
            lag: Lag between commutator and errors (positive = commutator leads)

        Returns:
            Tuple of (correlation, p-value)
        """
        if lag > 0:
            # Commutator leads errors
            comm = commutator_series[:-lag]
            err = errors[lag:]
        elif lag < 0:
            # Errors lead commutator
            comm = commutator_series[-lag:]
            err = errors[:lag]
        else:
            comm = commutator_series
            err = errors

        # Ensure same length
        min_len = min(len(comm), len(err))
        comm = comm[:min_len]
        err = err[:min_len]

        # Pearson correlation
        correlation, pvalue = stats.pearsonr(comm, err)

        return correlation, pvalue

    def granger_causality_test(
        self,
        commutator_series: np.ndarray,
        errors: np.ndarray,
        max_lag: int = 5
    ) -> Tuple[float, float]:
        """
        Test if commutator Granger-causes forecast errors.

        Uses simple VAR(p) approach with F-test.

        Args:
            commutator_series: Time series of commutator norms
            errors: Time series of forecast errors
            max_lag: Maximum lag to test

        Returns:
            Tuple of (F-statistic, p-value)
        """
        n = len(commutator_series)
        if n < max_lag * 3:
            logger.warning("Insufficient data for Granger causality test")
            return np.nan, 1.0

        # Build restricted model: errors ~ past errors only
        Y_restricted = errors[max_lag:]
        X_restricted = np.column_stack([
            errors[max_lag - i - 1:-i - 1] for i in range(max_lag)
        ])

        # Add constant
        X_restricted = np.column_stack([np.ones(len(Y_restricted)), X_restricted])

        # OLS for restricted model
        try:
            beta_r, residuals_r, rank_r, s_r = np.linalg.lstsq(X_restricted, Y_restricted, rcond=None)
            ssr_r = np.sum((Y_restricted - X_restricted @ beta_r) ** 2)
        except Exception:
            return np.nan, 1.0

        # Build unrestricted model: errors ~ past errors + past commutator
        X_unrestricted = np.column_stack([
            X_restricted,
            *[commutator_series[max_lag - i - 1:-i - 1] for i in range(max_lag)]
        ])

        # OLS for unrestricted model
        try:
            beta_u, residuals_u, rank_u, s_u = np.linalg.lstsq(X_unrestricted, Y_restricted, rcond=None)
            ssr_u = np.sum((Y_restricted - X_unrestricted @ beta_u) ** 2)
        except Exception:
            return np.nan, 1.0

        # F-test
        df1 = max_lag  # Number of added regressors
        df2 = len(Y_restricted) - X_unrestricted.shape[1]

        if ssr_u <= 0 or df2 <= 0:
            return np.nan, 1.0

        F_stat = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
        p_value = 1 - stats.f.cdf(F_stat, df1, df2)

        return F_stat, p_value

    def permutation_test(
        self,
        commutator_series: np.ndarray,
        errors: np.ndarray,
        n_permutations: int = 1000,
        seed: int = 42
    ) -> Tuple[float, float, float]:
        """
        Permutation test for significance of correlation.

        Shuffles the commutator series to test null hypothesis of no relationship.

        Args:
            commutator_series: Time series of commutator norms
            errors: Time series of forecast errors
            n_permutations: Number of permutations
            seed: Random seed

        Returns:
            Tuple of (actual correlation, permutation p-value, effect size)
        """
        rng = np.random.default_rng(seed)

        # Actual correlation
        actual_corr, _ = self.correlation_with_errors(commutator_series, errors)

        # Permutation distribution
        perm_corrs = []
        for _ in range(n_permutations):
            perm_comm = rng.permutation(commutator_series)
            perm_corr, _ = self.correlation_with_errors(perm_comm, errors)
            perm_corrs.append(perm_corr)

        perm_corrs = np.array(perm_corrs)

        # P-value: proportion of permutations with |correlation| >= |actual|
        p_value = np.mean(np.abs(perm_corrs) >= np.abs(actual_corr))

        # Effect size: how many std above permutation mean
        effect_size = (actual_corr - np.mean(perm_corrs)) / (np.std(perm_corrs) + 1e-10)

        return actual_corr, p_value, effect_size

    def validate_uncertainty_principle(
        self,
        X: np.ndarray,
        y: np.ndarray,
        forecaster,
        alpha: float = 0.05
    ) -> UncertaintyValidationResult:
        """
        Validate the uncertainty principle hypothesis.

        Tests whether higher commutator magnitude correlates with higher
        forecast errors, as predicted by the quantum uncertainty principle.

        Args:
            X: Feature data
            y: Target values
            forecaster: Fitted QCMLVolForecaster
            alpha: Significance level

        Returns:
            UncertaintyValidationResult with test outcomes
        """
        # Compute predictions and errors
        predictions = forecaster.predict(X)
        errors = np.abs(predictions - y)

        # Compute uncertainty bounds at each point
        uncertainty_bounds = np.array([
            forecaster.compute_uncertainty_bound(x) for x in X
        ])

        # Correlation between uncertainty bound and errors
        correlation, corr_pvalue = stats.pearsonr(uncertainty_bounds, errors)

        # Granger causality
        granger_f, granger_p = self.granger_causality_test(uncertainty_bounds, errors)

        # Permutation test
        actual_corr, perm_pvalue, effect_size = self.permutation_test(
            uncertainty_bounds, errors
        )

        # Determine significance
        is_significant = (
            corr_pvalue < alpha and
            correlation > 0 and  # Positive correlation expected
            perm_pvalue < alpha
        )

        # Compute permutation statistics for reporting
        rng = np.random.default_rng(42)
        perm_corrs = []
        for _ in range(1000):
            perm_bounds = rng.permutation(uncertainty_bounds)
            perm_corr, _ = stats.pearsonr(perm_bounds, errors)
            perm_corrs.append(perm_corr)

        return UncertaintyValidationResult(
            correlation=correlation,
            correlation_pvalue=corr_pvalue,
            granger_f_stat=granger_f,
            granger_pvalue=granger_p,
            permutation_mean=np.mean(perm_corrs),
            permutation_std=np.std(perm_corrs),
            actual_correlation=actual_corr,
            permutation_pvalue=perm_pvalue,
            is_significant=is_significant
        )

    def compute_effective_dimension(self, commutator: np.ndarray) -> float:
        """
        Compute effective dimension of commutator based on singular values.

        A higher effective dimension indicates the non-commutativity
        is spread across more dimensions.

        Args:
            commutator: Commutator matrix

        Returns:
            Effective dimension (participation ratio of singular values)
        """
        singular_values = np.linalg.svd(commutator, compute_uv=False)
        sv_normalized = singular_values / (np.sum(singular_values) + 1e-10)

        # Participation ratio
        return 1 / (np.sum(sv_normalized ** 2) + 1e-10)


if __name__ == "__main__":
    # Test the analyzer
    logging.basicConfig(level=logging.INFO)

    print("Testing Commutator Analyzer...")

    # Create mock forecaster for testing
    class MockForecaster:
        def __init__(self):
            np.random.seed(42)
            # Create random Hermitian operators
            A = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
            self.iv_op = (A + A.conj().T) / 2
            B = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
            self.rv_op = (B + B.conj().T) / 2

        def get_iv_operator(self):
            return self.iv_op

        def get_rv_operator(self):
            return self.rv_op

        def compute_uncertainty_bound(self, x):
            return np.abs(np.random.randn()) * 0.1

        def predict(self, X):
            return np.random.randn(len(X)) * 0.1

    # Test basic analysis
    analyzer = CommutatorAnalyzer()
    forecaster = MockForecaster()

    result = analyzer.analyze_commutator(forecaster)
    print(f"\nCommutator Analysis:")
    print(f"  Frobenius norm: {result.frobenius_norm:.6f}")
    print(f"  Spectral norm: {result.spectral_norm:.6f}")
    print(f"  Trace norm: {result.trace_norm:.6f}")
    print(f"  Is anti-Hermitian: {result.is_anti_hermitian}")
    print(f"  Max |eigenvalue|: {result.max_eigenvalue_abs:.6f}")

    # Test correlation analysis
    np.random.seed(42)
    comm_series = np.abs(np.random.randn(100))
    errors = comm_series * 0.5 + np.random.randn(100) * 0.3  # Correlated

    corr, pval = analyzer.correlation_with_errors(comm_series, errors)
    print(f"\nCorrelation Analysis:")
    print(f"  Correlation: {corr:.4f}")
    print(f"  P-value: {pval:.4e}")

    # Test permutation test
    actual, perm_p, effect = analyzer.permutation_test(comm_series, errors)
    print(f"\nPermutation Test:")
    print(f"  Actual correlation: {actual:.4f}")
    print(f"  Permutation p-value: {perm_p:.4f}")
    print(f"  Effect size: {effect:.4f}")

    # Test effective dimension
    eff_dim = analyzer.compute_effective_dimension(result.commutator)
    print(f"\nEffective Dimension: {eff_dim:.4f}")

    print("\nCommutator Analyzer tests passed!")
