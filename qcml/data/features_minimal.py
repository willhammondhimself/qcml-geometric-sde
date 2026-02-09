"""
Minimal Feature Engine for Quick Hypothesis Validation

Computes 5 essential features for testing the Chern number hypothesis
on 2008 crisis data before investing in full Phase 2 feature engineering.

Base Features (per symbol):
    1. Log returns - Core price dynamics
    2. Rolling volatility - Regime indicator
    3. Rolling mean - Momentum
    4. Correlation to benchmark - Cross-sectional relationship
    5. Relative strength - Cross-sectional performance

Optional Bond Features (if bond_col provided):
    6. Bond returns - Treasury price dynamics
    7. Bond volatility - Duration risk indicator
    8. Bond momentum - Treasury trend
    9. Equity-bond correlation - Flight-to-safety indicator
    10. Duration spread - Bond vol / Equity vol (rate sensitivity)

Author: QCML Research
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MinimalFeatureEngine:
    """
    Minimal features for hypothesis validation.

    Computes 5 essential features sufficient to test whether Chern number
    discontinuities can detect regime changes in real markets.

    Args:
        window: Rolling window size for volatility/mean/correlation (default: 20)

    Example:
        >>> engine = MinimalFeatureEngine(window=20)
        >>> prices = pd.DataFrame({'SPY': spy_prices, 'XLF': xlf_prices})
        >>> features = engine.create_feature_matrix(prices, benchmark_col='SPY')
        >>> print(features.shape)
    """

    def __init__(self, window: int = 20):
        self.window = window
        logger.info(f"MinimalFeatureEngine initialized with window={window}")

    def compute_returns(self, prices: pd.Series) -> pd.Series:
        """
        Compute log returns from price series.

        Args:
            prices: Price series

        Returns:
            Log returns series (first value is NaN)

        Example:
            >>> returns = engine.compute_returns(spy_prices)
        """
        return np.log(prices / prices.shift(1))

    def compute_volatility(
        self, returns: pd.Series, window: Optional[int] = None
    ) -> pd.Series:
        """
        Compute rolling volatility (standard deviation of returns).

        Args:
            returns: Returns series
            window: Rolling window size (default: self.window)

        Returns:
            Rolling volatility series

        Example:
            >>> vol = engine.compute_volatility(returns, window=20)
        """
        window = window or self.window
        return returns.rolling(window=window, min_periods=window).std()

    def compute_rolling_mean(
        self, returns: pd.Series, window: Optional[int] = None
    ) -> pd.Series:
        """
        Compute rolling mean return (momentum indicator).

        Args:
            returns: Returns series
            window: Rolling window size (default: self.window)

        Returns:
            Rolling mean returns series

        Example:
            >>> momentum = engine.compute_rolling_mean(returns, window=20)
        """
        window = window or self.window
        return returns.rolling(window=window, min_periods=window).mean()

    def compute_correlation(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        window: Optional[int] = None,
    ) -> pd.Series:
        """
        Compute rolling correlation to benchmark.

        Args:
            returns: Asset returns series
            benchmark_returns: Benchmark returns series (e.g., SPY)
            window: Rolling window size (default: self.window)

        Returns:
            Rolling correlation series

        Example:
            >>> corr = engine.compute_correlation(xlf_returns, spy_returns, window=20)
        """
        window = window or self.window
        return returns.rolling(window=window, min_periods=window).corr(benchmark_returns)

    def compute_relative_strength(
        self, returns: pd.Series, benchmark_returns: pd.Series
    ) -> pd.Series:
        """
        Compute relative strength (returns minus benchmark returns).

        Args:
            returns: Asset returns series
            benchmark_returns: Benchmark returns series

        Returns:
            Relative strength series

        Example:
            >>> rel_strength = engine.compute_relative_strength(xlf_returns, spy_returns)
        """
        return returns - benchmark_returns

    def create_feature_matrix(
        self, prices_df: pd.DataFrame, benchmark_col: str = "SPY",
        bond_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create feature matrix from price DataFrame.

        Computes all 5 minimal features for each symbol and combines
        into a single feature matrix suitable for QCML processing.

        If bond_col is provided, adds 5 bond/cross-asset features:
        - Bond returns, volatility, momentum
        - Equity-bond correlation
        - Duration spread (bond vol / equity vol)

        Args:
            prices_df: DataFrame with columns as symbols, index as dates
            benchmark_col: Column name of benchmark (default: 'SPY')
            bond_col: Optional bond ETF column (e.g., 'TLT'). If provided,
                     adds bond-specific features for rate crisis detection.

        Returns:
            DataFrame with features for each symbol, aligned by date.
            Shape: (n_dates, n_symbols * 5 + 5) if bond_col provided
            NaN values from warmup period are dropped.

        Example:
            >>> prices = pd.DataFrame({
            ...     'SPY': spy_prices,
            ...     'XLF': xlf_prices,
            ...     'BAC': bac_prices,
            ...     'TLT': tlt_prices
            ... })
            >>> features = engine.create_feature_matrix(
            ...     prices, benchmark_col='SPY', bond_col='TLT'
            ... )
            >>> print(f"Features: {features.columns.tolist()}")
        """
        logger.info(
            f"Creating feature matrix for {len(prices_df.columns)} symbols, "
            f"benchmark={benchmark_col}" +
            (f", bond={bond_col}" if bond_col else "")
        )

        if benchmark_col not in prices_df.columns:
            raise ValueError(f"Benchmark column '{benchmark_col}' not in DataFrame")

        if bond_col is not None and bond_col not in prices_df.columns:
            raise ValueError(f"Bond column '{bond_col}' not in DataFrame")

        # Compute benchmark returns first
        benchmark_returns = self.compute_returns(prices_df[benchmark_col])

        feature_dfs = []

        for col in prices_df.columns:
            prices = prices_df[col]
            returns = self.compute_returns(prices)

            # Compute all 5 features
            features = pd.DataFrame(
                {
                    f"{col}_returns": returns,
                    f"{col}_volatility": self.compute_volatility(returns),
                    f"{col}_momentum": self.compute_rolling_mean(returns),
                    f"{col}_corr_benchmark": self.compute_correlation(
                        returns, benchmark_returns
                    ),
                    f"{col}_rel_strength": self.compute_relative_strength(
                        returns, benchmark_returns
                    ),
                }
            )

            feature_dfs.append(features)

        # Combine all features
        result = pd.concat(feature_dfs, axis=1)

        # Add bond/cross-asset features if bond_col provided
        if bond_col is not None:
            logger.info(f"Adding bond/cross-asset features using {bond_col}")

            bond_returns = self.compute_returns(prices_df[bond_col])
            bond_vol = self.compute_volatility(bond_returns)
            equity_vol = self.compute_volatility(benchmark_returns)

            # Create bond feature set
            bond_features = pd.DataFrame({
                'bond_returns': bond_returns,
                'bond_volatility': bond_vol,
                'bond_momentum': self.compute_rolling_mean(bond_returns),
                'equity_bond_corr': self.compute_correlation(
                    benchmark_returns, bond_returns
                ),
                'duration_spread': bond_vol / (equity_vol + 1e-8),  # Rate sensitivity
            })

            result = pd.concat([result, bond_features], axis=1)

        # Drop warmup period (rows with NaN)
        n_before = len(result)
        result = result.dropna()
        n_after = len(result)

        logger.info(
            f"Feature matrix created: {result.shape[0]} samples, "
            f"{result.shape[1]} features (dropped {n_before - n_after} warmup rows)"
        )

        return result

    def validate_features(self, features: pd.DataFrame) -> dict:
        """
        Validate feature matrix for common issues.

        Args:
            features: Feature DataFrame

        Returns:
            Dictionary with validation results

        Example:
            >>> validation = engine.validate_features(features)
            >>> if validation['is_valid']:
            ...     print("Features ready for QCML")
        """
        issues = []

        # Check for NaN values
        nan_count = features.isna().sum().sum()
        if nan_count > 0:
            issues.append(f"Contains {nan_count} NaN values")

        # Check for infinite values
        inf_count = np.isinf(features.values).sum()
        if inf_count > 0:
            issues.append(f"Contains {inf_count} infinite values")

        # Check for constant columns
        constant_cols = features.columns[features.std() == 0].tolist()
        if constant_cols:
            issues.append(f"Constant columns: {constant_cols}")

        # Check for reasonable value ranges
        for col in features.columns:
            if "returns" in col or "rel_strength" in col:
                # Returns should be within reasonable bounds
                if features[col].abs().max() > 0.5:  # 50% daily return is extreme
                    issues.append(f"{col} has extreme values (max abs: {features[col].abs().max():.4f})")
            elif "volatility" in col:
                # Volatility should be positive
                if (features[col] < 0).any():
                    issues.append(f"{col} has negative values")
            elif "corr" in col:
                # Correlation should be in [-1, 1]
                if (features[col].abs() > 1.001).any():  # Small tolerance
                    issues.append(f"{col} has correlations outside [-1, 1]")

        return {
            "is_valid": len(issues) == 0,
            "n_samples": len(features),
            "n_features": len(features.columns),
            "issues": issues,
            "feature_stats": features.describe().to_dict(),
        }
