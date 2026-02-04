"""
Data Preprocessing Module for QCML Regime Detection

Production-grade preprocessing with:
- Missing data handling (forward fill, interpolation, dropping)
- Outlier detection and treatment (rolling z-score, clipping)
- Normalization methods (rolling z-score, cross-sectional, MinMax)
- Walk-forward validation with embargo period (critical for regime detection)

Design Decisions:
- Walk-forward validation prevents lookahead bias in regime detection
- Embargo period handles autocorrelation in financial data
- Rolling normalization avoids using future information

Author: QCML Research
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Generator, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class NormalizationMethod(Enum):
    """Normalization methods for feature preprocessing."""
    ROLLING_ZSCORE = auto()  # Default: rolling z-score (no lookahead)
    CROSS_SECTIONAL = auto()  # Cross-sectional standardization
    MINMAX = auto()  # Min-max scaling (rolling)
    NONE = auto()  # No normalization


class OutlierMethod(Enum):
    """Methods for outlier detection and treatment."""
    ROLLING_ZSCORE = auto()  # Rolling z-score detection
    IQR = auto()  # Interquartile range method
    CLIP = auto()  # Simple clipping at fixed percentiles
    NONE = auto()  # No outlier treatment


class MissingDataMethod(Enum):
    """Methods for handling missing data."""
    FORWARD_FILL = auto()  # Forward fill with limit
    INTERPOLATE = auto()  # Linear interpolation
    DROP = auto()  # Drop rows with missing data
    FILL_MEAN = auto()  # Fill with rolling mean


@dataclass
class PreprocessingConfig:
    """
    Configuration for data preprocessing.

    Args:
        normalization: Normalization method (default: ROLLING_ZSCORE)
        norm_window: Window for rolling normalization (default: 252)
        outlier_method: Outlier treatment method (default: ROLLING_ZSCORE)
        outlier_window: Window for outlier detection (default: 60)
        outlier_threshold: Z-score threshold for outliers (default: 5.0)
        clip_percentile: Percentile for clipping outliers (default: 99.5)
        missing_method: Missing data handling (default: FORWARD_FILL)
        ffill_limit: Maximum forward fill periods (default: 5)
        train_window: Training window size for walk-forward (default: 252)
        test_window: Test window size for walk-forward (default: 21)
        embargo_days: Gap between train and test to prevent leakage (default: 5)
        min_train_samples: Minimum training samples required (default: 126)
        expanding_train: Use expanding training window (default: False)
        min_periods_ratio: Minimum periods as ratio of window (default: 0.5)

    Example:
        >>> config = PreprocessingConfig(
        ...     normalization=NormalizationMethod.ROLLING_ZSCORE,
        ...     train_window=252,
        ...     test_window=21,
        ...     embargo_days=5
        ... )
    """
    # Normalization settings
    normalization: NormalizationMethod = NormalizationMethod.ROLLING_ZSCORE
    norm_window: int = 252

    # Outlier settings
    outlier_method: OutlierMethod = OutlierMethod.ROLLING_ZSCORE
    outlier_window: int = 60
    outlier_threshold: float = 5.0
    clip_percentile: float = 99.5

    # Missing data settings
    missing_method: MissingDataMethod = MissingDataMethod.FORWARD_FILL
    ffill_limit: int = 5

    # Walk-forward settings
    train_window: int = 252  # ~1 year of trading days
    test_window: int = 21  # ~1 month of trading days
    embargo_days: int = 5  # Gap to prevent leakage
    min_train_samples: int = 126  # ~6 months minimum
    expanding_train: bool = False  # Rolling vs expanding window

    # General settings
    min_periods_ratio: float = 0.5


@dataclass
class WalkForwardFold:
    """
    Container for a single walk-forward fold.

    Attributes:
        fold_index: Index of this fold
        train_start: Start index of training period
        train_end: End index of training period (exclusive)
        test_start: Start index of test period
        test_end: End index of test period (exclusive)
        train_features: Training feature DataFrame
        train_prices: Training price Series
        test_features: Test feature DataFrame
        test_prices: Test price Series
        train_dates: Training date range
        test_dates: Test date range
    """
    fold_index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_features: pd.DataFrame
    train_prices: pd.Series
    test_features: pd.DataFrame
    test_prices: pd.Series
    train_dates: Tuple[pd.Timestamp, pd.Timestamp]
    test_dates: Tuple[pd.Timestamp, pd.Timestamp]


class DataPreprocessor:
    """
    Data preprocessing pipeline for QCML feature engineering.

    Handles missing data, outliers, and normalization with proper
    temporal ordering to prevent lookahead bias.

    Args:
        config: PreprocessingConfig with all settings

    Example:
        >>> preprocessor = DataPreprocessor()
        >>> cleaned = preprocessor.fit_transform(features)
        >>>
        >>> # Walk-forward validation
        >>> for fold in preprocessor.walk_forward_split(features, prices):
        ...     train_X, train_p = fold.train_features, fold.train_prices
        ...     test_X, test_p = fold.test_features, fold.test_prices
        ...     # Train model on train_X, evaluate on test_X
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self._fitted = False
        self._fit_stats: Dict = {}
        logger.info(f"DataPreprocessor initialized: norm={self.config.normalization.name}")

    # =========================================================================
    # MISSING DATA HANDLING
    # =========================================================================

    def handle_missing_data(
        self,
        df: pd.DataFrame,
        method: Optional[MissingDataMethod] = None
    ) -> pd.DataFrame:
        """
        Handle missing data in feature DataFrame.

        Args:
            df: Feature DataFrame with potential missing values
            method: Missing data method (default: config.missing_method)

        Returns:
            DataFrame with missing data handled

        Example:
            >>> clean_df = preprocessor.handle_missing_data(features)
        """
        method = method or self.config.missing_method
        result = df.copy()

        initial_nans = result.isna().sum().sum()
        if initial_nans == 0:
            logger.debug("No missing data found")
            return result

        logger.info(f"Handling {initial_nans} missing values with method: {method.name}")

        if method == MissingDataMethod.FORWARD_FILL:
            result = result.ffill(limit=self.config.ffill_limit)

        elif method == MissingDataMethod.INTERPOLATE:
            result = result.interpolate(method='linear', limit_direction='forward')

        elif method == MissingDataMethod.DROP:
            result = result.dropna()

        elif method == MissingDataMethod.FILL_MEAN:
            # Rolling mean to avoid lookahead
            for col in result.columns:
                mask = result[col].isna()
                if mask.any():
                    rolling_mean = result[col].rolling(
                        window=20,
                        min_periods=1
                    ).mean()
                    result.loc[mask, col] = rolling_mean.loc[mask]

        remaining_nans = result.isna().sum().sum()
        if remaining_nans > 0:
            logger.warning(f"{remaining_nans} NaN values remain after handling")

        return result

    # =========================================================================
    # OUTLIER DETECTION AND TREATMENT
    # =========================================================================

    def detect_outliers(
        self,
        df: pd.DataFrame,
        method: Optional[OutlierMethod] = None
    ) -> pd.DataFrame:
        """
        Detect outliers in feature DataFrame.

        Args:
            df: Feature DataFrame
            method: Outlier detection method (default: config.outlier_method)

        Returns:
            Boolean DataFrame where True indicates outlier

        Example:
            >>> outlier_mask = preprocessor.detect_outliers(features)
            >>> print(f"Outliers found: {outlier_mask.sum().sum()}")
        """
        method = method or self.config.outlier_method

        if method == OutlierMethod.NONE:
            return pd.DataFrame(False, index=df.index, columns=df.columns)

        elif method == OutlierMethod.ROLLING_ZSCORE:
            min_periods = max(1, int(self.config.outlier_window * self.config.min_periods_ratio))
            rolling_mean = df.rolling(
                window=self.config.outlier_window,
                min_periods=min_periods
            ).mean()
            rolling_std = df.rolling(
                window=self.config.outlier_window,
                min_periods=min_periods
            ).std()

            zscore = (df - rolling_mean) / rolling_std.replace(0, np.nan)
            return zscore.abs() > self.config.outlier_threshold

        elif method == OutlierMethod.IQR:
            min_periods = max(1, int(self.config.outlier_window * self.config.min_periods_ratio))
            q1 = df.rolling(window=self.config.outlier_window, min_periods=min_periods).quantile(0.25)
            q3 = df.rolling(window=self.config.outlier_window, min_periods=min_periods).quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            return (df < lower) | (df > upper)

        elif method == OutlierMethod.CLIP:
            # Detect based on percentiles
            lower = df.quantile(1 - self.config.clip_percentile / 100)
            upper = df.quantile(self.config.clip_percentile / 100)
            return (df < lower) | (df > upper)

        return pd.DataFrame(False, index=df.index, columns=df.columns)

    def treat_outliers(
        self,
        df: pd.DataFrame,
        method: Optional[OutlierMethod] = None
    ) -> pd.DataFrame:
        """
        Treat outliers in feature DataFrame.

        Args:
            df: Feature DataFrame with potential outliers
            method: Outlier treatment method (default: config.outlier_method)

        Returns:
            DataFrame with outliers treated (clipped or replaced)

        Example:
            >>> treated_df = preprocessor.treat_outliers(features)
        """
        method = method or self.config.outlier_method

        if method == OutlierMethod.NONE:
            return df.copy()

        result = df.copy()

        if method == OutlierMethod.ROLLING_ZSCORE:
            # Clip at ± threshold standard deviations
            min_periods = max(1, int(self.config.outlier_window * self.config.min_periods_ratio))
            rolling_mean = df.rolling(
                window=self.config.outlier_window,
                min_periods=min_periods
            ).mean()
            rolling_std = df.rolling(
                window=self.config.outlier_window,
                min_periods=min_periods
            ).std()

            lower = rolling_mean - self.config.outlier_threshold * rolling_std
            upper = rolling_mean + self.config.outlier_threshold * rolling_std

            result = result.clip(lower=lower, upper=upper, axis=1)

        elif method == OutlierMethod.IQR:
            min_periods = max(1, int(self.config.outlier_window * self.config.min_periods_ratio))
            q1 = df.rolling(window=self.config.outlier_window, min_periods=min_periods).quantile(0.25)
            q3 = df.rolling(window=self.config.outlier_window, min_periods=min_periods).quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            result = result.clip(lower=lower, upper=upper, axis=1)

        elif method == OutlierMethod.CLIP:
            for col in result.columns:
                lower = result[col].quantile(1 - self.config.clip_percentile / 100)
                upper = result[col].quantile(self.config.clip_percentile / 100)
                result[col] = result[col].clip(lower=lower, upper=upper)

        outlier_count = self.detect_outliers(df, method).sum().sum()
        logger.info(f"Treated {outlier_count} outliers with method: {method.name}")

        return result

    # =========================================================================
    # NORMALIZATION
    # =========================================================================

    def normalize(
        self,
        df: pd.DataFrame,
        method: Optional[NormalizationMethod] = None,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Normalize feature DataFrame.

        Args:
            df: Feature DataFrame
            method: Normalization method (default: config.normalization)
            fit: Whether to fit normalization parameters (True for training)

        Returns:
            Normalized DataFrame

        Example:
            >>> # Fit and transform training data
            >>> train_norm = preprocessor.normalize(train_features, fit=True)
            >>> # Transform test data using fitted parameters
            >>> test_norm = preprocessor.normalize(test_features, fit=False)
        """
        method = method or self.config.normalization

        if method == NormalizationMethod.NONE:
            return df.copy()

        result = df.copy()

        if method == NormalizationMethod.ROLLING_ZSCORE:
            min_periods = max(1, int(self.config.norm_window * self.config.min_periods_ratio))
            rolling_mean = df.rolling(
                window=self.config.norm_window,
                min_periods=min_periods
            ).mean()
            rolling_std = df.rolling(
                window=self.config.norm_window,
                min_periods=min_periods
            ).std()

            result = (df - rolling_mean) / rolling_std.replace(0, np.nan)

            if fit:
                # Store last values for potential future use
                self._fit_stats['rolling_mean_last'] = rolling_mean.iloc[-1]
                self._fit_stats['rolling_std_last'] = rolling_std.iloc[-1]

        elif method == NormalizationMethod.CROSS_SECTIONAL:
            # Standardize across columns (cross-sectionally) at each time step
            row_mean = df.mean(axis=1)
            row_std = df.std(axis=1)

            result = df.sub(row_mean, axis=0).div(row_std.replace(0, np.nan), axis=0)

        elif method == NormalizationMethod.MINMAX:
            min_periods = max(1, int(self.config.norm_window * self.config.min_periods_ratio))
            rolling_min = df.rolling(
                window=self.config.norm_window,
                min_periods=min_periods
            ).min()
            rolling_max = df.rolling(
                window=self.config.norm_window,
                min_periods=min_periods
            ).max()

            range_ = rolling_max - rolling_min
            result = (df - rolling_min) / range_.replace(0, np.nan)

            if fit:
                self._fit_stats['rolling_min_last'] = rolling_min.iloc[-1]
                self._fit_stats['rolling_max_last'] = rolling_max.iloc[-1]

        if fit:
            self._fit_stats['method'] = method
            self._fitted = True

        logger.debug(f"Normalized features with method: {method.name}")

        return result

    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================

    def fit_transform(
        self,
        df: pd.DataFrame,
        handle_missing: bool = True,
        treat_outliers: bool = True,
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Apply full preprocessing pipeline.

        Args:
            df: Feature DataFrame
            handle_missing: Whether to handle missing data
            treat_outliers: Whether to treat outliers
            normalize: Whether to normalize features

        Returns:
            Preprocessed DataFrame

        Example:
            >>> preprocessor = DataPreprocessor()
            >>> clean_features = preprocessor.fit_transform(raw_features)
        """
        result = df.copy()

        if handle_missing:
            result = self.handle_missing_data(result)

        if treat_outliers:
            result = self.treat_outliers(result)

        if normalize:
            result = self.normalize(result, fit=True)

        # Final check: drop any remaining NaN from warmup
        n_before = len(result)
        result = result.dropna()
        n_dropped = n_before - len(result)

        if n_dropped > 0:
            logger.info(f"Dropped {n_dropped} rows with remaining NaN values")

        logger.info(f"Preprocessing complete: {result.shape[0]} samples, {result.shape[1]} features")

        return result

    def transform(
        self,
        df: pd.DataFrame,
        handle_missing: bool = True,
        treat_outliers: bool = True,
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Apply preprocessing without fitting (use previously fitted parameters).

        Args:
            df: Feature DataFrame
            handle_missing: Whether to handle missing data
            treat_outliers: Whether to treat outliers
            normalize: Whether to normalize features

        Returns:
            Preprocessed DataFrame
        """
        result = df.copy()

        if handle_missing:
            result = self.handle_missing_data(result)

        if treat_outliers:
            result = self.treat_outliers(result)

        if normalize:
            result = self.normalize(result, fit=False)

        return result.dropna()

    # =========================================================================
    # WALK-FORWARD VALIDATION
    # =========================================================================

    def walk_forward_split(
        self,
        features: pd.DataFrame,
        prices: pd.Series,
        train_window: Optional[int] = None,
        test_window: Optional[int] = None,
        embargo_days: Optional[int] = None,
        expanding: Optional[bool] = None
    ) -> Iterator[WalkForwardFold]:
        """
        Generate walk-forward validation folds with embargo period.

        Walk-forward validation is critical for regime detection to prevent
        lookahead bias. The embargo period prevents information leakage from
        autocorrelated financial time series.

        Args:
            features: Feature DataFrame (index must be DatetimeIndex)
            prices: Price series aligned with features
            train_window: Training window size (default: config.train_window)
            test_window: Test window size (default: config.test_window)
            embargo_days: Gap between train/test (default: config.embargo_days)
            expanding: Use expanding window (default: config.expanding_train)

        Yields:
            WalkForwardFold objects containing train/test splits

        Example:
            >>> for fold in preprocessor.walk_forward_split(features, prices):
            ...     print(f"Fold {fold.fold_index}:")
            ...     print(f"  Train: {fold.train_dates[0]} to {fold.train_dates[1]}")
            ...     print(f"  Test:  {fold.test_dates[0]} to {fold.test_dates[1]}")
            ...
            ...     # Train model
            ...     model.fit(fold.train_features, fold.train_prices)
            ...
            ...     # Evaluate
            ...     predictions = model.predict(fold.test_features)
        """
        train_window = train_window or self.config.train_window
        test_window = test_window or self.config.test_window
        embargo_days = embargo_days or self.config.embargo_days
        expanding = expanding if expanding is not None else self.config.expanding_train

        # Ensure aligned indices
        if not isinstance(features.index, pd.DatetimeIndex):
            logger.warning("Features index is not DatetimeIndex, attempting conversion")
            features = features.copy()
            features.index = pd.to_datetime(features.index)

        if not isinstance(prices.index, pd.DatetimeIndex):
            prices = prices.copy()
            prices.index = pd.to_datetime(prices.index)

        # Align features and prices
        common_idx = features.index.intersection(prices.index)
        features = features.loc[common_idx]
        prices = prices.loc[common_idx]

        n_samples = len(features)

        # Minimum size check
        min_required = train_window + embargo_days + test_window
        if n_samples < min_required:
            raise ValueError(
                f"Insufficient data: {n_samples} samples, need at least {min_required} "
                f"(train={train_window}, embargo={embargo_days}, test={test_window})"
            )

        logger.info(
            f"Walk-forward split: {n_samples} samples, "
            f"train={train_window}, test={test_window}, embargo={embargo_days}, "
            f"expanding={expanding}"
        )

        fold_index = 0

        # Starting position for first test set
        if expanding:
            # Start when we have minimum training samples
            current_test_start = self.config.min_train_samples + embargo_days
        else:
            # Start when we have full training window
            current_test_start = train_window + embargo_days

        while current_test_start + test_window <= n_samples:
            # Calculate indices
            if expanding:
                train_start = 0
            else:
                train_start = current_test_start - embargo_days - train_window

            train_end = current_test_start - embargo_days
            test_start = current_test_start
            test_end = min(current_test_start + test_window, n_samples)

            # Skip if train or test is too small
            train_size = train_end - train_start
            test_size = test_end - test_start

            if train_size < self.config.min_train_samples:
                current_test_start += test_window
                continue

            if test_size < 1:
                break

            # Extract data
            train_features = features.iloc[train_start:train_end]
            train_prices = prices.iloc[train_start:train_end]
            test_features = features.iloc[test_start:test_end]
            test_prices = prices.iloc[test_start:test_end]

            # Create fold
            fold = WalkForwardFold(
                fold_index=fold_index,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_features=train_features,
                train_prices=train_prices,
                test_features=test_features,
                test_prices=test_prices,
                train_dates=(train_features.index[0], train_features.index[-1]),
                test_dates=(test_features.index[0], test_features.index[-1])
            )

            yield fold

            fold_index += 1
            current_test_start += test_window

        logger.info(f"Generated {fold_index} walk-forward folds")

    def get_walk_forward_info(
        self,
        n_samples: int,
        train_window: Optional[int] = None,
        test_window: Optional[int] = None,
        embargo_days: Optional[int] = None,
        expanding: Optional[bool] = None
    ) -> Dict:
        """
        Get information about walk-forward splits without generating data.

        Args:
            n_samples: Total number of samples
            train_window: Training window size
            test_window: Test window size
            embargo_days: Embargo period
            expanding: Use expanding window

        Returns:
            Dictionary with split information

        Example:
            >>> info = preprocessor.get_walk_forward_info(500)
            >>> print(f"Expected folds: {info['n_folds']}")
        """
        train_window = train_window or self.config.train_window
        test_window = test_window or self.config.test_window
        embargo_days = embargo_days or self.config.embargo_days
        expanding = expanding if expanding is not None else self.config.expanding_train

        min_required = train_window + embargo_days + test_window

        if n_samples < min_required:
            return {
                'n_folds': 0,
                'n_samples': n_samples,
                'min_required': min_required,
                'sufficient_data': False,
                'message': f"Need {min_required} samples, have {n_samples}"
            }

        if expanding:
            first_test_start = self.config.min_train_samples + embargo_days
        else:
            first_test_start = train_window + embargo_days

        available_for_testing = n_samples - first_test_start
        n_folds = available_for_testing // test_window

        return {
            'n_folds': n_folds,
            'n_samples': n_samples,
            'train_window': train_window,
            'test_window': test_window,
            'embargo_days': embargo_days,
            'expanding': expanding,
            'first_test_start': first_test_start,
            'min_required': min_required,
            'sufficient_data': True,
            'coverage': (n_folds * test_window) / n_samples
        }

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def validate_preprocessing(self, df: pd.DataFrame) -> Dict:
        """
        Validate preprocessed data for common issues.

        Args:
            df: Preprocessed DataFrame

        Returns:
            Dictionary with validation results
        """
        issues = []

        # Check for NaN
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            issues.append(f"Contains {nan_count} NaN values")

        # Check for infinite values
        inf_count = np.isinf(df.values).sum()
        if inf_count > 0:
            issues.append(f"Contains {inf_count} infinite values")

        # Check for constant columns
        constant_cols = df.columns[df.std() == 0].tolist()
        if constant_cols:
            issues.append(f"Constant columns: {constant_cols}")

        # Check normalization (if rolling z-score, most values should be in [-3, 3])
        if self.config.normalization == NormalizationMethod.ROLLING_ZSCORE:
            extreme_ratio = ((df.abs() > 3).sum().sum()) / df.size
            if extreme_ratio > 0.1:
                issues.append(f"High ratio of extreme values ({extreme_ratio:.2%} > |3|)")

        return {
            'is_valid': len(issues) == 0,
            'n_samples': len(df),
            'n_features': len(df.columns),
            'issues': issues,
            'stats': {
                'mean_range': (df.mean().min(), df.mean().max()),
                'std_range': (df.std().min(), df.std().max()),
                'skew_range': (df.skew().min(), df.skew().max()),
            }
        }

    def get_config_summary(self) -> Dict:
        """Get summary of preprocessing configuration."""
        return {
            'normalization': self.config.normalization.name,
            'norm_window': self.config.norm_window,
            'outlier_method': self.config.outlier_method.name,
            'outlier_window': self.config.outlier_window,
            'outlier_threshold': self.config.outlier_threshold,
            'missing_method': self.config.missing_method.name,
            'ffill_limit': self.config.ffill_limit,
            'train_window': self.config.train_window,
            'test_window': self.config.test_window,
            'embargo_days': self.config.embargo_days,
            'expanding_train': self.config.expanding_train
        }


def create_preprocessor(
    normalization: str = "rolling_zscore",
    train_window: int = 252,
    test_window: int = 21,
    embargo_days: int = 5,
    **kwargs
) -> DataPreprocessor:
    """
    Factory function to create DataPreprocessor with common settings.

    Args:
        normalization: "rolling_zscore", "cross_sectional", "minmax", or "none"
        train_window: Training window for walk-forward
        test_window: Test window for walk-forward
        embargo_days: Embargo period
        **kwargs: Additional config parameters

    Returns:
        Configured DataPreprocessor

    Example:
        >>> preprocessor = create_preprocessor(
        ...     normalization="rolling_zscore",
        ...     train_window=252,
        ...     test_window=21
        ... )
    """
    norm_map = {
        'rolling_zscore': NormalizationMethod.ROLLING_ZSCORE,
        'cross_sectional': NormalizationMethod.CROSS_SECTIONAL,
        'minmax': NormalizationMethod.MINMAX,
        'none': NormalizationMethod.NONE
    }

    normalization_method = norm_map.get(normalization.lower(), NormalizationMethod.ROLLING_ZSCORE)

    config = PreprocessingConfig(
        normalization=normalization_method,
        train_window=train_window,
        test_window=test_window,
        embargo_days=embargo_days,
        **kwargs
    )

    return DataPreprocessor(config)
