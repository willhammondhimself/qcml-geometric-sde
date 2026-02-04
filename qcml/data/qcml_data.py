"""
QCML Data Interface Module

Dataset interface for integrating market data with the QCML framework.
Provides format conversion and helper functions for regime detection research.

Classes:
    QCMLDataset: Dataset wrapper for QCML framework integration

Functions:
    load_crisis_dataset: Load pre-configured crisis datasets
    create_multi_timeframe_dataset: Create datasets at multiple resolutions
"""

import logging
from typing import Dict, List, Optional, Tuple, Set, Union
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FeatureEngine, FeatureConfig, FeatureCategory
from .features_minimal import MinimalFeatureEngine
from .preprocessing import DataPreprocessor, PreprocessingConfig, NormalizationMethod, OutlierMethod
from .storage import ParquetDataStore

logger = logging.getLogger(__name__)


class QCMLDataset:
    """
    Dataset interface for QCML framework

    Wraps feature matrices and price data in a format compatible with
    QCML geometry learning and regime detection modules.

    Args:
        features: DataFrame with feature data (MultiIndex: symbol, timestamp)
        prices: Series or DataFrame with price data
        times: DatetimeIndex for temporal ordering
        metadata: Dictionary with dataset metadata (symbols, crisis info, etc.)

    Example:
        >>> features = pd.DataFrame(np.random.randn(100, 10))
        >>> prices = pd.Series(np.cumsum(np.random.randn(100)))
        >>> times = pd.date_range("2024-01-01", periods=100, freq="D")
        >>> dataset = QCMLDataset(features, prices, times, {'universe': 'SP500'})
        >>>
        >>> # Use with QCML modules
        >>> X, p, t = dataset.to_qcml_format()
        >>> geometry = QCMLGeometry(n_features=X.shape[1])
        >>> geometry.fit_operators(X)
    """

    def __init__(
        self,
        features: pd.DataFrame,
        prices: pd.Series,
        times: pd.DatetimeIndex,
        metadata: Dict
    ):
        # Validate inputs
        if len(features) != len(prices) != len(times):
            raise ValueError("Features, prices, and times must have same length")

        self.features = features
        self.prices = prices
        self.times = pd.DatetimeIndex(times)
        self.metadata = metadata

        logger.info(f"QCMLDataset created: {len(features)} samples, "
                   f"{features.shape[1] if len(features.shape) > 1 else 0} features")

    @property
    def X(self) -> np.ndarray:
        """
        Feature matrix (n_samples, n_features)

        Returns:
            NumPy array of features suitable for QCML operators
        """
        return self.features.values

    @property
    def prices_array(self) -> np.ndarray:
        """
        Price series (n_samples,)

        Returns:
            NumPy array of prices
        """
        if isinstance(self.prices, pd.Series):
            return self.prices.values
        elif isinstance(self.prices, pd.DataFrame):
            # If multiple symbols, return first column or flatten
            if self.prices.shape[1] == 1:
                return self.prices.iloc[:, 0].values
            else:
                logger.warning("Multiple price columns - using first column")
                return self.prices.iloc[:, 0].values
        else:
            return np.array(self.prices)

    @property
    def times_array(self) -> np.ndarray:
        """
        Timestamps (n_samples,) as numpy datetime64

        Returns:
            NumPy array of timestamps
        """
        return self.times.values

    @property
    def returns(self) -> np.ndarray:
        """
        Calculate returns from prices

        Returns:
            NumPy array of returns (n_samples-1,)
        """
        prices = self.prices_array
        return np.diff(np.log(prices))

    @property
    def n_samples(self) -> int:
        """Number of samples in dataset"""
        return len(self.features)

    @property
    def n_features(self) -> int:
        """Number of features"""
        return self.features.shape[1] if len(self.features.shape) > 1 else 0

    def to_qcml_format(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (X, prices, times) for QCML modules

        Returns:
            Tuple of (feature_matrix, price_array, timestamp_array)

        Example:
            >>> X, prices, times = dataset.to_qcml_format()
            >>> geometry = QCMLGeometry(n_features=X.shape[1])
            >>> geometry.fit_operators(X)
        """
        return self.X, self.prices_array, self.times_array

    def split_by_date(
        self,
        split_date: str
    ) -> Tuple['QCMLDataset', 'QCMLDataset']:
        """
        Split into before/after for regime detection

        Args:
            split_date: Date in YYYY-MM-DD format

        Returns:
            Tuple of (before_dataset, after_dataset)

        Example:
            >>> # Split at Lehman Brothers collapse
            >>> before, after = dataset.split_by_date("2008-09-15")
            >>> chern_before = detector.compute_chern_number(before.X)
            >>> chern_after = detector.compute_chern_number(after.X)
        """
        split_timestamp = pd.Timestamp(split_date)

        # Find split index
        split_idx = (self.times <= split_timestamp).sum()

        if split_idx == 0 or split_idx == len(self.times):
            raise ValueError(f"Split date {split_date} is outside dataset range")

        # Split features
        features_before = self.features.iloc[:split_idx]
        features_after = self.features.iloc[split_idx:]

        # Split prices
        if isinstance(self.prices, pd.Series):
            prices_before = self.prices.iloc[:split_idx]
            prices_after = self.prices.iloc[split_idx:]
        else:
            prices_before = self.prices[:split_idx]
            prices_after = self.prices[split_idx:]

        # Split times
        times_before = self.times[:split_idx]
        times_after = self.times[split_idx:]

        # Create metadata for splits
        meta_before = {**self.metadata, 'split': 'before', 'split_date': split_date}
        meta_after = {**self.metadata, 'split': 'after', 'split_date': split_date}

        dataset_before = QCMLDataset(features_before, prices_before, times_before, meta_before)
        dataset_after = QCMLDataset(features_after, prices_after, times_after, meta_after)

        return dataset_before, dataset_after

    def get_window(
        self,
        start_idx: int,
        end_idx: Optional[int] = None
    ) -> 'QCMLDataset':
        """
        Extract a windowed subset of the dataset

        Args:
            start_idx: Start index (inclusive)
            end_idx: End index (exclusive), None means end of dataset

        Returns:
            New QCMLDataset with windowed data

        Example:
            >>> # Get last 30 days
            >>> recent = dataset.get_window(-30, None)
        """
        # Handle None end_idx
        actual_end = end_idx if end_idx is not None else len(self.features)

        features_window = self.features.iloc[start_idx:end_idx]

        if isinstance(self.prices, pd.Series):
            prices_window = self.prices.iloc[start_idx:end_idx]
        else:
            prices_window = self.prices[start_idx:end_idx]

        times_window = self.times[start_idx:end_idx]

        meta_window = {
            **self.metadata,
            'window': f"{start_idx}:{end_idx}",
            'window_size': len(features_window)
        }

        return QCMLDataset(features_window, prices_window, times_window, meta_window)

    def describe(self) -> Dict:
        """
        Get dataset statistics

        Returns:
            Dictionary with dataset statistics

        Example:
            >>> stats = dataset.describe()
            >>> print(f"Date range: {stats['start_date']} to {stats['end_date']}")
        """
        return {
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'start_date': str(self.times[0]),
            'end_date': str(self.times[-1]),
            'symbols': self.metadata.get('symbols', 'unknown'),
            'mean_return': np.mean(self.returns) if len(self.returns) > 0 else None,
            'volatility': np.std(self.returns) if len(self.returns) > 0 else None,
            'metadata': self.metadata
        }

    def __repr__(self) -> str:
        return (f"QCMLDataset(n_samples={self.n_samples}, "
                f"n_features={self.n_features}, "
                f"date_range={self.times[0]} to {self.times[-1]})")

    def __len__(self) -> int:
        return self.n_samples


def load_crisis_dataset(
    crisis_name: str,
    lookback_months: int = 6,
    lookahead_months: int = 6,
    data_dir: str = "./data",
    use_full_features: bool = True,
    feature_categories: Optional[Set[FeatureCategory]] = None,
    normalize: bool = True,
    benchmark_col: str = "SPY"
) -> QCMLDataset:
    """
    Load pre-configured crisis datasets with full feature engineering.

    Args:
        crisis_name: "2008_crisis", "2020_covid", "2022_rates", "2010_flash_crash"
        lookback_months: Months before crisis event
        lookahead_months: Months after crisis event
        data_dir: Base data directory
        use_full_features: Use full 30+ feature engine (True) or minimal 5 features (False)
        feature_categories: Feature categories to include (default: ALL)
        normalize: Whether to apply rolling z-score normalization
        benchmark_col: Benchmark symbol for cross-sectional features

    Returns:
        QCMLDataset ready for analysis

    Example:
        >>> # Load with full features (default)
        >>> dataset = load_crisis_dataset("2008_crisis", lookback_months=6)
        >>> before, after = dataset.split_by_date("2008-09-15")
        >>>
        >>> # Load with minimal features for quick testing
        >>> dataset = load_crisis_dataset("2008_crisis", use_full_features=False)
        >>>
        >>> # Load specific feature categories
        >>> dataset = load_crisis_dataset(
        ...     "2008_crisis",
        ...     feature_categories={FeatureCategory.VOLATILITY, FeatureCategory.MOMENTUM}
        ... )

    Crisis Dates:
        - 2008_crisis: Lehman Brothers collapse (Sept 15, 2008)
        - 2020_covid: COVID crash (March 16, 2020)
        - 2022_rates: Rate hike regime (March 16, 2022)
        - 2010_flash_crash: Flash crash (May 6, 2010)
    """
    logger.info(f"Loading crisis dataset: {crisis_name}")

    # Define crisis dates
    crisis_dates = {
        "2008_crisis": "2008-09-15",  # Lehman Brothers
        "2020_covid": "2020-03-16",   # COVID crash
        "2022_rates": "2022-03-16",   # Fed rate hike
        "2010_flash_crash": "2010-05-06"  # Flash crash
    }

    # Define crisis-specific universes
    crisis_universes = {
        "2008_crisis": ['SPY', 'XLF', 'BAC', 'JPM', 'C', 'GS', 'MS', 'WFC', 'USB', 'PNC'],
        "2020_covid": ['SPY', 'QQQ', 'XLF', 'XLE', 'XLK', 'XLV', 'XLY', 'IWM'],
        "2022_rates": ['SPY', 'XLF', 'XLRE', 'XLU', 'TLT', 'IEF', 'SHY'],
        "2010_flash_crash": ['SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLK']
    }

    if crisis_name not in crisis_dates:
        raise ValueError(f"Unknown crisis: {crisis_name}. Available: {list(crisis_dates.keys())}")

    crisis_date = pd.Timestamp(crisis_dates[crisis_name])
    universe = crisis_universes.get(crisis_name, ['SPY'])

    # Calculate date range
    start_date = crisis_date - pd.DateOffset(months=lookback_months)
    end_date = crisis_date + pd.DateOffset(months=lookahead_months)

    logger.info(f"Crisis period: {start_date.date()} to {end_date.date()}, universe: {universe}")

    # Try to load data from ParquetDataStore
    data_path = Path(data_dir)
    store = None
    ohlcv_df = None
    prices_df = None

    try:
        store = ParquetDataStore(base_path=str(data_path))
        ohlcv_df = store.load_daily_bars(
            symbols=universe,
            start_date=str(start_date.date()),
            end_date=str(end_date.date())
        )

        if not ohlcv_df.empty:
            # Pivot to get prices DataFrame (symbols as columns)
            prices_df = ohlcv_df['close'].unstack(level=0)
            logger.info(f"Loaded {len(prices_df)} days of data for {len(prices_df.columns)} symbols")
        else:
            logger.warning("No data found in ParquetDataStore")

    except Exception as e:
        logger.warning(f"Could not load from ParquetDataStore: {e}")
        logger.info("Attempting to generate synthetic data for testing...")

    # If no data loaded, create synthetic data for testing
    if prices_df is None or prices_df.empty:
        logger.warning("Generating synthetic crisis data for testing purposes")
        prices_df, ohlcv_df = _generate_synthetic_crisis_data(
            crisis_date, start_date, end_date, universe
        )

    # Ensure benchmark is in the data
    if benchmark_col not in prices_df.columns:
        available = list(prices_df.columns)
        logger.warning(f"Benchmark {benchmark_col} not found. Available: {available}")
        benchmark_col = available[0] if available else "SPY"

    # Compute features
    if use_full_features:
        # Full feature engineering (30+ features)
        feature_config = FeatureConfig(
            categories=feature_categories or {FeatureCategory.ALL},
            benchmark_col=benchmark_col
        )
        engine = FeatureEngine(config=feature_config)

        # Use OHLCV data if available for full features
        features = engine.create_feature_matrix(
            prices_df,
            ohlcv_df=ohlcv_df,
            benchmark_col=benchmark_col
        )
    else:
        # Minimal features (5 features per symbol)
        engine = MinimalFeatureEngine(window=20)
        features = engine.create_feature_matrix(prices_df, benchmark_col=benchmark_col)

    # Apply preprocessing if requested
    if normalize:
        preprocessor = DataPreprocessor(PreprocessingConfig(
            normalization=NormalizationMethod.ROLLING_ZSCORE,
            norm_window=20,  # ~1 month - smaller window to preserve more data
            outlier_method=OutlierMethod.NONE,  # Skip outlier treatment to preserve data
        ))
        # Only normalize, don't run full pipeline which drops all NaN
        features = preprocessor.handle_missing_data(features)
        features = preprocessor.normalize(features, fit=True)

        # Drop only the initial rows where most columns have NaN (warmup period)
        # Find first row where majority of columns are not NaN
        nan_ratio = features.isna().mean(axis=1)
        valid_mask = nan_ratio < 0.5  # Keep rows with less than 50% NaN
        features = features.loc[valid_mask]

        # Fill any remaining sparse NaN with forward fill
        features = features.ffill().bfill()

    # Create price series (benchmark for regime detection)
    benchmark_prices = prices_df[benchmark_col].loc[features.index]

    # Create metadata
    metadata = {
        'crisis': crisis_name,
        'crisis_date': str(crisis_date),
        'start_date': str(start_date),
        'end_date': str(end_date),
        'lookback_months': lookback_months,
        'lookahead_months': lookahead_months,
        'symbols': list(prices_df.columns),
        'benchmark': benchmark_col,
        'n_features': features.shape[1],
        'use_full_features': use_full_features,
        'normalized': normalize,
        'feature_categories': [c.name for c in (feature_categories or {FeatureCategory.ALL})]
    }

    logger.info(
        f"Created crisis dataset: {features.shape[0]} samples, "
        f"{features.shape[1]} features"
    )

    return QCMLDataset(features, benchmark_prices, features.index, metadata)


def _generate_synthetic_crisis_data(
    crisis_date: pd.Timestamp,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    universe: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic crisis data for testing when real data is unavailable.

    Creates realistic-looking price data with a regime change at the crisis date.
    """
    np.random.seed(42)

    # Generate business days
    dates = pd.bdate_range(start=start_date, end=end_date)
    n_days = len(dates)

    # Find crisis index
    crisis_idx = (dates <= crisis_date).sum()

    prices_data = {}
    ohlcv_data = []

    for symbol in universe:
        # Pre-crisis: lower volatility, slight uptrend
        pre_crisis_vol = 0.015
        pre_crisis_drift = 0.0003

        # Post-crisis: higher volatility, downtrend initially
        post_crisis_vol = 0.045
        post_crisis_drift = -0.002

        returns = np.zeros(n_days)
        returns[:crisis_idx] = np.random.normal(
            pre_crisis_drift, pre_crisis_vol, crisis_idx
        )
        returns[crisis_idx:] = np.random.normal(
            post_crisis_drift, post_crisis_vol, n_days - crisis_idx
        )

        # Initial price based on symbol
        initial_price = 100 + hash(symbol) % 50

        prices = initial_price * np.exp(np.cumsum(returns))

        # Generate OHLCV
        for i, (date, price) in enumerate(zip(dates, prices)):
            # Synthetic OHLC with realistic ranges
            daily_vol = pre_crisis_vol if i < crisis_idx else post_crisis_vol
            high = price * (1 + abs(np.random.normal(0, daily_vol)))
            low = price * (1 - abs(np.random.normal(0, daily_vol)))
            open_ = price * (1 + np.random.normal(0, daily_vol * 0.5))

            ohlcv_data.append({
                'symbol': symbol,
                'timestamp': date,
                'open': open_,
                'high': max(high, open_, price),
                'low': min(low, open_, price),
                'close': price,
                'volume': int(np.random.uniform(1e6, 1e8))
            })

        prices_data[symbol] = pd.Series(prices, index=dates)

    # Create DataFrames
    prices_df = pd.DataFrame(prices_data)

    ohlcv_df = pd.DataFrame(ohlcv_data)
    ohlcv_df['timestamp'] = pd.to_datetime(ohlcv_df['timestamp'])
    ohlcv_df = ohlcv_df.set_index(['symbol', 'timestamp']).sort_index()

    return prices_df, ohlcv_df


def create_multi_timeframe_dataset(
    symbols: List[str],
    start_date: str,
    end_date: str,
    timeframes: List[str] = ["1d", "1h"],
    data_dir: str = "./data"
) -> Dict[str, QCMLDataset]:
    """
    Create datasets at multiple resolutions

    Args:
        symbols: List of ticker symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        timeframes: List of timeframes (e.g., ["1d", "1h", "5m"])
        data_dir: Base data directory

    Returns:
        Dictionary mapping timeframe to QCMLDataset

    Example:
        >>> datasets = create_multi_timeframe_dataset(
        ...     ["AAPL", "MSFT"],
        ...     "2024-01-01",
        ...     "2024-01-31",
        ...     timeframes=["1d", "1h"]
        ... )
        >>> daily_dataset = datasets["1d"]
        >>> hourly_dataset = datasets["1h"]
    """
    logger.info(f"Creating multi-timeframe datasets for {len(symbols)} symbols")

    # This is a placeholder - actual implementation would:
    # 1. Load data for each timeframe from storage
    # 2. Compute features at each resolution
    # 3. Create QCMLDataset for each timeframe

    logger.warning("Multi-timeframe dataset creation not yet implemented - requires Phase 2 features")

    # Return empty dict for now
    return {}


def align_multi_timeframe_features(
    datasets: Dict[str, QCMLDataset],
    alignment_method: str = "resample"
) -> QCMLDataset:
    """
    Align and combine features from multiple timeframes

    Args:
        datasets: Dictionary mapping timeframe to QCMLDataset
        alignment_method: "resample", "interpolate", or "forward_fill"

    Returns:
        Combined QCMLDataset with multi-timeframe features

    Example:
        >>> datasets = create_multi_timeframe_dataset(symbols, start, end, ["1d", "1h"])
        >>> combined = align_multi_timeframe_features(datasets, method="resample")
    """
    logger.warning("Multi-timeframe alignment not yet implemented")
    raise NotImplementedError("Phase 2 feature")


def create_synthetic_qcml_dataset(
    n_samples: int = 1000,
    n_features: int = 10,
    regime_change_idx: Optional[int] = None,
    noise_level: float = 0.1,
    seed: int = 42
) -> QCMLDataset:
    """
    Create synthetic dataset for testing QCML pipeline

    Args:
        n_samples: Number of samples
        n_features: Number of features
        regime_change_idx: Index where regime changes (for testing detection)
        noise_level: Gaussian noise standard deviation
        seed: Random seed

    Returns:
        Synthetic QCMLDataset

    Example:
        >>> # Create dataset with regime change at midpoint
        >>> dataset = create_synthetic_qcml_dataset(
        ...     n_samples=1000,
        ...     n_features=10,
        ...     regime_change_idx=500
        ... )
        >>> detector = TopologicalRegimeDetector(geometry, window_size=50)
        >>> transitions = detector.detect_transitions(dataset.X)
        >>> # Should detect transition near index 500
    """
    logger.info(f"Creating synthetic dataset: {n_samples} samples, {n_features} features")

    np.random.seed(seed)

    # Generate features
    if regime_change_idx is not None:
        # Create two different regimes
        features1 = np.random.randn(regime_change_idx, n_features)
        features2 = np.random.randn(n_samples - regime_change_idx, n_features) * 2  # Different variance
        features = np.vstack([features1, features2])
    else:
        features = np.random.randn(n_samples, n_features)

    # Add noise
    features += np.random.randn(n_samples, n_features) * noise_level

    # Generate synthetic prices (geometric Brownian motion)
    returns = np.random.randn(n_samples) * 0.01  # 1% daily volatility
    if regime_change_idx is not None:
        # Higher volatility after regime change
        returns[regime_change_idx:] *= 2

    prices = 100 * np.exp(np.cumsum(returns))

    # Generate times (daily)
    times = pd.date_range("2020-01-01", periods=n_samples, freq="D")

    # Convert to DataFrame/Series
    features_df = pd.DataFrame(
        features,
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    prices_series = pd.Series(prices, name='close')

    metadata = {
        'type': 'synthetic',
        'n_samples': n_samples,
        'n_features': n_features,
        'regime_change_idx': regime_change_idx,
        'noise_level': noise_level,
        'seed': seed
    }

    return QCMLDataset(features_df, prices_series, times, metadata)
