"""
Data Storage Module

Efficient Parquet-based storage and retrieval for time series financial data
with partitioning, compression, and caching support.

Classes:
    ParquetDataStore: Parquet-based time series data storage
    CacheManager: In-memory and disk caching for frequently accessed data
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

import pandas as pd
import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    logging.warning("pyarrow not installed. ParquetDataStore will not be available.")

logger = logging.getLogger(__name__)


class ParquetDataStore:
    """
    Parquet-based time series data storage

    Provides efficient storage and retrieval of financial time series data
    with automatic partitioning, compression, and metadata management.

    Args:
        base_path: Root directory for data storage (default: "./data")
        compression: Compression algorithm ("snappy", "gzip", "brotli", "none")

    Example:
        >>> store = ParquetDataStore(base_path="./data")
        >>> store.save_daily_bars(df, symbol="AAPL")
        >>> loaded = store.load_daily_bars(["AAPL"], "2024-01-01", "2024-01-31")
    """

    def __init__(
        self,
        base_path: str = "./data",
        compression: str = "snappy"
    ):
        if not PYARROW_AVAILABLE:
            raise ImportError("pyarrow required. Install with: pip install pyarrow")

        self.base_path = Path(base_path)
        self.compression = compression

        # Create directory structure
        self._create_directory_structure()

        logger.info(f"ParquetDataStore initialized at {self.base_path}")

    def _create_directory_structure(self):
        """Create standard directory structure for data storage"""
        directories = [
            self.base_path / "raw" / "equities" / "daily",
            self.base_path / "raw" / "equities" / "minute",
            self.base_path / "raw" / "options",
            self.base_path / "features",
            self.base_path / "processed" / "qcml_ready",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def save_daily_bars(
        self,
        df: pd.DataFrame,
        symbol: str,
        partition_cols: Optional[List[str]] = None
    ):
        """
        Save daily OHLCV data partitioned by date

        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
                Index should be DatetimeIndex or have 'timestamp' column
            symbol: Ticker symbol
            partition_cols: Optional partition columns (e.g., ["year", "month"])

        Example:
            >>> df = pd.DataFrame({
            ...     'open': [100, 101],
            ...     'close': [102, 103],
            ...     'timestamp': pd.date_range('2024-01-01', periods=2)
            ... })
            >>> store.save_daily_bars(df, "AAPL")
        """
        logger.info(f"Saving daily bars for {symbol} ({len(df)} rows)")

        # Ensure timestamp column
        df = df.copy()
        if 'timestamp' not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df['timestamp'] = df.index
            else:
                raise ValueError("DataFrame must have 'timestamp' column or DatetimeIndex")

        # Add partitioning columns if requested
        if partition_cols:
            if 'year' in partition_cols:
                df['year'] = df['timestamp'].dt.year
            if 'month' in partition_cols:
                df['month'] = df['timestamp'].dt.month
            if 'day' in partition_cols:
                df['day'] = df['timestamp'].dt.day

        # Save path
        save_path = self.base_path / "raw" / "equities" / "daily" / f"{symbol}.parquet"

        # Write parquet
        df.to_parquet(
            save_path,
            engine='pyarrow',
            compression=self.compression,
            index=False
        )

        logger.info(f"Saved to {save_path}")

    def load_daily_bars(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load daily bars for date range

        Args:
            symbols: List of ticker symbols
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (optional)

        Returns:
            DataFrame with MultiIndex (symbol, timestamp) and OHLCV columns

        Example:
            >>> df = store.load_daily_bars(["AAPL", "MSFT"], "2024-01-01", "2024-01-31")
            >>> print(df.loc["AAPL"].head())
        """
        logger.info(f"Loading daily bars for {len(symbols)} symbols")

        all_data = []

        for symbol in symbols:
            file_path = self.base_path / "raw" / "equities" / "daily" / f"{symbol}.parquet"

            if not file_path.exists():
                logger.warning(f"No data file found for {symbol}")
                continue

            try:
                df = pd.read_parquet(file_path, engine='pyarrow')

                # Add symbol column
                df['symbol'] = symbol

                # Filter by date range
                if start_date or end_date:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])

                    if start_date:
                        df = df[df['timestamp'] >= pd.Timestamp(start_date)]
                    if end_date:
                        df = df[df['timestamp'] <= pd.Timestamp(end_date)]

                if not df.empty:
                    all_data.append(df)

            except Exception as e:
                logger.error(f"Failed to load {symbol}: {e}")
                continue

        if not all_data:
            logger.warning("No data loaded")
            return pd.DataFrame()

        # Combine and set MultiIndex
        result = pd.concat(all_data, ignore_index=True)
        result['timestamp'] = pd.to_datetime(result['timestamp'])
        result = result.set_index(['symbol', 'timestamp']).sort_index()

        logger.info(f"Loaded {len(result)} total rows")
        return result

    def save_features(
        self,
        df: pd.DataFrame,
        feature_set_name: str,
        metadata: Dict[str, Any]
    ):
        """
        Save feature matrix with metadata

        Args:
            df: Feature DataFrame
            feature_set_name: Name for this feature set (e.g., "technical_v1")
            metadata: Dictionary with feature set metadata (columns, params, etc.)

        Example:
            >>> metadata = {
            ...     'feature_count': len(df.columns),
            ...     'symbols': df.index.get_level_values(0).unique().tolist(),
            ...     'date_range': [str(df.index.min()), str(df.index.max())]
            ... }
            >>> store.save_features(features_df, "technical_v1", metadata)
        """
        logger.info(f"Saving feature set: {feature_set_name}")

        feature_dir = self.base_path / "features" / feature_set_name
        feature_dir.mkdir(parents=True, exist_ok=True)

        # Save data
        data_path = feature_dir / "data.parquet"
        df.to_parquet(
            data_path,
            engine='pyarrow',
            compression=self.compression
        )

        # Save metadata
        metadata_path = feature_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Saved feature set to {feature_dir}")

    def load_features(
        self,
        feature_set_name: str,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load feature matrix and metadata

        Args:
            feature_set_name: Name of feature set to load
            symbols: Optional list of symbols to filter
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Tuple of (feature_df, metadata_dict)

        Example:
            >>> features, meta = store.load_features("technical_v1", symbols=["AAPL"])
            >>> print(f"Loaded {len(features)} rows with {len(features.columns)} features")
        """
        logger.info(f"Loading feature set: {feature_set_name}")

        feature_dir = self.base_path / "features" / feature_set_name

        if not feature_dir.exists():
            raise FileNotFoundError(f"Feature set not found: {feature_set_name}")

        # Load data
        data_path = feature_dir / "data.parquet"
        df = pd.read_parquet(data_path, engine='pyarrow')

        # Load metadata
        metadata_path = feature_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Apply filters
        if symbols:
            if isinstance(df.index, pd.MultiIndex):
                df = df[df.index.get_level_values(0).isin(symbols)]
            else:
                logger.warning("Cannot filter by symbols - not a MultiIndex")

        if start_date or end_date:
            # Assume second level of MultiIndex is timestamp
            if isinstance(df.index, pd.MultiIndex):
                timestamps = df.index.get_level_values(1)
                mask = pd.Series(True, index=df.index)

                if start_date:
                    mask &= timestamps >= pd.Timestamp(start_date)
                if end_date:
                    mask &= timestamps <= pd.Timestamp(end_date)

                df = df[mask]

        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} features")
        return df, metadata

    def list_available_symbols(self, data_type: str = "daily") -> List[str]:
        """
        List available symbols in storage

        Args:
            data_type: "daily" or "minute"

        Returns:
            List of available ticker symbols
        """
        path = self.base_path / "raw" / "equities" / data_type

        if not path.exists():
            return []

        symbols = [f.stem for f in path.glob("*.parquet")]
        return sorted(symbols)

    def list_feature_sets(self) -> List[str]:
        """
        List available feature sets

        Returns:
            List of feature set names
        """
        features_path = self.base_path / "features"

        if not features_path.exists():
            return []

        feature_sets = [d.name for d in features_path.iterdir() if d.is_dir()]
        return sorted(feature_sets)


class CacheManager:
    """
    Cache frequently accessed data

    Provides in-memory and disk-based caching for expensive operations
    like correlation matrices and universe definitions.

    Args:
        cache_dir: Directory for disk cache (default: "./data/cache")
        max_memory_items: Maximum items to keep in memory cache

    Example:
        >>> cache = CacheManager()
        >>> cache.cache_universe(["AAPL", "MSFT", "GOOGL"])
        >>> universe = cache.get_universe()
    """

    def __init__(
        self,
        cache_dir: str = "./data/cache",
        max_memory_items: int = 100
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_memory_items = max_memory_items
        self._memory_cache: Dict[str, Any] = {}

        logger.info(f"CacheManager initialized at {self.cache_dir}")

    def cache_universe(self, symbols: List[str]):
        """
        Cache current trading universe

        Args:
            symbols: List of ticker symbols
        """
        logger.info(f"Caching universe with {len(symbols)} symbols")

        cache_file = self.cache_dir / "universe.json"

        data = {
            'symbols': symbols,
            'timestamp': datetime.now().isoformat(),
            'count': len(symbols)
        }

        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)

        # Also cache in memory
        self._memory_cache['universe'] = symbols

    def get_universe(self) -> Optional[List[str]]:
        """
        Get cached trading universe

        Returns:
            List of symbols or None if not cached
        """
        # Try memory cache first
        if 'universe' in self._memory_cache:
            return self._memory_cache['universe']

        # Try disk cache
        cache_file = self.cache_dir / "universe.json"

        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
                symbols = data['symbols']

                # Update memory cache
                self._memory_cache['universe'] = symbols

                return symbols

        return None

    def cache_correlations(
        self,
        corr_matrix: pd.DataFrame,
        date: str
    ):
        """
        Cache correlation matrices

        Args:
            corr_matrix: Correlation matrix DataFrame
            date: Date identifier (YYYY-MM-DD)
        """
        logger.info(f"Caching correlation matrix for {date}")

        cache_file = self.cache_dir / f"corr_{date}.parquet"
        corr_matrix.to_parquet(cache_file, compression='snappy')

    def get_correlations(
        self,
        date: str
    ) -> Optional[pd.DataFrame]:
        """
        Get cached correlation matrix

        Args:
            date: Date identifier (YYYY-MM-DD)

        Returns:
            Correlation matrix or None if not cached
        """
        cache_file = self.cache_dir / f"corr_{date}.parquet"

        if cache_file.exists():
            return pd.read_parquet(cache_file)

        return None

    def clear_cache(self):
        """Clear all caches (memory and disk)"""
        logger.info("Clearing all caches")

        # Clear memory
        self._memory_cache.clear()

        # Clear disk
        for file in self.cache_dir.glob("*"):
            if file.is_file():
                file.unlink()

        logger.info("Caches cleared")

    def get_cache_size(self) -> Dict[str, int]:
        """
        Get cache size information

        Returns:
            Dictionary with cache statistics
        """
        disk_files = list(self.cache_dir.glob("*"))
        disk_size = sum(f.stat().st_size for f in disk_files if f.is_file())

        return {
            'memory_items': len(self._memory_cache),
            'disk_files': len(disk_files),
            'disk_size_mb': disk_size / (1024 * 1024)
        }
