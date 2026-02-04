"""
Unit tests for QCML data pipeline (Phase 1)

Tests cover:
    - Data acquisition module
    - Storage module
    - QCML data interface
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from qcml.data import (
    QCMLDataset,
    ParquetDataStore,
    CacheManager,
    UniverseManager,
    create_synthetic_qcml_dataset
)


class TestQCMLDataset:
    """Tests for QCMLDataset class"""

    def test_dataset_creation(self):
        """Test basic dataset creation"""
        features = pd.DataFrame(np.random.randn(100, 10))
        prices = pd.Series(np.random.randn(100) + 100)
        times = pd.date_range("2024-01-01", periods=100, freq="D")

        dataset = QCMLDataset(features, prices, times, {'test': True})

        assert dataset.n_samples == 100
        assert dataset.n_features == 10
        assert len(dataset) == 100

    def test_to_qcml_format(self):
        """Test format conversion for QCML modules"""
        features = pd.DataFrame(np.random.randn(100, 10))
        prices = pd.Series(np.random.randn(100) + 100)
        times = pd.date_range("2024-01-01", periods=100, freq="D")

        dataset = QCMLDataset(features, prices, times, {})

        X, p, t = dataset.to_qcml_format()

        assert isinstance(X, np.ndarray)
        assert isinstance(p, np.ndarray)
        assert isinstance(t, np.ndarray)
        assert X.shape == (100, 10)
        assert p.shape == (100,)
        assert t.shape == (100,)

    def test_split_by_date(self):
        """Test splitting dataset by date"""
        features = pd.DataFrame(np.random.randn(100, 10))
        prices = pd.Series(np.random.randn(100) + 100)
        times = pd.date_range("2024-01-01", periods=100, freq="D")

        dataset = QCMLDataset(features, prices, times, {})

        # Split at midpoint
        split_date = "2024-02-19"
        before, after = dataset.split_by_date(split_date)

        assert before.n_samples < dataset.n_samples
        assert after.n_samples < dataset.n_samples
        assert before.n_samples + after.n_samples == dataset.n_samples

    def test_get_window(self):
        """Test windowed subset extraction"""
        features = pd.DataFrame(np.random.randn(100, 10))
        prices = pd.Series(np.random.randn(100) + 100)
        times = pd.date_range("2024-01-01", periods=100, freq="D")

        dataset = QCMLDataset(features, prices, times, {})

        # Get last 30 samples
        window = dataset.get_window(-30, None)

        assert window.n_samples == 30
        assert window.n_features == 10

    def test_returns_calculation(self):
        """Test returns property"""
        features = pd.DataFrame(np.random.randn(100, 10))
        prices = pd.Series(np.linspace(100, 110, 100))  # Linearly increasing
        times = pd.date_range("2024-01-01", periods=100, freq="D")

        dataset = QCMLDataset(features, prices, times, {})

        returns = dataset.returns

        assert len(returns) == 99  # n-1 returns
        assert np.all(returns > 0)  # All positive for increasing prices

    def test_describe(self):
        """Test dataset description"""
        features = pd.DataFrame(np.random.randn(100, 10))
        prices = pd.Series(np.random.randn(100) + 100)
        times = pd.date_range("2024-01-01", periods=100, freq="D")

        dataset = QCMLDataset(features, prices, times, {'universe': 'TEST'})

        stats = dataset.describe()

        assert stats['n_samples'] == 100
        assert stats['n_features'] == 10
        assert 'start_date' in stats
        assert 'end_date' in stats
        assert 'mean_return' in stats


class TestParquetDataStore:
    """Tests for ParquetDataStore class"""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_store_initialization(self, temp_data_dir):
        """Test store initialization creates directory structure"""
        store = ParquetDataStore(base_path=temp_data_dir)

        assert Path(temp_data_dir).exists()
        assert (Path(temp_data_dir) / "raw" / "equities" / "daily").exists()
        assert (Path(temp_data_dir) / "features").exists()

    def test_save_and_load_daily_bars(self, temp_data_dir):
        """Test save/load round-trip for daily bars"""
        store = ParquetDataStore(base_path=temp_data_dir)

        # Create sample data
        df = pd.DataFrame({
            'timestamp': pd.date_range("2024-01-01", periods=10, freq="D"),
            'open': np.random.randn(10) + 100,
            'high': np.random.randn(10) + 101,
            'low': np.random.randn(10) + 99,
            'close': np.random.randn(10) + 100,
            'volume': np.random.randint(1e6, 1e7, 10)
        })

        # Save
        store.save_daily_bars(df, symbol="TEST")

        # Load
        loaded = store.load_daily_bars(["TEST"])

        assert not loaded.empty
        assert len(loaded) == 10
        assert all(col in loaded.columns for col in ['open', 'high', 'low', 'close', 'volume'])

    def test_save_and_load_features(self, temp_data_dir):
        """Test save/load round-trip for feature sets"""
        store = ParquetDataStore(base_path=temp_data_dir)

        # Create sample features
        features = pd.DataFrame(
            np.random.randn(100, 5),
            columns=[f"feature_{i}" for i in range(5)]
        )

        metadata = {
            'feature_count': 5,
            'n_samples': 100
        }

        # Save
        store.save_features(features, "test_features", metadata)

        # Load
        loaded_features, loaded_meta = store.load_features("test_features")

        assert loaded_features.shape == features.shape
        assert loaded_meta['feature_count'] == 5

    def test_list_available_symbols(self, temp_data_dir):
        """Test listing available symbols"""
        store = ParquetDataStore(base_path=temp_data_dir)

        # Save some data
        df = pd.DataFrame({
            'timestamp': pd.date_range("2024-01-01", periods=10),
            'close': np.random.randn(10) + 100
        })

        store.save_daily_bars(df, "AAPL")
        store.save_daily_bars(df, "MSFT")

        symbols = store.list_available_symbols()

        assert "AAPL" in symbols
        assert "MSFT" in symbols
        assert len(symbols) == 2


class TestCacheManager:
    """Tests for CacheManager class"""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary directory for cache testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_cache_universe(self, temp_cache_dir):
        """Test universe caching"""
        cache = CacheManager(cache_dir=temp_cache_dir)

        symbols = ["AAPL", "MSFT", "GOOGL"]
        cache.cache_universe(symbols)

        # Retrieve from cache
        cached_symbols = cache.get_universe()

        assert cached_symbols == symbols

    def test_cache_correlations(self, temp_cache_dir):
        """Test correlation matrix caching"""
        cache = CacheManager(cache_dir=temp_cache_dir)

        # Create sample correlation matrix
        corr = pd.DataFrame(
            np.random.randn(5, 5),
            index=[f"S{i}" for i in range(5)],
            columns=[f"S{i}" for i in range(5)]
        )

        date = "2024-01-15"
        cache.cache_correlations(corr, date)

        # Retrieve from cache
        cached_corr = cache.get_correlations(date)

        assert cached_corr is not None
        pd.testing.assert_frame_equal(corr, cached_corr)

    def test_clear_cache(self, temp_cache_dir):
        """Test cache clearing"""
        cache = CacheManager(cache_dir=temp_cache_dir)

        # Add some data
        cache.cache_universe(["AAPL", "MSFT"])

        # Clear
        cache.clear_cache()

        # Should return None
        assert cache.get_universe() is None

    def test_cache_size(self, temp_cache_dir):
        """Test cache size reporting"""
        cache = CacheManager(cache_dir=temp_cache_dir)

        cache.cache_universe(["AAPL", "MSFT", "GOOGL"])

        size_info = cache.get_cache_size()

        assert size_info['memory_items'] >= 0
        assert size_info['disk_files'] >= 0


class TestUniverseManager:
    """Tests for UniverseManager class"""

    def test_get_sp500_constituents(self):
        """Test S&P 500 constituents"""
        manager = UniverseManager()

        symbols = manager.get_sp500_constituents()

        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert "AAPL" in symbols  # Should include major stocks

    def test_get_sector_etfs(self):
        """Test sector ETF mapping"""
        manager = UniverseManager()

        sectors = manager.get_sector_etfs()

        assert isinstance(sectors, dict)
        assert 'Technology' in sectors
        assert 'Financials' in sectors
        assert 'XLK' in sectors['Technology']
        assert 'XLF' in sectors['Financials']

    def test_get_liquid_universe(self):
        """Test liquid universe filtering"""
        manager = UniverseManager()

        liquid = manager.get_liquid_universe(min_volume=1e6, min_price=5.0)

        assert isinstance(liquid, list)
        assert len(liquid) > 0

    def test_get_crisis_universe(self):
        """Test crisis-specific universes"""
        manager = UniverseManager()

        crisis_2008 = manager.get_crisis_universe("2008_financial")
        crisis_2020 = manager.get_crisis_universe("2020_covid")

        assert isinstance(crisis_2008, list)
        assert isinstance(crisis_2020, list)
        assert "XLF" in crisis_2008  # Financials for 2008
        assert "SPY" in crisis_2020  # Broad market for COVID


class TestSyntheticData:
    """Tests for synthetic data generation"""

    def test_create_synthetic_dataset(self):
        """Test synthetic dataset creation"""
        dataset = create_synthetic_qcml_dataset(
            n_samples=1000,
            n_features=10,
            regime_change_idx=500,
            seed=42
        )

        assert dataset.n_samples == 1000
        assert dataset.n_features == 10
        assert dataset.metadata['regime_change_idx'] == 500

    def test_synthetic_regime_change(self):
        """Test regime change is reflected in data"""
        dataset = create_synthetic_qcml_dataset(
            n_samples=1000,
            n_features=10,
            regime_change_idx=500,
            seed=42
        )

        # Split at regime change
        before, after = dataset.split_by_date(dataset.times[500].strftime("%Y-%m-%d"))

        # Different statistics before/after
        before_std = np.std(before.returns)
        after_std = np.std(after.returns)

        # After should have higher volatility (by design)
        assert after_std > before_std


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
