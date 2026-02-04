# QCML Data Pipeline Documentation

Professional-grade data pipeline for feeding real financial market data into the QCML-Geometric SDE framework.

## Overview

The QCML data pipeline bridges the gap between raw market data and the QCML framework's topological regime detection capabilities. It provides:

- **Data Acquisition**: Fetching from Polygon.io and Alpaca APIs
- **Efficient Storage**: Parquet-based time series storage with partitioning
- **Feature Engineering**: Technical indicators and cross-sectional features (Phase 2)
- **QCML Integration**: Seamless integration with existing QCML modules

## Architecture

```
Raw Market Data (APIs)
    ↓
Data Acquisition (acquisition.py)
    ↓
Parquet Storage (storage.py)
    ↓
Feature Engineering (features.py - Phase 2)
    ↓
QCML Dataset (qcml_data.py)
    ↓
QCML Framework (Geometry, SDE, Regime Detection)
```

## Phase 1: Core Infrastructure (✓ Complete)

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Quick Start

```python
from qcml.data import (
    PolygonDataSource,
    ParquetDataStore,
    QCMLDataset
)

# 1. Fetch data
source = PolygonDataSource()  # Reads POLYGON_API_KEY from env
df = source.fetch_equities(
    symbols=["AAPL", "MSFT"],
    start_date="2024-01-01",
    end_date="2024-01-31",
    timeframe="1d"
)

# 2. Store data
store = ParquetDataStore(base_path="./data")
for symbol in df.index.get_level_values(0).unique():
    symbol_data = df.loc[symbol]
    store.save_daily_bars(symbol_data, symbol=symbol)

# 3. Create QCML dataset (with features from Phase 2)
# For now, use synthetic data:
from qcml.data import create_synthetic_qcml_dataset

dataset = create_synthetic_qcml_dataset(
    n_samples=1000,
    n_features=10,
    regime_change_idx=500
)

# 4. Use with QCML framework
from qcml import QCMLGeometry, TopologicalRegimeDetector

X, prices, times = dataset.to_qcml_format()

geometry = QCMLGeometry(n_features=X.shape[1])
geometry.fit_operators(X)

detector = TopologicalRegimeDetector(geometry, window_size=50)
transitions = detector.detect_transitions(X, times=times)
```

### Modules

#### `acquisition.py` - Data Fetching

**PolygonDataSource**
- Primary API client for Polygon.io
- Fetches OHLCV data with automatic rate limiting
- Exponential backoff retry logic

```python
source = PolygonDataSource()

# Fetch daily data
df = source.fetch_equities(
    symbols=["AAPL", "MSFT", "GOOGL"],
    start_date="2023-01-01",
    end_date="2023-12-31",
    timeframe="1d"  # or "1h", "5m"
)
```

**AlpacaDataSource**
- Fallback API client for Alpaca
- Same interface as Polygon for easy swapping

```python
source = AlpacaDataSource()

df = source.fetch_bars(
    symbols=["AAPL"],
    start="2023-01-01",
    end="2023-12-31",
    timeframe="1Day"
)
```

**UniverseManager**
- Manage trading universes and symbol lists
- Pre-defined universes (S&P 500, sectors, crisis-specific)

```python
manager = UniverseManager()

sp500 = manager.get_sp500_constituents()
sectors = manager.get_sector_etfs()
crisis_2008 = manager.get_crisis_universe("2008_financial")
```

#### `storage.py` - Parquet Storage

**ParquetDataStore**
- Efficient Parquet-based storage with compression
- Automatic directory structure management
- Metadata tracking for feature sets

```python
store = ParquetDataStore(base_path="./data")

# Save daily bars
store.save_daily_bars(df, symbol="AAPL")

# Load data
loaded = store.load_daily_bars(
    symbols=["AAPL", "MSFT"],
    start_date="2023-01-01",
    end_date="2023-12-31"
)

# Save features (Phase 2)
store.save_features(features_df, "technical_v1", metadata)
features, meta = store.load_features("technical_v1")
```

**CacheManager**
- In-memory and disk caching for frequently accessed data
- Cache correlation matrices, universes, etc.

```python
cache = CacheManager(cache_dir="./data/cache")

# Cache universe
cache.cache_universe(["AAPL", "MSFT"])
universe = cache.get_universe()

# Cache correlations
cache.cache_correlations(corr_matrix, date="2023-12-31")
corr = cache.get_correlations("2023-12-31")
```

#### `qcml_data.py` - QCML Integration

**QCMLDataset**
- Dataset wrapper for QCML framework integration
- Provides numpy arrays in expected format
- Supports splitting, windowing, and analysis

```python
dataset = QCMLDataset(
    features=features_df,
    prices=prices_series,
    times=pd.date_range("2023-01-01", periods=len(features_df)),
    metadata={'universe': 'SP500'}
)

# Convert to QCML format
X, prices, times = dataset.to_qcml_format()

# Split by date (for regime detection)
before, after = dataset.split_by_date("2023-06-15")

# Get windowed subset
recent = dataset.get_window(-30, None)  # Last 30 samples

# Statistics
stats = dataset.describe()
```

**Helper Functions**

```python
# Load pre-configured crisis datasets (Phase 2+)
dataset = load_crisis_dataset(
    crisis_name="2008_crisis",
    lookback_months=6,
    lookahead_months=6
)

# Create multi-timeframe datasets (Phase 2+)
datasets = create_multi_timeframe_dataset(
    symbols=["AAPL", "MSFT"],
    start_date="2023-01-01",
    end_date="2023-12-31",
    timeframes=["1d", "1h"]
)

# Create synthetic data for testing
dataset = create_synthetic_qcml_dataset(
    n_samples=1000,
    n_features=10,
    regime_change_idx=500,  # Regime change at index 500
    seed=42
)
```

## Data Storage Structure

```
data/
├── raw/
│   ├── equities/
│   │   ├── daily/
│   │   │   ├── AAPL.parquet
│   │   │   ├── MSFT.parquet
│   │   │   └── ...
│   │   └── minute/
│   │       └── AAPL.parquet
│   └── options/
│       └── ...
├── features/
│   ├── technical_v1/
│   │   ├── data.parquet
│   │   └── metadata.json
│   └── cross_sectional_v1/
│       ├── data.parquet
│       └── metadata.json
└── processed/
    └── qcml_ready/
        ├── crisis_2008/
        ├── crisis_2020/
        └── full_history/
```

## Environment Variables

Create `.env` file from `.env.example`:

```bash
# Required (Phase 1)
POLYGON_API_KEY=your_polygon_api_key

# Optional (fallback)
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key

# Storage
DATA_PATH=./data
PARQUET_COMPRESSION=snappy

# Universe
MIN_VOLUME=1000000
MIN_PRICE=5.0

# QCML
HILBERT_DIM=4
REGIME_WINDOW_SIZE=30
```

## Examples

### Example 1: Basic Data Fetching

```python
from qcml.data import PolygonDataSource, ParquetDataStore

# Fetch S&P 500 data for 2023
source = PolygonDataSource()
store = ParquetDataStore()

symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

for symbol in symbols:
    df = source.fetch_equities(
        symbols=[symbol],
        start_date="2023-01-01",
        end_date="2023-12-31",
        timeframe="1d"
    )

    store.save_daily_bars(df.loc[symbol], symbol=symbol)
    print(f"Saved {symbol}")
```

### Example 2: Synthetic Regime Detection

See `examples/demo_phase1_integration.py` for a complete demo showing:
- Synthetic dataset creation with regime change
- Storage and retrieval
- QCML geometry learning
- Topological regime detection
- Visualization of results

Run with:
```bash
cd qcml-geometric-sde
python examples/demo_phase1_integration.py
```

### Example 3: Crisis Analysis Preparation

```python
from qcml.data import UniverseManager, PolygonDataSource

manager = UniverseManager()
source = PolygonDataSource()

# Get 2008 crisis universe
symbols = manager.get_crisis_universe("2008_financial")

# Fetch data around Lehman Brothers collapse
df = source.fetch_equities(
    symbols=symbols,
    start_date="2008-03-01",  # 6 months before
    end_date="2009-03-01",    # 6 months after
    timeframe="1d"
)

# Phase 2: Add features and create QCMLDataset
# Then analyze Chern number discontinuity at Sept 15, 2008
```

## Testing

Run unit tests:

```bash
pytest tests/test_data_pipeline.py -v
```

Tests cover:
- QCMLDataset creation and operations
- Parquet storage round-trips
- Cache management
- Universe management
- Synthetic data generation

## Phase 2: Feature Engineering (Coming Next)

Phase 2 will add:

1. **Technical Indicators** (`features.py`)
   - MACD, RSI, Bollinger Bands, ATR
   - Volume indicators (OBV, VWAP)
   - Moving averages (SMA, EMA)

2. **Cross-Sectional Features** (`features.py`)
   - Relative strength vs. benchmark
   - Rolling correlations
   - Sector momentum
   - Dispersion measures

3. **Preprocessing** (`preprocessing.py`)
   - Missing data handling
   - Normalization (rolling z-score, cross-sectional)
   - Walk-forward train/test splits

4. **Historical Crisis Datasets**
   - 2008 financial crisis
   - 2020 COVID crash
   - 2022 rate hike regime
   - Ready-to-use datasets for research

## API Reference

See inline docstrings in each module for detailed API documentation.

## Performance Notes

- **Polygon.io rate limits**: Free tier = 5 requests/minute. Add delays or upgrade.
- **Parquet compression**: `snappy` is fast, `gzip` is smaller. Default = `snappy`.
- **Cache**: Store frequently accessed data to avoid redundant API calls.
- **Batch operations**: Fetch multiple symbols in single calls when possible.

## Troubleshooting

**Issue**: `polygon package not installed`
- **Fix**: `pip install polygon-api-client`

**Issue**: `POLYGON_API_KEY not found`
- **Fix**: Create `.env` file with your API key

**Issue**: `No data returned for symbol`
- **Fix**: Check symbol exists, date range is valid, API subscription covers asset

**Issue**: `Rate limit exceeded`
- **Fix**: Add delays between requests or upgrade API plan

## Contributing

When adding new features:
1. Add unit tests to `tests/test_data_pipeline.py`
2. Update this documentation
3. Follow existing code style and patterns
4. Ensure backward compatibility

## License

See main project LICENSE file.

## Contact

For questions or issues:
- Open an issue on GitHub
- See project README for contact information
