# Installation Guide

## Prerequisites

- Python 3.8+
- pip package manager

## Quick Install

```bash
# Navigate to project directory
cd qcml-geometric-sde

# Install all dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env

# Edit .env with your API keys
# POLYGON_API_KEY=your_key_here
```

## Dependencies

### Core (Existing)
- numpy>=1.24.0
- scipy>=1.10.0
- pandas>=2.0.0
- torch>=2.0.0
- matplotlib>=3.7.0

### Phase 1 (New)
- polygon-api-client>=1.12.0
- alpaca-py>=0.9.0
- python-dotenv>=1.0.0
- pyarrow>=10.0.0
- fastparquet>=2023.0.0
- tqdm>=4.65.0
- requests>=2.31.0
- joblib>=1.3.0

### Development
- pytest>=7.4.0
- pytest-cov>=4.1.0
- jupyter>=1.0.0
- ipykernel>=6.25.0

## Optional: TA-Lib (Phase 2)

For faster technical analysis (Phase 2), install TA-Lib C library:

**macOS:**
```bash
brew install ta-lib
pip install ta-lib
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libta-lib0-dev
pip install ta-lib
```

**Windows:**
Download from: http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-msvc.zip

## API Keys

### Polygon.io (Primary)

1. Sign up at https://polygon.io/
2. Get API key from https://polygon.io/dashboard/api-keys
3. Add to `.env`:
   ```bash
   POLYGON_API_KEY=your_polygon_api_key_here
   ```

**Free Tier**: 5 requests/minute, delayed data
**Paid Tier**: Higher limits, real-time data

### Alpaca (Optional Fallback)

1. Sign up at https://alpaca.markets/
2. Get API keys from dashboard
3. Add to `.env`:
   ```bash
   ALPACA_API_KEY=your_alpaca_api_key
   ALPACA_SECRET_KEY=your_alpaca_secret_key
   ```

**Free Tier**: Unlimited historical data (delayed)

## Verification

Test installation:

```python
# Test imports
from qcml import (
    QCMLGeometry,
    TopologicalRegimeDetector
)

from qcml.data import (
    QCMLDataset,
    ParquetDataStore,
    UniverseManager,
    create_synthetic_qcml_dataset
)

print("✓ All imports successful")

# Test synthetic data
dataset = create_synthetic_qcml_dataset(n_samples=100, n_features=5)
print(f"✓ Synthetic dataset: {dataset.n_samples} samples")

# Test with API key (if available)
try:
    from qcml.data import PolygonDataSource
    source = PolygonDataSource()
    print("✓ Polygon API configured")
except Exception as e:
    print(f"⚠ Polygon API not configured: {e}")
```

## Run Demo

```bash
# Phase 1 integration demo
python examples/demo_phase1_integration.py
```

Expected output:
- Creates synthetic dataset
- Stores in Parquet
- Learns QCML geometry
- Detects regime transitions
- Generates visualization

## Run Tests

```bash
# All tests
pytest tests/ -v

# Just data pipeline tests
pytest tests/test_data_pipeline.py -v

# With coverage
pytest tests/ --cov=qcml.data --cov-report=html
```

## Troubleshooting

### Import Error: No module named 'torch'
```bash
pip install torch>=2.0.0
```

### Import Error: No module named 'polygon'
```bash
pip install polygon-api-client
```

### Error: POLYGON_API_KEY not found
- Create `.env` file in project root
- Add `POLYGON_API_KEY=your_key_here`
- Never commit `.env` to git (it's in .gitignore)

### Rate Limit Errors
- Free tier = 5 requests/minute
- Add delays between requests
- Or upgrade API plan

### Permission Errors on ./data
```bash
mkdir -p data
chmod 755 data
```

## Development Setup

For development with auto-reload:

```bash
# Install in editable mode
pip install -e .

# Install dev dependencies
pip install pytest pytest-cov jupyter ipython

# Run Jupyter notebooks
jupyter notebook notebooks/
```

## Next Steps

After installation:

1. Run demo: `python examples/demo_phase1_integration.py`
2. Read docs: `docs/DATA_PIPELINE.md`
3. Try fetching real data (with API key)
4. Explore notebooks: `notebooks/`

## Support

For issues:
- Check `docs/DATA_PIPELINE.md`
- Review `PHASE1_COMPLETE.md`
- Check GitHub issues
