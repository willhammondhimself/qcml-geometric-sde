# Geometric Observables for Financial Regime Detection

A spectral metric learning framework that detects financial crises using
differential geometry — without labeled data.

## Key Results

- **Multi-Lag Fidelity** matches supervised Random Forest on Friedman rank (4.42)
  despite requiring no crisis labels
- Walk-forward validation: **86% detection rate**, 9-day median delay
- Tested across **12 historical crises** (2007-2024) and **5 ETFs**
- Computes faster than RF (0.26-0.77s vs 1.07s per window)
- MLF wins 5/12 crises RF misses (SVB d=0.94 vs RF d=0.12)

## Interactive Demo

Try the interactive demo: `streamlit run demo/app.py`

Pre-cache data first: `python demo/cache_data.py`

## How It Works

Financial time series are embedded into a projective Hilbert space via spectral
metric learning. Three geometric observables measure manifold deformation during
market stress:

1. **Berry Phase Rate** — curvature of the data path on the manifold
2. **QFI Determinant** — metric volume element (distinguishability between states)
3. **Multi-Lag Fidelity** — state overlap decay across multiple time horizons

These spike during regime transitions without requiring any labeled crisis data.

## Project Structure

```
qcml_geometry/          Core library (pure math, no I/O)
  core.py               QCMLGeometry: metric tensor, Berry curvature, Chern numbers
  observables.py        Berry/QFI/MLF regime detectors
  indicators.py         Spectral gap, energy, fidelity indicators
  topology.py           Topological regime detectors

experiments/            Reproducible experiment scripts
  regime_comparison.py  Main 11-method x 12-crisis pipeline
  walk_forward_evaluation.py
  baselines.py          RF, VolZ, CUSUM, HMM, BOCPD, Isolation Forest
  data_loader.py        Polygon API + feature engineering
  honest_hpo_sweep.py   Optuna HPO with consistency penalty

demo/                   Interactive Streamlit app
  app.py                Main demo (dark navy theme, Plotly charts)
  cache_data.py         One-time data caching

poster/                 APS Global Physics Summit 2026 poster
paper/                  LaTeX paper (25 pages, 3 theorems)
tests/                  pytest suite
```

## Quick Start

```bash
pip install -r requirements.txt
echo "POLYGON_API_KEY=your_key" > .env

# Run the full comparison pipeline
python experiments/regime_comparison.py --causal

# Run tests
pytest tests/ -x -q
```

## Paper

25-page paper with 3 theorems, 1 proposition, and 33 references.
Available in `paper/qcml_geometric_sde.tex`.

## Author

Will Hammond | Pitzer College | whammond@pitzer.edu
