# Geometric Observables for Financial Regime Detection

A spectral metric learning framework that detects financial crises using
differential geometry — without labeled data.

## Key Results

- **36-method comparison** across **17 historical crises** (2007–2024)
- **Reduced Purity** (d = 0.834) ranks #1 overall, followed by Hamilton MS (d = 0.713) and CUSUM (d = 0.625)
- Friedman chi² = 219.31, p < 0.0001 — methods are not interchangeable
- QCML observables are **orthogonal** to classical baselines (mean |ρ| = 0.13), enabling complementary fusion
- **Regime-Adaptive fusion** generalizes to holdout crises (d = 0.783) while top individuals collapse (Reduced Purity drops 66%)
- Berry Phase Rate leads regime transitions by **90 days** vs. Random Forest's 6 days

## How It Works

Financial time series are embedded into a projective Hilbert space via spectral
metric learning. **17 geometric observatory channels** measure manifold
deformation during market stress, organized by geometric family:

| Family | Observables |
|--------|------------|
| **Metric** | Fubini-Study velocity, QFI determinant, multi-lag fidelity, geodesic curvature |
| **Curvature** | Berry phase rate, sectional curvature, Ricci scalar rate, curvature rate |
| **Spectral** | Spectral gap, spectral entropy, spectral complexity, effective state dimension |
| **Topological** | Chern number, reduced state purity, QGT phase rigidity |
| **Information** | Hamiltonian sensitivity, speed limit ratio |

These spike during regime transitions without requiring any labeled crisis data.

## Project Structure

```
qcml_geometry/              Core library (pure math, no I/O)
  core.py                   QCMLGeometry: metric tensor, Berry curvature, Chern numbers
  observables.py            17 geometric regime detectors
  indicators.py             Spectral gap, energy, fidelity indicators
  topology.py               Topological regime detectors
  fusion.py                 Composite signal fusion
  info_geometry.py          Information-geometric utilities
  adaptive_threshold.py     Online adaptive thresholding
  online_detection.py       Streaming regime detection

experiments/                Reproducible experiment scripts
  regime_comparison.py      Main 36-method × 17-crisis pipeline
  fusion_experiments.py     Multi-channel fusion experiments
  runner.py                 Incremental cell-based experiment runner
  config.yaml               Experiment configuration
  baselines.py              RF, VolZ, CUSUM, HMM, BOCPD, IF, GARCH, Hamilton MS, EWMA, ...
  data_loader.py            yfinance + feature engineering (17 crises)
  holdout_evaluation.py     Holdout crisis evaluation
  lead_time_analysis.py     Lead time measurement
  observatory_analysis.py   Orthogonality matrix + oracle fusion
  backtest/                 Walk-forward backtest suite (9 files)

demo/                       Interactive Streamlit app
  app.py                    Main demo (dark navy theme, Plotly charts)
  cache_data.py             One-time data caching

paper/                      LaTeX paper (~48 pages, 3 theorems, 1 proposition, 45+ refs)
  qcml_geometric_sde.tex    Main document
  references.bib            Bibliography
  tables/                   Auto-generated LaTeX tables (9 files)

poster/                     APS Global Physics Summit 2026 poster
tests/                      pytest suite (14 test files)
scripts/                    Verification utilities
```

## Quick Start

```bash
pip install -r requirements.txt

# Run the full 36-method comparison (quick mode, ~10 min)
python experiments/regime_comparison.py --causal

# Run tests
pytest tests/ -x -q

# Interactive demo
python demo/cache_data.py
streamlit run demo/app.py
```

## Makefile Targets

```bash
make test              # Run all unit tests
make rebuild           # Incremental experiments + tables + compile paper
make paper             # Compile LaTeX paper
make paper-full        # Regenerate tables from JSON + compile
make review            # Deploy multi-agent paper review
make verify            # Check paper numbers vs source data
make pre-submit        # Full pre-submission gate check
make clean             # Remove build artifacts
```

## Paper

~48-page paper with 3 theorems, 1 proposition, and 45+ references.
Source: `paper/qcml_geometric_sde.tex`

### Citation

```bibtex
@article{hammond2026geometric,
  title   = {Geometric Observables for Financial Regime Detection},
  author  = {Hammond, Will},
  year    = {2026},
  note    = {Pitzer College}
}
```

## Author

Will Hammond — Pitzer College — whammond@pitzer.edu
