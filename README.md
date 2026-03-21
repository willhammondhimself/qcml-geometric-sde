# Geometric observables for financial regime detection

We embed prices and covariates into a projective Hilbert space, then read off geometric quantities (Berry-type rates, spectra, curvature proxies) as regime indicators. No hand-labeled crisis mask is required for scoring.

## Key results

- 46 detectors compared on 17 historical crises (2000-2024).
- Walk-forward Berry Phase Rate: Cohen's d about 0.72 under nested HPO (no lookahead in the protocol we ship).
- Friedman test on the big panel: χ² = 233.1, p < 10⁻¹⁶ (methods are not exchangeable).
- Geometry channels sit far from classical baselines in correlation space: mean |ρ| ≈ 0.13 vs the classical stack in the orthogonality run.
- Regime-Adaptive fusion holds up on holdout crises (d ≈ 0.78 in the paper highlights) while single-channel stars such as Reduced Purity fall apart there (roughly 0.83 in-sample median d vs about 0.26 on the frozen holdout fusion table).
- Lead-time example in the paper: Berry Phase Rate about 90 days ahead of the RF benchmark on the median crisis, RF about 6 days (details in the lead-time experiment output).

## How it works

Spectral metric learning builds the embedding. The **46-method** leaderboard adds classical baselines to **19** geometric detectors evaluated on the crisis panel; **17** of those geometric streams participate in fusion. Three are marked dead in code (near-zero *d* on all 17 crises) and excluded from `ACTIVE_CHANNELS` in [`qcml_geometry/fusion.py`](qcml_geometry/fusion.py): **QGT Phase Rigidity**, **Berry Velocity Coupling**, **Curvature Rate**.

Fusion taxonomy (display names match `OBSERVABLE_FAMILIES`):

| Family | Observables |
|--------|-------------|
| Holonomy | Berry Phase Rate, Geometric Phase Rate |
| Metric | QFI Determinant, Hamiltonian Sensitivity |
| State Dynamics | Multi-Lag Fidelity, Reduced Purity, Quantum Relative Entropy |
| Kinematics | Geodesic Velocity, Speed Limit Ratio |
| Spectral | Spectral Entropy, Spectral Complexity, Effective State Dim, Level Spacing Ratio |
| Curvature | Sectional Curvature Sign, Geodesic Curvature |
| Topology | QCML Chern, Dimensionality Collapse |

Implementations and HPO keys live in [`qcml_geometry/observables.py`](qcml_geometry/observables.py); paper tables are authoritative for headline *d* values.

## Project Structure

```
qcml_geometry/              Core library (pure math, no I/O)
  core.py                   QCMLGeometry: metric tensor, Berry curvature, Chern numbers
  observables.py            Geometric regime detectors (19 in main panel; 17 fused)
  indicators.py             Spectral gap, energy, fidelity indicators
  topology.py               Topological regime detectors
  fusion.py                 Composite signal fusion
  info_geometry.py          Information-geometric utilities
  adaptive_threshold.py     Online adaptive thresholding
  online_detection.py       Streaming regime detection

experiments/                Reproducible experiment scripts
  regime_comparison.py      Main 46-method x 17-crisis pipeline
  fusion_experiments.py     Multi-channel fusion experiments
  runner.py                 Incremental cell-based experiment runner
  config.yaml               Experiment configuration
  baselines.py              RF, VolZ, CUSUM, HMM, BOCPD, IF, GARCH, Hamilton MS, EWMA, ...
  data_loader.py            yfinance + feature engineering (17 crises)
  holdout_evaluation.py     Holdout crisis evaluation
  lead_time_analysis.py     Lead time measurement
  observatory_analysis.py   Orthogonality matrix + oracle fusion
  backtest/                 Walk-forward backtest suite

demo/                       Interactive Streamlit app
paper/                      LaTeX paper (~50 pages, 3 theorems, 1 proposition, ~44 refs)
tests/                      pytest suite (10 test_*.py files)
scripts/                    Verification utilities
archive/                    Dead experiments, old code
```

## Quick Start

```bash
pip install -r requirements.txt

# Run the full 46-method comparison (quick mode, ~10 min)
python experiments/regime_comparison.py --causal

# Run tests
pytest tests/ -x -q

# Interactive demo
python demo/cache_data.py
streamlit run demo/app.py
```

## Reproducibility

Paper claims tied to numbers are checked with `make verify` (see `memory/results_registry.yaml`). **Three canonical JSON runs are tracked in git** under `experiments/outputs/` (see `experiments/outputs/README.md`); everything else there is ignored. Clone → install → `make verify` should pass without rerunning the full pipeline.

Video assets under `media/` are generated (e.g. `make video`); that directory is gitignored.

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

~50 pages, 3 theorems, 1 proposition, ~44 references.
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

Will Hammond, Pitzer College, whammond@pitzer.edu
