# Geometric observables for financial regime detection

QCML, an unsupervised ML framework, uses Hilbert space geometry to uncover intrinsic structure in data, rather than relying on labels. It learns manifold curvature and topological invariants from Hamiltonian ground state eigenfunctions. This is a three-paper series; Paper 1 focuses on four geometric observables (Berry Phase Rate, Spectral Entropy, Reduced State Purity, Hamiltonian Sensitivity) and their walk-forward evaluation.

## Key results (Paper 1)

- 46 detectors compared on 17 historical crises (2000-2024).
- Walk-forward Berry Phase Rate: Cohen's d ≈ 0.72 under nested HPO with 30% fewer false alarms than Random Forest (2.5 vs 3.6/yr).
- Offline: Reduced Purity d = 0.83 (rank 1/46), Absorption Ratio d = 0.80 (classical benchmark, rank 2), Berry Phase Rate d = 0.61 (rank 9).
- Friedman test on the big panel: χ² = 233.1, p < 10⁻¹⁶ (methods are not exchangeable).
- Geometry channels sit far from classical baselines in correlation space: mean |ρ| ≈ 0.13.
- Lead-time example: Berry Phase Rate ~90 days ahead of the RF benchmark on the median crisis (retrospective methodology; walk-forward median is 4 days).

### Upcoming papers

- **Paper 2**: Full 19-channel geometric observatory with orthogonality analysis and per-crisis specialization.
- **Paper 3**: Adaptive fusion strategies (Regime-Adaptive fusion d ≈ 0.78 on holdout crises).

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
paper/                      LaTeX paper (~25 pages, 3 theorems, 1 proposition, ~44 refs)
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

**Paper 1**: ~25 pages, 3 theorems, 1 proposition, ~44 references.
Source: `paper/qcml_geometric_sde.tex`
Full 19-channel version archived as `paper/qcml_geometric_sde_full.tex`.

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
