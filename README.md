# Geometric observables for financial regime detection

QCML, an unsupervised ML framework, uses Hilbert space geometry to uncover intrinsic structure in data, rather than relying on labels. It learns manifold curvature and topological invariants from Hamiltonian ground state eigenfunctions. This is a three-paper series; Paper 1 focuses on four geometric observables (Berry Phase Rate, Spectral Entropy, Reduced State Purity, Hamiltonian Sensitivity) and their walk-forward evaluation.

## Key results (Paper 1)

- 46 detectors compared on 17 historical crises (2000-2024).
- Walk-forward Berry Phase Rate: Cohen's d ≈ 0.72 under nested HPO with ~67% fewer false alarms than Random Forest (1.2 vs 3.6/yr).
- Offline: Reduced Purity d = 0.83 (rank 1/46), Absorption Ratio d = 0.80 (classical benchmark, rank 2), Berry Phase Rate d = 0.61 (rank 9).
- Friedman test on the big panel: χ²₄₅ = 220.84, p < 10⁻¹⁶ (methods are not exchangeable).
- Geometry channels sit far from classical baselines in correlation space: mean |ρ| ≈ 0.22.
- Lead-time: walk-forward Berry Phase Rate detects 5/7 crisis windows at a 24-day median delay and 1.2 false alarms/yr; the large full-sample retrospective lead (≈186-day median) reflects contemporaneous sensitivity, not predictive lead time.

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

## Project Structure (Paper 1 Scope)

```
qcml_geometry/              Core library (pure math, no I/O)
  core.py                   QCMLGeometry: Hamiltonian, ground state, Berry curvature, metric tensor
  observables.py            Geometric regime detectors (4 headline + 15 extended)
  fusion.py                 Channel taxonomy + fusion methods
  indicators.py             Spectral gap, energy, fidelity indicators
  topology.py               Topological regime detectors
  info_geometry.py          Information-geometric utilities

experiments/                Paper 1 experiment scripts
  regime_comparison.py      Main 46-method x 17-crisis pipeline
  data_loader.py            yfinance + feature engineering (17 crises)
  baselines.py              RF, GARCH, HMM, CUSUM, BOCPD, IF, EWMA, ...
  evaluation.py             Cohen's d, Friedman test, bootstrap CI
  generate_paper_figures.py Narrative panels (2008 GFC, 2020 COVID, 2022 rates)
  lead_time_analysis.py     Lead time measurement
  backtest/                 Walk-forward backtest suite
  config.yaml               Experiment configuration

paper/                      LaTeX paper (~25 pages, 3 theorems, 1 proposition, ~44 refs)
tests/                      pytest suite
scripts/                    Verification utilities (make verify, make pre-submit)
demo/                       Interactive Streamlit app

paper2_staging/             Deferred work (Papers 2/3: observatory, fusion)
archive/                    Dead experiments, old code
qcml_course.html            Interactive course (open in browser)
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
Full 19-channel version archived in `paper2_staging/paper/qcml_geometric_sde_full.tex`.

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
