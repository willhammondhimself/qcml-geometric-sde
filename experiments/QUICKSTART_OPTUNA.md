# Quick Start: Optuna Causal Regime Optimization

**Goal**: Find optimal hyperparameters for QCML regime detectors under causal constraint (no lookahead bias).

## TL;DR - Run Everything

```bash
# Make script executable (one time)
chmod +x experiments/run_full_causal_optuna.sh

# Run full pipeline (Phase A + optional Phase B + evaluation)
./experiments/run_full_causal_optuna.sh
```

**Total time**: ~2-3 hours (Phase A) + ~30 min (Phase B, optional) + ~60-90 min (evaluation)

**Output**: Results in `experiments/outputs/regime_detection/optuna_causal/` and `.../causal_optimized/`

---

## Step-by-Step (Manual Control)

### Step 1: Quick Test (Verify Setup)
```bash
# 2 trials, 1 method, 4 crises (~1 minute)
python experiments/optuna_causal_regime.py \
  --phase A \
  --n-trials 2 \
  --crises quick \
  --methods Berry
```

**Expected output**:
```
Best median_d: 1.6142
Best params: {'hilbert_dim': 16, 'n_pca_components': 15, ...}
Results exported to: .../berry_phase_rate_phase_a_results.json
```

### Step 2: Full Phase A Optimization
```bash
# 200 trials × 3 methods × 12 crises (~45-60 minutes)
python experiments/optuna_causal_regime.py \
  --phase A \
  --n-trials 200 \
  --crises all \
  --methods all \
  --storage sqlite:///experiments/outputs/regime_detection/optuna_causal/causal_optuna.db
```

**What it does**:
- Optimizes `hilbert_dim`, `n_pca_components`, `operator_method`, `rolling_window` under causal constraint
- Uses TPE sampler (smarter than grid search)
- Saves results to JSON + SQLite (can resume if interrupted)

**Check progress**:
```bash
# Watch log output
tail -f /tmp/optuna_test.log

# Check saved results
ls experiments/outputs/regime_detection/optuna_causal/*_phase_a_results.json
```

### Step 3: Phase B (Optional - Expanding Window)
```bash
# 150 trials × 3 methods (~30 minutes)
python experiments/optuna_causal_regime.py \
  --phase B \
  --n-trials 150 \
  --crises all \
  --methods all \
  --storage sqlite:///experiments/outputs/regime_detection/optuna_causal/causal_optuna.db
```

**What it does**:
- Loads best Phase A configs
- Optimizes `expanding_refit_interval` on top
- Tests if adaptive refitting improves performance

### Step 4: Full Evaluation
```bash
# 6 conditions × 3 methods × 12 crises (~60-90 minutes)
python experiments/causal_optimized_evaluation.py \
  --optuna-dir experiments/outputs/regime_detection/optuna_causal
```

**With Phase B results**:
```bash
python experiments/causal_optimized_evaluation.py \
  --optuna-dir experiments/outputs/regime_detection/optuna_causal \
  --optuna-dir-phase-b experiments/outputs/regime_detection/optuna_causal
```

**What it does**:
- Evaluates 6 conditions (original, Phase A, Phase B, expanding 20/30, RF)
- Computes statistical tests (Wilcoxon, Friedman)
- Generates summary report

**Check results**:
```bash
# View summary
cat experiments/outputs/regime_detection/causal_optimized/causal_eval_optuna_*_summary.txt

# View full results (JSON)
cat experiments/outputs/regime_detection/causal_optimized/causal_eval_optuna_*.json | jq .statistical_tests
```

---

## Quick Reference: Command Options

### Optuna Optimization (`optuna_causal_regime.py`)
```bash
--phase {A,B,all}          # Optimization phase (default: A)
--n-trials N               # Number of Optuna trials (default: 200)
--crises {quick,all}       # quick: 4 crises, all: 12 crises
--methods {Berry,QFI,MultiLag,all}  # Methods to optimize
--storage sqlite:///path   # SQLite storage for resume (optional)
```

### Evaluation (`causal_optimized_evaluation.py`)
```bash
--optuna-dir PATH          # Directory with Phase A results (required)
--optuna-dir-phase-b PATH  # Directory with Phase B results (optional)
--causal-config PATH       # Grid search results (alternative to Optuna)
--quick                    # Reduced bootstrap/permutation (faster, less rigorous)
```

---

## Expected Results

### Phase A: Best Configs
```json
{
  "Berry Phase Rate": {
    "hilbert_dim": 16,
    "n_pca_components": 15,
    "operator_method": "pca_inspired",
    "rolling_window": 20,
    "median_d": 1.61
  },
  "QFI Determinant": {
    "hilbert_dim": 12,
    "n_pca_components": 8,
    "operator_method": "pca_inspired",
    "rolling_window": 20,
    "median_d": 1.52
  },
  "Multi-Lag Fidelity": {
    "hilbert_dim": 12,
    "n_pca_components": 15,
    "operator_method": "pca_inspired",
    "rolling_window": 20,
    "median_d": 1.48
  }
}
```

**Key findings** (hypothesized):
- `pca_inspired` operators outperform `random` under causal constraint
- Higher `hilbert_dim` (12-16) compensates for less training data
- Optimal configs **differ** from non-causal defaults

### Evaluation: Statistical Superiority
```
Wilcoxon vs RF (Berry Phase Rate, Phase A):
  wins: 8, losses: 3, ties: 1
  p-value: 0.023 (significant at α=0.05)
  median_d: 1.31 vs 1.13 (RF)

Friedman ranking (all methods):
  chi-sq: 28.4, p-value: 0.001
  Best: Berry Phase Rate (Phase A)
```

---

## Troubleshooting

### "No module named 'optuna'"
```bash
pip install optuna
```

### "Phase A results not found"
Make sure Phase A completed:
```bash
ls experiments/outputs/regime_detection/optuna_causal/*_phase_a_results.json
```

### SQLite database locked
Another process is using the study:
```bash
# Check running processes
ps aux | grep optuna_causal_regime

# Kill if needed
kill <PID>
```

### Results show all NaN
Check logs for data loading errors:
```bash
grep -i "error\|warning" /tmp/optuna_test.log
```

---

## Next Steps After Completion

1. **Compare to grid search**:
   ```bash
   # Grid search best
   cat experiments/outputs/regime_detection/causal_sensitivity/causal_stage2_*.json | \
     jq '.["Berry Phase Rate"] | to_entries | max_by(.value.mean_d)'

   # Optuna best
   cat experiments/outputs/regime_detection/optuna_causal/berry_phase_rate_phase_a_results.json | \
     jq '{best_value, best_params}'
   ```

2. **Update paper** (Section 7.18):
   - Replace grid search configs with Optuna-optimized
   - Add runtime comparison (Optuna 2-3h vs grid 4-5h)
   - Report any novel config discoveries

3. **Re-run improved defaults comparison**:
   ```bash
   python experiments/improved_defaults_comparison.py \
     --config experiments/outputs/regime_detection/optuna_causal/berry_phase_rate_phase_a_results.json
   ```

---

## Files Created

| File | Purpose |
|------|---------|
| `optuna_causal_regime.py` | Main optimization script |
| `run_full_causal_optuna.sh` | Bash orchestrator |
| `OPTUNA_CAUSAL_README.md` | Comprehensive documentation |
| `QUICKSTART_OPTUNA.md` | This file (quick reference) |
| `causal_optimized_evaluation.py` (updated) | Evaluation with Optuna support |

---

## Questions?

See full documentation: `experiments/OPTUNA_CAUSAL_README.md`
