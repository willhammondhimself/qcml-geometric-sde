#!/bin/bash
# Full causal Optuna optimization + evaluation pipeline
#
# This script orchestrates the complete workflow:
#   1. Phase A: Optimize base hyperparameters (causal constraint)
#   2. Phase B: Optimize expanding window (optional)
#   3. Evaluation: 6 conditions × 3 methods × 12 crises
#
# Usage:
#   chmod +x experiments/run_full_causal_optuna.sh
#   ./experiments/run_full_causal_optuna.sh

set -e

# Configuration
OUTPUT_BASE="experiments/outputs/regime_detection"
OPTUNA_DIR="${OUTPUT_BASE}/optuna_causal"
EVAL_DIR="${OUTPUT_BASE}/causal_optimized"

echo "=========================================================================="
echo "CAUSAL OPTUNA OPTIMIZATION + EVALUATION PIPELINE"
echo "=========================================================================="
echo ""
echo "Output directories:"
echo "  - Optuna optimization: ${OPTUNA_DIR}/"
echo "  - Evaluation: ${EVAL_DIR}/"
echo ""

# Create output directories
mkdir -p "${OPTUNA_DIR}"
mkdir -p "${EVAL_DIR}"

# Phase A: Causal Base Hyperparameter Optimization
echo "=========================================================================="
echo "PHASE A: Causal Base Hyperparameter Optimization"
echo "=========================================================================="
echo "Optimizing hilbert_dim, n_pca_components, operator_method, rolling_window"
echo "Search space: 4 × 4 × 2 × 3 = 96 combinations (TPE sampling)"
echo "Trials: 200 per method (Berry, QFI, Multi-Lag)"
echo "Crises: 12 (all available)"
echo "Estimated time: ~45-60 minutes"
echo ""

python experiments/optuna_causal_regime.py \
  --phase A \
  --n-trials 200 \
  --crises all \
  --methods all \
  --storage "sqlite:///${OPTUNA_DIR}/causal_optuna.db"

echo ""
echo "Phase A complete! Results saved to: ${OPTUNA_DIR}/"
echo ""

# Phase B: Expanding Window Optimization (optional)
echo "=========================================================================="
echo "PHASE B: Expanding Window Optimization (Optional)"
echo "=========================================================================="
read -p "Run Phase B (expanding window optimization)? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Optimizing expanding_refit_interval on top of Phase A results..."
    echo "Trials: 150 per method"
    echo "Estimated time: ~30 minutes"
    echo ""

    python experiments/optuna_causal_regime.py \
      --phase B \
      --n-trials 150 \
      --crises all \
      --methods all \
      --storage "sqlite:///${OPTUNA_DIR}/causal_optuna.db"

    PHASE_B_FLAG="--optuna-dir-phase-b ${OPTUNA_DIR}"
    echo ""
    echo "Phase B complete!"
    echo ""
else
    PHASE_B_FLAG=""
    echo "Skipping Phase B."
    echo ""
fi

# Full Evaluation: 6 Conditions × 3 Methods × 12 Crises
echo "=========================================================================="
echo "FULL EVALUATION"
echo "=========================================================================="
echo "Conditions:"
echo "  1. Original (non-causal) improved defaults"
echo "  2. Optuna Phase A optimized (causal constraint)"
if [[ -n "${PHASE_B_FLAG}" ]]; then
    echo "  2b. Optuna Phase B optimized (expanding window)"
fi
echo "  3. Expanding window (interval=20) with Phase A configs"
echo "  4. Expanding window (interval=30) with Phase A configs"
echo "  5. Random Forest baseline (leave-one-crisis-out)"
echo ""
echo "Methods: 3 (Berry Phase Rate, QFI Determinant, Multi-Lag Fidelity)"
echo "Crises: 12"
echo "Statistical tests: Wilcoxon signed-rank, Friedman"
echo "Estimated time: ~60-90 minutes"
echo ""

python experiments/causal_optimized_evaluation.py \
  --optuna-dir "${OPTUNA_DIR}" \
  ${PHASE_B_FLAG}

echo ""
echo "=========================================================================="
echo "PIPELINE COMPLETE!"
echo "=========================================================================="
echo ""
echo "Results saved to:"
echo "  - Optuna optimization: ${OPTUNA_DIR}/"
echo "  - Evaluation: ${EVAL_DIR}/"
echo ""
echo "Next steps:"
echo "  1. Review Optuna results: ${OPTUNA_DIR}/*_results.json"
echo "  2. Review evaluation summary: ${EVAL_DIR}/causal_eval_optuna_*_summary.txt"
echo "  3. Compare vs grid search: experiments/outputs/regime_detection/causal_sensitivity/"
echo ""
