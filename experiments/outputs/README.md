# Experiment outputs

Most files under this directory are **gitignored** so the repo does not fill with reruns.

## Canonical runs (committed)

These three JSON files are **not** ignored and back `make verify` / `memory/results_registry.yaml`:

| File | Role |
|------|------|
| `regime_detection/causal_comparison_20260311_010639.json` | Main 46×17 causal comparison (`Makefile` `CANONICAL_JSON`) |
| `fusion/fusion_results_20260304_101523.json` | In-sample fusion table |
| `fusion/fusion_results_20260304_101842.json` | Holdout fusion (4 crises) |

After a full pipeline refresh that changes headline numbers, regenerate these, rerun `make verify`, update the registry/Makefile paths if filenames change, and commit.

## Everything else

Dated `causal_comparison_*.json`, walk-forward JSON, etc. are local caches. Safe to delete once a newer canonical run is promoted, or park copies under `archive/experiment_runs/` if you want an audit trail outside git.
