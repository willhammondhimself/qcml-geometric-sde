.PHONY: install install-dev test test-unit test-integration lint format experiments paper clean help

PYTHON ?= python
PIP ?= pip

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Installation ──────────────────────────────────────────────────

install:  ## Install core package
	$(PIP) install -e .

install-dev:  ## Install with all optional dependencies
	$(PIP) install -e ".[all]"

install-data:  ## Install with data acquisition deps
	$(PIP) install -e ".[data]"

# ── Testing ───────────────────────────────────────────────────────

test:  ## Run all unit tests
	$(PYTHON) -m pytest tests/ -v --ignore=tests/test_crisis_validation.py -x

test-unit:  ## Run fast unit tests only (no API calls)
	$(PYTHON) -m pytest tests/ -v -m "not integration and not slow" -x

test-integration:  ## Run integration tests (requires API keys)
	$(PYTHON) -m pytest tests/ -v -m integration

# ── Code Quality ──────────────────────────────────────────────────

lint:  ## Check code style
	$(PYTHON) -m ruff check qcml_geometry/ experiments/ tests/

format:  ## Format code with black
	$(PYTHON) -m black qcml_geometry/ experiments/ tests/

# ── Experiments ───────────────────────────────────────────────────

experiments: experiments-comparison experiments-theorems  ## Run core experiments

experiments-comparison:  ## Run 17-method regime comparison (quick)
	$(PYTHON) experiments/regime_comparison.py --quick

experiments-theorems:  ## Run theorem validation (quick)
	$(PYTHON) experiments/theorem_validation.py --quick

experiments-full:  ## Run all experiments on all 16 crises
	$(PYTHON) experiments/regime_comparison.py --full
	$(PYTHON) experiments/theorem_validation.py
	$(PYTHON) experiments/learned_operator_training.py --full
	$(PYTHON) experiments/online_detection_diagnosis.py

# ── Paper ─────────────────────────────────────────────────────────

paper:  ## Compile LaTeX paper
	cd paper && pdflatex -interaction=nonstopmode qcml_geometric_sde.tex
	cd paper && bibtex qcml_geometric_sde
	cd paper && pdflatex -interaction=nonstopmode qcml_geometric_sde.tex
	cd paper && pdflatex -interaction=nonstopmode qcml_geometric_sde.tex

# ── Cleanup ───────────────────────────────────────────────────────

clean:  ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
