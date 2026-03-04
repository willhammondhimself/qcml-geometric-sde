.PHONY: install install-dev test test-unit test-integration lint format \
       experiments paper clean help \
       rebuild rebuild-force paper-full review verify verify-citations \
       pre-submit snapshot diff registry-summary validate clear-cache

PYTHON ?= python
PIP ?= pip
PAPER_DIR = paper
SCRIPTS_DIR = scripts

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

# ── Experiments (Legacy) ─────────────────────────────────────────

experiments: experiments-comparison experiments-theorems  ## Run core experiments (legacy)

experiments-comparison:  ## Run 17-method regime comparison (quick)
	$(PYTHON) experiments/regime_comparison.py --quick

experiments-theorems:  ## Run theorem validation (quick)
	$(PYTHON) experiments/theorem_validation.py --quick

experiments-full:  ## Run all experiments on all 16 crises
	$(PYTHON) experiments/regime_comparison.py --full
	$(PYTHON) experiments/theorem_validation.py
	$(PYTHON) experiments/learned_operator_training.py --full
	$(PYTHON) experiments/online_detection_diagnosis.py

# ── Incremental Pipeline ─────────────────────────────────────────

rebuild:  ## End-to-end rebuild (only changed cells recompute)
	$(PYTHON) experiments/runner.py
	$(PYTHON) $(PAPER_DIR)/populate_paper.py --compile

rebuild-force:  ## Full rebuild ignoring cache
	$(PYTHON) experiments/runner.py --force
	$(PYTHON) $(PAPER_DIR)/populate_paper.py --compile

clear-cache:  ## Clear experiment cell cache
	$(PYTHON) experiments/runner.py --clear-cache

validate:  ## Run post-experiment validation checks
	$(PYTHON) experiments/validate.py

registry-summary:  ## Show experiment registry summary
	$(PYTHON) -m experiments.registry list

# ── Paper Pipeline ───────────────────────────────────────────────

paper:  ## Compile LaTeX paper
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode qcml_geometric_sde.tex
	cd $(PAPER_DIR) && bibtex qcml_geometric_sde
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode qcml_geometric_sde.tex
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode qcml_geometric_sde.tex

paper-full:  ## Generate tables from latest JSON + compile paper
	$(PYTHON) $(PAPER_DIR)/populate_paper.py --compile --copy-figures

# ── Review Pipeline ──────────────────────────────────────────────

review:  ## Deploy multi-agent paper review
	bash $(PAPER_DIR)/review/run_review.sh

# ── Verification ─────────────────────────────────────────────────

verify:  ## Verify paper numbers match source data
	$(PYTHON) $(SCRIPTS_DIR)/verify_paper_numbers.py

verify-citations:  ## Verify bibliography against Semantic Scholar
	$(PYTHON) $(PAPER_DIR)/verify_citations.py

# ── Version Diffing ──────────────────────────────────────────────

snapshot:  ## Tag current paper version for diffing
	@SNAP=$$(date +%Y%m%d_%H%M%S) && \
	git tag "paper-snapshot-$$SNAP" && \
	echo "Tagged: paper-snapshot-$$SNAP"

diff:  ## Generate latexdiff between last two snapshots
	@TAGS=$$(git tag -l 'paper-snapshot-*' --sort=-creatordate | head -2) && \
	OLD=$$(echo "$$TAGS" | tail -1) && \
	NEW=$$(echo "$$TAGS" | head -1) && \
	echo "Diffing $$OLD → $$NEW" && \
	git show "$$OLD:paper/qcml_geometric_sde.tex" > /tmp/paper_old.tex && \
	git show "$$NEW:paper/qcml_geometric_sde.tex" > /tmp/paper_new.tex && \
	latexdiff /tmp/paper_old.tex /tmp/paper_new.tex > $(PAPER_DIR)/diff.tex && \
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode diff.tex && \
	echo "Diff PDF: $(PAPER_DIR)/diff.pdf"

# ── Pre-Submission ───────────────────────────────────────────────

pre-submit:  ## Full pre-submission gate check
	@echo "=== Gate 1: Tests ===" && $(PYTHON) -m pytest tests/ -v --ignore=tests/test_crisis_validation.py -x
	@echo "=== Gate 2: Lint ===" && $(PYTHON) -m ruff check qcml_geometry/ experiments/ tests/
	@echo "=== Gate 3: Paper Numbers ===" && $(PYTHON) $(SCRIPTS_DIR)/verify_paper_numbers.py
	@echo "=== Gate 4: Paper Compile ===" && cd $(PAPER_DIR) && \
		pdflatex -interaction=nonstopmode qcml_geometric_sde.tex && \
		bibtex qcml_geometric_sde && \
		pdflatex -interaction=nonstopmode qcml_geometric_sde.tex && \
		pdflatex -interaction=nonstopmode qcml_geometric_sde.tex
	@echo "=== Gate 5: Citation Verification ===" && $(PYTHON) $(PAPER_DIR)/verify_citations.py --dry-run
	@echo "=== All gates passed ==="

# ── Cleanup ───────────────────────────────────────────────────────

clean:  ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-paper:  ## Remove LaTeX auxiliary files
	cd $(PAPER_DIR) && rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk diff.tex diff.pdf
