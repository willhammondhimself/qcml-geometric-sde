.PHONY: install install-dev test test-unit test-integration lint format \
       experiments paper clean help pipeline-diagram \
       rebuild rebuild-force paper-full paper-anon review verify verify-citations \
       pre-submit snapshot diff registry-summary validate clear-cache \
       pipeline pipeline-quick pipeline-full canonical dashboard \
       video-preview video-hq video-combine video \
       review-status review-extract review-verify review-loop

PYTHON ?= python
PIP ?= pip
PAPER_DIR = paper
SCRIPTS_DIR = scripts
CANONICAL_JSON ?= experiments/outputs/regime_detection/causal_comparison_20260311_010639.json

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

experiments-comparison:  ## Legacy: regime_comparison.py --quick (4 crises, subset of methods)
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
	$(PYTHON) $(PAPER_DIR)/populate_paper.py --json $(CANONICAL_JSON) --compile

rebuild-force:  ## Full rebuild ignoring cache
	$(PYTHON) experiments/runner.py --force
	$(PYTHON) $(PAPER_DIR)/populate_paper.py --json $(CANONICAL_JSON) --compile

clear-cache:  ## Clear experiment cell cache
	$(PYTHON) experiments/runner.py --clear-cache

validate:  ## Run post-experiment validation checks
	$(PYTHON) experiments/validate.py

registry-summary:  ## Show experiment registry summary
	$(PYTHON) -m experiments.registry list

# ── Figures ──────────────────────────────────────────────────────

pipeline-diagram:  ## Compile TikZ pipeline diagram to PDF + PNG
	cd $(PAPER_DIR)/figures && pdflatex -interaction=nonstopmode pipeline_diagram.tex \
		&& pdftoppm -png -r 300 -singlefile pipeline_diagram.pdf pipeline_diagram \
		&& rm -f pipeline_diagram.aux pipeline_diagram.log

# ── Paper Pipeline ───────────────────────────────────────────────

paper:  ## Compile LaTeX paper
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode qcml_geometric_sde.tex
	cd $(PAPER_DIR) && bibtex qcml_geometric_sde
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode qcml_geometric_sde.tex
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode qcml_geometric_sde.tex

paper-full:  ## Generate tables from canonical JSON + compile paper
	$(PYTHON) $(PAPER_DIR)/populate_paper.py --json $(CANONICAL_JSON) --compile --copy-figures

paper-anon:  ## Compile anonymized paper for double-blind submission
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode qcml_geometric_sde_anon.tex
	cd $(PAPER_DIR) && bibtex qcml_geometric_sde_anon
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode qcml_geometric_sde_anon.tex
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode qcml_geometric_sde_anon.tex

# ── Review Pipeline ──────────────────────────────────────────────

review:  ## Deploy multi-agent paper review (ARGS=--quick for 2 reviewers)
	bash $(PAPER_DIR)/review/run_review.sh $(ARGS)

review-status:  ## Show review issue tracking status
	$(PYTHON) $(SCRIPTS_DIR)/review_fix_loop.py status

review-extract:  ## Extract issues from latest synthesis into registry
	$(PYTHON) $(SCRIPTS_DIR)/extract_review_issues.py

review-verify:  ## Verify fixes and promote issue status
	$(PYTHON) $(SCRIPTS_DIR)/review_fix_loop.py verify

review-loop:  ## Full review cycle: review → extract → status
	bash $(PAPER_DIR)/review/run_review.sh $(ARGS)
	$(PYTHON) $(SCRIPTS_DIR)/extract_review_issues.py
	$(PYTHON) $(SCRIPTS_DIR)/review_fix_loop.py status

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

# ── Pipeline ─────────────────────────────────────────────────────

pipeline:  ## Full experiment pipeline (run → register → validate → compile → verify)
	$(PYTHON) experiments/pipeline.py --mode default

pipeline-quick:  ## Quick pipeline (subset of methods/crises)
	$(PYTHON) experiments/pipeline.py --mode quick

pipeline-full:  ## Full pipeline with maximum bootstrap
	$(PYTHON) experiments/pipeline.py --mode full

canonical:  ## List canonical JSON references
	$(PYTHON) -m experiments.registry canonical list

dashboard:  ## Show research status dashboard
	@echo "=== Latest Experiments ===" && $(PYTHON) -m experiments.registry list
	@echo "" && echo "=== Canonical JSONs ===" && $(PYTHON) -m experiments.registry canonical list

# ── Pre-Submission ───────────────────────────────────────────────

pre-submit:  ## Full pre-submission gate check (8 gates)
	$(PYTHON) $(SCRIPTS_DIR)/pre_submit_gate.py

# ── Cleanup ───────────────────────────────────────────────────────

clean:  ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-paper:  ## Remove LaTeX auxiliary files
	cd $(PAPER_DIR) && rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk diff.tex diff.pdf

# ── Video ────────────────────────────────────────────────────────

video-preview:  ## Render all Manim scenes (low quality preview)
	manim -ql --disable_caching paper/manim_qcml_explainer.py

video-hq:  ## Render all Manim scenes (high quality)
	manim -qh paper/manim_qcml_explainer.py

video-combine:  ## Combine rendered scenes into single video
	cd media/videos/manim_qcml_explainer/480p15 && \
	printf "file '%s'\n" TitleScene.mp4 PipelineScene.mp4 HamiltonianScene.mp4 \
		GroundStateEvolutionScene.mp4 ObservatoryScene.mp4 DeadSignalScene.mp4 \
		> concat.txt && \
	ffmpeg -y -f concat -safe 0 -i concat.txt -c copy ../QCMLExplainer_combined.mp4 && \
	rm concat.txt

video:  video-preview video-combine  ## Preview render + combine
