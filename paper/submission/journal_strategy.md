# Multi-journal submission strategy — QF, JFE-Oxford, Physica A

## The hard constraint

You cannot submit to two journals simultaneously. Every journal makes you sign a statement that the work isn't under review elsewhere. The strategy below is **sequential, not parallel** — one base manuscript with three thin overlay packs, retargeted within ~3 days of any rejection.

## What's common vs. what's journal-specific

The base manuscript is ~85% the same across all three. Differences:

| Component | Common? | Notes |
|---|---|---|
| LaTeX class wrapper | NO | `elsarticle` (QF / Physica A) vs. T&F `interact` (JFE-Oxford allows either, but `interact` is the official template) |
| Abstract | NO | Lead emphasis differs sharply per journal (see below) |
| Intro front-half (lines ~100–175) | PARTIAL | Framing of "why this matters" is journal-specific |
| Intro back-half (related work, contributions) | YES | Minor citation reordering only |
| §2 Framework, §3 Observables, §4 Data/Methods, §5 Results, §6 Discussion | YES | Identical |
| Theorem block placement | PARTIAL | In-body for Physica A; statements-in-body / proofs-in-appendix for QF and JFE |
| All figures, tables, bibliography | YES | Identical |
| Keywords | NO | Different per journal — affects editorial routing |
| Cover letter | NO | Different per journal |

So roughly **22 of 25 pages are shared**. The overlay per journal is ~3 pages of front matter + a cover letter.

## Per-journal positioning matrix

### Quantitative Finance (T&F)

**Lead with**: empirical OOS result (Berry walk-forward d = 0.72 + ~67% fewer false alarms + orthogonality |ρ|≈0.13).
**De-emphasize**: theorems in abstract, "QCML" terminology.
**Keywords (in order)**: regime detection, financial crises, walk-forward validation, Berry curvature, Fubini–Study metric, spectral metric learning.
**Cover-letter emphasis**: practitioner audience, classical baselines named, false-alarm reduction as risk-overlay.
**Theorem placement**: statements in §2 body; proofs in appendix.
**Title**: keep current — *Geometric Observables for Financial Regime Detection*.

### Journal of Financial Econometrics (Oxford)

**Lead with**: statistical apparatus (Friedman χ² = 220.84 with Iman–Davenport, Nemenyi post-hoc, percentile bootstrap CI on Berry, null-model permutation p = 0.045) AND the OOS d. Frame the contribution as "a *label-free*, *non-parametric* alternative to Hamilton-style Markov-switching detectors with formal walk-forward guarantees."
**De-emphasize**: physics terminology in front matter — "Hilbert space" and "Berry curvature" survive but are introduced as *one possible parameterization* of a learned spectral embedding, not the main pitch.
**Keywords (in order)**: regime detection, walk-forward validation, multiple-comparison correction, nonlinear time-series, Markov-switching alternatives, spectral methods.
**Cover-letter emphasis**: rigor of the comparison protocol, multiple-testing discipline (Holm–Bonferroni + BH-FDR + Friedman + Nemenyi all in `experiments/evaluation.py`), reproducibility (canonical JSON, `make verify`).
**Theorem placement**: statements compressed into §2; full proofs in appendix.
**Title suggestion (lighter on geometry)**: *A Label-Free Spectral Detector for Financial Regime Shifts: Walk-Forward Evaluation Against 46 Baselines*.

### Physica A (Statistical Mechanics and its Applications)

**Lead with**: the geometric/physics framing — Berry curvature, Fubini–Study pullback metric, Hamiltonian sensitivity, spectral gap as order parameter — *and* the empirical headline.
**De-emphasize**: nothing. Physica A welcomes the formalism.
**Keywords (in order)**: econophysics, Berry curvature, quantum geometry, financial crises, regime detection, complex systems.
**Cover-letter emphasis**: the construction adapts ideas from quantum geometry (Provost–Vallée 1980, Berry 1984) to a financial data manifold; cite Bouchaud / Mantegna–Stanley / Sornette lineage in §1.1; the empirical evaluation is what makes this a Physica A paper, not a methods-only paper.
**Theorem placement**: in body, with proofs visible (or compact appendix).
**Title suggestion (more physics-flavored)**: *Geometric Observables from a Learned Hilbert-Space Embedding for Financial Regime Detection*.

## Concrete file structure

```
paper/
├── qcml_geometric_sde.tex          # canonical base (current)
├── submission/
│   ├── qf_review.md                 # editorial pass (existing)
│   ├── journal_strategy.md          # this file
│   ├── qf/
│   │   ├── wrapper.tex              # \documentclass[3p]{elsarticle} + frontmatter
│   │   ├── abstract.tex             # QF-tuned abstract (~145 words)
│   │   ├── intro_frame.tex          # opening 1.5 pages
│   │   ├── keywords.tex
│   │   └── cover_letter.md
│   ├── jfe/
│   │   ├── wrapper.tex              # \documentclass{interact} (T&F interact)
│   │   ├── abstract.tex             # JFE-tuned abstract
│   │   ├── intro_frame.tex
│   │   ├── keywords.tex
│   │   └── cover_letter.md
│   └── physica_a/
│       ├── wrapper.tex              # \documentclass[5p]{elsarticle} + Physica A bibstyle
│       ├── abstract.tex
│       ├── intro_frame.tex
│       ├── keywords.tex
│       └── cover_letter.md
```

The shared body content (§2 onward) is `\input{}`'d from `paper/qcml_geometric_sde.tex` (or a refactored `paper/qcml_geometric_sde_body.tex` extracted from §2 onward). Three top-level wrapper files compile to three submission PDFs.

Cost estimate: **one weekend** to refactor the base into shared `_body.tex` + three wrappers; then ~2 hours per journal to write each abstract / intro frame / cover letter overlay.

## Recommended submission sequence

The sequence question is the harder one. Three plausible orderings:

**Option A — QF first, Physica A as backup (skip JFE-Oxford).** Best EV if you want a publication you can defend professionally with low time risk.
- QF → ~3–4mo to first decision; if R&R, ~6–9mo total; acceptance rate ~25–30%
- If reject → Physica A → ~6–8wk to first decision; near-guaranteed acceptance
- **Total worst case: ~6 months. Total expected case: ~4–5 months.**

**Option B — JFE-Oxford first, QF middle, Physica A backup.** Highest-prestige shot first.
- JFE-Oxford → ~3–6mo; reject probability is genuinely high (~70%) for this paper given the formalism friction
- If reject → QF → 3–4mo
- If reject → Physica A → 6–8wk
- **Total worst case: ~12 months. Expected case: ~7–9 months.**

**Option C — QF first, JFE-Oxford as upgrade only on QF reject, Physica A backup.** Hybrid.
- QF → 3–4mo
- If reject → JFE-Oxford → 3–6mo (cycle is fresh; the QF reject doesn't prejudice JFE)
- If reject → Physica A → 6–8wk
- **Total worst case: ~12 months. Expected case: ~5–6 months.**

**My recommendation: Option A (QF → Physica A).** Reasons:
1. QF is the venue where this paper, *as currently constructed*, has the best fit-to-acceptance ratio. The reframing in `qf_review.md` is enough to make it a serious shot.
2. JFE-Oxford is a 30%-acceptance shot with a ~6-month cycle. The expected time cost is high, and the JFE referee pool will not love the geometric formalism — you'd be revising for econometricians who fundamentally don't care about Berry curvature.
3. Physica A as backup means you have a near-certain publication outcome if QF rejects, in 6–8 weeks. That's a better safety net than wandering through three top venues.
4. If you really want the JFE-Oxford prestige bid, the right time is *Paper 2* (the deferred fusion / observatory paper), where the statistical apparatus will be even more mature and the geometric formalism will already have a published precedent.

## Practical pre-submission checklist

Before any first submission:
1. Do the editorial reframe from `qf_review.md` (abstract, §1 compression, caveat placement).
2. Run `make pre-submit` (8-gate check) and resolve all blockers.
3. Reconcile Friedman χ² in prose: registry says 220.84; check that no body text still says 233.1.
4. Resolve the open `first_swing_report.md` items: 18 vs 17 crises at line 1089; "5 of 9 windows" phrasing in abstract; 4 overfull hboxes.
5. Refactor `paper/qcml_geometric_sde.tex` → `qcml_geometric_sde_body.tex` + three wrappers (or just maintain three full top-level files; pick whichever is less error-prone for you).
6. Pick an order, write the cover letter for the primary, submit.
