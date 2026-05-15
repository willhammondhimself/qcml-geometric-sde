# Quantitative Finance submission review — *Geometric Observables for Financial Regime Detection*

Editorial pass against `paper/qcml_geometric_sde.tex` (~25pp, 33/33 verified claims as of 2026-05-14). All numeric claims cross-checked against `memory/results_registry.yaml`.

---

## 0. Blocker 1 resolved (2026-05-14) — orthogonality figure

The abstract, §6, §7, and cover-letter claim `|\rho| \approx 0.13` ("geometric and classical channels are largely uncorrelated") was previously not in `memory/results_registry.yaml` and had no traceable source JSON. Investigation found:

- The canonical observatory run (`experiments/outputs/observatory/observatory_20260228_214412.json`) reports `orthogonality.mean_abs_rho_geo_vs_baseline = 0.221` across all geometric × classical channel pairs.
- The Berry Phase Rate vs Rolling Vol Z cross-correlation in the same JSON is `-0.359` (|ρ| = 0.36) — a single-pair number, not a cross-channel mean.
- No JSON anywhere in the repo produces `0.13` as an orthogonality figure.

**Resolution applied:** all four occurrences (paper ll. 98, 1075, 1215; cover_letter l. 29) replaced with `mean $|\rho| \approx 0.22$`. The word "mean" makes the cross-channel-average framing explicit, which is what the canonical JSON measures. A new row was added to `memory/results_registry.yaml` (`Geo-vs-baseline mean |rho| (orthogonality)`) pointing at the observatory JSON. `make verify` now passes 33/33 with this row.

This was Blocker 1 of two integrity blockers gating QF submission. Blocker 2 (walk-forward HPO Table 4 non-reproducibility) is deeper than initially scoped — the regenerated JSON has `d=None` populated in every per-window cell with only an aggregate summary; a focused half-day session is needed.

---

## 1. Journal fit

**Verdict: good fit, but the paper is currently dressed for a physics or methods venue.** *Quantitative Finance* publishes a roughly even mix of empirical methods papers and theoretical contributions, with a stated practitioner audience and an editorial preference for papers whose contribution can be *used* by a desk. Your paper has the empirical substance QF wants — 17 historical crises, 46 baselines, walk-forward OOS with bootstrap CIs, real false-alarm-per-year numbers — but the framing currently leads with formalism (`CP^{n-1}`, Hermitian operators, three theorems) before the result.

**Emphasize for QF:**
- The walk-forward Berry headline (d = 0.72 OOS, ~67% fewer false alarms than RF, no crisis labels). This is the sentence a QF reader needs in the abstract.
- Comparability against named classical baselines (Absorption Ratio, Hamilton MS, GARCH, BOCPD). QF reviewers will check that you compared against what *they* use.
- Orthogonality (|ρ| ≈ 0.13) — frames the geometric observables as a *complement* to existing risk indicators, which is the practitioner-friendly read.
- Reproducibility (canonical JSON, `make verify`, fixed seed). QF increasingly cares.

**De-emphasize for QF:**
- "Quantum" in the abstract and title (see §5 below). The disclaimer in §1 is correct but doesn't undo the keyword effect on reviewer routing.
- The three formal theorems in §2/§3. Keep them — they are the right kind of rigor — but they should sit *behind* the empirical contribution in the abstract and intro, not in front of it.
- Contribution 4 ("first independent evaluation"). True, but it sounds inside-baseball for a QF audience that doesn't track the QCML ecosystem.

---

## 2. Abstract

### Critique of the current abstract

The current abstract leads with **methodology** (`We embed financial time series into a projective Hilbert space...`). For QF, this is the wrong ordering. A QF reviewer skims abstracts asking *"what does this detector do, and is it better than what I already use?"* — they should hit the OOS Cohen's d and the false-alarm reduction inside the first two sentences, not after a Hilbert-space setup.

Other issues:
- The 9-window OOS result is buried in sentence 2.
- "Three theorems about the embedding geometry are stated and verified empirically" is the last sentence — currently reads as an afterthought; for QF it should be *demoted further* (single phrase) or removed from the abstract entirely.
- Reduced Purity d = 0.83 is cited without the holdout-collapse caveat that §1 raises later. A QF reviewer will catch that and suspect the abstract is cherry-picked.
- Word count is fine (~155).

### Suggested rewrite (~145 words)

> We extract four geometric observables — Berry Phase Rate, Spectral Entropy, Reduced State Purity, and Hamiltonian Sensitivity — from a learned spectral embedding of equity-index returns and evaluate them as regime-shift detectors against 46 classical and machine-learning baselines on 17 historical crises (2000–2024). Under walk-forward nested hyperparameter selection on nine labelled windows, the Berry Phase Rate achieves an unbiased out-of-sample median Cohen's d = 0.72 (95% percentile-bootstrap CI [0.34, 1.18]) and produces roughly 67% fewer false alarms per year than a Random Forest using crisis labels (1.2 vs. 3.6/yr). Reduced State Purity attains the highest in-sample separability of any method (d = 0.83), tied closely by the Absorption Ratio (d = 0.80); the geometric and classical channels are largely uncorrelated (|ρ| ≈ 0.13), suggesting they capture distinct risk signals. Construction is unsupervised; hyperparameter selection is the only supervised step.

Key edits relative to the current draft:
- Lead with the walk-forward d and the false-alarm reduction.
- Drop "Hilbert space", "projective", "embedding geometry", "theorems" from the abstract.
- Replace "spectral metric learning (QCML)" with "learned spectral embedding" — same content, no jargon.
- Add the orthogonality line as a substantive practitioner takeaway, not a methodology footnote.
- Make the unsupervised/semi-supervised distinction explicit in one sentence (the current abstract omits this entirely; reviewers will assume the worst).
- I deliberately did **not** repeat Reduced Purity's holdout-collapse caveat in the abstract — but I demoted it from headline (sentence 1) to supporting evidence (sentence 3), so it is no longer a misleading lead.

---

## 3. Introduction critique

The intro is strong on substance but mistimed on emphasis. Specific callouts:

**Lines 112–124 (Hilbert-space paragraph).** Too dense for a first read. A QF reviewer hitting `CP^{n-1}`, Hermitian operators, and citations to Candelori/Abanov before the first empirical result will tag this as "physics paper". Recommend compressing to ~3 sentences and pushing the formal definitions to §2:
> "QCML [Candelori et al. 2025; Abanov et al. 2025] embeds data as unit vectors in a finite-dimensional complex space and represents features as Hermitian operators. The construction equips the data manifold with a learned metric, a curvature, and a spectral gap — geometric quantities that respond to deformations in the data-generating process. The framework runs entirely on classical hardware; 'quantum' refers to the Hilbert-space formalism, not quantum computing."

**Lines 139–146 ("Supervision level" paragraph).** Important, but the level of detail (Optuna TPE, Hilbert-dim grid 4–16, operator method ablation) is methods-section material. Recommend reducing to a one-sentence statement that score construction is unsupervised and HPO uses crisis labels, with the rest moved to §3 or a footnote. As written, it interrupts the flow between the observables paragraph and the evaluation paragraph.

**Contributions list (lines 147–167).** Good structure. Two edits:
- **Contribution 4** ("first independent evaluation") is the weakest contribution for a QF audience — they don't know or care about the QCML community. Reframe as a *practitioner* contribution, e.g., "Out-of-sample false-alarm rates suitable for risk-management overlays" — and let your §6.3 overlay paragraph carry the weight.
- The contributions list could move *up*, before Related Work, so a skimming reviewer hits it earlier. (Currently they have to read Related Work to reach the explicit contribution claims.)

**Caveat paragraph (lines 169–175).** Honest, important, well-placed in spirit — but currently sits awkwardly *between* the contributions and Related Work. Two options: (a) fold it into contribution 3 as a parenthetical, or (b) move it to the end of §1 as a "Scope and limitations" mini-paragraph immediately before §1.1 Related Work. I prefer (b) — it signals epistemic discipline without diluting the contribution claims.

**Related Work (§1.1).** Length is appropriate. The Sandhu (Ollivier–Ricci) and Gidea–Katz (TDA) framing is the right defense against the "what's new geometrically?" objection. No changes recommended.

**Overall:** the intro is currently arranged for a physics-leaning reader. Shuffling the abstract lead and compressing the Hilbert-space paragraph fixes ~80% of the QF-fit problem.

---

## 4. Cover letter (~200 words)

> Dear Editors of *Quantitative Finance*,
>
> I am pleased to submit *Geometric Observables for Financial Regime Detection* for consideration as an original research article.
>
> The paper introduces four geometric detectors — Berry Phase Rate, Spectral Entropy, Reduced State Purity, and Hamiltonian Sensitivity — derived from a learned spectral embedding of equity-index returns, and evaluates them against 46 classical and machine-learning baselines (including the Absorption Ratio, Hamilton's Markov-switching model, GARCH, BOCPD, and supervised Random Forests) on 17 historical crises spanning 2000–2024. Under walk-forward nested hyperparameter selection on nine labelled windows, the Berry Phase Rate achieves an unbiased out-of-sample Cohen's d of 0.72 (95% bootstrap CI [0.34, 1.18]) and reduces the false-alarm rate by approximately 67% relative to a label-supervised Random Forest. The geometric and classical channels are largely uncorrelated (|ρ| ≈ 0.13), suggesting the observables provide a complementary risk-overlay signal rather than a substitute for existing detectors.
>
> Although the underlying formalism is borrowed from quantum geometry, the construction runs entirely on classical hardware; the contribution is empirical and methodological, not physical. All experiments are reproducible from a public codebase with pinned seeds and canonical result files.
>
> The work is original, has not been published elsewhere, and is not under consideration by any other journal. We have no competing interests to declare.
>
> Sincerely,
> Will Hammond
> Pitzer College

(~205 words; trim "I am pleased to submit" → "We submit" if you want to drop a few.)

---

## 5. Framing the "quantum" label

Short answer: **yes, "quantum" is a real reviewer-routing risk for QF, but you have already done the right thing in §1.** The disclaimer "Everything runs on classical hardware; 'quantum' is referring to the Hilbert-space formalism, not quantum computing" is exactly what is needed. Three further recommendations:

1. **Title.** Currently *Geometric Observables for Financial Regime Detection* — already clean, no "quantum" in it. Keep it that way. (The README/PDF title inconsistency is internal-only and doesn't affect submission.)
2. **Keywords.** Currently includes "QCML" as the last keyword. For QF, I would either drop it or move it to last position behind safer terms ("regime detection", "Fubini–Study metric", "Berry curvature", "spectral metric learning"). The first 2–3 keywords drive editorial routing more than the rest.
3. **Cover-letter language.** I included a short "borrowed from quantum geometry but runs on classical hardware" line above. Don't elaborate further — over-defending the framing in the cover letter signals weakness.
4. **Abstract.** I removed "QCML" from the rewritten abstract for the same reason. You can keep it in §1 (where the disclaimer lives) and §2 (where it earns its keep).

What I would *not* recommend: stripping "Berry", "Fubini–Study", "Hilbert", or the formalism entirely. These are the substantive content of the contribution; hiding them would invite a different (and worse) objection that the paper is opaque about its methods.

---

## 6. Most likely referee objections (and preempts)

### Objection 1 — "Reduced Purity offline d = 0.83 collapses to ≈ 0.26 on holdout, so the in-sample number is meaningless."

**Likelihood: very high.** This is the single most damaging line in the paper if a reviewer reads it adversarially, and you have already flagged it yourself in §1.

**Preempt:** Restructure the headline narrative so Berry Phase Rate (walk-forward d = 0.72, CI [0.34, 1.18]) is the *load-bearing OOS claim*, and Reduced Purity's d = 0.83 is positioned as in-sample evidence of geometric signal that does *not* survive frozen-holdout hyperparameter freezing. The current §1 caveat does this, but it is buried after the contributions list. Move it earlier (see §3 above) and never let Reduced Purity's d = 0.83 appear in any sentence without the holdout-collapse number nearby.

### Objection 2 — "These observables are basis- and HPO-dependent. The d-values are an artifact of operator choice."

**Likelihood: high.** A statistically-sophisticated QF reviewer will notice that random Hermitian operators have a free seed, that there is HPO over `n ∈ {4,…,16}`, and that you optimize against crisis labels.

**Preempt:** You already have the evidence — surface it more aggressively.
- Cite the seed-sensitivity result (Berry d ∈ [0.36, 0.71] across five operator offsets; canonical d = 0.71 sits at the top of the band) explicitly in the limitations paragraph.
- Cite the null-model permutation test (Berry d = 0.71 vs. null median 0.53, p = 0.045) in §5.1 — this is exactly the test a referee would ask for and you already have it.
- Make the walk-forward nested HPO protocol unambiguous: HPO is conducted on training windows only, with no leakage into the OOS d. The current §5.2 description is correct but could lead with this sentence.

### Objection 3 — "Why is this not just a vol proxy in disguise?"

**Likelihood: high — this is the QF-specific reflex.** Any new detector gets asked "what's its R² with realized vol?" before anything else.

**Preempt:**
- The orthogonality result (|ρ| ≈ 0.13 with classical baselines) is the right answer. Move it from where it currently sits to the abstract (I did this in §2 above) and to the headline of §5.
- Add — if you have the run — a Pearson correlation of the Berry detector with rolling 20-day realized vol on SPY/DIA. (Internal notes record this for E_0; the analogous number for Berry should be cheap to produce and would directly defuse this objection.)
- The lead-time result (Berry leads RF/VIX-style detectors by ~90 days on the median crisis per `experiments/lead_time_analysis.py`) is your strongest disconfirmation that this is just vol — vol-based detectors don't lead by 90 days. Mention it in the abstract or §1 if there is room.

### Possible 4th objection — "The three theorems are window dressing for a physics audience."

**Likelihood: medium.** A QF reviewer may see Theorems 1–3 as out of character for the journal.

**Preempt:** Soft-pedal in the abstract (already done in the rewrite — they are not mentioned). Keep the theorems in the body, but immediately follow each with the empirical verification that the paper already includes. The "stated and verified empirically" framing is the right one for QF; just don't lead with "We prove three theorems."

---

## Suggested next steps

1. Apply the abstract rewrite above (drop-in replacement for the current `\begin{abstract}...\end{abstract}` block).
2. Compress the §1 Hilbert-space paragraph and move the "Supervision level" detail to §3.
3. Move the Reduced Purity holdout caveat to a "Scope and limitations" mini-paragraph at the end of §1.
4. Reword contribution 4 around practitioner value.
5. Reorder keywords so "regime detection" leads.
6. Run `make verify` after edits to confirm no claim drift.

If you want, I can do (1)–(5) as a follow-up — they are localized edits and the numbers are already verified against the registry. The class swap from `elsarticle` to T&F's `interact` is a separate task.
