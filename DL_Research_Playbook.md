# The Deep Learning Research Playbook
*A portable, reusable guide to running a DL research project from gap-finding to a published paper — productively and defensibly. Distilled from a real physics-informed-GNN project plus current field best practices. Drop this into a new project's repo (e.g. next to `CLAUDE.md`) and it inherits the methodology.*

> **How to use this.** It's phase-structured. Each phase has a **goal**, **steps**, **why it matters**, a **checklist**, and **pitfalls** (many with a real war-story from the source project). Swap the domain/architecture — the method holds. The single most important meta-lesson: *neural nets fail silently* (Karpathy), so the whole playbook is built around being paranoid, using trivial baselines as truth-anchors, and verifying before scaling.

---

## Phase 0 — Frame the research (find a defensible, publishable gap)

**Goal:** a one-paragraph thesis + 2–4 research questions that are novel, defensible, and feasible under your constraints.

**Steps.**
1. **Reframe "which domain" → "which transferable contribution."** Top venues reward a method insight, not an application. "Technique X applied to domain Y" usually belongs in a domain venue or workshop. Ask: *what would another researcher reuse?*
2. **Pick a contribution type:** architecture, learning strategy (loss/regularization/training), representation, efficiency, or **diagnosis** (characterizing when/why methods fail — cheap and respected).
3. **Apply the "cheap-by-construction" filter if compute-limited.** When you can't out-scale big labs, choose a contribution whose value is an *idea*, not GPUs (a diagnosis + a cheap fix). A small, sharp, reproducible result beats a big, fragile one.
4. **Frame the thesis as a hypothesis to TEST, not a fact to assert.** "We hypothesize X; if instead Y, that is still a finding." This is honest and survives review.
5. **Write 2–4 RQs**: typically *diagnose* (does the effect exist / how does it manifest?), *isolate* (is it your mechanism, not a confound?), *fix* (does the intervention work, and generalize?).

**The gap-finding loop (repeatable):**
broad scan of the area → check if it's a *maturing* field (most obvious ideas already have a flagship paper) → **position against the 2–3 nearest neighbors** (name exactly what each one *misses*) → integrity check (is the thesis backed by real papers, or invented?).

**Pitfalls.**
- Choosing a domain application and calling it novelty. *(War story: our first instinct — "PI-GNN for traffic" — was already done by Lin et al. 2024; the novelty had to move to a diagnosis + fix.)*
- Asserting a thesis you haven't tested. Keep the "claim-to-test" framing.
- Ignoring that the strongest neighbors define your novelty *boundary* — read them first.

---

## Phase 1 — Literature review with integrity

**Goal:** a related-work map that separates *what you cite* from *what you must demonstrate*, with every reference verified.

**Steps.**
1. **Split established building blocks (cite confidently) from your novel synthesis (must demonstrate).** Your contribution is usually a *new combination/interaction* of established parts.
2. **Verify every reference against the actual paper**, not a search snippet. Use confidence tiers; flag anything unverified. *(War story: a load-bearing citation turned out to be **withdrawn on arXiv** — caught only by opening the source. Reproduce nothing you haven't read.)*
3. **Position against the closest neighbors with one differentiation sentence each** ("X does A but not B; we do B").
4. **Watch terminology traps** — same words, different problem (e.g., "physics-informed" as a *cure for* a pathology vs. a model *suffering* it).

**Tools:** arXiv, Semantic Scholar, Papers with Code (for datasets/baselines + code), Connected Papers / Google Scholar (citation graph).

**Pitfalls (Lones 2024):** not reviewing the literature (reinventing/missing context); assuming your idea is new without checking. **Copyright/honesty:** paraphrase; never fabricate a citation.

---

## Phase 2 — Data & problem setup (before you touch a model)

**Goal:** a clean, leak-free dataset and an evaluation protocol with trivial anchors — *set up before modeling*.

**Steps (Karpathy "become one with the data" + Lones):**
1. **Inspect the data by hand.** Distributions, units, ranges, class balance, and especially **missing values and how they're coded** (0? NaN? sentinel?).
2. **Split FIRST, then preprocess.** Fit scalers/feature-selection on **train only**; apply to val/test. *(Data leakage from preprocessing-before-split is the #1 reproducibility killer — Lones, Kapoor & Narayanan.)*
3. **Respect structure in the split:** time series → **temporal split, never shuffle across the boundary**; grouped data → keep groups (e.g., same subject) in one fold; imbalanced → stratify.
4. **Handle missing data explicitly** — mask it in the loss *and* the metrics; don't let the model be graded on unpredictable sentinels.
5. **Define metrics + trivial anchors up front:** mean-predictor, persistence/last-value (time series), majority-class (classification), random. *These are your truth-anchors for every later result.*

**Pitfalls.** *(War story: coding missing readings as "0 mph" and scoring the model on them inflated error ~35% and flattened the hyperparameter sweep — it looked like "the model has no HP sensitivity" but was really "the metric is dominated by unpredictable zeros." Masking fixed it.)* Also: don't smooth a per-node/per-pixel regression output across neighbors — it averages away the signal.

---

## Phase 3 — Baselines (the foundation everything is measured against)

**Goal:** correct, honest baselines — both a simple one and a strong one.

**Steps (Karpathy skeleton + our ladder method):**
1. **Build the end-to-end train/eval skeleton with a dumb baseline first.** Fix the random seed.
2. **Verify the pipeline before trusting numbers:** check the loss at initialization equals the theoretical chance value; **overfit a single batch** (if you can't, something is broken); add significant digits to your eval.
3. **Sanity-check every result against the trivial anchors.** *A model that can't beat mean/persistence is not learning — no matter how low the raw number looks.*
4. **Build a ladder** where each rung adds one mechanism (e.g., base → +residual → +temporal). Each rung isolates a contribution.
5. **Weak vs strong baseline — build both.** The weak/plain model is your **microscope** (where the failure you study is visible); the strong model is your **credibility anchor** (proves your method still matters). *Beating a handicapped baseline is fake novelty a reviewer will see through.*
6. **Make comparisons fair:** uniform training budget (same epochs) + a per-model hyperparameter sweep, best chosen **by validation**. Report the config.
7. **Verify code on synthetic data before expensive runs** (shapes, backward pass, a known-answer sanity check).

**Signatures of a broken model (memorize these):**
- Results **nearly identical across a wide hyperparameter sweep** → it's not learning (converged to a trivial optimum).
- **RMSE ≈ the target's std** (or accuracy ≈ majority class) → it's predicting a constant.
- **Worse than a trivial baseline** → a bug or a modeling flaw, not a "weak but valid" result.

**Pitfalls (Lones "compare models fairly"):** unequal tuning budgets; comparing to a strawman; reporting a number with no anchor; not reproducing your own baseline's known value from the literature (if one exists) as a pipeline check.

---

## Phase 4 — The contribution (method → diagnosis → fix)

**Goal:** demonstrate your hypothesis and that the gain is *attributable to your idea*.

**Steps.**
1. **Diagnose before you fix.** Measure the phenomenon (e.g., error vs. a structural variable; an internal quantity vs. depth). Diagnosis-first work is cheap, respected, and gives your fix a target.
2. **Control confounds.** Before crediting your mechanism, rule out mundane explanations — optimizer strength, model capacity, more epochs. *(War story: a neighboring paper argued the effect we study is "optimization-induced." Our RQ2 exists solely to show it survives a stronger optimizer — otherwise the whole contribution is dismissible.)*
3. **Change one thing at a time (ablations).** Every design choice earns its place through an ablation.
4. **Validate the fix on TWO substrates:** the weak one (expect a large, clear gain → proves the mechanism) *and* a strong one (expect a smaller gain → proves it still matters). **If it helps on the weak substrate but vanishes on the strong one, that is a critical finding to surface early**, not to hide.
5. **Watch for the "cure that hides the disease."** A capacity/architecture add-on can accidentally fix the thing you're studying by a different route (e.g., a temporal module bypassing the spatial pathway you're diagnosing). Keep the clean substrate as the microscope.

**Pitfalls:** attributing a gain to your idea when it came from extra capacity/tuning; a single lucky seed; testing only where your method looks good.

---

## Phase 5 — Robust evaluation

**Goal:** conclusions that hold up.

**Steps (Lones "robustly evaluate" + "report results"):**
- **Multiple seeds**; report mean ± std / error bars. One run is an anecdote.
- **Multiple metrics** — no single number captures a model; report the tradeoffs (e.g., accuracy vs. cost/latency/constraint-satisfaction).
- **Slice performance** — where does it fail? (subgroups, hard cases, tail).
- **Statistical significance with care** (appropriate tests, corrected for multiple comparisons); don't over-read tiny differences.
- **Use a proper held-out test set once, at the end.** Don't tune on test.

---

## Phase 6 — Reproducibility & engineering

**Goal:** anyone (including future-you and graders) can reproduce your numbers.

**Steps.**
- **Determinism:** fix seeds (data, init, shuffling); pin dependency versions; note hardware.
- **Repo hygiene — the Papers-with-Code ML Code Completeness Checklist (aim for all 5):** (1) training script, (2) evaluation script, (3) pretrained models/weights, (4) a results table/plot + a script to reproduce it, (5) a README with setup + commands.
- **Experiment tracking:** W&B / MLflow / TensorBoard for metrics; a config file (YAML/argparse) per run so every result maps to a config.
- **Keep a decision log / project notes** — *what* you did and *why*, updated as you go. Record the non-obvious calls (a masking fix, a withdrawn citation, why baseline X over Y) — a teammate or reviewer *will* ask.
- **Context handoff files** (e.g., `CLAUDE.md`) so agentic tools / new collaborators start with full context.

---

## Phase 7 — Writing & publishing

**Goal:** a paper that claims exactly what you showed and is easy to trust.

**Structure:** Abstract → Introduction (problem, gap, contributions as a bulleted list) → Related Work (established vs. your delta, with differentiation sentences) → Method → Experimental Setup (data, baselines, protocol) → Results (with ablations + anchors) → Discussion/Limitations → Conclusion → References.

**Rules (Lones "report your results" + NeurIPS Paper Checklist):**
- **Claim only what you demonstrated.** Report negative/partial results honestly — they're often the most credible part.
- **State limitations explicitly.** The NeurIPS checklist makes this near-mandatory; reviewers reward it.
- **Be transparent / reproducible:** the checklist asks whether code, data, and instructions to reproduce main results are provided. Answer "yes" wherever possible.
- **Don't reconstruct or over-quote sources;** paraphrase and cite.

**Venues (calibrate ambition to resources):**
- Main track (ICML / NeurIPS / ICLR) — demands strong novelty + rigor + often theory; very competitive.
- **Workshops (e.g., NeurIPS ML4PS, ICML sci-ML tracks)** — prestigious, realistic on-ramp, short-paper (~4 pages), and the standard first step. Land a workshop, then extend to main track / a domain journal.
- Post preprints to arXiv; reviewing is usually on OpenReview.

---

## Debugging deep learning (Karpathy-distilled + case study)

Core stance: **neural nets fail silently** — a misconfigured net usually still trains to *something*, so be "thorough, defensive, paranoid" (Karpathy). Concrete moves:
- **Visualize exactly what enters the model** (`y = model(x)`) — decode the raw tensor. Catches preprocessing/orientation bugs.
- **Verify loss at init** = theoretical chance value.
- **Overfit one batch** to ~0 loss — if you can't, the model/loss/data-plumbing is broken.
- **Simplify, then generalize** — write the specific loopy version, confirm it, then vectorize and confirm the same result.
- **Use gradient checks** to confirm dependency structure (e.g., an autoregressive output at time *t* must not depend on *t+1*; a per-example loss must give gradient only on that example — catches batch-dimension mixing).

**Case study (the "flat sweep" collapse).** Symptoms: MAE ~2–3× worse than a trivial baseline, and *all* hyperparameter configs within 0.02 of each other. Diagnosis path: compared to anchors (RMSE ≈ target std → predicting the mean); found two causes — (a) the output layer was smoothing predictions across graph neighbors, (b) missing values coded as 0 were unmasked and dominated the metric. Fixes: node-wise (non-smoothing) readout + mask missing in loss/metrics. Result: the model started learning and the sweep separated. *Lesson: when a sweep is suspiciously flat, suspect the metric/data, not "the model has no sensitivity."*

---

## The universal checklists (copy-paste into each project)

**Before running any experiment:**
- [ ] Inspected the raw data by hand; know units, ranges, and how missing is coded.
- [ ] Split before preprocessing; scalers fit on train only.
- [ ] Correct split type (temporal / grouped / stratified).
- [ ] Trivial anchors defined (mean / persistence / majority / random).
- [ ] Seed fixed; deps pinned.
- [ ] Loss-at-init checked; overfit-one-batch passes; code verified on synthetic data.

**Before trusting a result:**
- [ ] Does it beat the trivial anchors? By how much?
- [ ] Is the sweep suspiciously flat (→ not learning)? RMSE ≈ std (→ predicting mean)?
- [ ] Multiple seeds run; variance reported.
- [ ] The gain is attributable to my idea (confounds controlled, ablated)?
- [ ] Comparisons fair (equal budget + per-model tuning)?

**Before submission:**
- [ ] Every claim is supported by an experiment in the paper.
- [ ] Limitations stated explicitly.
- [ ] Code: training + eval scripts, results-repro script, README (Code Completeness Checklist).
- [ ] All references verified (nothing withdrawn/fabricated); related work differentiates from nearest neighbors.
- [ ] Reproducibility/ethics checklist completed (venue-specific).

---

## Key resources

- **A. Karpathy, "A Recipe for Training Neural Networks"** (karpathy.github.io/2019/04/25/recipe/) — the debugging bible.
- **M. Lones, "Avoiding common machine learning pitfalls,"** Patterns (Cell Press) 2024 / arXiv:2108.02497 — five stages, updated annually; the single best pitfalls reference.
- **Kapoor & Narayanan, "Leakage and the Reproducibility Crisis in ML-based Science,"** Patterns 2023 — an 8-type taxonomy of data leakage.
- **NeurIPS Paper Checklist** (blog.neurips.cc) — reproducibility/limitations/ethics; mirror it even for other venues.
- **Papers with Code — ML Code Completeness Checklist** — the 5-item repo standard.
- Tooling: Weights & Biases / MLflow / TensorBoard (tracking); Hydra/argparse (configs); arXiv, Semantic Scholar, Papers with Code, Connected Papers (lit search).

---

## The distilled principles (the one-liners to remember)

1. Neural nets fail silently — be paranoid; verify before you scale.
2. A result that can't beat a dumb baseline is not a result.
3. Frame the thesis as a hypothesis to test — a negative result is still a finding.
4. Novelty is a *transferable idea*, not a domain application.
5. When compute-limited, make the contribution cheap by construction.
6. Split before preprocessing; the test set is sacred.
7. Mask missing data in loss *and* metrics.
8. Build both a weak baseline (microscope) and a strong one (credibility).
9. Make comparisons fair or don't make them.
10. Diagnose before you fix; control confounds; ablate one change at a time.
11. Validate the fix on a weak *and* a strong substrate.
12. Verify every citation against the real paper.
13. Keep a decision log; write down *why*, not just *what*.
14. Report limitations — it builds trust, not doubt.

---

### Appendix — this playbook mapped onto the source project (a worked example)
- **Phase 0:** MIP → PINN pivot; reframed to a cheap *diagnosis + fix* of oversmoothing in physics-informed GNNs; thesis kept as claim-to-test; RQ1 diagnose / RQ2 isolate-from-optimizer / RQ3 fix.
- **Phase 1:** verified 7 papers; separated established (PINN spectral bias, GNN oversmoothing) from the novel synthesis (their compounding on graphs); caught a withdrawn reference.
- **Phase 2:** METR-LA; 8.1% missing coded 0 → masked; temporal 70/10/20 split; anchors = mean (11.5) + persistence (3.0).
- **Phase 3:** GCN baseline; caught the "flat sweep" collapse; built the ladder (plain → residual → temporal) with fair budgets.
- **Phase 4:** PI-GCN (physics loss) → diagnosis → fix on plain *and* strong substrate (in progress).
- **Phase 6–7:** `PROJECT_NOTES.md` decision log + `CLAUDE.md` handoff; target NeurIPS ML4PS workshop first.
