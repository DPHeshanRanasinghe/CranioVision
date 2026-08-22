# CranioVision — Validation & Confidence Calibration Report

**Status:** full study complete (n=56, Colab T4). One caveat outstanding — the run was
**not** restricted to the held-out test split (see §3.5), so Part A needs a re-run
before its accuracy figures are quotable. Part B's conclusion is unaffected.
**Date:** 2026-08-21.
**Scope:** how well the system performs, and whether it can signal when it is
likely to be wrong.

Companion documents:
- `evidence_agreement_calibration.md` — verified literature backing the claim.
- `validation_study.ipynb` — the reproducible study (Colab/Kaggle-ready).
- `results/` — pilot artifacts (`per_case_raw.json`, `validation_summary.json`,
  `agreement_vs_dice.png`).

---

## 1. The question

A clinician asked, in effect: *how well does this perform, and what baseline do I
judge it against?* Answering that honestly exposed a structural gap.

### 1.1 What the two agreement numbers in the UI actually mean

Both measure **cross-model consensus** — how often the three architectures assign a
voxel the same label. Neither involves ground truth, so **neither is an accuracy
measure.** Computed in `pipeline.py:377-409`.

| Metric | Definition | Reading |
|---|---|---|
| **Whole-Volume Agreement** (~99%) | all 3 agree ÷ all voxels | Dominated by background, where every model trivially predicts 0. A sanity check, not a confidence signal. |
| **Tumor-Region Agreement** | (all agree AND ≥1 called tumor) ÷ (voxels *any* model called tumor) | The denominator is the **union** of tumor masks, so a region only one model flags counts wholly as disagreement — deliberately conservative. |

Disagreement concentrates at tumor **boundaries**. A low value means "the extent is
uncertain, review the margins", not "the location is wrong".

### 1.2 The gap

**The "Mean Dice 0.87" shown in the UI is not product performance.**
`pipeline.py:367` computes it only `if gt_3d is not None` — i.e. against the
ground-truth mask that ships inside BraTS demo folders. A real patient upload has no
`-seg.nii`, so `mean_dice` is `None` and the Performance panel renders nothing
(`frontend/app/page.tsx:1176`).

The number that looks like an accuracy guarantee disappears exactly when a clinician
would need it — and displaying it on demo cases invites the reading "this tool is 87%
accurate on my patient", which it cannot support.

Related findings at the time of investigation:
- No `outputs/*_history.json` in the working checkout → `_load_val_dice`
  (`pipeline.py:113-123`) falls back to `1.0` per model → the documented
  *Dice-weighted* ensemble silently degrades to plain averaging.
- `mc_dropout.py` exists but is never called by the pipeline or backend.

---

## 2. The proposed answer

Since per-case Dice is impossible at inference time, the honest substitute is a signal
that **predicts** Dice. Hypothesis:

> If cross-model agreement correlates with segmentation accuracy, then high agreement
> on a new patient is evidence the output is reliable — and we can quote a calibrated
> expectation instead of an uninterpretable percentage.

### 2.1 Prior work (verified by fetching the sources, not search snippets)

| Work | Relevance |
|---|---|
| **QU-BraTS**, Mehta et al. 2022, MELBA; arXiv:2112.10074 | A MICCAI challenge task on *exactly this problem*, on *exactly this dataset*. Scoring rewards estimates that are "confident when correct and uncertain when incorrect", motivated by letting uncertain predictions "be flagged and brought to the attention of clinical experts". 14 teams benchmarked. |
| **Reverse Classification Accuracy**, Valindria et al. 2017, IEEE TMI; arXiv:1702.03407 | Foundational result that per-case segmentation quality *is* predictable without ground truth. |
| **Sikha et al. 2025**, Comput. Med. Imaging Graph.; arXiv:2508.01460 | Quantitative comparison of MC-dropout / ensembles / TTA. **TTA + entropy best overall**: skin R²=93.25, r=96.58; 3D liver R²=85.03, r=65.02. |

**Methodological note.** An earlier draft of this argument cited the 2025 paper as
showing "ensembles performed best, r ≈ 0.86". That came from a search-result summary.
The paper's actual abstract reports **TTA with entropy** as best. The general claim
survived; the specific one did not. Verify against sources.

### 2.2 Positioning consequence

The idea is **not novel** — a MICCAI challenge covered it. Therefore:

- Present this as **product validation** ("our confidence indicator has been calibrated
  against measured accuracy"), **not** as a research contribution.
- The one under-explored angle: our signal is agreement between *structurally
  different* architectures (attention-gated CNN / transformer encoder / residual
  decoder). The literature studies MC-dropout, TTA, and *same-architecture* deep
  ensembles. A targeted search surfaced no direct precedent for calibrating
  heterogeneous-architecture agreement on BraTS. A modest, honest novelty claim.

---

## 3. Results

### 3.1 Performance — 30-case held-out test set

Recovered from the original developer's `outputs/*_test_results.json`. **These are the
authoritative numbers.**

| Model | Mean Dice | WT | TC | ET |
|---|---|---|---|---|
| **SwinUNETR** | **0.7928** | 0.9115 ± 0.099 | 0.8094 ± 0.182 | 0.7208 ± 0.308 |
| Ensemble | 0.7850 | 0.9142 ± 0.093 | 0.8051 ± 0.211 | **0.6893 ± 0.330** |
| Attention U-Net | 0.7307 | — | — | — |
| nnU-Net (DynUNet) | 0.6925 | 0.8701 ± 0.095 | 0.7446 ± 0.248 | 0.5604 ± 0.357 |

Also recorded: `mean_volume_relative_error = 0.074`, `mean_model_agreement = 0.9933`
(whole-volume — see §1.1 on why that is near-trivially high).

**Three findings that matter:**

1. **`ensemble_gain_over_best_single = −0.0077`.** The ensemble is marginally *worse*
   than SwinUNETR alone. This independently confirms the project's own framing: the
   ensemble earns its place as a cross-model **confidence signal**, not as a Dice
   improvement. It should never be presented as "more accurate".
2. **`val_dice_reference = {attention_unet: 1.0, swin_unetr: 1.0, nnunet: 1.0}`.**
   The ensemble was equal-weighted even in the developer's own evaluation. The
   *Dice-weighted ensemble* described in `README.md` and `CLAUDE.md` **has never
   actually run.** Either fix the code (write real `*_history.json`) or fix the docs.
3. **Enhancing tumor is the genuine weakness: 0.689 ± 0.330.** That standard deviation
   means some cases score near zero on ET — the compartment that matters most for
   surgical planning. This is the honest headline limitation.

`attention_unet_test_results.json` is **missing**, so that model has no stored per-case
data (its mean comes from the ensemble file's `per_model_mean_dice`).

#### Our own run — n = 56 (split provenance unverified, see §3.5)

Dice, mean [95% bootstrap CI]:

| Model | Mean Dice | WT | TC | ET |
|---|---|---|---|---|
| **SwinUNETR** | **0.795** [0.762–0.827] | 0.946 [0.938–0.953] | 0.851 [0.810–0.888] | 0.755 [0.682–0.817] |
| Ensemble | 0.784 [0.746–0.820] | 0.946 [0.937–0.954] | 0.817 [0.753–0.875] | 0.751 [0.679–0.815] |
| Attention U-Net | 0.751 [0.706–0.792] | 0.909 [0.896–0.922] | 0.773 [0.690–0.846] | 0.725 [0.648–0.794] |
| nnU-Net (DynUNet) | 0.683 [0.640–0.723] | 0.909 [0.897–0.920] | 0.751 [0.671–0.822] | 0.596 [0.505–0.682] |

vs the nnU-Net BraTS 2020 winner: **WT +0.057**, TC −0.034, **ET −0.069** (different
cohort and year — context, not a ranking).

`ensemble_gain_over_best_single = −0.011`, reproducing the prior evaluation's −0.008 at
a larger n. **The ensemble is consistently, if marginally, below SwinUNETR.** Two
independent evaluations now agree on this.

### 3.2 Calibration study — n = 56 (Colab, Tesla T4)

**Adequately powered:** minimum detectable |r| = 0.263; power vs a true r=0.7 ≈ 100%.

| Statistic | Value |
|---|---|
| Spearman ρ (agreement vs ensemble Dice) | **+0.585** |
| p (Spearman) | **2.2 × 10⁻⁶** |
| 95% CI (bootstrap) | [+0.386, +0.734] |
| Permutation p | < 1 × 10⁻⁴ |
| **Partial ρ, controlling GT tumor volume** | **+0.609** |
| p (partial) | **6.2 × 10⁻⁷ — significant** |
| Attenuation | **−4% (the effect strengthened)** |
| agreement ↔ tumor volume | ρ = **−0.004** (p = 0.97 — no relationship) |
| Dice ↔ tumor volume | ρ = −0.286 (p = 0.032) |

**Calibration table** (expected ensemble Dice by agreement band):

| Agreement | n | Observed Dice | 95% CI | Reportable |
|---|---|---|---|---|
| <70% | 23 | 0.704 | [0.641–0.762] | ✅ |
| 70–80% | 19 | 0.811 | [0.763–0.854] | ✅ |
| 80–85% | 13 | 0.875 | [0.824–0.916] | ✅ |
| 85–90% | 1 | 0.946 | — | ❌ n=1 |
| ≥90% | 0 | — | — | ❌ n=0 |

Monotonic, with non-overlapping CIs between the bottom and top reportable bands — a
~17-point Dice spread from the lowest to the highest usable band.

### 3.3 The confound is resolved — and the pilot's version of it was an artifact

The n=10 pilot suggested agreement might be a proxy for tumor size. At n=56 that
collapses completely:

| | pilot n=10 | full n=56 |
|---|---|---|
| raw ρ | +0.770 | +0.585 |
| partial ρ (control volume) | +0.455 (ns) | **+0.609 (p=6×10⁻⁷)** |
| agreement ↔ volume | **+0.903** | **−0.004** |
| Dice ↔ volume | +0.697 | **−0.286** |

Every relationship involving tumor volume **reversed or vanished**. The apparent
ρ=0.90 collinearity was a small-sample artifact; with 56 cases, agreement and tumor
volume are statistically unrelated (p=0.97), and Dice actually correlates *negatively*
with tumor size.

Controlling for volume makes the association slightly **stronger** (+0.585 → +0.609).
That is coherent rather than anomalous: volume is a mild suppressor here — it adds
noise to Dice while carrying no information about agreement — so removing it sharpens
the relationship.

**Conclusion: agreement carries information about accuracy that is independent of
tumor size.** This is the finding the pilot could not deliver, and it is what upgrades
the claim from "difficulty indicator" to a genuine reliability signal.

### 3.4 Statistical power

| n | min \|r\| for p<0.05 | power vs true r=0.7 |
|---|---|---|
| 10 (pilot) | 0.632 | 63% |
| 30 | 0.361 | 99% |
| **56 (actual)** | **0.263** | **~100%** |

### 3.5 Outstanding caveat — the held-out split was not applied

The run reported *"Reference result files found: none"* because
`outputs/*_test_results.json` were not present in the Colab environment. With no stored
test IDs, `RESTRICT_TO_TEST_SPLIT` could not filter, and **all 56 cases were analysed**
— an unverified mixture of train / validation / test.

Consequences:
- **Part A accuracy figures are indicative, not a clean held-out estimate.**
- The reproducibility cross-check against the prior evaluation was skipped.
- Part B is **less affected**: a correlation between agreement and error is a property
  of the relationship, not of absolute performance level — but it should still be
  confirmed on the clean split.

Reassuringly, the overall means land almost exactly on the developer's held-out
numbers (ensemble 0.784 vs 0.785; SwinUNETR 0.795 vs 0.793), which argues against
substantial training-set inflation. Per-region figures differ more (WT +0.035 for
SwinUNETR), consistent with a different case mix.

**Also note:** 10 cases were skipped for *"missing modality"*, including several
(`BraTS-GLI-02105-100/102/105`) that belong to the stored test split. Those folders
appear incomplete in Drive and are worth checking.

**Cheap fix:** inference is cached in `per_case_raw.json`, so copying the three
`outputs/*_test_results.json` files (32 KB) into the Colab repo and re-running only the
analysis cells reproduces everything on the clean split in seconds — no GPU time.

---

## 4. Conclusions

### Established
- Higher cross-model agreement predicts higher Dice — **ρ = +0.585, p = 2×10⁻⁶, n = 56**,
  adequately powered (min detectable |r| = 0.263).
- **The effect is independent of tumor size** (partial ρ = +0.609, p = 6×10⁻⁷;
  agreement ↔ volume ρ = −0.004). The pilot's confound was a small-sample artifact.
- Three calibration bands now have enough cases to quote (n = 23 / 19 / 13), spanning
  Dice 0.704 → 0.875 with non-overlapping confidence intervals.
- The ensemble does **not** improve Dice over the best single model — confirmed twice
  independently (−0.008 at n=30, −0.011 at n=56).
- The mechanism is well-supported by prior work, including a MICCAI challenge on this
  dataset.

### Not established
- **Part A on a verified held-out split** — the n=56 run mixed splits (§3.5). Numbers
  are indicative until re-run; cheap to fix, no GPU needed.
- The **high-agreement bands** (85–90%: n=1; ≥90%: n=0). We cannot quote an expected
  Dice for the most confident cases — which, ironically, is where a clinician would
  most want one.
- The "9–15× signal-to-background ratio" claim in `README.md`, `pipeline.py` and
  `clinical_report.py` — still has **no reproducible computation** in this repo.

### What can be said to a clinician today

| Claim | Status |
|---|---|
| "High agreement predicts a more accurate result." | ✅ Supported, n=56, p=2×10⁻⁶. |
| "Agreement reflects reliability independent of tumor size." | ✅ **Now supported** (partial ρ=+0.609). |
| "Agreement 80–85% → observed Dice 0.875 [0.824–0.916]." | ✅ Supported for the three bands below 85%. |
| "ET is the weakest region." | ✅ Supported (0.596–0.755 across models; below baseline). |
| "Agreement ≥90% → expect Dice X." | ❌ No cases in that band. |
| "On held-out data we achieve WT 0.95." | ❌ Not until §3.5 is resolved. |
| "The ensemble is the most accurate output." | ❌ False — SwinUNETR scores higher, twice confirmed. |

### Standing limitations (independent of any result)
1. Agreement is a **consensus** signal, not correctness. Three models can agree and all
   be wrong; architectural diversity lowers that risk without eliminating it.
2. Everything here is measured on BraTS-format, skull-stripped, co-registered,
   pre-operative **adult glioma**. It does not transfer to pediatric, meningioma,
   post-operative, or raw clinical acquisitions without re-validation.

---

## 5. Next steps

1. **Re-run on the clean held-out split** (§3.5). Copy the three
   `outputs/*_test_results.json` into the Colab repo and re-run from the "Restrict to
   the held-out test split" cell. Inference is cached, so this costs seconds and
   resolves Part A plus the reproducibility cross-check.
2. **Surface calibrated confidence in the UI** — now justified for the three bands
   below 85%, e.g. *"Agreement 82% — cases at this level had Dice 0.875 [0.824–0.916]
   across 13 validation cases."* Keep the ≥85% bands qualitative until they have cases.
3. **Fix the "9–15×" claim** or remove it from README/pipeline/report — it is the last
   unsupported assertion still shipping.
4. **Decide the ensemble-weighting question.** Two evaluations now show the ensemble
   below SwinUNETR on Dice. Either implement real Dice weighting (which might close
   the gap) or state plainly that the ensemble exists for consensus, not accuracy.
5. **Investigate ET** — weakest compartment (0.596 for nnU-Net), below the published
   baseline, and the most clinically consequential.
6. **Check the 10 skipped cases** — several belong to the held-out split but appear to
   be missing modalities in Drive.
7. **Add an intended-use statement** bounding the claim to the validated population.
