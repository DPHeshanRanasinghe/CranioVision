# Evidence base — model agreement as a confidence signal

Supporting document for the claim CranioVision wants to make to clinicians:

> **"When the three models agree strongly on a new patient, you can place more
> confidence in the output."**

Verification status per playbook Phase 1 step 2: every entry below marked
*verified* was checked by fetching the paper's own abstract page on
2026-08-21, not taken from a search snippet. Confidence tier noted per entry.

**A correction that motivates this file:** an earlier draft of this argument
cited the 2025 Sikha et al. paper as showing "ensembles performed best,
Pearson ≈ 0.86". That came from a search-result summary. The paper's actual
abstract reports **Test-Time Augmentation with entropy** as the best overall
method. The general claim survives; the specific one did not. Do not cite
from snippets.

---

## 1. What the literature already establishes (cite confidently)

- **Mehta, Filos, Baid, et al., 2022. "QU-BraTS: MICCAI BraTS 2020 Challenge on
  Quantifying Uncertainty in Brain Tumor Segmentation — Analysis of Ranking
  Scores and Benchmarking Results."** *Journal of Machine Learning for
  Biomedical Imaging (MELBA)*; arXiv:2112.10074. 92 authors.
  Tier: **high** (verified via direct fetch, not withdrawn).
  A dedicated BraTS challenge task on exactly this problem — on exactly the
  dataset CranioVision uses. Its evaluation score "rewards uncertainty
  estimates that produce high confidence in correct assertions and those that
  assign low confidence levels at incorrect assertions", and penalises
  under-confident correct assertions. Benchmarked 14 teams and concluded that
  uncertainty estimates provide complementary value to segmentation
  algorithms, with the explicit clinical motivation that uncertain predictions
  can "be flagged and brought to the attention of clinical experts."
  **Why it matters here:** this is the strongest possible backing — a MICCAI
  challenge established the premise on our dataset. It also means the *idea*
  is not novel, and should not be presented as such.

- **Valindria, Lavdas, Bai, Kamnitsas, Aboagye, Rockall, Rueckert, Glocker,
  2017. "Reverse Classification Accuracy: Predicting Segmentation Performance
  in the Absence of Ground Truth."** *IEEE Transactions on Medical Imaging*;
  arXiv:1702.03407. Tier: **high** (verified via direct fetch, not withdrawn).
  Establishes the foundational result that "it is indeed possible to predict
  the quality of individual segmentations, in the absence of ground truth",
  and frames it for "automatic processing pipelines in clinical routine."
  Note: the abstract does not report specific correlation coefficients — do
  not attribute numbers to this paper without reading the full text.

- **Sikha O K, Riera-Marín, Galdran, García Lopez, Rodríguez-Comas, Piella,
  González Ballester, 2025. "Uncertainty-Aware Segmentation Quality Prediction
  via Deep Learning Bayesian Modeling."** *Computerized Medical Imaging and
  Graphics*, DOI 10.1016/j.compmedimag.2025.102547; arXiv:2508.01460.
  Tier: **high** (verified via direct fetch, not withdrawn).
  Compares MC-dropout, ensembles and Test-Time Augmentation against confidence
  maps, entropy, mutual information and expected pairwise KL divergence, for
  predicting segmentation quality without ground truth. **TTA with entropy
  performed best overall**: skin lesion R² 93.25 / Pearson 96.58; 3D liver
  R² 85.03 / Pearson 65.02.
  **Why it matters here:** recent, quantitative evidence that uncertainty
  signals genuinely predict Dice — but note the liver (3D) correlation is much
  weaker than the 2D skin result, which is a caution for our 3D setting. It
  also indicates ensembles were *not* the top performer, so we should not
  claim ensemble agreement is the best available signal, only that it is a
  legitimate one that we can calibrate.

---

## 2. What is NOT established — the part we must demonstrate

None of the above tells a clinician how much to trust **CranioVision's**
output. Specifically, no located work covers:

1. **Heterogeneous-architecture agreement as the confidence signal.** The
   literature studies MC-dropout, TTA, and same-architecture deep ensembles.
   CranioVision's signal is disagreement between three *structurally
   different* decoders (attention-gated CNN / transformer encoder /
   residual-with-deep-supervision). A targeted search surfaced no direct
   precedent for calibrating that specific quantity on BraTS. This is a
   modest, honest novelty claim — the mechanism differs, not the goal.
2. **Our numbers.** Even if the effect is universal, the mapping from *our*
   agreement values to *our* expected Dice is system-specific and must be
   measured on our checkpoints.

**Framing consequence:** present this to clinicians as **product validation**
("our confidence indicator has been calibrated against measured accuracy"),
not as a research discovery. The literature is the reason to *expect* the
effect; our study is the evidence that it holds here.

---

## 3. Study design (see `validation_study.ipynb`)

Per case, with ground truth available:
- **x** = tumor-region agreement (existing metric — unanimous tumor voxels /
  union of all model tumor masks; see `pipeline.py`).
- **y** = ensemble Dice (and HD95) against GT.
- Test H0: no monotonic relationship, via Spearman ρ with bootstrap CI.
- Expect **ρ significantly positive** for Dice (equivalently, negative for
  error). Report the calibration table (Dice observed within agreement bins),
  which is what the UI would actually surface.

### Confound to control
Agreement and Dice are both driven by case difficulty — large, well-defined
tumors are both easier to segment *and* easier to agree on. A raw correlation
may therefore reflect "this case was easy" rather than agreement carrying
independent information. Mitigation: report partial correlation controlling
for GT tumor volume, and inspect the relationship within size strata.
Operationally the raw correlation is still what the UI needs (it only has to
*predict*), but the honest scientific claim requires the confound noted.

### Statistical power — the binding constraint
Minimum |r| for p < 0.05 (two-tailed), and power to detect a true r = 0.7:

| n   | min \|r\| for significance | power at true r=0.7 |
|-----|---------------------------|---------------------|
| 10  | 0.632                     | 63%                 |
| 20  | 0.444                     | 95%                 |
| 30  | 0.361                     | 99%                 |
| 50  | 0.279                     | >99%                |

With **n = 10** only a very strong effect is detectable, and a null result
would be uninformative (37% chance of missing a real, strong effect). Treat
n=10 as a **pilot**. At **n ≈ 30** the study becomes adequately powered for a
strong-to-moderate effect, which is the stated near-term data target.

### What would falsify / weaken the claim
- ρ not significantly different from zero at n≈30 → the signal does not carry
  usable information; do not ship the confidence framing.
- ρ significant but vanishing after controlling for tumor volume → agreement
  is a proxy for tumor size, not model confidence. Still usable operationally
  but must be described as such, not as "model confidence".
- Strong ρ but flat calibration inside the clinically relevant high-agreement
  band → the signal separates easy from hard cases but cannot discriminate
  among the cases a clinician would actually act on.

---

## 4. To verify before citing (do not use yet)

- Lakshminarayanan, Pritzel, Blundell, 2017, "Simple and Scalable Predictive
  Uncertainty Estimation using Deep Ensembles" — the standard citation for
  ensemble-based uncertainty. **Not yet fetched/verified.**
- Jungo & Reyes, "Assessing Reliability and Challenges of Uncertainty
  Estimations for Medical Image Segmentation" (MICCAI 2019) — relevant
  cautionary work on uncertainty reliability. **Not yet fetched/verified.**
- Roy et al., Bayesian QuickNAT — uncertainty for QC in brain MRI.
  **Not yet fetched/verified.**
- Full text of QU-BraTS for the specific ranking metric definition, if we want
  to report our signal in a form comparable to that challenge.
