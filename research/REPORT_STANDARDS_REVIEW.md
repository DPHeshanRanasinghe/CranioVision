# Clinical report — standards review and gap analysis

**Question:** does the CranioVision PDF actually give clinicians what they need?

**Reviewed:** `BraTS-GLI-02063-101_23a5dd54.pdf` (5 pages), generated 2026-08-22,
against ACR reporting requirements, RANO 2.0, and AI-transparency expectations.

**Short answer:** the *analysis* is strong and in places exceeds what a
conventional report offers (multi-model comparison, eloquent-cortex proximity,
explainability). The *report* is weaker than the analysis: it is organised as a
data dump rather than a clinical document, and is missing several elements a
radiologist would consider mandatory — most importantly an **Impression**.

---

## 1. What the standards require

### 1.1 ACR — Communication of Diagnostic Imaging Findings

The ACR practice parameter enumerates the components of a diagnostic imaging
report as: *"patient identification, relevant clinical information, description
of the procedure and technique, comparison with prior studies, findings,
impression, recommendations, and report authentication."*

Scored against the current PDF:

| # | ACR component | CranioVision | Notes |
|---|---|---|---|
| 1 | Patient identification | ⚠️ partial | Case ID only. No MRN / DOB / accession / study date. Fine for de-identified research, insufficient as a clinical document. |
| 2 | Relevant clinical information (indication) | ❌ **missing** | No indication field at all. A report with no stated question cannot be judged as answering one. |
| 3 | Procedure and technique | ⚠️ partial | Grad-CAM methodology is described; the **MRI acquisition** is not (sequences, slice thickness, contrast agent, field strength). |
| 4 | Comparison with priors | ❌ **missing** | No prior-study field. In neuro-oncology this is the central axis of interpretation. |
| 5 | Findings | ✅ strong | Volumes, per-region anatomy, laterality, eloquent proximity, model agreement. |
| 6 | **Impression** | ❌ **missing** | The most-read section of any radiology report. Referring clinicians read the impression first, sometimes only. |
| 7 | Recommendations | ⚠️ partial | One good recommendation (fMRI/DTI) buried in the eloquent-cortex page; not a distinct section. |
| 8 | Report authentication | ⚠️ partial | Timestamp present; no reviewing-clinician block, no software/model version for traceability. |

**4 of 8 fully met.** The two hardest misses are **Impression** and
**Comparison**.

### 1.2 RANO 2.0 (neuro-oncology response assessment)

RANO 2.0 (2023) is how gliomas are actually assessed in practice and trials.
Points that bear directly on this report:

- **Primary measurement is the 2-D maximum cross-sectional (bidimensional)
  product**; volumetric measurement is an *option* where resources allow. We
  report **volume only**. A neuro-oncologist comparing against prior imaging or
  a trial protocol cannot use our numbers directly.
- **Response categories are defined by change over time** against a baseline
  (post-radiation scan in RANO 2.0). A single-timepoint report cannot state
  response — and ours correctly does not claim to, but it also does not say so.
- **T2/FLAIR is excluded from enhancing-tumour assessment** for IDH-wild-type
  tumours. Our headline "Total tumour volume 76.71 cm³" is dominated by
  **edema (68.12 cm³)**. Presenting that as "total tumour" risks being read as
  tumour burden in the RANO sense, which it is not.
- **Acquisition should follow the Brain Tumour Imaging Protocol (BTIP)**
  (post-contrast T1w ≤1.5 mm, T2/FLAIR ≤4 mm axial). We should state whether
  the input complied, or that compliance was not verified.

### 1.3 AI-specific transparency

Regulatory and editorial guidance converges on: disclose model identity and
version, state the intended use, state the evidence base and its limits, and
make the human-review step explicit. Current state:

- ❌ No software or model version anywhere in the PDF — a report that cannot be
  traced to the code that produced it is not auditable.
- ❌ **Page 5 asserts a "9–15× signal-to-background ratio"** for the Grad-CAM
  explainer. This number has **no reproducible computation in the repository**
  (see `PROJECT_NOTES.md`). It is currently being asserted to clinicians.
- ⚠️ "Research use only" appears only as a small footer, not as a prominent
  status banner.
- ✅ The agreement wording was corrected to describe consensus rather than
  confidence, and states plainly that agreement is not accuracy.

---

## 2. Clinical-content problems found in the current output

**a. Dice scores are shown to clinicians.** Page 2 lists WT/TC/ET Dice per
model (0.9456, 0.9417, …). Those exist only because this demo case ships a
ground-truth mask. On a real patient there is no ground truth, so the column is
either blank or — worse — a stale demo number. Even when valid it is a
*research* metric; a clinician may read "0.9456" as confidence in this
patient's result.

**b. "Total tumour volume" conflates compartments.** 89% of the headline volume
is edema. Report the compartments with equal prominence, or lead with enhancing
+ necrotic core (the RANO-relevant components).

**c. No uncertainty on the volumes.** We know from validation that
volume relative error averages ~7.4%. Quoting `76.71 cm³` to four significant
figures implies a precision the method does not have.

**d. Eloquent-cortex distances are only in a figure.** The clinical note names
five at-risk regions but the actual millimetre distances live in a plot. Those
numbers should be in a table — that is the surgically actionable content.

**e. Laterality is stated once.** Given laterality errors are among the most
serious reporting errors — and this project shipped a left/right mirroring bug
in the 3D viewer — laterality deserves prominent, redundant statement.

---

## 3. What the report does *better* than a conventional one

Worth keeping and highlighting, not just fixing:

- **Multi-model comparison with explicit disagreement.** Conventional AI tools
  emit one mask. Showing three architectures plus where they diverge is a real
  contribution, now backed by measured calibration (ρ = 0.585, n = 56).
- **Automated eloquent-cortex proximity** with an explicit instruction that
  fMRI/DTI is required before surgical use — appropriately hedged.
- **Explainability page.** Rare in clinical reports; correctly framed as
  qualitative.
- **Volumetric quantification** — RANO 2.0 permits volumetrics as an option, so
  this is forward-looking, provided the 2-D measures are also given.

---

## 4. Recommended changes, in priority order

### P0 — correctness and safety
1. **Delete the "9–15×" claim.** Unverified; currently asserted to clinicians.
2. **Add an Impression** at the top of page 1 — 3–5 lines synthesising
   compartment volumes, location/laterality, eloquent risk, and consensus.
3. **Suppress Dice when no ground truth exists**, and label it as a research
   metric against a supplied reference when it does.
4. **Prominent status banner**: AI-generated, research use, requires clinician
   review — not just a footer.

### P1 — ACR completeness
5. Header block with **Indication**, **Technique**, **Comparison**, study date
   — printing "Not provided" is honest and shows the operator what is missing.
6. **Recommendations** as its own section.
7. **Authentication block**: software version, model versions, atlas version,
   and a `Reviewed by / Date` line for the responsible clinician.

### P2 — clinical fit
8. **2-D bidimensional measurement** alongside volume, for RANO comparability.
9. **Uncertainty on volumes** (±, from the measured ~7.4% relative error).
10. **Eloquent distances as a table**, not only a figure.
11. State that response assessment requires a prior study and is not attempted.

### P3 — design
12. Typographic hierarchy: impression > findings > supporting data.
13. Compartment volumes as a visual breakdown, not just a table row.
14. Consistent colour semantics (risk = red, informational = navy).

---

## 5. Honest positioning

Nothing here makes the tool unsafe as **decision support** — it is explicitly
not a diagnostic report, and the clinician remains responsible. But if the
claim is "a report a doctor could use", the missing Impression, indication and
comparison are the difference between a *results printout* and a *clinical
document*. The P0 and P1 items close most of that gap without new modelling
work.
