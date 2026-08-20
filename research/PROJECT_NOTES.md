# Project Notes — decision log

Running record of research decisions and why, per the Deep Learning Research Playbook
(`DL_Research_Playbook.md`), Phase 6: "keep a decision log... record the non-obvious
calls... a teammate or reviewer *will* ask." Newest entries at the bottom.

---

## 2026-08-03 — Paper framing decision

**Decision:** Target a workshop-style paper (e.g. a MICCAI-adjacent or clinical-ML
workshop, ~4 pages) centered on a **diagnosis-style ML finding**, not a systems paper.

**Why:** The playbook is explicit that novelty must be a *transferable idea*, not a
domain application ("technique X applied to domain Y" reads as engineering, not
research, to an ML reviewer). CranioVision's README currently frames the contribution
as "first accessible platform combining multi-model segmentation + atlas anatomy +
eloquent cortex + XAI" — real engineering value, but not a novel *method* or *finding*
on its own.

The strongest candidate for an actual research contribution already exists in the
repo's own design decisions: Grad-CAM reliability varies sharply across the three
segmentation architectures (Attention U-Net: reliable; SwinUNETR, nnU-Net/DynUNet:
unreliable), which is why the pipeline uses a single surrogate explainer
(`pipeline.py` `XAI_EXPLAINER_NAME`). This is a cheap-by-construction contribution
(no new training runs required) and fits a diagnose → isolate → fix arc.

**How to apply:** The systems/clinical-utility material (multi-model consensus, atlas
registration, eloquent-cortex risk, PDF reporting) becomes Introduction/motivation and
a secondary contribution — not the headline claim. See `phase0_framing.md`.

---

## 2026-08-03 — Open issue: the "9-15x signal-to-background ratio" claim is unverified

**Finding:** `README.md`, `src/cranovision/pipeline.py` (module docstring), and
`src/cranovision/reporting/clinical_report.py:610` all assert that Attention U-Net's
Grad-CAM heatmaps achieve a "9-15x signal-to-background ratio" versus "unreliable"
heatmaps from SwinUNETR/nnU-Net. There is **no computation script for this number
anywhere in the tracked repo**, and the untracked local directories that might hold it
(`notebooks/`, `outputs/`) are not present in this checkout either.

**Why this matters:** Per the playbook's "before trusting a result" checklist, a
number with no reproducible computation is not a citable result — it's an anecdote.
This is exactly the kind of claim a reviewer will ask to see the methodology for.

**Status:** Open. Asked the user (2026-08-03) where/how this was originally computed.
Until answered, treat "9-15x" as a **prior informal observation to be re-derived**,
not a result to cite as-is. Wrote `src/cranovision/analysis/gradcam_reliability.py`
to (re)compute it rigorously (multiple test cases, per-class, mean ± std) — needs to
be run by the user against local checkpoints + BraTS test data, since this environment
has no models/data/GPU (`models/`, `data/`, `outputs/`, `atlas_data/` are all absent
from this checkout per `.gitignore`).

**How to apply:** Don't cite "9-15x" in the paper draft until it's been reproduced by
the script above (or the user points to where it was originally measured and that
methodology can be described/cited).

---

## 2026-08-03 — Literature positioning: the real novelty gap

**Finding (Phase 1 lit scan):** The literature already well-establishes that Grad-CAM
degrades on Vision Transformers specifically — e.g. Wu et al. 2024 ("On the
Faithfulness of Vision Transformer Explanations," arXiv:2404.01415) and Panboonyuen
2026 ("Seeing Isn't Always Believing," arXiv:2601.12826, tested ResNet-50/101,
DenseNet-161, EfficientNet-B0, ViT-Base on lung CT) both report Grad-CAM working
reasonably on CNNs but degrading sharply on ViT-style attention. That's the field's
dominant CNN-good/ViT-bad narrative — see `related_work.md`.

**Why this matters:** CranioVision's own design notes claim nnU-Net (DynUNet) — a
**CNN**, not a transformer — is *also* Grad-CAM-unreliable, same as SwinUNETR. If that
holds up under rigorous quantification, it directly contradicts the simple
CNN-vs-transformer story and is the genuine, defensible novelty: something about
decoder architecture *within* the CNN family (e.g. attention gates vs. plain residual
blocks, deep supervision, instance norm) governs Grad-CAM reliability, not
conv-vs-attention alone.

**How to apply:** RQ1/RQ2 in `phase0_framing.md` are written to test this directly.
If the quantification script instead shows nnU-Net is *actually* fine and the original
"unreliable" claim was an artifact (bad target-layer selection, undertrained checkpoint,
etc.), that is itself the finding — report it, don't force the original narrative.

---

## 2026-08-03 — Wrote `src/cranovision/analysis/gradcam_reliability.py`

**What:** Reproducible measurement script for RQ1 — computes three metrics per case/
model/class: `signal_background_ratio` (reproduces the informal "9-15x" number with a
defined formula), `concentration_ratio` (spatial precision), and `deletion_aopc`
(perturbation-based faithfulness — the one that should actually anchor paper claims,
since the first two are correlational, not causal). Runs all 3 models sequentially
(same memory discipline as `pipeline.py`), writes per-case raw values plus paired
Wilcoxon signed-rank comparisons (Attention U-Net vs. each other model) to
`OUTPUTS_DIR/gradcam_reliability_study.json`.

**Status:** Written, syntax-checked, **not yet run** — this environment has no
`models/`, `data/`, or GPU (see earlier open-issue entry). Needs the user to run it
locally or on Kaggle against trained checkpoints + the BraTS test split, then hand the
output JSON back for Results-section analysis.

**Dependency management:** user asked (2026-08-03) to use a project-local venv
(`.venv/` at repo root, `python -m venv .venv`) rather than installing packages
globally — applies to any future `pip install` in this repo, including whatever's
needed to run this script if not using the local `ml_env_fixed` conda env.

---

## 2026-08-03 — Compute/provenance constraints clarified

**Findings:**
- User is **not the original author** of this repo — it was handed to them to
  research. They do not know where the "9-15x" claim originally came from.
- User has **no local GPU**; heavy runs (retraining, the reliability script) will
  happen on **Google Colab**, not the Kaggle T4 the README's "Tools Used" table
  implies the original author used.
- Unknown (blocking): whether trained checkpoints (`attention_unet_best.pth`,
  `swin_unetr_best.pth`, `nnunet_best.pth`) exist anywhere accessible to the user, or
  whether all 3 models need to be retrained from scratch. This is the single biggest
  fork in near-term effort — asked the user directly (2026-08-03), answer pending.
- Unknown (blocking): whether the user already has BraTS 2024 data access (Synapse
  registration/data-use agreement) or needs to apply for it.

**How to apply:** Don't assume either branch (have-checkpoints vs. must-retrain) until
confirmed. If retraining is needed, that itself becomes a Phase 6 reproducibility
exercise worth documenting, but changes the paper timeline substantially.

---

## 2026-08-20 — Research track pull request submission

**Action:** Prepared formal pull request for the `docs/research` branch targeting `main`.
**Details:** Includes Phase 0–1 research framing (`phase0_framing.md`), literature review (`related_work.md`), decision log (`PROJECT_NOTES.md`), research methodology guide (`DL_Research_Playbook.md`), and the reproducible RQ1 evaluation script (`gradcam_reliability.py`).

