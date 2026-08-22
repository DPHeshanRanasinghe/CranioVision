# Phase 1 — Related work

Verification status per playbook Phase 1 step 2 ("verify every reference against the
actual paper, not a search snippet"): all entries below were checked by fetching the
arXiv/publisher page directly, not taken from search-result summaries alone. Confidence
tier noted per entry. This list is a starting scan, not exhaustive — expand before
submission.

## Established building blocks (cite confidently)

- **Selvaraju et al., 2017. "Grad-CAM: Visual Explanations from Deep Networks via
  Gradient-based Localization." ICCV 2017 / arXiv:1610.02391.**
  Tier: high (canonical, already in CranioVision's README references). The base
  method CranioVision's `grad_cam.py` implements (3D-extended, patch-based).

- **Oktay et al., 2018. "Attention U-Net: Learning Where to Look for the Pancreas."**
  arXiv:1804.03999. Tier: high (already cited). Establishes attention-gated skip
  connections — the architectural feature RQ2 isolates.

- **Hatamizadeh et al., 2022. "Swin UNETR."** arXiv:2201.01266. Tier: high (already
  cited). Transformer-encoder segmentation architecture under test.

- **Isensee et al., 2021. "nnU-Net."** Nature Methods 18(2):203-211. Tier: high
  (already cited). CranioVision's DynUNet-style third model follows nnU-Net's
  architectural plan.

- **Vinogradova et al., 2020. "Towards Interpretable Semantic Segmentation via
  Gradient-weighted Class Activation Mapping" (Seg-Grad-CAM).** AAAI 2020 workshop.
  Tier: medium — not yet directly verified via fetch, but this is the standard
  citation for extending Grad-CAM from classification to per-pixel/per-region
  segmentation outputs, which is what `compute_grad_cam`'s `target_mask` parameter
  does. **Action: verify before citing.**

## Novel synthesis — what must be demonstrated, not just cited

- **Wu, Kang, Tang, Hong, Yan, 2024. "On the Faithfulness of Vision Transformer
  Explanations." arXiv:2404.01415.** Tier: high (verified via direct fetch,
  2026-08-03). Introduces SaCo, a faithfulness metric showing standard attribution
  metrics "struggle to differentiate advanced explanation methods from Random
  Attribution" for ViT explanations, and that gradient + multi-layer aggregation
  markedly improves faithfulness over naive single-layer attention.
  **Differentiation:** studies 2D natural-image ViT classifiers; does not touch 3D
  medical segmentation or compare against non-transformer CNN *decoder variants*
  (e.g. attention-gated vs. plain residual). CranioVision's RQ1/RQ2 test whether the
  same faithfulness gap CranioVision observes is actually transformer-specific (as
  this paper's framing implies) or also present in a non-transformer CNN (nnU-Net).

- **Panboonyuen, 2026. "Seeing Isn't Always Believing: Analysis of Grad-CAM
  Faithfulness and Localization Reliability in Lung Cancer CT Classification."**
  arXiv:2601.12826 (Jan 2026). Tier: high (verified via direct fetch, 2026-08-03).
  Tests ResNet-50/101, DenseNet-161, EfficientNet-B0, ViT-Base on lung CT
  classification; finds Grad-CAM "effectively highlights salient tumor regions in
  most convolutional networks" but "interpretive fidelity significantly degrades for
  Vision Transformer models due to non-local attention behavior."
  **Differentiation:** reinforces the field's CNN-good/ViT-bad framing on a
  classification task; the four CNNs tested are all plain feed-forward classifiers,
  none are segmentation decoders with attention gating or deep supervision. If
  CranioVision's nnU-Net result holds, it's a direct counter-example to "most
  convolutional networks" being uniformly reliable — worth explicitly citing this
  paper's claim and contrasting it.

- **Chen, Hu, Shi, Ho, 2023. "AME-CAM: Attentive Multiple-Exit CAM for Weakly
  Supervised Segmentation on MRI Brain Tumor." arXiv:2306.14505.** Tier: high
  (verified via direct fetch, 2026-08-03). Proposes a new CAM variant combining
  multi-scale activations to fix CAM's low-resolution problem in BraTS weakly
  supervised segmentation; shows improved Dice vs. competing weak-supervision
  methods.
  **Differentiation:** proposes a fix without diagnosing which existing architectures
  the standard method already fails on or why — it's solution-first. CranioVision is
  diagnosis-first: characterizing *when* standard Grad-CAM already works well enough
  (Attention U-Net) vs. when it doesn't, as a cheaper prerequisite to deciding whether
  a fix like AME-CAM is even necessary for a given architecture.

## Not yet verified / to chase before submission

- A quantitative comparison paper surfaced by search ("A Quantitative Evaluation of
  Gradient-Based Visual Explainability Methods Across Convolutional and
  Transformer-Based Vision Models," MDPI Electronics) — relevant for methodology
  (which faithfulness metrics to adopt: AOPC, Max-Sensitivity, insertion AUC, point
  localization accuracy) but not yet fetched/verified. Useful for RQ1's metric choice.
- Search for nnU-Net-specific interpretability work directly (little found so far —
  may itself be evidence of a gap, or may mean the search terms need refinement).
- Search for "signal-to-background ratio" or similar SNR-style Grad-CAM metrics
  specifically, to see if CranioVision's original metric has precedent or needs
  justifying as a novel evaluation choice.

## Terminology traps (playbook Phase 1 pitfall)

"Reliable"/"unreliable" as used in CranioVision's existing docstrings is informal —
no citation, no defined metric, no statistical test. Before the paper can claim
anything, this needs to become one of the standard faithfulness metrics above
(localization accuracy, perturbation-based AOPC, etc.), not the ad hoc
signal-to-background ratio, unless SBR itself can be justified/precedented.
