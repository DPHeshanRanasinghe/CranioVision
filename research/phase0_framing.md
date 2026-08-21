# Phase 0 — Framing

Status: **draft, thesis is a claim to test, not yet confirmed.** See
`PROJECT_NOTES.md` for the open issue this framing depends on (the unverified
9-15x SBR number) before treating any RQ below as answered.

## Working title (draft)

*"Not All CNNs Are Equal: Diagnosing Architecture-Dependent Grad-CAM Reliability in
Multi-Model 3D Brain Tumor Segmentation"*

(Working — will likely change once RQ1 is actually answered by the quantification
script, since the title currently presupposes the nnU-Net-is-also-unreliable finding
holds up.)

## Thesis (hypothesis to test, not asserted fact)

We hypothesize that Grad-CAM reliability in 3D tumor segmentation is **not**
determined by the coarse conv-vs-attention distinction the current literature
emphasizes, but by finer decoder-architecture properties — specifically, that
**explicit spatial attention gating (as in Attention U-Net) produces measurably more
faithful and spatially concentrated Grad-CAM heatmaps than either a transformer
encoder (SwinUNETR) or a plain residual/deep-supervision decoder (nnU-Net-style
DynUNet), even though the latter two are architecturally very different from each
other.**

If instead nnU-Net turns out to be Grad-CAM-reliable once rigorously measured (i.e.
the original "unreliable" observation doesn't replicate under a proper metric and
multiple test cases), that is still a finding: it would mean the original informal
observation was measurement noise or an implementation artifact (e.g. target-layer
selection), and the paper's contribution shifts to "how not to misdiagnose Grad-CAM
reliability in multi-architecture segmentation pipelines."

## Research questions

**RQ1 (diagnose).** Does Grad-CAM heatmap quality — measured by a faithfulness/
localization metric, not just informal visual inspection — differ significantly
across Attention U-Net, SwinUNETR, and nnU-Net (DynUNet) when all three are trained
on the same task, data, and label space? *(Re-derives and rigorously tests the
existing 9-15x SBR claim — see open issue in PROJECT_NOTES.md.)*

**RQ2 (isolate).** Is any observed gap attributable to attention-gating specifically
(Attention U-Net's differentiator from both other architectures), rather than to a
confound such as parameter count, validation Dice, or target-layer depth/receptive
field? *(Controls: pick target layers at matched depth; check correlation between
Grad-CAM quality and each model's raw segmentation Dice — a model that's simply
"better" could produce cleaner activations for reasons unrelated to gating.)*

**RQ3 (fix / practical implication).** Does a single shared surrogate explainer
(always running Grad-CAM through the reliable architecture, regardless of which
model's prediction is being explained — CranioVision's current production design)
preserve per-model explanatory value, or does it systematically mis-explain
predictions the surrogate model itself would not have made? *(i.e., quantify the
cost of the surrogate-explainer shortcut, not just assert it's fine.)*

## Nearest neighbors (see `related_work.md` for full positioning)

- Wu et al. 2024, arXiv:2404.01415 — Grad-CAM faithfulness gap is a *known*
  phenomenon for ViT, but studied on 2D classification, not 3D medical segmentation,
  and frames the gap as CNN-vs-transformer rather than testing whether it holds
  *within* the CNN family.
- Panboonyuen 2026, arXiv:2601.12826 — same CNN-good/ViT-bad story, 2D lung CT
  classification, 5 architectures none of which are U-Net-style segmentation decoders.
- Chen et al. 2023 (AME-CAM), arXiv:2306.14505 — CAM-based weakly supervised
  segmentation on BraTS, but proposes a *new* CAM variant rather than diagnosing
  *why* standard Grad-CAM reliability differs across architectures.

None of the located work tests Grad-CAM reliability across CNN-decoder variants
(attention-gated vs. plain residual) in 3D multi-modal medical segmentation, or
quantifies the practical cost of a shared-surrogate-explainer design in a
multi-model ensemble pipeline. That is the gap this paper occupies, contingent on
RQ1 actually replicating.

## What would kill this framing

- If the quantification script (RQ1) shows nnU-Net's Grad-CAM is actually comparable
  to Attention U-Net's, the "not all CNNs are equal" thesis collapses. Don't force it;
  pivot to the ensemble-agreement-as-confidence-signal angle instead (README's other
  documented empirical result) or report the negative/correcting finding honestly.
- If Grad-CAM quality turns out to correlate strongly with raw Dice (RQ2 confound),
  the story becomes "better segmentation → better Grad-CAM as a side effect,"
  which is a weaker, less mechanistic claim — still publishable as a diagnosis, but
  changes the framing from "attention gating helps XAI" to "XAI quality tracks
  segmentation quality, beware conflating them."
