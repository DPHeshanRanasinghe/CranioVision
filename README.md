# 🧠 CranioVision

**AI-Assisted Clinical Platform for 3D Brain Tumor Segmentation, Uncertainty Quantification, and Explainable Volumetric Analysis**

CranioVision is an end-to-end pipeline that takes multi-modal brain MRI volumes (T1, T1c, T2, FLAIR) and produces **clinically interpretable** tumor segmentations from four independent prediction sources — three deep learning models plus a consensus ensemble — letting the radiologist choose which output to trust. Each prediction is paired with per-voxel confidence estimates, model attention heatmaps, and per-region tumor volumes in cm³.

The project targets a real gap in clinical tooling: there is no accessible, open platform combining automated 3D tumor segmentation with multi-model second-opinion logic, uncertainty quantification, and XAI in one deployable system. Commercial systems (BrainLab, iPlan) are prohibitively expensive; open research prototypes (DSNet, TumorPrism3D) stop at single-model inference.

---

## ✨ Core Capabilities

| Capability | What it does | Why it matters clinically |
|---|---|---|
| **3-model segmentation** | Predicts tumor regions with Attention U-Net, SwinUNETR, and nnU-Net-style DynUNet | Three independent opinions — radiologist sees if models agree |
| **Weighted ensemble** | Combines all three via Dice-proportional soft voting | Provides a fourth "consensus" prediction with built-in confidence signal |
| **Volume quantification** | Per-region tumor volume in cm³ with 7.5% mean relative error vs. ground truth | Required for RANO tumor response assessment |
| **MC Dropout uncertainty** | Per-voxel confidence via 5–20 stochastic forward passes | Flags unreliable predictions for radiologist review |
| **Grad-CAM XAI** | Shows what the model attends to for each tumor class | Builds clinical trust; validates model is not looking at noise |
| **Model agreement map** | Highlights voxels where all 3 models agree vs. disagree | Direct visual signal of which regions warrant manual review |

---

## 📊 Final Results (BraTS 2024 small subset, 140 train / 30 val / 30 test)

### Per-model performance on test set (n=30)

| Model | Params | Val Dice | Test mean Dice | Edema | Enhancing | Necrotic |
|---|---|---|---|---|---|---|
| Attention U-Net | 23.6M | 0.7642 | 0.7308 | 0.858 | 0.676 | 0.658 |
| **SwinUNETR** ⭐ | 62.2M | 0.8219 | **0.7929** | **0.903** | **0.721** | **0.755** |
| nnU-Net DynUNet | 31.4M | 0.7562 | 0.6925 | 0.845 | 0.560 | 0.672 |
| **Ensemble (3-model)** | — | — | 0.7853 | 0.906 | 0.690 | 0.760 |

⭐ SwinUNETR is the strongest single model. The ensemble matches its performance closely (within 1%) and adds an agreement-based confidence signal not available from any single model.

### BraTS standard region Dice (test set)

| Region | Attention U-Net | SwinUNETR | nnU-Net | Ensemble |
|---|---|---|---|---|
| Whole Tumor (WT) | 0.862 | 0.912 | 0.870 | **0.914** |
| Tumor Core (TC) | 0.711 | 0.810 | 0.745 | 0.806 |
| Enhancing Tumor (ET) | 0.676 | 0.721 | 0.560 | 0.690 |

### Multi-model agreement (test set)

| Metric | Value |
|---|---|
| Mean unanimous voxel fraction | **99.33%** |
| Minimum unanimous voxel fraction (hardest case) | 98.39% |
| Mean volume relative error vs. ground truth | 7.47% |

The very high agreement across three architecturally diverse models (CNN, Transformer, Residual U-Net) is itself a clinically valuable signal. When all three models agree on a voxel, the prediction can be trusted deeply. The 0.7% of voxels where models disagree are typically tumor boundaries — exactly where radiologist review adds the most value.

### Why we present all 4 predictions, not just the ensemble

Our test-set evaluation shows the Dice-weighted ensemble achieves 0.7853 — statistically equivalent to SwinUNETR alone (0.7929). When one model is significantly stronger than its partners, classical ensembling provides minimal Dice gain. **We therefore use the ensemble not to maximize Dice, but as a consensus signal.** All four predictions (three models + ensemble) are presented in the clinical interface; the radiologist decides which output to use, with explicit visualization of where they agree and disagree.

---

## 🏗️ Architecture

```
                      MRI input (T1, T1c, T2, FLAIR)
                                    │
                                    ▼
              ┌─────────────────────────────────────────┐
              │        Preprocessing (MONAI)            │
              │  • Orientation → RAS                     │
              │  • Z-score normalization (per-modality)  │
              │  • Foreground crop                       │
              └─────────────────────────────────────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           ▼                        ▼                        ▼
  ┌────────────────┐      ┌─────────────────┐      ┌────────────────┐
  │ Attention U-Net│      │   SwinUNETR      │      │ nnU-Net-style  │
  │    (23.6M)     │      │    (62.2M)       │      │    (31.4M)     │
  └────────────────┘      └─────────────────┘      └────────────────┘
           │                        │                        │
           └────────────────────────┼────────────────────────┘
                                    ▼
              ┌─────────────────────────────────────────┐
              │   Dice-weighted soft voting (ensemble)  │
              └─────────────────────────────────────────┘
                                    │
       ┌────────────────────┬───────┴────────┬─────────────────────┐
       ▼                    ▼                ▼                     ▼
  ┌──────────┐      ┌────────────┐    ┌────────────┐      ┌──────────────┐
  │ Volumes  │      │ MC Dropout │    │  Grad-CAM  │      │  Agreement   │
  │  (cm³)   │      │ uncertainty│    │ explainer  │      │     map      │
  └──────────┘      └────────────┘    └────────────┘      └──────────────┘
                                    │
                                    ▼
              Radiologist interface — all 4 predictions shown,
              user selects which to trust per case
```

---

## 📂 Repository structure

```
CranioVision/
├── src/cranovision/
│   ├── config.py                     Paths + hyperparameters
│   ├── data/
│   │   ├── dataset.py                BraTS loader + train/val/test splits
│   │   └── transforms.py             MONAI preprocessing + augmentation
│   ├── models/
│   │   ├── attention_unet.py         MONAI AttentionUnet factory
│   │   ├── swin_unetr.py             MONAI SwinUNETR factory
│   │   └── nnunet_model.py           DynUNet with nnU-Net plan
│   ├── training/
│   │   ├── metrics.py                Dice + BraTS region metrics
│   │   └── trainer.py                Unified training loop
│   └── inference/
│       ├── predict.py                Single-model inference + volumes
│       ├── mc_dropout.py             Stochastic uncertainty
│       ├── grad_cam.py               3D patch-based Grad-CAM
│       └── ensemble.py               Weighted soft voting + agreement
├── notebooks/
│   ├── train_*.ipynb                 Training notebooks (3 models)
│   ├── inference_*.ipynb             Per-model + ensemble test eval
│   ├── *_demo.ipynb                  Per-model demo notebooks
│   ├── mc_dropout_viz.ipynb          Uncertainty visualization
│   ├── gradcam_viz.ipynb             XAI heatmap visualization
│   ├── single_model_demo.ipynb       Single-model pipeline demo
│   ├── full_demo.ipynb               Full 3-model ensemble pipeline
│   ├── inference_ensemble.ipynb      Ensemble test-set evaluation
│   └── comparison_report.ipynb       Side-by-side model comparison
├── tests/
│   └── test_inference.py             pytest suite (22/22 passing)
├── models/                           Checkpoints (gitignored)
├── outputs/                          Predictions, curves, JSON reports
├── data/                             BraTS data (gitignored)
└── requirements.txt
```

---

## 🚀 Getting started

### Prerequisites
- Python 3.11+
- CUDA 11.8+ GPU (inference: 4GB+; training: 16GB+)
- Access to BraTS 2024 dataset

### Installation

```bash
git clone https://github.com/DPHeshanRanasinghe/CranioVision.git
cd CranioVision
conda create -n cranovision python=3.11
conda activate cranovision
pip install -r requirements.txt
```

### Running the full pipeline

```bash
# Place all 3 trained checkpoints in models/
# Run the full 3-model ensemble pipeline on a single case
jupyter lab notebooks/full_demo.ipynb

# Or evaluate the ensemble across the entire test set
jupyter lab notebooks/inference_ensemble.ipynb
```

### Running tests

```bash
pytest -m "not slow"   # 22 fast tests (~1 min)
pytest                 # full suite incl. GPU inference (~5 min)
```

---

## 🎯 Training strategy

CranioVision uses a **local-code / cloud-train** workflow:

1. All code lives in the repo — models, training notebooks, inference scripts
2. Training runs on Kaggle T4 GPU (16GB free tier) via a small launcher notebook that clones the repo, installs dependencies, and executes the in-repo training notebook
3. Trained checkpoints are downloaded back to local `models/` for inference

Each architecture lives on its own feature branch (`feature/attention-unet`, `feature/SwinUNETR`, `feature/nnU-Net`) and merges into `dev` for ensemble work.

---

## 🔬 Design decisions

**Three architecturally diverse models, not three identical ones.** Attention U-Net (CNN with attention gates), SwinUNETR (Transformer encoder), and DynUNet (residual U-Net) make different errors on different cases. This diversity is what makes the agreement signal meaningful — three identical models would always agree.

**Dice-proportional ensemble weights, not equal voting.** Each model's vote is weighted by its validation Dice (SwinUNETR ≈ 0.35, Attention U-Net ≈ 0.33, nnU-Net ≈ 0.32). Equal voting would give the weakest model too much influence.

**Ensemble for confidence, not raw performance.** When one model is significantly stronger than its partners, classical ensembling provides minimal Dice gain. We use the ensemble's agreement statistic as a clinical confidence indicator — when all three models agree on a voxel, that voxel can be trusted; disagreement flags review.

**MC Dropout over deep-ensemble uncertainty.** Single trained model, multiple stochastic passes. Cheaper than training a second ensemble and equally informative for boundary uncertainty.

**Patch-based Grad-CAM.** Full-volume Grad-CAM requires ~7GB of gradient storage. We locate the tumor via a cheap forward pass, crop a 128³ patch around it, compute Grad-CAM there, and stitch back. Fits in 2GB.

**Memory-efficient inference for 4GB GPUs.** Inference loads one model at a time. Predictions are softmax-saved to disk, models are freed, and final voting happens on CPU. The full 3-model ensemble runs on a GTX 1650 4GB without memory pressure.

---

## 📖 Models

### Attention U-Net (Oktay et al., 2018)
Standard U-Net with attention gates at each skip connection. The gates suppress irrelevant features (skull, healthy tissue) and emphasize tumor regions. Strong baseline for diffuse structures like edema. High recall on whole tumor.

### SwinUNETR (Hatamizadeh et al., 2022)
Swin Transformer encoder paired with a CNN decoder. The transformer's self-attention captures global context — especially useful for small enhancing-tumor cores that conventional CNNs struggle with. Consistently tops BraTS leaderboards. **Strongest single model in our pipeline.**

### nnU-Net-style DynUNet (Isensee et al., 2021)
Residual-block encoder/decoder with instance normalization. Uses nnU-Net's architectural plan for brain MRI (6-level encoder, filter schedule 32→320, deep supervision). Provides architectural diversity to the ensemble — different inductive biases, different failure modes.

---

## 🛠️ Tools used

| Domain | Tool |
|---|---|
| Deep learning | PyTorch 2.5 |
| Medical imaging primitives | MONAI 1.5.2 |
| Volume IO | nibabel, SimpleITK |
| Visualization | matplotlib, Plotly (Mesh3d), scikit-image |
| Experiment tracking | JSON histories + matplotlib curves |
| Testing | pytest |
| Version control | Git + GitHub (multi-branch workflow) |
| Compute | Local GTX 1650 (dev), Kaggle T4 (training) |

---

## 📅 Project status

### Phase 1 — Foundation + Training (✅ complete)
- [x] Data pipeline (loaders, splits, transforms)
- [x] Unified trainer + metrics + sliding-window val
- [x] Three model architectures coded
- [x] Attention U-Net trained (0.764 val / 0.731 test)
- [x] SwinUNETR trained (0.822 val / 0.793 test)
- [x] nnU-Net DynUNet trained (0.756 val / 0.693 test)

### Phase 2 — Inference + XAI + Ensemble (✅ complete)
- [x] Per-model inference + volume quantification
- [x] MC Dropout uncertainty
- [x] Patch-based Grad-CAM explainability
- [x] Weighted soft-voting ensemble + agreement analysis
- [x] Memory-efficient pipeline for 4GB GPUs
- [x] Full demo notebook (3-model ensemble + uncertainty + XAI)
- [x] Ensemble test-set evaluation (0.7853 mean Dice, 99.33% agreement)
- [x] 22/22 pytest test suite passing

### Phase 3 — Clinical Intelligence (in progress)
- [ ] Atlas registration (MNI152) — anatomical tumor location
- [ ] Per-model XAI extension (Grad-CAM for SwinUNETR + nnU-Net)
- [ ] Consensus attention map across all 3 models
- [ ] Structured PDF clinical report (radiologist workflow)

### Phase 4 — Clinical Platform (planned)
- [ ] FastAPI backend
- [ ] React + VTK.js frontend (3D viewer with model selector)
- [ ] DICOM import/export
- [ ] Hospital workflow integration

---

## 👤 Author

**Heshan Ranasinghe**
University of Moratuwa, Faculty of Engineering
Communication Network Engineering, Batch 23

---

## 📚 Key references

1. Oktay et al. (2018). *Attention U-Net: Learning Where to Look for the Pancreas.* arXiv:1804.03999
2. Hatamizadeh et al. (2022). *Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images.* arXiv:2201.01266
3. Isensee et al. (2021). *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.* Nature Methods 18(2):203-211
4. Gal & Ghahramani (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.* ICML 2016
5. Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017
6. Baid et al. (2021). *The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification.* arXiv:2107.02314

---

## 📄 License

This project is for academic research purposes. Clinical deployment requires additional validation and regulatory approval.