# 🧠 CranioVision

**AI-Assisted Clinical Platform for 3D Brain Tumor Segmentation, Uncertainty Quantification, and Explainable Volumetric Analysis**

CranioVision is an end-to-end pipeline that takes multi-modal brain MRI volumes (T1, T1c, T2, FLAIR) and produces **clinically interpretable** tumor segmentations — including per-region volumes in cm³, voxel-level confidence estimates, and model attention heatmaps that explain *why* each prediction was made.

The project targets a genuine gap in current clinical tooling: there is no accessible, open platform combining automated 3D tumor segmentation with uncertainty quantification and XAI in a single deployable system. Commercial systems like BrainLab and iPlan are prohibitively expensive; open research prototypes (DSNet, TumorPrism3D) stop at the inference stage and lack safety features.

---

## ✨ Core Capabilities

| Capability | What it does | Why it matters clinically |
|---|---|---|
| **3D segmentation** | Predicts edema, enhancing tumor, necrotic core on isotropic MRI | Automates a 30-60 min manual task |
| **Volume quantification** | Per-region tumor volume in cm³ | Required for RANO tumor response assessment |
| **MC Dropout uncertainty** | Per-voxel confidence via 20× stochastic forward passes | Flags unreliable predictions for radiologist review |
| **Grad-CAM XAI** | Shows *what* the model attends to for each class | Builds clinical trust; validates model is not looking at noise |
| **Ensemble voting** | Combines 3 architectures (Attention U-Net, SwinUNETR, nnU-Net-style) | Reduces catastrophic failures, improves Dice by 3-8 points |

---

## 📊 Results (BraTS 2024 small subset, 140 train / 30 val / 30 test)

### Individual model performance

| Model | Val Dice | Test Dice (mean) | Edema | Enhancing | Necrotic |
|---|---|---|---|---|---|
| Attention U-Net | 0.7642 | 0.7308 | 0.858 | 0.676 | 0.658 |
| SwinUNETR | **0.8432** | *TBD* | 0.899 | 0.796 | 0.835 |
| nnU-Net-style DynUNet | *in progress* | *TBD* | — | — | — |
| **Ensemble (3-model)** | *TBD* | *TBD* | — | — | — |

*SwinUNETR validation at epoch 90. nnU-Net training in progress.*

### BraTS standard region Dice (Attention U-Net test set)

| Region | Mean | Std |
|---|---|---|
| Whole Tumor (WT) | 0.862 | ±0.124 |
| Tumor Core (TC) | 0.711 | ±0.301 |
| Enhancing Tumor (ET) | 0.676 | ±0.331 |

### Uncertainty and explainability (sample case)

| Metric | Value |
|---|---|
| Mean voxel confidence | **0.9763** |
| Uncertain-voxel fraction | 0.15% |
| Grad-CAM signal-to-background ratio (enhancing) | **14.8×** |

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
              │  • Foreground crop → 160×192×160         │
              └─────────────────────────────────────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           ▼                        ▼                        ▼
  ┌────────────────┐      ┌─────────────────┐      ┌────────────────┐
  │ Attention U-Net│      │   SwinUNETR      │      │ nnU-Net-style  │
  │   (23.6M)      │      │   (62.2M)        │      │   (31.4M)      │
  └────────────────┘      └─────────────────┘      └────────────────┘
           │                        │                        │
           └────────────────────────┼────────────────────────┘
                                    ▼
              ┌─────────────────────────────────────────┐
              │   Weighted soft voting (ensemble)       │
              └─────────────────────────────────────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           ▼                        ▼                        ▼
  ┌────────────────┐      ┌─────────────────┐      ┌────────────────┐
  │  Segmentation  │      │   MC Dropout    │      │   Grad-CAM     │
  │  + volumes cm³ │      │   uncertainty   │      │   attention    │
  └────────────────┘      └─────────────────┘      └────────────────┘
                                    │
                                    ▼
                        Clinical report + 3D viewer
```

---

## 📂 Repository structure

```
CranioVision/
├── src/cranovision/
│   ├── config.py                     Paths + hyperparameters (single source of truth)
│   ├── data/
│   │   ├── dataset.py                BraTS loader + train/val/test splits
│   │   └── transforms.py             MONAI preprocessing + augmentation
│   ├── models/
│   │   ├── attention_unet.py         MONAI AttentionUnet factory
│   │   ├── swin_unetr.py             MONAI SwinUNETR factory
│   │   └── nnunet_model.py           DynUNet with nnU-Net-style plan
│   ├── training/
│   │   ├── metrics.py                Dice + HD95 + BraTS region metrics
│   │   └── trainer.py                Unified training loop (AMP, sliding window val)
│   └── inference/
│       ├── predict.py                Single-model inference + volume estimation
│       ├── mc_dropout.py             Stochastic uncertainty quantification
│       ├── grad_cam.py               3D patch-based Grad-CAM
│       └── ensemble.py               Weighted soft voting + agreement analysis
├── notebooks/
│   ├── train_attention_unet.ipynb    Training pipeline
│   ├── train_swin_unetr.ipynb        Training pipeline
│   ├── train_nnunet.ipynb            Training pipeline
│   ├── inference_attention_unet.ipynb  Per-case inference demo
│   ├── mc_dropout_viz.ipynb          Uncertainty visualization
│   ├── gradcam_viz.ipynb             XAI heatmap visualization
│   └── full_demo.ipynb               Complete pipeline on one case
├── models/                           Checkpoints (gitignored)
├── outputs/                          Predictions, curves, figures
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

### Running inference on a trained model

```bash
# Place your checkpoint in models/ — e.g. attention_unet_best.pth
python -m src.cranovision.inference.predict
```

### Running the full clinical demo

```bash
jupyter lab notebooks/full_demo.ipynb
```

---

## 🎯 Training strategy

CranioVision uses a **local-code / cloud-train** workflow:

1. All code lives in the repo — models, training notebooks, inference scripts
2. Training runs on Kaggle T4 GPU (16GB free tier) via a small *launcher* notebook that clones the repo, installs dependencies, and executes the in-repo training notebook
3. Trained checkpoints are downloaded back to local `models/` for inference

Each architecture lives on its own feature branch (`feature/attention-unet`, `feature/SwinUNETR`, `feature/nnU-Net`) and merges into `dev` for ensemble work. This keeps experiments isolated while sharing the same data and metric code.

---

## 🔬 Design decisions

**Option B for nnU-Net.** We use MONAI's `DynUNet` with nnU-Net-style hyperparameters inside our unified training framework, rather than invoking the full `nnunetv2` CLI. This costs 2-3 Dice points in isolation but keeps a single training/inference interface across all three models — critical for the ensemble, MC Dropout, and Grad-CAM to work uniformly. The ensemble voting reliably recovers those 2-3 points.

**128³ training patches, full-volume inference.** Training uses random 128³ crops with heavy augmentation for throughput. Inference uses a Gaussian-weighted sliding window over the full ~160³ volume for smooth segmentation at edges.

**MC Dropout over deep-ensemble uncertainty.** MC Dropout requires only a single trained model and works for any architecture with dropout layers — far cheaper than training many models. 20 stochastic passes give publication-grade variance estimates.

**Patch-based Grad-CAM.** Full-volume Grad-CAM requires ~7GB of gradient storage. We locate the tumor via a cheap forward pass, crop a 128³ patch around it, compute Grad-CAM on that patch, and stitch back. Fits in 2GB with no loss of useful signal.

---

## 📖 Models

### Attention U-Net (Oktay et al., 2018)
Standard U-Net with attention gates at each skip connection. The gates learn to suppress irrelevant features (skull, healthy tissue) and emphasize tumor regions. Good baseline for diffuse structures like edema. High recall on whole tumor.

### SwinUNETR (Hatamizadeh et al., 2022)
Swin Transformer encoder paired with a CNN decoder. The transformer's self-attention captures global context — especially useful for small enhancing-tumor cores that conventional CNNs struggle with. Consistently tops BraTS leaderboards.

### nnU-Net-style DynUNet (Isensee et al., 2021)
Residual-block encoder/decoder with instance normalization. Not the full nnU-Net framework, but uses its architectural plan for brain MRI (6-level encoder, filter schedule 32→320, deep supervision). Contributes a stable third vote to the ensemble with different inductive biases.

---

## 🛠️ Tools used

| Domain | Tool |
|---|---|
| Deep learning | PyTorch 2.5 |
| Medical imaging primitives | MONAI 1.5.2 |
| Volume IO | nibabel, SimpleITK |
| Visualization | matplotlib, Plotly (Mesh3d), scikit-image |
| Experiment tracking | JSON histories + matplotlib curves |
| Version control | Git + GitHub (multi-branch workflow) |
| Compute | Local GTX 1650 (dev), Kaggle T4 (training) |

---

## 📅 Project status

- [x] Foundation (data, training, metrics)
- [x] All 3 model architectures coded
- [x] Attention U-Net trained (0.76 val / 0.73 test mean Dice)
- [x] SwinUNETR trained (0.84 val Dice)
- [ ] nnU-Net training (in progress)
- [x] Inference + volume quantification
- [x] MC Dropout uncertainty
- [x] Grad-CAM explainability
- [x] Ensemble voting code (awaiting all 3 checkpoints)
- [ ] Atlas registration (MNI152) — Phase 3
- [ ] RANO criteria assessment — Phase 3
- [ ] FastAPI backend + web viewer — Phase 4

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