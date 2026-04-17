# CranioVision

AI-assisted clinical pipeline for 3D brain tumor MRI segmentation, uncertainty quantification, and explainability.

## Overview

CranioVision is a medical imaging pipeline that takes 3D MRI brain scans, runs them through an ensemble of deep learning segmentation models, quantifies prediction uncertainty, generates visual explanations, and reports tumor volume and location — all designed to assist neuroradiologists and surgeons in clinical decision-making.

## Architecture (Phase 1 + 2)

```
MRI upload (.nii.gz)
   ↓
Preprocessing (MONAI transforms, 4-channel stacking)
   ↓
Parallel ensemble segmentation
   ├── Attention U-Net
   ├── SwinUNETR
   └── nnU-Net
   ↓
Soft voting → final segmentation mask
   ↓
Monte Carlo Dropout (uncertainty map)
   ↓
Grad-CAM (XAI heatmap)
   ↓
Clinical metrics (volume, RANO, region)
   ↓
Report + interactive 3D visualization
```

## Dataset

BraTS 2024 — Glioma sub-region segmentation (edema, enhancing tumor, necrotic core) from 4-modality MRI (T1, T1c, T2, FLAIR).

## Repo Structure

```
CranioVision/
├── src/cranovision/      # Python package
│   ├── config.py         # Paths & hyperparameters
│   ├── data/             # Dataset loaders, transforms
│   ├── models/           # Network definitions
│   ├── training/         # Training loops, metrics
│   ├── inference/        # Ensemble, MC Dropout, Grad-CAM
│   └── utils/            # Helpers
├── notebooks/            # Exploration & training notebooks
├── configs/              # YAML configs
├── data/                 # Raw & processed data (gitignored)
├── models/               # Saved checkpoints (gitignored)
└── outputs/              # Figures, reports, logs (gitignored)
```

## Branch Workflow

- `main` — Production-ready
- `dev` — Integration branch
- `feature/attention-unet` — Attention U-Net training
- `feature/SwinUNETR` — SwinUNETR training
- `feature/nnU-Net` — nnU-Net training

## Environment

Local conda env: `ml_env_fixed` (Python 3.11, PyTorch 2.5.1 + CUDA 11.8, MONAI 1.5.2)

## Status

🚧 **Under active development** — Phase 1 + 2 scope.

## Author

DP Heshan Ranasinghe — Faculty of Engineering, University of Moratuwa
