# CranioVision ML Scope

This repository stage focuses only on the machine learning core for 3D brain MRI tumor segmentation using BraTS-style NIfTI data.

## In Scope

- Dataset discovery and validation for 4 MRI modalities plus a segmentation mask
- Preprocessing and augmentation with MONAI
- Baseline 3D segmentation training and validation
- Config-driven model selection with a simple path to add more architectures
- Sliding-window inference for full 3D volumes
- Metric reporting, checkpointing, and prediction visualization
- Lightweight notebooks for exploration and preprocessing sanity checks

## Out of Scope For This Stage

- DICOM ingestion
- Atlas registration
- Uncertainty estimation
- Explainability tooling
- Vision-language report generation
- Frontend or application integration

## Current Assumptions

- Data is organized in case folders, following the pattern already visible in the repo notebook:
  - Example case folder: `BraTS-GLI-02105-105`
  - Example file names: `BraTS-GLI-02105-105-t1c.nii`, `BraTS-GLI-02105-105-seg.nii`
- Dataset paths must be set in config files before running training or notebooks.
- If a dataset variant uses different modality suffixes, update `configs/default.yaml` rather than editing code.
