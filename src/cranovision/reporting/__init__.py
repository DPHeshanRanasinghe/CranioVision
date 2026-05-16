"""
CranioVision — Clinical PDF report generation.

Public API:
  generate_clinical_report(...)  -> creates a 4-page clinical PDF

The report follows a radiologist-friendly white-background layout while
preserving meaningful color coding in medical figures (segmentations,
Grad-CAM heatmaps, eloquent-risk badges).
"""
from .clinical_report import generate_clinical_report

__all__ = ["generate_clinical_report"]