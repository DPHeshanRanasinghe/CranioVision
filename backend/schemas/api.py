"""
CranioVision — Pydantic schemas for backend API.

All request bodies and response payloads have typed schemas so the Next.js
frontend can consume them with full TypeScript type safety.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# JOB LIFECYCLE
# -----------------------------------------------------------------------------

class CreateJobResponse(BaseModel):
    """Returned by POST /upload after a folder is accepted."""
    job_id: str = Field(..., description="Unique job identifier (UUID)")
    case_id: str = Field(..., description="BraTS case ID parsed from folder name")
    cached_registration: bool = Field(
        ...,
        description="True if atlas registration is pre-cached (faster pipeline)",
    )
    status: str = Field("queued", description="Initial status: 'queued'")


class JobProgress(BaseModel):
    """One progress update event sent over SSE."""
    stage: str = Field(..., description="Current stage name")
    percent: int = Field(..., ge=0, le=100, description="Completion percentage")
    message: str = Field(..., description="Human-readable status")
    timestamp: float = Field(..., description="Unix timestamp")


class JobStatus(BaseModel):
    """Returned by GET /jobs/{id}/status — non-streaming snapshot."""
    job_id: str
    case_id: str
    state: str = Field(
        ..., description="One of: queued | running | done | failed"
    )
    latest_progress: Optional[JobProgress] = None
    error: Optional[str] = None
    elapsed_seconds: Optional[float] = None


# -----------------------------------------------------------------------------
# ANALYSIS RESULT
# -----------------------------------------------------------------------------

class VolumeBreakdown(BaseModel):
    """Per-region tumor volumes in cm^3."""
    edema: float
    enhancing_tumor: float
    necrotic_core: float
    total_tumor: float


class DicePerClass(BaseModel):
    """Per-class Dice scores (only present if GT available)."""
    edema: Optional[float] = None
    enhancing_tumor: Optional[float] = None
    necrotic_core: Optional[float] = None


class BratsRegions(BaseModel):
    """BraTS standard region Dice scores."""
    WT: Optional[float] = None
    TC: Optional[float] = None
    ET: Optional[float] = None


class ModelPredictionMetrics(BaseModel):
    """Per-prediction metrics returned for each of 4 outputs."""
    model_name: str
    display_name: str
    volumes_cm3: VolumeBreakdown
    mean_dice: Optional[float] = None
    dice_per_class: Optional[DicePerClass] = None
    brats_regions: Optional[BratsRegions] = None


class EloquentRegionInfo(BaseModel):
    distance_mm: Optional[float] = None
    risk_level: str
    involved: bool
    function: str
    deficit_if_damaged: str


class AnatomyInfo(BaseModel):
    primary_region: str
    primary_pct: float
    lateralization: str
    left_hemisphere_pct: float
    right_hemisphere_pct: float
    total_volume_cm3: float
    lobes: Dict[str, Dict[str, float]]
    regions_involved: List[Any] = []


class AtlasResult(BaseModel):
    anatomy: Optional[AnatomyInfo] = None
    eloquent: Optional[Dict[str, EloquentRegionInfo]] = None
    error: Optional[str] = None
    shared_across_predictions: Optional[bool] = None


class AgreementInfo(BaseModel):
    unanimous_fraction: float
    n_models_compared: int


class AnalysisResult(BaseModel):
    """Returned by GET /jobs/{id}/result when the pipeline is done."""
    case_id: str
    available_models: List[str]
    weights_used: Dict[str, float]
    per_model_metrics: Dict[str, ModelPredictionMetrics]
    agreement: AgreementInfo
    atlas: Dict[str, AtlasResult]
    elapsed_seconds: float


# -----------------------------------------------------------------------------
# XAI
# -----------------------------------------------------------------------------

class XaiRequest(BaseModel):
    """Body for POST /jobs/{id}/explain."""
    model_name: str = Field(
        ...,
        description=(
            "Which prediction's tumor regions to explain. "
            "One of: attention_unet | swin_unetr | nnunet | ensemble."
        ),
    )


class XaiResult(BaseModel):
    """Returned by GET /jobs/{id}/xai/{prediction_name}."""
    prediction_being_explained: str
    explainer_model: str
    target_layer: str
    heatmap_paths: Dict[str, str] = Field(
        ...,
        description=(
            "Class -> path of saved heatmap NIfTI. Frontend fetches these to "
            "overlay on the 3D viewer."
        ),
    )


# -----------------------------------------------------------------------------
# REPORT
# -----------------------------------------------------------------------------

class ReportRequest(BaseModel):
    """Body for POST /jobs/{id}/report — picks which prediction to feature."""
    prediction_to_feature: str = Field(
        "ensemble",
        description=(
            "Which prediction headlines page 1 + page 3 of the report. "
            "Default: ensemble."
        ),
    )