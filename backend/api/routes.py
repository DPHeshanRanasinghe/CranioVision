"""
CranioVision — HTTP routes.

All endpoints under one router:
    POST   /upload                              — accept BraTS folder, start job
    GET    /jobs/{job_id}/status                — non-streaming snapshot
    GET    /jobs/{job_id}/progress              — Server-Sent Events stream
    GET    /jobs/{job_id}/result                — final analysis dict
    POST   /jobs/{job_id}/explain               — trigger lazy XAI
    POST   /jobs/{job_id}/report                — generate clinical PDF
    GET    /jobs/{job_id}/report.pdf            — download generated PDF

Viewer artifact endpoints (NEW):
    GET    /jobs/{job_id}/t1c.nii.gz            — base T1c MRI for viewer
    GET    /jobs/{job_id}/predictions/{model}.nii.gz   — prediction mask
    GET    /jobs/{job_id}/xai/{model}/{class}.nii.gz   — Grad-CAM heatmap
"""
from __future__ import annotations

import asyncio
import json
import shutil
import time
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse

from schemas.api import (
    CreateJobResponse, JobProgress, JobStatus,
    XaiRequest, ReportRequest,
)
from services.case_parser import parse_case_folder, is_registration_cached
from services.job_manager import get_manager


router = APIRouter()

import os
UPLOAD_ROOT = Path(os.environ.get("UPLOAD_ROOT", "/tmp/cranovision_uploads"))
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

REPORT_ROOT = Path(os.environ.get("REPORT_ROOT", "/tmp/cranovision_reports"))
REPORT_ROOT.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# UPLOAD
# -----------------------------------------------------------------------------

@router.post("/upload", response_model=CreateJobResponse)
async def upload_case(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(400, "No files uploaded")

    job_dir = UPLOAD_ROOT / f"upload_{int(time.time() * 1000)}"
    job_dir.mkdir(parents=True, exist_ok=True)

    for upf in files:
        target = job_dir / Path(upf.filename).name
        with open(target, "wb") as f:
            shutil.copyfileobj(upf.file, f)

    try:
        case_dict = parse_case_folder(job_dir)
    except ValueError as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(400, f"Invalid case folder: {e}")

    case_id = case_dict["case_id"]
    cached = is_registration_cached(case_id)

    mgr = get_manager()
    job = await mgr.create_job(
        case_id=case_id,
        case_dict=case_dict,
        cached_registration=cached,
    )
    await mgr.run_pipeline(job.job_id)

    return CreateJobResponse(
        job_id=job.job_id,
        case_id=case_id,
        cached_registration=cached,
        status="queued",
    )


# -----------------------------------------------------------------------------
# JOB STATUS / PROGRESS / RESULT
# -----------------------------------------------------------------------------

@router.get("/jobs/{job_id}/status", response_model=JobStatus)
async def job_status(job_id: str):
    job = get_manager().get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    latest = job.latest_progress
    progress_obj = None
    if latest is not None:
        progress_obj = JobProgress(
            stage=latest.stage,
            percent=latest.percent,
            message=latest.message,
            timestamp=latest.timestamp,
        )

    return JobStatus(
        job_id=job.job_id,
        case_id=job.case_id,
        state=job.state,
        latest_progress=progress_obj,
        error=job.error,
        elapsed_seconds=job.elapsed_seconds,
    )


@router.get("/jobs/{job_id}/progress")
async def job_progress_sse(job_id: str, request: Request):
    mgr = get_manager()
    if mgr.get(job_id) is None:
        raise HTTPException(404, "Job not found")

    async def event_stream():
        q = await mgr.subscribe(job_id)
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue

                payload = {
                    "stage": event.stage,
                    "percent": event.percent,
                    "message": event.message,
                    "timestamp": event.timestamp,
                }
                yield f"data: {json.dumps(payload)}\n\n"

                if event.percent >= 100 or event.stage in ("done", "error"):
                    break
        finally:
            mgr.unsubscribe(job_id, q)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/jobs/{job_id}/result")
async def job_result(job_id: str):
    job = get_manager().get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.state != "done":
        raise HTTPException(409, f"Job not complete (state: {job.state})")
    if job.result is None:
        raise HTTPException(500, "Job done but no result")

    result = job.result
    json_result = {
        "case_id": result["case_id"],
        "available_models": result["available_models"],
        "weights_used": result["weights_used"],
        "agreement": result["agreement"],
        "elapsed_seconds": result["elapsed_seconds"],
        "per_model_metrics": result["per_model_metrics"],
        "atlas": _serialize_atlas(result.get("atlas", {})),
        "prediction_names": list(result["predictions"].keys()),
    }
    return JSONResponse(json_result)


def _serialize_atlas(atlas: dict) -> dict:
    out = {}
    for key, val in atlas.items():
        if isinstance(val, dict):
            out[key] = val
        else:
            out[key] = str(val)
    return out


# -----------------------------------------------------------------------------
# VIEWER ARTIFACT ENDPOINTS — for the 3D viewer in the frontend
# -----------------------------------------------------------------------------

@router.get("/jobs/{job_id}/t1c.nii.gz")
async def download_t1c(job_id: str):
    """Patient T1c MRI as the base layer for the 3D viewer."""
    job = get_manager().get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    t1c_path = job.artifacts_dir / "t1c.nii.gz"
    if not t1c_path.exists():
        raise HTTPException(
            404,
            f"T1c artifact not yet available. "
            f"Pipeline state: {job.state}",
        )
    return FileResponse(
        t1c_path,
        media_type="application/gzip",
        filename=f"{job.case_id}_t1c.nii.gz",
    )


@router.get("/jobs/{job_id}/predictions/{model_name}.nii.gz")
async def download_prediction(job_id: str, model_name: str):
    """
    Prediction mask as a NIfTI file for the 3D viewer.
    model_name in: attention_unet | swin_unetr | nnunet | ensemble
    """
    job = get_manager().get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    valid_models = {"attention_unet", "swin_unetr", "nnunet", "ensemble"}
    if model_name not in valid_models:
        raise HTTPException(
            400, f"Unknown model. Choose from {valid_models}"
        )

    pred_path = job.artifacts_dir / "predictions" / f"{model_name}.nii.gz"
    if not pred_path.exists():
        raise HTTPException(
            404,
            f"Prediction artifact not yet available. "
            f"Pipeline state: {job.state}",
        )
    return FileResponse(
        pred_path,
        media_type="application/gzip",
        filename=f"{job.case_id}_{model_name}_mask.nii.gz",
    )


@router.get("/jobs/{job_id}/xai/{model_name}/{class_name}.nii.gz")
async def download_xai_heatmap(
    job_id: str, model_name: str, class_name: str,
):
    """
    Grad-CAM heatmap as a NIfTI file for viewer overlay.
    class_name in: edema | enhancing | necrotic
    """
    job = get_manager().get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    valid_classes = {"edema", "enhancing", "necrotic"}
    if class_name not in valid_classes:
        raise HTTPException(
            400, f"Unknown class. Choose from {valid_classes}"
        )

    hm_path = (
        job.artifacts_dir / "xai" / model_name / f"{class_name}.nii.gz"
    )
    if not hm_path.exists():
        raise HTTPException(
            404,
            f"XAI heatmap not yet available. "
            f"Call POST /jobs/{job_id}/explain first.",
        )
    return FileResponse(
        hm_path,
        media_type="application/gzip",
        filename=f"{job.case_id}_{model_name}_{class_name}_heatmap.nii.gz",
    )


# -----------------------------------------------------------------------------
# XAI
# -----------------------------------------------------------------------------

@router.post("/jobs/{job_id}/explain")
async def trigger_xai(job_id: str, body: XaiRequest):
    mgr = get_manager()
    job = mgr.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.state != "done":
        raise HTTPException(
            409, f"Pipeline not complete (state: {job.state})"
        )

    try:
        xai = await mgr.run_xai(job_id, body.model_name)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    if xai is None:
        raise HTTPException(500, "XAI failed for unknown reason")

    return {
        "prediction_being_explained": xai["prediction_being_explained"],
        "explainer_model": xai["explainer_model"],
        "target_layer": xai["target_layer"],
        "ready": True,
        "heatmap_urls": {
            "edema": f"/jobs/{job_id}/xai/{body.model_name}/edema.nii.gz",
            "enhancing": f"/jobs/{job_id}/xai/{body.model_name}/enhancing.nii.gz",
            "necrotic": f"/jobs/{job_id}/xai/{body.model_name}/necrotic.nii.gz",
        },
    }


# -----------------------------------------------------------------------------
# REPORT
# -----------------------------------------------------------------------------

@router.post("/jobs/{job_id}/report")
async def generate_report(job_id: str, body: ReportRequest):
    job = get_manager().get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.state != "done":
        raise HTTPException(
            409, f"Pipeline not complete (state: {job.state})"
        )

    from src.cranovision.reporting import generate_clinical_report

    mgr = get_manager()
    xai = job.xai_results.get(body.prediction_to_feature)
    if xai is None:
        try:
            xai = await mgr.run_xai(job_id, body.prediction_to_feature)
        except Exception:
            xai = None

    output_path = REPORT_ROOT / f"{job.case_id}_{job.job_id[:8]}.pdf"

    loop = asyncio.get_running_loop()
    def _gen():
        return generate_clinical_report(
            case_id=job.case_id,
            analysis_result=job.result,
            xai_result=xai,
            prediction_to_feature=body.prediction_to_feature,
            image=job.image_tensor,
            output_path=output_path,
        )
    pdf_path = await loop.run_in_executor(None, _gen)

    return {
        "ready": True,
        "filename": pdf_path.name,
        "size_kb": round(pdf_path.stat().st_size / 1024, 1),
        "download_url": f"/jobs/{job_id}/report.pdf",
    }


@router.get("/jobs/{job_id}/report.pdf")
async def download_report(job_id: str):
    job = get_manager().get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    pdf_path = REPORT_ROOT / f"{job.case_id}_{job.job_id[:8]}.pdf"
    if not pdf_path.exists():
        raise HTTPException(
            404,
            f"Report not generated yet for job {job_id}. "
            f"Call POST /jobs/{job_id}/report first.",
        )

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=pdf_path.name,
    )