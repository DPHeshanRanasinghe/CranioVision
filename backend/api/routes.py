"""
CranioVision — HTTP routes.

All endpoints under one router:
    GET    /cache/status                        — demo cache contents + health
    POST   /cache/flush                         — drop every cached case
    GET    /demo-cases                          — list locally available demo cases
    POST   /demo-cases/{case_id}/run            — start job on a demo case
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

3D mesh endpoints (Atlas mode — MNI152 standard space, shared across patients):
    GET    /jobs/{job_id}/meshes/brain.glb              — brain shell mesh (Three.js)
    GET    /jobs/{job_id}/meshes/brain.json             — brain shell mesh (Plotly)
    GET    /jobs/{job_id}/meshes/{model}/manifest.json  — which classes exist
    GET    /jobs/{job_id}/meshes/{model}/{class}.glb    — per-class tumor mesh (Three.js)
    GET    /jobs/{job_id}/meshes/{model}/{class}.json   — per-class tumor mesh (Plotly)

3D mesh endpoints (Patient mode — patient's own native space):
    GET    /jobs/{job_id}/meshes/native/brain.json              — brain shell mesh
    GET    /jobs/{job_id}/meshes/native/{model}/manifest.json   — which classes exist
    GET    /jobs/{job_id}/meshes/native/{model}/{class}.json    — per-class tumor mesh
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
    CreateJobResponse, DemoCasesResponse, JobProgress, JobStatus,
    XaiRequest, ReportRequest,
)
from services.case_parser import (
    list_demo_cases, parse_case_folder, is_registration_cached,
)
from services.job_manager import get_manager


router = APIRouter()

import os
UPLOAD_ROOT = Path(os.environ.get("UPLOAD_ROOT", "/tmp/cranovision_uploads"))
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

REPORT_ROOT = Path(os.environ.get("REPORT_ROOT", "/tmp/cranovision_reports"))
REPORT_ROOT.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# DEMO CACHE ADMIN
# -----------------------------------------------------------------------------

@router.get("/cache/status")
async def cache_status():
    """What is warmed, how big it is, and whether Redis is reachable."""
    from services import demo_cache
    return demo_cache.stats()


@router.post("/cache/flush")
async def cache_flush():
    """Drop every cached case. The next run of each recomputes from scratch."""
    from services import demo_cache
    return demo_cache.flush()


# -----------------------------------------------------------------------------
# DEMO CASES
# -----------------------------------------------------------------------------

@router.get("/demo-cases", response_model=DemoCasesResponse)
async def demo_cases():
    return DemoCasesResponse(cases=list_demo_cases())


@router.post("/demo-cases/{case_id}/run", response_model=CreateJobResponse)
async def run_demo_case(case_id: str):
    from src.cranovision.config import RAW_DATA_DIR

    case_dir = RAW_DATA_DIR / case_id
    if not case_dir.is_dir():
        raise HTTPException(404, f"Unknown demo case: {case_id}")

    try:
        case_dict = parse_case_folder(case_dir)
    except ValueError as e:
        raise HTTPException(400, f"Invalid demo case folder: {e}")

    parsed_case_id = case_dict["case_id"]
    cached = is_registration_cached(parsed_case_id)

    mgr = get_manager()
    job = await mgr.create_job(
        case_id=parsed_case_id,
        case_dict=case_dict,
        cached_registration=cached,
    )
    await mgr.run_pipeline(job.job_id)

    return CreateJobResponse(
        job_id=job.job_id,
        case_id=parsed_case_id,
        cached_registration=cached,
        status="queued",
    )


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


def build_result_json(job) -> dict:
    """The exact payload GET /result returns. Shared with the demo cache so a
    cached replay is byte-identical to a fresh run."""
    result = job.result
    return {
        "case_id": result["case_id"],
        "available_models": result["available_models"],
        "weights_used": result["weights_used"],
        "agreement": result["agreement"],
        "elapsed_seconds": result["elapsed_seconds"],
        "per_model_metrics": result["per_model_metrics"],
        "atlas": _serialize_atlas(result.get("atlas", {})),
        "prediction_names": list(result["predictions"].keys()),
    }


@router.get("/jobs/{job_id}/result")
async def job_result(job_id: str):
    job = get_manager().get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.state != "done":
        raise HTTPException(409, f"Job not complete (state: {job.state})")

    # Cache replay: `result` holds torch tensors and cannot be rehydrated, so
    # the stored payload is served verbatim instead.
    if job.cached_result_json is not None:
        return JSONResponse(job.cached_result_json)

    if job.result is None:
        raise HTTPException(500, "Job done but no result")
    return JSONResponse(build_result_json(job))


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
# 3D MESH ENDPOINTS — for the three.js viewer
# -----------------------------------------------------------------------------

VALID_MODEL_NAMES = {"attention_unet", "swin_unetr", "nnunet", "ensemble"}
VALID_CLASS_NAMES = {"edema", "enhancing", "necrotic"}


@router.get("/jobs/{job_id}/meshes/brain.glb")
async def download_brain_mesh(job_id: str):
    """Semi-transparent brain shell mesh (shared across models)."""
    job = get_manager().get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    mesh_path = job.artifacts_dir / "meshes" / "brain.glb"
    if not mesh_path.exists():
        raise HTTPException(
            404,
            f"Brain mesh not yet available. Pipeline state: {job.state}",
        )
    return FileResponse(
        mesh_path,
        media_type="model/gltf-binary",
        filename=f"{job.case_id}_brain.glb",
    )


@router.get("/jobs/{job_id}/meshes/brain.json")
async def download_brain_mesh_json(job_id: str):
    """Brain shell mesh as Plotly Mesh3d JSON ({x,y,z,i,j,k})."""
    job = get_manager().get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    json_path = job.artifacts_dir / "meshes" / "brain.json"
    if not json_path.exists():
        raise HTTPException(
            404,
            f"Brain mesh JSON not yet available. Pipeline state: {job.state}",
        )
    return FileResponse(json_path, media_type="application/json")


# --- Patient mode (native space) ---
#
# MUST be registered BEFORE the parameterized atlas routes below. FastAPI
# matches in registration order, and "/meshes/native/brain.json" also matches
# "/meshes/{model_name}/{class_name}.json" (model_name="native",
# class_name="brain") — if the atlas route is registered first it swallows the
# request and rejects it as an unknown model, so the Patient-mode brain shell
# silently 400s while the deeper 6-segment native routes below still work.
# That asymmetry is exactly what "tumor renders but brain doesn't" looks like.

@router.get("/jobs/{job_id}/meshes/native/brain.json")
async def download_native_brain_mesh_json(job_id: str):
    """Patient-native brain shell mesh as Plotly Mesh3d JSON."""
    job = get_manager().get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    json_path = job.artifacts_dir / "meshes_native" / "brain.json"
    if not json_path.exists():
        raise HTTPException(
            404,
            f"Patient-native brain mesh not available. Pipeline state: {job.state}",
        )
    return FileResponse(json_path, media_type="application/json")


@router.get("/jobs/{job_id}/meshes/native/{model_name}/manifest.json")
async def native_mesh_manifest(job_id: str, model_name: str):
    """Per-model manifest for patient-native meshes."""
    job = get_manager().get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if model_name not in VALID_MODEL_NAMES:
        raise HTTPException(400, f"Unknown model. Choose from {VALID_MODEL_NAMES}")

    manifest_path = (
        job.artifacts_dir / "meshes_native" / model_name / "manifest.json"
    )
    if not manifest_path.exists():
        raise HTTPException(
            404,
            f"Patient-native mesh manifest not available. Pipeline state: {job.state}",
        )
    return FileResponse(manifest_path, media_type="application/json")


@router.get("/jobs/{job_id}/meshes/native/{model_name}/{class_name}.json")
async def download_native_class_mesh_json(
    job_id: str, model_name: str, class_name: str,
):
    """
    Per-class tumor mesh in patient-native space as Plotly-Mesh3d JSON.
    `class_name` ∈ {edema, enhancing, necrotic, heatmap} — 'heatmap' is the
    Grad-CAM surface, built on demand by POST /jobs/{id}/explain.
    """
    job = get_manager().get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if model_name not in VALID_MODEL_NAMES:
        raise HTTPException(400, f"Unknown model. Choose from {VALID_MODEL_NAMES}")
    if class_name not in VALID_CLASS_NAMES and class_name != "heatmap":
        raise HTTPException(400, f"Unknown class. Choose from {VALID_CLASS_NAMES}")

    json_path = (
        job.artifacts_dir / "meshes_native" / model_name / f"{class_name}.json"
    )
    if not json_path.exists():
        raise HTTPException(
            404,
            f"Mesh JSON not available for {model_name}/{class_name} in patient-native "
            f"space. This class may be empty, or the prediction couldn't be aligned "
            f"to native space (see manifest.json's 'source' field).",
        )
    return FileResponse(json_path, media_type="application/json")


# --- Atlas mode (MNI152 standard space) ---

@router.get("/jobs/{job_id}/meshes/{model_name}/manifest.json")
async def mesh_manifest(job_id: str, model_name: str):
    """
    Per-model mesh manifest. The frontend reads this to know which class
    meshes exist before trying to fetch them (a case may have no necrotic
    core, etc).
    """
    job = get_manager().get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if model_name not in VALID_MODEL_NAMES:
        raise HTTPException(400, f"Unknown model. Choose from {VALID_MODEL_NAMES}")

    manifest_path = (
        job.artifacts_dir / "meshes" / model_name / "manifest.json"
    )
    if not manifest_path.exists():
        raise HTTPException(
            404,
            f"Mesh manifest not yet available. Pipeline state: {job.state}",
        )
    return FileResponse(manifest_path, media_type="application/json")


@router.get("/jobs/{job_id}/meshes/{model_name}/{class_name}.glb")
async def download_class_mesh(
    job_id: str, model_name: str, class_name: str,
):
    """Per-class tumor mesh as a binary glTF."""
    job = get_manager().get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if model_name not in VALID_MODEL_NAMES:
        raise HTTPException(400, f"Unknown model. Choose from {VALID_MODEL_NAMES}")
    # heatmap.glb is a Task-2 addition; allow it through without joining the
    # tumor-class set so we don't need a second route.
    if class_name not in VALID_CLASS_NAMES and class_name != "heatmap":
        raise HTTPException(400, f"Unknown class. Choose from {VALID_CLASS_NAMES}")

    mesh_path = (
        job.artifacts_dir / "meshes" / model_name / f"{class_name}.glb"
    )
    if not mesh_path.exists():
        raise HTTPException(
            404,
            f"Mesh not available for {model_name}/{class_name}. "
            f"This class may be empty for this case.",
        )
    return FileResponse(
        mesh_path,
        media_type="model/gltf-binary",
        filename=f"{job.case_id}_{model_name}_{class_name}.glb",
    )


@router.get("/jobs/{job_id}/meshes/{model_name}/{class_name}.json")
async def download_class_mesh_json(
    job_id: str, model_name: str, class_name: str,
):
    """
    Per-class tumor mesh (or heatmap mesh) as Plotly-Mesh3d JSON.
    `class_name` ∈ {edema, enhancing, necrotic, heatmap}.
    """
    job = get_manager().get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if model_name not in VALID_MODEL_NAMES:
        raise HTTPException(400, f"Unknown model. Choose from {VALID_MODEL_NAMES}")
    if class_name not in VALID_CLASS_NAMES and class_name != "heatmap":
        raise HTTPException(400, f"Unknown class. Choose from {VALID_CLASS_NAMES}")

    json_path = (
        job.artifacts_dir / "meshes" / model_name / f"{class_name}.json"
    )
    if not json_path.exists():
        raise HTTPException(
            404,
            f"Mesh JSON not available for {model_name}/{class_name}. "
            f"This class may be empty (or XAI not yet generated for heatmap).",
        )
    return FileResponse(json_path, media_type="application/json")


# -----------------------------------------------------------------------------
# XAI
# -----------------------------------------------------------------------------

@router.post("/jobs/{job_id}/explain")
async def trigger_xai(job_id: str, body: XaiRequest):
    import traceback
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
    except Exception as e:
        tb = traceback.format_exc()
        print(
            "[xai] /explain failed for job=%s model=%s: %s\n%s"
            % (job_id, body.model_name, type(e).__name__, tb),
            flush=True,
        )
        raise HTTPException(
            500, f"XAI failed: {type(e).__name__}: {e}"
        )
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

def _rehydrate_for_report(job):
    """
    Rebuild (analysis_result, image) for a job that was served from the demo
    cache. Returns (None, None) when the inputs are no longer available.

    Cheap: re-running the val transforms is preprocessing only — no model
    inference — so this costs a couple of seconds, not minutes.
    """
    try:
        import numpy as np
        import nibabel as nib
        import torch

        raw_dir = job.artifacts_dir / "predictions_raw"
        if not raw_dir.is_dir():
            return None, None

        predictions = {}
        for f in sorted(raw_dir.glob("*.nii.gz")):
            arr = np.asarray(nib.load(str(f)).dataobj).astype(np.int64)
            predictions[f.name.replace(".nii.gz", "")] = torch.from_numpy(arr)
        if not predictions:
            return None, None

        cached = dict(job.cached_result_json or {})
        analysis = {
            "case_id": cached.get("case_id", job.case_id),
            "predictions": predictions,
            "per_model_metrics": cached.get("per_model_metrics", {}),
            "agreement": cached.get("agreement", {}),
            "atlas": cached.get("atlas", {}),
            "weights_used": cached.get("weights_used", {}),
            "available_models": cached.get("available_models", []),
            "elapsed_seconds": cached.get("elapsed_seconds", 0),
        }

        image = job.image_tensor
        if image is None:
            from src.cranovision.data import get_val_transforms
            image = get_val_transforms()(job.case_dict)["image"]
            job.image_tensor = image
        return analysis, image
    except Exception as e:
        print(f"[report] rehydrate failed: {type(e).__name__}: {e}", flush=True)
        return None, None


@router.post("/jobs/{job_id}/report")
async def generate_report(job_id: str, body: ReportRequest):
    job = get_manager().get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.state != "done":
        raise HTTPException(
            409, f"Pipeline not complete (state: {job.state})"
        )

    # Served from cache: the analysis tensors are gone, but the PDF that was
    # generated on the original run came back with the entry.
    if job.cached_report_path and job.cached_report_path.exists():
        p = job.cached_report_path
        return {
            "ready": True,
            "filename": p.name,
            "size_kb": round(p.stat().st_size / 1024, 1),
            "download_url": f"/jobs/{job_id}/report.pdf",
            "cached": True,
        }

    from src.cranovision.reporting import generate_clinical_report

    mgr = get_manager()

    # A cache-replayed job has no torch tensors in memory, so rebuild exactly
    # what the report needs: the preprocessed 4-channel image (re-derived from
    # the case files — cheap, no inference) and the predictions on that same
    # preprocessed grid (kept in artifacts/predictions_raw for this purpose).
    analysis_result = job.result
    image_tensor = job.image_tensor
    if analysis_result is None and job.cached_result_json is not None:
        analysis_result, image_tensor = _rehydrate_for_report(job)
        if analysis_result is None:
            raise HTTPException(
                409,
                "This case was served from the demo cache without a stored "
                "report, and its inputs could not be re-derived. Re-warm it: "
                "python scripts/warm_demo_cache.py --cases " + job.case_id,
            )

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
            analysis_result=analysis_result,
            xai_result=xai,
            prediction_to_feature=body.prediction_to_feature,
            image=image_tensor,
            output_path=output_path,
        )
    try:
        pdf_path = await loop.run_in_executor(None, _gen)
    except Exception as e:
        import traceback
        print(f"[report] generation failed for {job.case_id}: "
              f"{type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        raise HTTPException(500, f"Report generation failed: {type(e).__name__}: {e}")

    # Fold the PDF into this case's cache entry, so a cached replay can serve
    # the report too. Without this the entry is warm for analysis + XAI but a
    # judge clicking "Generate report" would still wait for a fresh render.
    try:
        job.cached_report_path = pdf_path
        mgr._store_in_cache(job)
    except Exception as e:
        print(f"[demo-cache] report cache refresh failed (harmless): {e}", flush=True)

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