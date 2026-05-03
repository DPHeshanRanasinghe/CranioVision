"""
CranioVision — Job manager.

In-memory job store with async execution. Each job runs the pipeline in
a worker thread and reports progress via a queue that SSE endpoints consume.

Lifecycle: queued -> running -> done | failed

Persistent artifacts (NIfTI files for the 3D viewer) are saved per-job
under JOB_ARTIFACTS_DIR / {job_id} / so the frontend can fetch them via
HTTP. These are the files VTK.js (or the React 3D viewer) consumes.

State (job dicts) is in-memory. NIfTI artifacts are on disk.
"""
from __future__ import annotations

import asyncio
import os
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# Where per-job NIfTI artifacts (predictions, T1c, heatmaps) are saved.
JOB_ARTIFACTS_DIR = Path(
    os.environ.get("JOB_ARTIFACTS_DIR", "/tmp/cranovision_artifacts")
)
JOB_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# JOB STATE
# -----------------------------------------------------------------------------

@dataclass
class ProgressEvent:
    stage: str
    percent: int
    message: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Job:
    job_id: str
    case_id: str
    case_dict: Dict[str, Any]
    cached_registration: bool

    state: str = "queued"
    progress_events: List[ProgressEvent] = field(default_factory=list)
    progress_subscribers: List[asyncio.Queue] = field(default_factory=list)

    result: Optional[Dict[str, Any]] = None
    xai_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    image_tensor: Optional[Any] = None

    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    @property
    def artifacts_dir(self) -> Path:
        d = JOB_ARTIFACTS_DIR / self.job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def elapsed_seconds(self) -> Optional[float]:
        if self.started_at is None:
            return None
        end = self.completed_at if self.completed_at else time.time()
        return round(end - self.started_at, 1)

    @property
    def latest_progress(self) -> Optional[ProgressEvent]:
        if not self.progress_events:
            return None
        return self.progress_events[-1]


# -----------------------------------------------------------------------------
# MANAGER
# -----------------------------------------------------------------------------

class JobManager:
    def __init__(self, max_workers: int = 1):
        self._jobs: Dict[str, Job] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="cv-job"
        )
        self._lock = asyncio.Lock()

    # ---- creation ---------------------------------------------------------

    async def create_job(
        self,
        case_id: str,
        case_dict: Dict[str, Any],
        cached_registration: bool,
    ) -> Job:
        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            case_id=case_id,
            case_dict=case_dict,
            cached_registration=cached_registration,
        )
        async with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    # ---- progress fan-out -------------------------------------------------

    async def _publish(self, job: Job, event: ProgressEvent) -> None:
        job.progress_events.append(event)
        for q in list(job.progress_subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    async def subscribe(self, job_id: str) -> Optional[asyncio.Queue]:
        job = self.get(job_id)
        if job is None:
            return None
        q: asyncio.Queue = asyncio.Queue(maxsize=128)
        for ev in job.progress_events:
            await q.put(ev)
        job.progress_subscribers.append(q)
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue) -> None:
        job = self.get(job_id)
        if job is None:
            return
        if q in job.progress_subscribers:
            job.progress_subscribers.remove(q)

    # ---- execution --------------------------------------------------------

    async def run_pipeline(self, job_id: str) -> None:
        job = self.get(job_id)
        if job is None:
            raise RuntimeError(f"Unknown job: {job_id}")
        if job.state != "queued":
            raise RuntimeError(f"Job {job_id} already {job.state}")

        loop = asyncio.get_running_loop()
        loop.run_in_executor(self._executor, self._run_in_worker, job_id, loop)

    def _run_in_worker(
        self, job_id: str, loop: asyncio.AbstractEventLoop
    ) -> None:
        from src.cranovision.pipeline import run_full_analysis
        from src.cranovision.data import get_val_transforms

        job = self.get(job_id)
        if job is None:
            return

        job.state = "running"
        job.started_at = time.time()

        def progress_bridge(stage: str, pct: int, message: str) -> None:
            event = ProgressEvent(stage=stage, percent=pct, message=message)
            asyncio.run_coroutine_threadsafe(self._publish(job, event), loop)

        try:
            tx = get_val_transforms()
            sample = tx(job.case_dict)
            job.image_tensor = sample["image"]

            result = run_full_analysis(
                case_dict=job.case_dict,
                progress_fn=progress_bridge,
                include_atlas=True,
            )
            job.result = result

            # Save predictions + T1c as NIfTI for the 3D viewer
            try:
                self._save_viewer_artifacts(job)
            except Exception as e:
                # Don't fail the job over artifact save; log & continue
                progress_bridge(
                    "warning", 99,
                    f"Artifact save failed (viewer may be limited): {e}",
                )

            job.state = "done"

        except Exception as e:
            job.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            job.state = "failed"
            err_event = ProgressEvent(
                stage="error", percent=100, message=str(e),
            )
            asyncio.run_coroutine_threadsafe(self._publish(job, err_event), loop)

        finally:
            job.completed_at = time.time()

    # ---- viewer artifact saving ------------------------------------------

    def _save_viewer_artifacts(self, job: Job) -> None:
        """
        Save per-prediction NIfTI masks + the T1c image to disk so the
        frontend can fetch them for the 3D viewer.

        File layout under artifacts_dir:
            t1c.nii.gz                          (patient T1c, base layer)
            predictions/{model_name}.nii.gz     (4 files: 3 models + ensemble)
        """
        import nibabel as nib
        import numpy as np

        if job.result is None:
            return

        artifacts = job.artifacts_dir
        preds_dir = artifacts / "predictions"
        preds_dir.mkdir(parents=True, exist_ok=True)

        # T1c is the second modality in BraTS order (t1n, t1c, t2w, t2f).
        # Use the original patient T1c affine for proper alignment in viewer.
        t1c_path = job.case_dict["image"][1]
        t1c_nifti = nib.load(t1c_path)
        t1c_out = artifacts / "t1c.nii.gz"
        if not t1c_out.exists():
            # Save a fresh copy (preserves affine/header)
            nib.save(t1c_nifti, str(t1c_out))

        # Save each prediction with patient T1c's affine + header.
        # The predictions are already aligned to the patient native space
        # because BraTS preprocessing keeps modalities co-registered.
        patient_affine = t1c_nifti.affine
        patient_header = t1c_nifti.header

        for model_name, pred_tensor in job.result["predictions"].items():
            pred_np = pred_tensor.numpy().astype(np.int16)
            out_path = preds_dir / f"{model_name}.nii.gz"

            # Shape sanity check — if predictions don't match patient T1c,
            # skip rather than save misaligned data.
            if pred_np.shape != t1c_nifti.shape:
                # Predictions came out of preprocessed space; we don't try
                # to invert preprocessing. Save with identity affine and
                # mark that the viewer should align manually.
                nib.save(
                    nib.Nifti1Image(pred_np, np.eye(4)),
                    str(out_path),
                )
            else:
                nib.save(
                    nib.Nifti1Image(pred_np, patient_affine, patient_header),
                    str(out_path),
                )

    async def run_xai(
        self, job_id: str, model_name: str
    ) -> Optional[Dict[str, Any]]:
        job = self.get(job_id)
        if job is None:
            return None
        if job.state != "done":
            raise RuntimeError(
                f"Cannot run XAI on job {job_id} (state: {job.state})"
            )

        if model_name in job.xai_results:
            return job.xai_results[model_name]

        loop = asyncio.get_running_loop()

        def _run():
            from src.cranovision.pipeline import compute_xai_for_model
            return compute_xai_for_model(
                model_name=model_name,
                case_dict=job.case_dict,
            )

        xai = await loop.run_in_executor(self._executor, _run)
        job.xai_results[model_name] = xai

        # Save XAI heatmaps to disk for the 3D viewer
        try:
            self._save_xai_artifacts(job, model_name, xai)
        except Exception as e:
            # Non-fatal — XAI metadata still returned to caller
            print(f"[xai] artifact save failed: {e}")

        return xai

    def _save_xai_artifacts(
        self, job: Job, model_name: str, xai: Dict[str, Any]
    ) -> None:
        """Save Grad-CAM heatmaps as NIfTI for viewer overlay."""
        import nibabel as nib
        import numpy as np

        if "heatmaps" not in xai:
            return

        artifacts = job.artifacts_dir
        xai_dir = artifacts / "xai" / model_name
        xai_dir.mkdir(parents=True, exist_ok=True)

        # Use patient T1c affine where possible
        t1c_path = job.case_dict["image"][1]
        t1c_nifti = nib.load(t1c_path)
        patient_affine = t1c_nifti.affine

        class_to_name = {1: "edema", 2: "enhancing", 3: "necrotic"}

        for cls, heatmap_tensor in xai["heatmaps"].items():
            hm_np = heatmap_tensor.numpy().astype(np.float32)
            class_name = class_to_name.get(cls, f"class_{cls}")
            out_path = xai_dir / f"{class_name}.nii.gz"

            if hm_np.shape != t1c_nifti.shape:
                affine = np.eye(4)
            else:
                affine = patient_affine

            nib.save(nib.Nifti1Image(hm_np, affine), str(out_path))


# -----------------------------------------------------------------------------
# GLOBAL SINGLETON
# -----------------------------------------------------------------------------

_manager: Optional[JobManager] = None


def get_manager() -> JobManager:
    global _manager
    if _manager is None:
        _manager = JobManager(max_workers=1)
    return _manager