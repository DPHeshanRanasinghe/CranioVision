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
    # Set only when the job was served from the demo cache. The full `result`
    # dict cannot be rebuilt (it holds torch tensors), so the routes serve
    # this pre-serialized payload instead. See services/demo_cache.py.
    cached_result_json: Optional[Dict[str, Any]] = None
    cached_report_path: Optional[Path] = None
    cache_key: Optional[str] = None
    served_from_cache: bool = False
    # The val-transform Compose that produced image_tensor. Retained so XAI
    # (which runs later, on demand) can invert the same preprocessing to map
    # Grad-CAM heatmaps back into patient-native space.
    transforms: Optional[Any] = None

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

        # Demo cache: a hit replays a previously computed run in <1s instead
        # of ~5 min. Any failure here falls through to the real pipeline.
        try:
            from services import demo_cache
            job.cache_key = demo_cache.case_key(job.case_dict)
            entry = demo_cache.get(job.cache_key) if job.cache_key else None
            if entry:
                await self._serve_from_cache(job, entry)
                return
        except Exception as e:
            print(f"[demo-cache] lookup failed, running normally: {e}", flush=True)

        loop = asyncio.get_running_loop()
        loop.run_in_executor(self._executor, self._run_in_worker, job_id, loop)

    async def _serve_from_cache(self, job: Job, entry: Dict[str, Any]) -> None:
        """
        Rehydrate a completed job from Redis: unpack its artifacts, restore the
        result payload, XAI and PDF, then replay the pipeline stages quickly so
        the UI animates as usual rather than snapping to 100%.

        Stage messages are suffixed "(cached)" — the audience should be able to
        see that this is a replay, not a fresh 5-minute run.
        """
        import tarfile, io

        from services import demo_cache

        job.state = "running"
        job.started_at = time.time()
        meta = entry["meta"]

        if entry.get("artifacts"):
            demo_cache._untar_to(entry["artifacts"], job.artifacts_dir)

        job.cached_result_json = meta.get("result")
        job.served_from_cache = True
        for model, payload in (meta.get("xai") or {}).items():
            job.xai_results[model] = payload

        if entry.get("report"):
            # Restore under THIS job's canonical filename, not the one from the
            # job that originally generated it. GET /report.pdf derives the path
            # from the current job id, so keeping the old name made the cached
            # download 404 even though the PDF was right there.
            from api.routes import REPORT_ROOT
            dest = REPORT_ROOT / f"{job.case_id}_{job.job_id[:8]}.pdf"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(entry["report"])
            job.cached_report_path = dest

        stages = [
            ("preprocess", 15, "Preprocessing volumes (cached)"),
            ("inference", 45, "Loading model predictions (cached)"),
            ("ensemble", 65, "Combining predictions (cached)"),
            ("agreement", 80, "Computing model agreement (cached)"),
            ("atlas", 92, "Atlas anatomy + eloquent cortex (cached)"),
        ]
        for stage, pct, msg in stages:
            await self._publish(job, ProgressEvent(stage=stage, percent=pct, message=msg))
            await asyncio.sleep(0.18)

        job.state = "done"
        job.completed_at = time.time()
        await self._publish(job, ProgressEvent(
            stage="done", percent=100,
            message=f"Analysis complete (cached · {job.case_id})",
        ))
        print(f"[demo-cache] HIT {job.case_id} key={job.cache_key[:12]} "
              f"served in {job.elapsed_seconds}s", flush=True)

    def _store_in_cache(self, job: Job) -> None:
        """
        Persist or ENRICH this job's cache entry. Never raises.

        A replayed job can still produce something the stored entry lacks — a
        case warmed without Grad-CAM, then explained later. Skipping those
        updates meant every replay recomputed XAI (~166s measured), which
        defeats the point of the cache. So we re-store whenever there is new
        content, and skip only when there genuinely is none.
        """
        try:
            from services import demo_cache
            from api.routes import build_result_json

            if job.result is not None:
                result_json = build_result_json(job)
            elif job.cached_result_json is not None:
                result_json = job.cached_result_json
            else:
                return

            key = job.cache_key or demo_cache.case_key(job.case_dict)
            if not key:
                return
            job.cache_key = key

            xai = {m: {k: v for k, v in x.items()
                       if k in ("prediction_being_explained",
                                "explainer_model", "target_layer")}
                   for m, x in job.xai_results.items()}
            report_path = job.cached_report_path

            if job.served_from_cache:
                # Only pay the re-pack cost when this run actually adds
                # something the entry does not already hold.
                existing = demo_cache.get(key)
                if existing:
                    meta = existing.get("meta", {})
                    has_new_xai = set(xai) - set(meta.get("xai") or {})
                    has_new_report = bool(report_path) and not meta.get("report_name")
                    if not has_new_xai and not has_new_report:
                        return

            demo_cache.put(
                key,
                case_id=job.case_id,
                result_json=result_json,
                artifacts_dir=job.artifacts_dir,
                xai=xai,
                report_path=report_path,
            )
        except Exception as e:
            print(f"[demo-cache] store failed (harmless): {e}", flush=True)

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
            # The pipeline emits ("done", 100) when ITS work finishes — but
            # the job still has viewer artifacts, per-model warps, and mesh
            # extraction left. The frontend treats stage=="done" or pct>=100
            # as completion and races to fetch /result, hitting a 409. Demote
            # the pipeline's terminal event so only the job-level finaliser
            # (line ~229) gets to report true completion.
            if stage == "done" and pct >= 100 and job.state == "running":
                stage, pct = "pipeline_done", 92
            event = ProgressEvent(stage=stage, percent=pct, message=message)
            asyncio.run_coroutine_threadsafe(self._publish(job, event), loop)

        try:
            tx = get_val_transforms()
            sample = tx(job.case_dict)
            job.image_tensor = sample["image"]
            job.transforms = tx

            result = run_full_analysis(
                case_dict=job.case_dict,
                progress_fn=progress_bridge,
                include_atlas=True,
            )
            job.result = result

            # Save predictions + T1c as NIfTI for the 3D viewer
            try:
                self._save_viewer_artifacts(job, tx)
            except Exception as e:
                # Don't fail the job over artifact save; log & continue
                progress_bridge(
                    "warning", 99,
                    f"Artifact save failed (viewer may be limited): {e}",
                )

            # Warp each per-model prediction from the preprocessed grid into
            # MNI space using the cached ANTs transforms. Without this every
            # model tab in the 3D viewer rendered the same shared GT mesh,
            # contradicting the per-model volumes in the metrics panel.
            try:
                progress_bridge(
                    "warp_predictions", 95,
                    "Warping per-model predictions to MNI...",
                )
                self._save_warped_predictions(job)
            except Exception as e:
                progress_bridge(
                    "warning", 96,
                    f"Per-model warp failed (viewer falls back to GT): {e}",
                )

            # Extract marching-cubes meshes (brain shell + per-class tumor)
            # so the 3D viewer can load .glb files instead of raw NIfTI.
            try:
                progress_bridge("meshes", 98, "Extracting 3D meshes...")
                self._save_mesh_artifacts(job)
            except Exception as e:
                progress_bridge(
                    "warning", 99,
                    f"Mesh extraction failed (viewer falls back): {e}",
                )

            # Patient-native meshes — the "Patient" view mode alongside the
            # standardized "Atlas" one above. Independent of atlas/ANTs
            # entirely, so this still works even when atlas registration
            # failed or antspyx isn't installed.
            try:
                progress_bridge(
                    "meshes_native", 99, "Building patient-native 3D meshes...",
                )
                self._save_native_mesh_artifacts(job)
            except Exception as e:
                progress_bridge(
                    "warning", 99,
                    f"Patient-native mesh extraction failed (Atlas view still works): {e}",
                )

            job.state = "done"
            # Terminal progress event — closes SSE on the client and lifts the
            # progress bar to 100%. Without this, the bar freezes at 98%
            # (the meshes stage) since no later stage publishes.
            progress_bridge("done", 100, "Analysis complete")

            # Warm the demo cache so the next run of this case is instant.
            # Deliberately after the terminal event: tarring ~43MB should not
            # delay the UI reaching 100%.
            self._store_in_cache(job)

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

    def _to_patient_native(self, job: Job, tx, volume) -> "Any":
        """
        Map one volume from get_val_transforms()' cropped/padded grid back to
        the patient's native grid by inverting the SAME transform pipeline
        (`tx`) that produced job.image_tensor.

        None of _base_transforms' ops (Orientationd, CropForegroundd,
        SpatialPadd, CenterSpatialCropd) resample or interpolate — they only
        reorder/crop/pad indices — so the inverse recovers the original grid
        exactly, for float heatmaps as well as integer label maps.

        Returns a numpy array in patient-native shape. Raises on failure so
        callers can decide their own fallback.

        The deep copies are load-bearing: tx.inverse() mutates the
        transform-info dicts inside applied_operations as it consumes them,
        so sharing them across calls corrupts state after the first one
        (confirmed: only the first of 4 sequential calls succeeded with a
        shallow copy, the rest raised inside CropForegroundd.inverse).
        """
        import copy
        import torch
        from monai.data.meta_tensor import MetaTensor

        vol = volume if torch.is_tensor(volume) else torch.as_tensor(volume)
        mt = MetaTensor(
            vol.unsqueeze(0).float(),
            meta=copy.deepcopy(job.image_tensor.meta),
        )
        mt.applied_operations = copy.deepcopy(job.image_tensor.applied_operations)
        return tx.inverse({"image": mt})["image"].numpy()[0]

    def _save_viewer_artifacts(self, job: Job, tx) -> None:
        """
        Save per-prediction NIfTI masks + the T1c image to disk so the
        frontend can fetch them for the 3D viewer.

        Predictions come out of get_val_transforms()' cropped/padded grid
        (job.image_tensor's shape), not patient T1c's native grid — every
        real case hits this (CropForegroundd always changes the shape), not
        just an edge case. We invert the SAME transform pipeline (`tx`) that
        produced job.image_tensor to place each prediction back into
        patient-native space exactly: none of _base_transforms' ops
        (Orientationd, CropForegroundd, SpatialPadd, CenterSpatialCropd)
        resample/interpolate, they only reorder/crop/pad indices, so the
        inverse recovers the original grid with no interpolation artifacts —
        verified directly against this pipeline before relying on it.

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

        patient_affine = t1c_nifti.affine
        patient_header = t1c_nifti.header

        # Also keep the predictions on their ORIGINAL preprocessed grid.
        # The clinical report overlays predictions on the preprocessed 4-channel
        # image (it indexes img[1] for T1c), so the native-space copies below
        # are the wrong grid for it. Without these a cache-replayed job could
        # never regenerate a report.
        raw_dir = artifacts / "predictions_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for model_name, pred_tensor in job.result["predictions"].items():
            try:
                arr = pred_tensor.numpy().astype(np.int16)
                nib.save(nib.Nifti1Image(arr, np.eye(4)),
                         str(raw_dir / f"{model_name}.nii.gz"))
            except Exception:
                pass

        for model_name, pred_tensor in job.result["predictions"].items():
            out_path = preds_dir / f"{model_name}.nii.gz"

            try:
                # Label map: round after inversion (the inverse only
                # crops/pads/reorders, but the tensor round-trips as float).
                pred_native = np.rint(
                    self._to_patient_native(job, tx, pred_tensor)
                ).astype(np.int16)

                if pred_native.shape != t1c_nifti.shape:
                    # Inversion didn't land on T1c's grid after all — don't
                    # silently save misaligned data.
                    raise RuntimeError(
                        f"inverted shape {pred_native.shape} != "
                        f"T1c shape {t1c_nifti.shape}"
                    )

                nib.save(
                    nib.Nifti1Image(pred_native, patient_affine, patient_header),
                    str(out_path),
                )
            except Exception:
                # Fall back to the raw preprocessed-grid array with an
                # identity affine — viewer-space features that need patient
                # alignment (e.g. Patient mesh mode) will report unavailable
                # for this model rather than render something misplaced.
                pred_np = pred_tensor.numpy().astype(np.int16)
                nib.save(
                    nib.Nifti1Image(pred_np, np.eye(4)),
                    str(out_path),
                )

    def _save_warped_predictions(self, job: Job) -> None:
        """
        Warp each per-model prediction from the BraTS-preprocessed grid to MNI
        space using the cached patient->MNI transforms. The mesh extractor
        then builds per-model tumor surfaces from these warped volumes, so
        switching tabs in the 3D viewer shows the actual model differences.

        Why we can use the cached transforms (computed against patient T1):
        ANTs apply_transforms operates in physical world coordinates, not
        voxel indices. As long as the prediction NIfTI carries the correct
        affine for its own grid (which MONAI's MetaTensor tracks through
        Orientation/Crop/Pad), the same patient->MNI warp applies cleanly.

        nearestNeighbor interpolation is mandatory: predictions are label
        volumes (1=edema, 2=enhancing, 3=necrotic). Linear interpolation
        smears small classes — necrotic in particular — into background.
        """
        import nibabel as nib
        import numpy as np

        if job.result is None:
            return

        preprocessed_affine = job.result.get("preprocessed_affine")
        if preprocessed_affine is None:
            return  # pipeline didn't expose the affine — skip silently

        # Cached registration paths
        project_root = Path(__file__).resolve().parents[2]
        cache_dir = project_root / "outputs" / "atlas_cache" / job.case_id
        fwd_warp = cache_dir / "fwd_warp.nii.gz"
        fwd_affine = cache_dir / "fwd_affine.mat"
        mni_path = (
            project_root
            / "atlas_data"
            / "MNI152NLin2009cAsym_1mm_T1_brain.nii.gz"
        )
        if not (fwd_warp.exists() and fwd_affine.exists() and mni_path.exists()):
            return  # no cache for this case — mesh extractor falls back to GT

        try:
            import ants
        except Exception:
            return

        artifacts = job.artifacts_dir
        out_dir = artifacts / "predictions_mni"
        out_dir.mkdir(parents=True, exist_ok=True)

        affine_np = np.asarray(preprocessed_affine, dtype=np.float64)
        if affine_np.ndim == 3:  # may carry a leading batch/channel dim
            affine_np = affine_np[0]

        fixed = ants.image_read(str(mni_path))

        for model_name, pred_tensor in job.result["predictions"].items():
            out_path = out_dir / f"{model_name}.nii.gz"
            if out_path.exists():
                continue

            pred_np = pred_tensor.detach().cpu().numpy().astype(np.int16)

            # Save the preprocessed-space prediction with its true patient
            # affine so ANTs can map it through the cached patient->MNI warp.
            tmp_path = out_dir / f"{model_name}_preprocessed.nii.gz"
            nib.save(nib.Nifti1Image(pred_np, affine_np), str(tmp_path))

            try:
                moving = ants.image_read(str(tmp_path))
                warped = ants.apply_transforms(
                    fixed=fixed,
                    moving=moving,
                    transformlist=[str(fwd_warp), str(fwd_affine)],
                    interpolator="nearestNeighbor",
                )
                ants.image_write(warped, str(out_path))
            finally:
                tmp_path.unlink(missing_ok=True)

    def _save_mesh_artifacts(self, job: Job) -> None:
        """
        Build the MNI brain shell (cached process-wide) and per-class tumor
        meshes from per-model predictions warped to MNI space. Falls back to
        the shared GT-warped mesh when the per-model warps aren't available
        (no cached registration, or pipeline failure earlier).
        """
        from services.mesh_extractor import extract_meshes_for_job

        if job.result is None:
            return

        gt_mask_path = job.case_dict.get("label")
        model_names = list(job.result.get("predictions", {}).keys())
        if not model_names:
            return

        extract_meshes_for_job(
            artifacts_dir=job.artifacts_dir,
            case_id=job.case_id,
            model_names=model_names,
            gt_mask_path=gt_mask_path,
            predictions_mni_dir=job.artifacts_dir / "predictions_mni",
        )

    def _save_native_mesh_artifacts(self, job: Job) -> None:
        """
        Build the brain shell + per-class tumor meshes in the patient's OWN
        native space (as opposed to _save_mesh_artifacts' MNI-warped ones),
        so the viewer can offer a "Patient" mode showing the real brain
        shape/size and tumor mass-effect alongside the standardized "Atlas"
        mode. Reuses t1c.nii.gz + predictions/*.nii.gz from
        _save_viewer_artifacts, which already carry the patient's real affine.
        """
        from services.mesh_extractor import extract_patient_meshes_for_job

        if job.result is None:
            return

        t1c_path = job.artifacts_dir / "t1c.nii.gz"
        if not t1c_path.exists():
            return  # _save_viewer_artifacts must have failed earlier

        model_names = list(job.result.get("predictions", {}).keys())
        if not model_names:
            return

        extract_patient_meshes_for_job(
            artifacts_dir=job.artifacts_dir,
            t1c_path=t1c_path,
            model_names=model_names,
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
        import traceback as _tb
        try:
            self._save_xai_artifacts(job, model_name, xai)
        except Exception as e:
            print(
                f"[xai] artifact save failed: {type(e).__name__}: {e}\n"
                f"{_tb.format_exc()}",
                flush=True,
            )

        # Build the magma-coloured tumor mesh for the Plotly viewer.
        try:
            self._save_heatmap_mesh(job, model_name, xai)
        except Exception as e:
            print(
                f"[xai] heatmap mesh build failed: {type(e).__name__}: {e}\n"
                f"{_tb.format_exc()}",
                flush=True,
            )

        # Same heatmap, mapped back to patient-native space so Grad-CAM also
        # works in the viewer's Patient mode. Independent of the atlas build
        # above — a failure here leaves Atlas-mode Grad-CAM working.
        try:
            self._save_native_heatmap_mesh(job, model_name, xai)
        except Exception as e:
            print(
                f"[xai] native heatmap mesh build failed: {type(e).__name__}: {e}\n"
                f"{_tb.format_exc()}",
                flush=True,
            )

        # Refresh the cache entry so a later run of this case replays the
        # Grad-CAM instantly too, not just the segmentation.
        self._store_in_cache(job)

        return xai

    def _save_native_heatmap_mesh(
        self, job: Job, model_name: str, xai: Dict[str, Any]
    ) -> None:
        """
        Build meshes_native/{model}/heatmap.glb — the Grad-CAM surface in the
        patient's own space, so Patient mode can show it aligned with the
        patient-native brain shell.

        Both the prediction and every heatmap volume go through the same
        verified transform inversion used for the native predictions, so the
        surface and its colouring stay in register.
        """
        import numpy as np
        import nibabel as nib
        from services.mesh_extractor import extract_native_heatmap_mesh_for_job

        if job.result is None or "predictions" not in job.result:
            return
        if not xai.get("heatmaps"):
            return
        if job.transforms is None or job.image_tensor is None:
            return  # can't invert without the pipeline that produced it

        pred_tensor = job.result["predictions"].get(model_name)
        if pred_tensor is None:
            return

        t1c_path = job.artifacts_dir / "t1c.nii.gz"
        if not t1c_path.exists():
            return
        patient_affine = nib.load(str(t1c_path)).affine

        tx = job.transforms
        pred_native = np.rint(
            self._to_patient_native(job, tx, pred_tensor)
        ).astype(np.int16)

        heatmaps_native: Dict[int, "np.ndarray"] = {}
        for cls, hm in xai["heatmaps"].items():
            arr = self._to_patient_native(job, tx, hm)
            heatmaps_native[int(cls)] = arr.astype(np.float32)

        extract_native_heatmap_mesh_for_job(
            artifacts_dir=job.artifacts_dir,
            model_name=model_name,
            prediction_native=pred_native,
            heatmaps_native_by_class=heatmaps_native,
            patient_affine=patient_affine,
        )

    def _save_heatmap_mesh(
        self, job: Job, model_name: str, xai: Dict[str, Any]
    ) -> None:
        """Build heatmap.glb (+ JSON sidecar) for `model_name`."""
        import numpy as np
        from services.mesh_extractor import extract_heatmap_mesh_for_job

        if job.result is None or "predictions" not in job.result:
            return
        if "heatmaps" not in xai or not xai["heatmaps"]:
            return

        pred_tensor = job.result["predictions"].get(model_name)
        if pred_tensor is None:
            return
        pred_np = (
            pred_tensor.numpy() if hasattr(pred_tensor, "numpy") else np.asarray(pred_tensor)
        ).astype(np.int16)

        heatmaps_np: Dict[int, "np.ndarray"] = {}
        for cls, hm in xai["heatmaps"].items():
            arr = hm.numpy() if hasattr(hm, "numpy") else np.asarray(hm)
            heatmaps_np[int(cls)] = arr.astype(np.float32)

        extract_heatmap_mesh_for_job(
            artifacts_dir=job.artifacts_dir,
            model_name=model_name,
            prediction=pred_np,
            heatmaps_by_class=heatmaps_np,
        )

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