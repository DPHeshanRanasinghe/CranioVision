"""
CranioVision — Mesh extraction for the 3D viewer.

Brain shell + per-class tumor meshes for the Plotly Mesh3d viewer. Both live
in MNI152 1mm voxel space (identity affine) so they always align.

Why MNI atlas + warped GT instead of patient T1n + raw prediction:
    - Patient T1n includes skull/scalp/eyes which confused Otsu and produced a
      noisy "brain" surface that didn't look like a brain.
    - Predictions live in the BraTS-preprocessed crop, not the patient native
      grid, so we couldn't easily warp them. The GT mask, however, lives in
      patient space and the cached fwd_warp + fwd_affine apply_transforms
      directly to it.
    - The MNI brain mask is already skull-stripped, so the shell extraction
      is just `vol > 0` + marching cubes — no thresholding.

Coordinate system:
    Identity affine in MNI 1mm voxel grid (~182 x 218 x 182). Both shell and
    tumor meshes derive from volumes in this grid, so they share coordinates
    in the viewer with no further transforms.

NOTE on heatmap meshes:
    The Grad-CAM heatmap mesh is still derived from the BraTS-preprocessed
    prediction grid, so it does NOT align with the MNI shell. Toggling the
    heatmap will show a tumor surface floating off-centre relative to the
    brain. Aligning that requires warping the heatmap volumes through the
    same cached transforms; deferred to a follow-up.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Project root: backend/services/mesh_extractor.py -> CranioVision/
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
MNI_BRAIN_PATH = _PROJECT_ROOT / "atlas_data" / "MNI152NLin2009cAsym_1mm_T1_brain.nii.gz"
ATLAS_CACHE_DIR = _PROJECT_ROOT / "outputs" / "atlas_cache"


# Class label -> (display_name, RGBA fill colour for fallback / face colour).
TUMOR_CLASSES: Dict[int, Tuple[str, Tuple[int, int, int, int]]] = {
    1: ("edema",     (255, 215,   0, 255)),  # yellow
    2: ("enhancing", (235,  51,  51, 255)),  # red
    3: ("necrotic",  ( 51, 115, 217, 255)),  # blue
}

TAUBIN_ITERATIONS = 5
BRAIN_SHELL_DECIMATE_FACES = 30_000
TUMOR_DECIMATE_FACES = 25_000

# Skip a class if fewer than this many voxels — marching_cubes can produce a
# degenerate mesh with isolated voxels and trimesh export errors out.
MIN_CLASS_VOXELS = 50


# Process-level cache for the MNI brain shell. The atlas is constant across
# every job, so we only ever extract the surface once per backend process.
_MNI_SHELL_MESH = None
_MNI_SHELL_BOUNDS: Optional[List[List[float]]] = None
# x-axis label direction in the MNI atlas storage order, derived once when
# the shell is extracted. Tuple of ('positive_x_label', 'negative_x_label')
# where each is 'L' or 'R' depending on the file's orientation. Used by the
# frontend to place anatomical L/R annotations correctly.
_MNI_X_AXIS_LABELS: Optional[Tuple[str, str]] = None


def _decimate(mesh, target_faces: int):
    """Quadric decimation via fast-simplification; no-op if unavailable."""
    if len(mesh.faces) <= target_faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=target_faces)
    except Exception:
        return mesh


# ---------------------------------------------------------------------------
# CORE EXTRACTION
# ---------------------------------------------------------------------------

def _marching_cubes_to_mesh(
    binary: np.ndarray,
    affine: np.ndarray,
    smoothing_iters: int = TAUBIN_ITERATIONS,
):
    """
    Marching cubes on a binary volume + Taubin smoothing.

    Returns a trimesh.Trimesh in the affine's coordinate space, or None if
    the volume is empty / produced a degenerate mesh.
    """
    import trimesh
    from skimage import measure

    if binary.sum() < MIN_CLASS_VOXELS:
        return None

    # Pad by 1 voxel so the surface closes at the volume boundary.
    padded = np.pad(binary.astype(np.uint8), 1, mode="constant")

    try:
        verts, faces, normals, _ = measure.marching_cubes(
            padded, level=0.5, allow_degenerate=False,
        )
    except (ValueError, RuntimeError):
        return None

    if len(verts) == 0 or len(faces) == 0:
        return None

    # Undo the 1-voxel pad so vertices live in the original index grid.
    verts = verts - 1.0

    # Map voxel indices -> world coords via the supplied affine.
    homog = np.hstack([verts, np.ones((verts.shape[0], 1))])
    world = (affine @ homog.T).T[:, :3]

    mesh = trimesh.Trimesh(
        vertices=world,
        faces=faces,
        vertex_normals=normals,
        process=False,
    )

    if smoothing_iters > 0 and len(mesh.faces) > 0:
        try:
            trimesh.smoothing.filter_taubin(mesh, iterations=smoothing_iters)
        except Exception:
            pass

    if len(mesh.faces) == 0:
        return None

    return mesh


def extract_class_mesh(
    pred_array: np.ndarray,
    class_value: int,
    affine: np.ndarray,
):
    """One mesh for one tumor class (1=edema, 2=enhancing, 3=necrotic)."""
    binary = (pred_array == class_value)
    mesh = _marching_cubes_to_mesh(binary, affine)
    if mesh is None:
        return None

    mesh = _decimate(mesh, TUMOR_DECIMATE_FACES)
    name, rgba = TUMOR_CLASSES.get(class_value, (f"class_{class_value}", (200, 200, 200, 255)))
    mesh.visual.face_colors = np.tile(np.array(rgba, dtype=np.uint8), (len(mesh.faces), 1))
    mesh.metadata["cranovision_class"] = name
    return mesh


def _largest_mesh_component(mesh):
    """Largest connected mesh component by volume (fallback area, then face count)."""
    try:
        parts = mesh.split(only_watertight=False)
    except Exception:
        return mesh
    if not parts:
        return mesh
    if len(parts) == 1:
        return parts[0]

    def _score(m):
        try:
            v = abs(float(m.volume))
            if v > 0:
                return v
        except Exception:
            pass
        try:
            return float(m.area)
        except Exception:
            return float(len(m.faces))

    return max(parts, key=_score)


def _x_axis_anatomical_labels(affine: np.ndarray) -> Tuple[str, str]:
    """
    Determine which end of the volume's first axis is patient Left vs Right.

    Returns (positive_x_label, negative_x_label) where each is 'L' or 'R'.
    nibabel uses RAS-world conventions, so axcodes[0] tells us whether
    increasing the first voxel index moves the world point toward Right
    ('R') or Left ('L'). Marching cubes operates in voxel-index space with
    an identity affine, so this mapping carries straight through to the
    mesh's +x direction.
    """
    import nibabel as nib

    try:
        axcodes = nib.aff2axcodes(affine)
    except Exception:
        return ("L", "R")  # safe default for ANTs / LPS pipelines

    first = axcodes[0]
    if first == "L":
        return ("L", "R")
    if first == "R":
        return ("R", "L")
    return ("L", "R")


def extract_brain_shell_mesh(*_args, **_kwargs):
    """
    Brain shell from the MNI152 1mm atlas template. Cached process-wide
    since the atlas never changes.

    Args/kwargs are accepted but ignored — the old signature took
    `(t1n_array, affine)`; some callers still pass them.
    """
    global _MNI_SHELL_MESH, _MNI_SHELL_BOUNDS, _MNI_X_AXIS_LABELS

    if _MNI_SHELL_MESH is not None:
        return _MNI_SHELL_MESH

    import nibabel as nib

    if not MNI_BRAIN_PATH.exists():
        return None

    img = nib.load(str(MNI_BRAIN_PATH))
    _MNI_X_AXIS_LABELS = _x_axis_anatomical_labels(img.affine)
    vol = np.asarray(img.dataobj).astype(np.float32)

    # MNI brain template is already skull-stripped — anything > 0 is brain.
    binary = vol > 0
    if binary.sum() < MIN_CLASS_VOXELS:
        return None

    affine = np.eye(4)  # voxel-space; warped tumor masks share this grid
    mesh = _marching_cubes_to_mesh(binary, affine, smoothing_iters=TAUBIN_ITERATIONS)
    if mesh is None:
        return None

    mesh = _largest_mesh_component(mesh)
    mesh = _decimate(mesh, BRAIN_SHELL_DECIMATE_FACES)
    mesh = _largest_mesh_component(mesh)

    mesh.visual.face_colors = np.tile(
        np.array([180, 180, 195, 255], dtype=np.uint8), (len(mesh.faces), 1),
    )
    mesh.metadata["cranovision_class"] = "brain_shell"

    _MNI_SHELL_MESH = mesh
    _MNI_SHELL_BOUNDS = mesh.bounds.tolist()
    return mesh


def get_mni_brain_bounds() -> Optional[List[List[float]]]:
    """Bounds of the MNI shell mesh (after extraction). None until first call."""
    return _MNI_SHELL_BOUNDS


def get_mni_x_axis_labels() -> Optional[Tuple[str, str]]:
    """(positive_x_label, negative_x_label) for the MNI shell, or None."""
    return _MNI_X_AXIS_LABELS


# ---------------------------------------------------------------------------
# GLB / JSON EXPORT
# ---------------------------------------------------------------------------

def export_glb(mesh, out_path: Path) -> None:
    """Write a single mesh as a binary glTF (.glb)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    glb = mesh.export(file_type="glb")
    out_path.write_bytes(glb)


def export_mesh_json(mesh, out_path: Path, intensity: Optional[np.ndarray] = None) -> None:
    """Write a Plotly-Mesh3d JSON sidecar: {x,y,z,i,j,k,intensity?}."""
    import json
    out_path.parent.mkdir(parents=True, exist_ok=True)

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    payload: Dict[str, list] = {
        "x": verts[:, 0].tolist(),
        "y": verts[:, 1].tolist(),
        "z": verts[:, 2].tolist(),
        "i": faces[:, 0].tolist(),
        "j": faces[:, 1].tolist(),
        "k": faces[:, 2].tolist(),
    }
    if intensity is not None:
        payload["intensity"] = np.asarray(intensity, dtype=np.float32).tolist()

    out_path.write_text(json.dumps(payload, separators=(",", ":")))


def export_mesh(mesh, glb_path: Path, intensity: Optional[np.ndarray] = None) -> None:
    """Write both the .glb and the .json sidecar for one mesh."""
    export_glb(mesh, glb_path)
    export_mesh_json(mesh, glb_path.with_suffix(".json"), intensity=intensity)


# ---------------------------------------------------------------------------
# HEATMAP MESH (Grad-CAM coloured tumor surface, preprocessed grid)
# ---------------------------------------------------------------------------

def _sample_volume_at_vertices(volume: np.ndarray, verts: np.ndarray) -> np.ndarray:
    from scipy.ndimage import map_coordinates
    coords = np.vstack([verts[:, 0], verts[:, 1], verts[:, 2]])
    return map_coordinates(volume, coords, order=1, mode="nearest", cval=0.0)


def _magma_colors(intensities: np.ndarray) -> np.ndarray:
    from matplotlib import cm
    rgba = cm.get_cmap("magma")(np.clip(intensities, 0.0, 1.0))
    return (rgba * 255).astype(np.uint8)


def extract_heatmap_mesh(
    prediction: np.ndarray,
    heatmaps_by_class: Dict[int, np.ndarray],
    affine: np.ndarray,
):
    import trimesh

    if not heatmaps_by_class:
        return None

    binary = (prediction > 0)
    if binary.sum() < MIN_CLASS_VOXELS:
        return None

    mesh = _marching_cubes_to_mesh(binary, affine, smoothing_iters=TAUBIN_ITERATIONS)
    if mesh is None:
        return None
    mesh = _decimate(mesh, TUMOR_DECIMATE_FACES)
    if len(mesh.vertices) == 0:
        return None

    sampled_stack = []
    for hm in heatmaps_by_class.values():
        if hm.shape != prediction.shape:
            continue
        sampled_stack.append(_sample_volume_at_vertices(hm, mesh.vertices))

    if not sampled_stack:
        return None

    intensities = np.maximum.reduce(sampled_stack)
    imax = float(intensities.max())
    if imax > 1e-8:
        intensities = intensities / imax
    else:
        intensities = np.zeros_like(intensities)

    colors = _magma_colors(intensities)
    mesh.visual.vertex_colors = colors
    mesh.metadata["cranovision_class"] = "heatmap"
    mesh.metadata["cranovision_intensity"] = intensities.astype(np.float32)
    return mesh


def extract_heatmap_mesh_for_job(
    artifacts_dir: Path,
    model_name: str,
    prediction: np.ndarray,
    heatmaps_by_class: Dict[int, np.ndarray],
) -> Optional[Path]:
    affine = np.eye(4)
    mesh = extract_heatmap_mesh(prediction, heatmaps_by_class, affine)
    if mesh is None:
        return None

    out_dir = artifacts_dir / "meshes" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "heatmap.glb"
    intensity = mesh.metadata.get("cranovision_intensity")
    export_mesh(mesh, out_path, intensity=intensity)
    return out_path


# ---------------------------------------------------------------------------
# WARPED TUMOR MESH (multi-class GT in MNI space, shared across model tabs)
# ---------------------------------------------------------------------------

def _warp_gt_to_mni_multiclass(
    case_id: str,
    gt_mask_path: str,
    cache_path: Path,
) -> Optional[np.ndarray]:
    """
    Warp the raw BraTS GT mask into MNI space and remap the labels into the
    project's internal scheme so TUMOR_CLASSES (1=edema, 2=enhancing,
    3=necrotic) lines up.

    Raw BraTS labels:    {0=bg, 2=edema, 3=enhancing, 4=necrotic}
    Internal labels:     {0=bg, 1=edema, 2=enhancing, 3=necrotic}
    (Raw label 1 — non-enhancing/atypical — is dropped to background, matching
    src/cranovision/config.py:LABEL_MAP. Without this remap the class IDs are
    off by one and necrotic disappears from the rendered tumor mesh.)

    Interpolator is `genericLabel` (NN-style for label maps), which is
    mandatory: linear interpolation would smear small classes like necrotic
    into adjacent labels and they'd vanish.

    Cached on disk per case at `cache_path`.
    """
    import nibabel as nib

    if cache_path.exists():
        try:
            img = nib.load(str(cache_path))
            return np.asarray(img.dataobj).astype(np.int16)
        except Exception:
            cache_path.unlink(missing_ok=True)

    case_cache = ATLAS_CACHE_DIR / case_id
    fwd_warp = case_cache / "fwd_warp.nii.gz"
    fwd_affine = case_cache / "fwd_affine.mat"

    if not (case_cache.exists() and fwd_warp.exists() and fwd_affine.exists()):
        return None
    if not Path(gt_mask_path).exists():
        return None

    try:
        import ants
    except Exception:
        return None

    fixed = ants.image_read(str(MNI_BRAIN_PATH))
    moving = ants.image_read(str(gt_mask_path))
    warped = ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=[str(fwd_warp), str(fwd_affine)],
        interpolator="genericLabel",
    )

    arr = warped.numpy().astype(np.int16)

    # Raw BraTS labels -> internal class indices (matches config.LABEL_MAP).
    remapped = np.zeros_like(arr, dtype=np.int16)
    remapped[arr == 2] = 1  # edema
    remapped[arr == 3] = 2  # enhancing
    remapped[arr == 4] = 3  # necrotic

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    remapped_img = warped.new_image_like(remapped.astype(np.float32))
    ants.image_write(remapped_img, str(cache_path))

    return remapped


def _load_binary_warped_mask(case_id: str) -> Optional[np.ndarray]:
    """Fallback: the binary warped_mask.nii.gz cached by the pipeline."""
    import nibabel as nib

    p = ATLAS_CACHE_DIR / case_id / "warped_mask.nii.gz"
    if not p.exists():
        return None
    try:
        return np.asarray(nib.load(str(p)).dataobj).astype(np.int16)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# JOB-LEVEL ORCHESTRATION
# ---------------------------------------------------------------------------

def _load_warped_prediction(path: Path) -> Optional[np.ndarray]:
    """Load a per-model prediction warped to MNI as int16. None on failure."""
    import nibabel as nib

    if not path.exists():
        return None
    try:
        return np.asarray(nib.load(str(path)).dataobj).astype(np.int16)
    except Exception:
        return None


def extract_meshes_for_job(
    artifacts_dir: Path,
    case_id: str,
    model_names: List[str],
    gt_mask_path: Optional[str] = None,
    predictions_mni_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, object]]:
    """
    Build the brain shell (MNI atlas) + per-class tumor meshes for every
    model.

    Source priority for each model's tumor surface:
      1. `predictions_mni_dir/{model}.nii.gz` — the model's own prediction
         warped to MNI (shows real per-model differences across tabs).
      2. GT mask warped to MNI (shared across all models — fallback only).
      3. Binary `warped_mask.nii.gz` from atlas_cache (single-surface fallback
         when there's no GT mask at all).

    File layout written under `artifacts_dir`:
        meshes/brain.glb + brain.json
        meshes/{model}/edema.glb + .json
        meshes/{model}/enhancing.glb + .json
        meshes/{model}/necrotic.glb + .json
        meshes/{model}/manifest.json
    """
    import json

    meshes_dir = artifacts_dir / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)

    affine = np.eye(4)

    # ---- brain shell (cached process-wide) ----
    brain_path = meshes_dir / "brain.glb"
    brain_json_path = brain_path.with_suffix(".json")
    brain_bounds = None
    shell = extract_brain_shell_mesh()
    if shell is not None:
        if not brain_path.exists() or not brain_json_path.exists():
            export_mesh(shell, brain_path)
        brain_bounds = shell.bounds.tolist()

    x_labels = get_mni_x_axis_labels() or ("L", "R")
    axis_labels = {"x_pos": x_labels[0], "x_neg": x_labels[1]}

    # ---- shared fallback: GT warped to MNI ----
    multiclass_cache = ATLAS_CACHE_DIR / case_id / "warped_mask_multiclass_v2.nii.gz"
    warped_multiclass: Optional[np.ndarray] = None
    if gt_mask_path:
        warped_multiclass = _warp_gt_to_mni_multiclass(
            case_id=case_id,
            gt_mask_path=gt_mask_path,
            cache_path=multiclass_cache,
        )

    binary_fallback_mask: Optional[np.ndarray] = None
    if warped_multiclass is None:
        binary_fallback_mask = _load_binary_warped_mask(case_id)

    # ---- per-model class meshes ----
    summary: Dict[str, Dict[str, object]] = {}
    for model_name in model_names:
        model_dir = meshes_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Prefer per-model warped prediction; this is the path that makes
        # tabs visibly different.
        per_model_volume: Optional[np.ndarray] = None
        if predictions_mni_dir is not None:
            per_model_volume = _load_warped_prediction(
                predictions_mni_dir / f"{model_name}.nii.gz"
            )

        source = "per_model_prediction" if per_model_volume is not None else None

        # Fall back to the shared GT-derived sources.
        if per_model_volume is None and warped_multiclass is not None:
            per_model_volume = warped_multiclass
            source = "shared_gt_multiclass"

        available: List[str] = []

        if per_model_volume is not None:
            for class_value, (class_name, _rgba) in TUMOR_CLASSES.items():
                out_path = model_dir / f"{class_name}.glb"
                json_path = out_path.with_suffix(".json")

                # Force re-extraction whenever the upstream source changes,
                # otherwise tabs would keep showing stale GT meshes after we
                # switch them to per-model predictions on a re-run.
                if out_path.exists() and json_path.exists():
                    out_path.unlink(missing_ok=True)
                    json_path.unlink(missing_ok=True)

                if (per_model_volume == class_value).sum() < MIN_CLASS_VOXELS:
                    continue
                mesh = extract_class_mesh(per_model_volume, class_value, affine)
                if mesh is None:
                    continue
                export_mesh(mesh, out_path)
                available.append(class_name)

        elif binary_fallback_mask is not None:
            # No GT and no per-model warp — single binary surface coloured as
            # 'enhancing' for the legend.
            class_name = "enhancing"
            out_path = model_dir / f"{class_name}.glb"
            json_path = out_path.with_suffix(".json")
            if out_path.exists() and json_path.exists():
                out_path.unlink(missing_ok=True)
                json_path.unlink(missing_ok=True)
            pseudo = ((binary_fallback_mask > 0).astype(np.int16)) * 2
            mesh = extract_class_mesh(pseudo, 2, affine)
            if mesh is not None:
                export_mesh(mesh, out_path)
                available.append(class_name)
            source = "binary_fallback"

        manifest: Dict[str, object] = {
            "model": model_name,
            "available_classes": available,
            "brain_bounds": brain_bounds,
            "axis_labels": axis_labels,
            "source": source,
        }
        (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        summary[model_name] = manifest

    return summary
