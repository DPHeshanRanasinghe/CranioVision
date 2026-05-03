"""
CranioVision — Uploaded folder parsing.

Takes an uploaded BraTS case folder (containing .nii or .nii.gz files for
4 modalities + optional GT) and turns it into a case dict the pipeline
understands.

Expected folder structure:
    BraTS-GLI-02143-102/
      BraTS-GLI-02143-102-t1n.nii
      BraTS-GLI-02143-102-t1c.nii
      BraTS-GLI-02143-102-t2w.nii
      BraTS-GLI-02143-102-t2f.nii
      BraTS-GLI-02143-102-seg.nii    (optional ground truth)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Required modalities in BraTS order
REQUIRED_MODALITIES = ("t1n", "t1c", "t2w", "t2f")
GT_SUFFIXES = ("seg", "label", "mask")


def _find_case_id_from_files(file_names: List[str]) -> Optional[str]:
    """Try to extract the BraTS case ID from a list of filenames."""
    for name in file_names:
        # BraTS-GLI-02143-102-t1n.nii  ->  BraTS-GLI-02143-102
        if "-t1n" in name.lower():
            stem = name.lower().split("-t1n")[0]
            # Recover the original casing by finding it in the actual name
            idx = name.lower().find(stem)
            return name[idx : idx + len(stem)]
        for mod in REQUIRED_MODALITIES:
            if f"-{mod}" in name.lower():
                stem = name.lower().split(f"-{mod}")[0]
                idx = name.lower().find(stem)
                return name[idx : idx + len(stem)]
    return None


def parse_case_folder(folder: Path) -> Dict[str, Any]:
    """
    Parse a folder into a case dict matching the pipeline schema.

    Returns
    -------
    dict with keys:
        case_id : str
        image   : list[str] of 4 modality paths in [t1n, t1c, t2w, t2f] order
        label   : str path to GT mask if present, else None

    Raises
    ------
    ValueError if required modalities are missing.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder}")

    files = [f for f in folder.iterdir() if f.is_file()
             and f.suffix.lower() in (".nii", ".gz")]

    case_id = (
        _find_case_id_from_files([f.name for f in files])
        or folder.name
    )

    # Find each modality
    modality_paths: Dict[str, Optional[Path]] = {m: None for m in REQUIRED_MODALITIES}
    label_path: Optional[Path] = None

    for f in files:
        name_lower = f.name.lower()
        for mod in REQUIRED_MODALITIES:
            if f"-{mod}." in name_lower or f"_{mod}." in name_lower:
                modality_paths[mod] = f
                break
        else:
            for suf in GT_SUFFIXES:
                if f"-{suf}." in name_lower or f"_{suf}." in name_lower:
                    label_path = f
                    break

    missing = [m for m, p in modality_paths.items() if p is None]
    if missing:
        raise ValueError(
            f"Missing required modalities in {folder}: {missing}. "
            f"Found files: {[f.name for f in files]}"
        )

    image_paths = [str(modality_paths[m]) for m in REQUIRED_MODALITIES]

    return {
        "case_id": case_id,
        "image": image_paths,
        "label": str(label_path) if label_path else None,
    }


def is_registration_cached(case_id: str) -> bool:
    """Check whether atlas registration is already cached for this case."""
    # Resolve path relative to project root, regardless of import context
    project_root = Path(__file__).resolve().parent.parent.parent
    cache_dir = project_root / "outputs" / "atlas_cache" / case_id

    if not cache_dir.exists():
        return False
    expected = ["warped_t1.nii.gz", "warped_mask.nii.gz",
                "fwd_affine.mat", "fwd_warp.nii.gz", "metadata.json"]
    return all((cache_dir / f).exists() for f in expected)