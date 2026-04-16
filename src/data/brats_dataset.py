"""BraTS-style case discovery and validation utilities."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping

import nibabel as nib
import numpy as np


class CaseValidationError(RuntimeError):
    """Raised when a BraTS case is missing required files or has invalid shapes."""


def _require_root(root_dir: str | Path) -> Path:
    root = Path(root_dir).expanduser().resolve()
    if not str(root_dir).strip():
        raise ValueError(
            "Dataset root_dir is empty. Set data.root_dir in configs/default.yaml first."
        )
    if not root.exists():
        raise FileNotFoundError(f"Dataset root directory does not exist: {root}")
    return root


def _find_case_file(
    case_dir: Path,
    case_id: str,
    aliases: list[str],
    extensions: list[str],
) -> Path | None:
    for alias in aliases:
        for extension in extensions:
            candidate = case_dir / f"{case_id}-{alias}{extension}"
            if candidate.exists():
                return candidate
    return None


def _load_spatial_shape(nifti_path: Path) -> tuple[int, int, int]:
    image = nib.load(str(nifti_path))
    shape = image.shape
    if len(shape) < 3:
        raise CaseValidationError(f"Expected a 3D NIfTI volume, found shape {shape} in {nifti_path}")
    return tuple(int(dim) for dim in shape[:3])


def discover_brats_cases(
    root_dir: str | Path,
    modality_aliases: Mapping[str, list[str]],
    label_aliases: list[str],
    file_extensions: list[str],
    expected_modalities: list[str],
    validate_shapes: bool = True,
) -> list[dict[str, Any]]:
    """Scan a BraTS-style directory and return MONAI-ready case dictionaries.

    Each case dictionary follows the MONAI dictionary dataset convention:
    `{"image": [modality_paths...], "label": label_path, "case_id": ..., ...}`.
    """

    root = _require_root(root_dir)
    case_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not case_dirs:
        raise FileNotFoundError(f"No case directories found under {root}")

    cases: list[dict[str, Any]] = []
    errors: list[str] = []

    for case_dir in case_dirs:
        case_id = case_dir.name
        modality_paths: dict[str, str] = {}

        try:
            for modality in expected_modalities:
                aliases = modality_aliases.get(modality, [modality])
                found = _find_case_file(case_dir, case_id, aliases, file_extensions)
                if found is None:
                    alias_text = ", ".join(aliases)
                    raise CaseValidationError(
                        f"Missing modality '{modality}' for case '{case_id}'. "
                        f"Checked aliases: {alias_text}"
                    )
                modality_paths[modality] = str(found)

            label_path = _find_case_file(case_dir, case_id, label_aliases, file_extensions)
            if label_path is None:
                raise CaseValidationError(
                    f"Missing segmentation label for case '{case_id}'. "
                    f"Checked aliases: {', '.join(label_aliases)}"
                )

            if validate_shapes:
                reference_shape: tuple[int, int, int] | None = None
                shape_sources = {**modality_paths, "label": str(label_path)}
                for name, path_string in shape_sources.items():
                    shape = _load_spatial_shape(Path(path_string))
                    if reference_shape is None:
                        reference_shape = shape
                    elif shape != reference_shape:
                        raise CaseValidationError(
                            f"Shape mismatch in case '{case_id}': '{name}' has shape {shape}, "
                            f"expected {reference_shape}"
                        )

            cases.append(
                {
                    "case_id": case_id,
                    "image": [modality_paths[modality] for modality in expected_modalities],
                    "label": str(label_path),
                    "modalities": modality_paths,
                }
            )
        except CaseValidationError as exc:
            errors.append(str(exc))

    if errors:
        joined = "\n".join(f"- {message}" for message in errors)
        raise CaseValidationError(
            "Dataset validation failed for one or more cases:\n"
            f"{joined}"
        )

    return cases


def split_cases(
    cases: list[dict[str, Any]],
    validation_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create a reproducible train/validation split from discovered cases."""

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    if len(cases) < 2:
        raise ValueError("At least two cases are required to create a train/validation split")

    shuffled = list(cases)
    random.Random(seed).shuffle(shuffled)

    val_count = max(1, int(round(len(shuffled) * validation_ratio)))
    val_cases = shuffled[:val_count]
    train_cases = shuffled[val_count:]

    if not train_cases:
        raise ValueError("Validation split is too large; training split became empty")

    return train_cases, val_cases


def write_split_manifest(
    train_cases: list[dict[str, Any]],
    val_cases: list[dict[str, Any]],
    output_dir: str | Path,
) -> Path:
    """Persist the case IDs used for the current experiment."""

    output_path = Path(output_dir).expanduser().resolve() / "splits.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_case_ids": [case["case_id"] for case in train_cases],
        "val_case_ids": [case["case_id"] for case in val_cases],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def load_case_arrays(case: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Load one case into NumPy arrays for notebooks and visualization."""

    arrays = {
        modality: nib.load(path).get_fdata().astype(np.float32)
        for modality, path in case["modalities"].items()
    }
    arrays["label"] = nib.load(case["label"]).get_fdata().astype(np.int16)
    return arrays


def inspect_case(case: Mapping[str, Any]) -> dict[str, Any]:
    """Return a lightweight summary for one BraTS-style case."""

    label_image = nib.load(case["label"])
    label_array = label_image.get_fdata()
    modality_summaries: dict[str, dict[str, Any]] = {}

    for modality, path in case["modalities"].items():
        image = nib.load(path)
        modality_summaries[modality] = {
            "path": path,
            "shape": tuple(int(dim) for dim in image.shape[:3]),
            "spacing": tuple(float(value) for value in image.header.get_zooms()[:3]),
        }

    return {
        "case_id": case["case_id"],
        "modalities": modality_summaries,
        "label_path": case["label"],
        "label_shape": tuple(int(dim) for dim in label_image.shape[:3]),
        "label_values": [int(value) for value in np.unique(label_array)],
    }
