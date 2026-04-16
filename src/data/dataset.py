"""BraTS-style dataset discovery, validation, and split helpers.

This module builds MONAI-ready dictionaries of the form
`{"image": [...], "label": ..., "case_id": ...}` from a directory that stores
one folder per case and one NIfTI file per modality/label inside each folder.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence, TypedDict, cast

import nibabel as nib
import numpy as np


class CaseValidationError(RuntimeError):
    """Raised when a BraTS case is missing required files or is internally inconsistent."""


class BraTSCase(TypedDict):
    """MONAI-ready sample record for one BraTS-style case."""

    case_id: str
    image: list[str]
    label: str
    modalities: dict[str, str]


class SplitManifest(TypedDict):
    """Serialized train/validation split manifest."""

    train_case_ids: list[str]
    val_case_ids: list[str]


def _normalize_root_dir(root_dir: str | Path) -> Path:
    root_text = str(root_dir).strip()
    if not root_text:
        raise ValueError("Dataset root_dir is empty. Set data.root_dir in your config first.")

    root = Path(root_text).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root path is not a directory: {root}")
    return root


def _normalize_expected_modalities(expected_modalities: Sequence[str]) -> list[str]:
    modalities = [str(modality).strip() for modality in expected_modalities if str(modality).strip()]
    if len(modalities) != 4:
        raise ValueError(
            "BraTS-style loading expects exactly 4 MRI modalities. "
            f"Received {len(modalities)} entries: {modalities}"
        )
    if len(set(modalities)) != len(modalities):
        raise ValueError(f"expected_modalities contains duplicates: {modalities}")
    return modalities


def _normalize_alias_map(
    modality_aliases: Mapping[str, Sequence[str]],
    expected_modalities: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    normalized: dict[str, tuple[str, ...]] = {}

    for modality in expected_modalities:
        raw_aliases = modality_aliases.get(modality, (modality,))
        aliases = tuple(alias.strip() for alias in raw_aliases if alias and alias.strip())
        if not aliases:
            raise ValueError(f"No filename aliases configured for modality '{modality}'")
        normalized[modality] = aliases

    return normalized


def _normalize_alias_list(aliases: Sequence[str], *, label: str) -> tuple[str, ...]:
    normalized = tuple(alias.strip() for alias in aliases if alias and alias.strip())
    if not normalized:
        raise ValueError(f"No filename aliases configured for {label}")
    return normalized


def _normalize_extensions(file_extensions: Sequence[str]) -> tuple[str, ...]:
    extensions = tuple(extension.strip() for extension in file_extensions if extension and extension.strip())
    if not extensions:
        raise ValueError("file_extensions must contain at least one value")
    return extensions


def _candidate_filenames(case_id: str, alias: str, extension: str) -> tuple[str, ...]:
    return (
        f"{case_id}-{alias}{extension}",
        f"{case_id}_{alias}{extension}",
        f"{alias}{extension}",
    )


def _find_unique_case_file(
    case_dir: Path,
    case_id: str,
    aliases: Sequence[str],
    extensions: Sequence[str],
) -> Path | None:
    matches: dict[Path, None] = {}

    for alias in aliases:
        for extension in extensions:
            for filename in _candidate_filenames(case_id=case_id, alias=alias, extension=extension):
                candidate = case_dir / filename
                if candidate.exists() and candidate.is_file():
                    matches[candidate.resolve()] = None

            # BraTS exports usually prefix files with the case ID, but we keep one
            # bounded wildcard fallback inside the case folder for minor naming drift.
            for candidate in case_dir.glob(f"*{alias}{extension}"):
                if candidate.is_file():
                    matches[candidate.resolve()] = None

    unique_matches = sorted(matches)
    if len(unique_matches) > 1:
        joined = ", ".join(path.name for path in unique_matches)
        raise CaseValidationError(
            f"Multiple files matched case '{case_id}' for aliases {list(aliases)}: {joined}"
        )
    if unique_matches:
        return unique_matches[0]
    return None


def _load_spatial_shape(nifti_path: Path) -> tuple[int, int, int]:
    image = nib.load(str(nifti_path))
    if len(image.shape) < 3:
        raise CaseValidationError(f"Expected a 3D NIfTI volume, found shape {image.shape} in {nifti_path}")
    return tuple(int(dim) for dim in image.shape[:3])


def _validate_case_shapes(case_id: str, paths: Mapping[str, Path]) -> None:
    reference_name: str | None = None
    reference_shape: tuple[int, int, int] | None = None

    for name, path in paths.items():
        shape = _load_spatial_shape(path)
        if reference_shape is None:
            reference_name = name
            reference_shape = shape
            continue
        if shape != reference_shape:
            raise CaseValidationError(
                f"Shape mismatch in case '{case_id}': '{name}' has shape {shape}, "
                f"but '{reference_name}' has shape {reference_shape}"
            )


def _select_cases_by_id(cases: Sequence[BraTSCase], case_ids: Sequence[str]) -> list[BraTSCase]:
    case_map = {case["case_id"]: case for case in cases}
    unknown_case_ids = sorted(set(case_ids) - set(case_map))
    if unknown_case_ids:
        raise ValueError(f"Split references unknown case IDs: {unknown_case_ids}")
    return [case_map[case_id] for case_id in case_ids]


def discover_brats_cases(
    root_dir: str | Path,
    modality_aliases: Mapping[str, Sequence[str]],
    label_aliases: Sequence[str],
    file_extensions: Sequence[str],
    expected_modalities: Sequence[str],
    validate_shapes: bool = True,
) -> list[BraTSCase]:
    """Discover BraTS-style cases under ``root_dir`` and return MONAI-ready records."""

    root = _normalize_root_dir(root_dir)
    modalities = _normalize_expected_modalities(expected_modalities)
    alias_map = _normalize_alias_map(modality_aliases, modalities)
    label_names = _normalize_alias_list(label_aliases, label="segmentation labels")
    extensions = _normalize_extensions(file_extensions)

    case_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not case_dirs:
        raise FileNotFoundError(f"No case directories found under {root}")

    cases: list[BraTSCase] = []
    errors: list[str] = []

    for case_dir in case_dirs:
        case_id = case_dir.name

        try:
            modality_paths: dict[str, Path] = {}
            for modality in modalities:
                found = _find_unique_case_file(
                    case_dir=case_dir,
                    case_id=case_id,
                    aliases=alias_map[modality],
                    extensions=extensions,
                )
                if found is None:
                    raise CaseValidationError(
                        f"Missing modality '{modality}' for case '{case_id}'. "
                        f"Checked aliases: {', '.join(alias_map[modality])}"
                    )
                modality_paths[modality] = found

            label_path = _find_unique_case_file(
                case_dir=case_dir,
                case_id=case_id,
                aliases=label_names,
                extensions=extensions,
            )
            if label_path is None:
                raise CaseValidationError(
                    f"Missing segmentation mask for case '{case_id}'. "
                    f"Checked aliases: {', '.join(label_names)}"
                )

            if validate_shapes:
                _validate_case_shapes(
                    case_id=case_id,
                    paths={**modality_paths, "label": label_path},
                )

            cases.append(
                {
                    "case_id": case_id,
                    "image": [str(modality_paths[modality]) for modality in modalities],
                    "label": str(label_path),
                    "modalities": {modality: str(modality_paths[modality]) for modality in modalities},
                }
            )
        except CaseValidationError as exc:
            errors.append(str(exc))

    if errors:
        raise CaseValidationError(
            "Dataset validation failed for one or more cases:\n"
            + "\n".join(f"- {message}" for message in errors)
        )

    return cases


def split_cases(
    cases: Sequence[BraTSCase],
    validation_ratio: float,
    seed: int,
    split_manifest_path: str | Path | None = None,
) -> tuple[list[BraTSCase], list[BraTSCase]]:
    """Create or restore a deterministic train/validation split.

    When ``split_manifest_path`` is provided, the split is restored from disk and
    validated against the discovered case IDs. Otherwise a seeded random split is
    produced using ``validation_ratio``.
    """

    if len(cases) < 2:
        raise ValueError("At least two cases are required to create a train/validation split")

    if split_manifest_path is not None:
        manifest = load_split_manifest(split_manifest_path)
        if len(manifest["train_case_ids"]) != len(set(manifest["train_case_ids"])):
            raise ValueError("Split manifest contains duplicate case IDs in train_case_ids")
        if len(manifest["val_case_ids"]) != len(set(manifest["val_case_ids"])):
            raise ValueError("Split manifest contains duplicate case IDs in val_case_ids")

        overlapping_case_ids = sorted(set(manifest["train_case_ids"]) & set(manifest["val_case_ids"]))
        if overlapping_case_ids:
            raise ValueError(
                "Split manifest contains case IDs in both train and val sets: "
                f"{overlapping_case_ids}"
            )

        discovered_case_ids = {case["case_id"] for case in cases}
        assigned_case_ids = set(manifest["train_case_ids"]) | set(manifest["val_case_ids"])
        missing_case_ids = sorted(discovered_case_ids - assigned_case_ids)
        if missing_case_ids:
            raise ValueError(
                "Split manifest does not cover all discovered cases. Missing case IDs: "
                f"{missing_case_ids}"
            )

        train_cases = _select_cases_by_id(cases, manifest["train_case_ids"])
        val_cases = _select_cases_by_id(cases, manifest["val_case_ids"])
        if not train_cases or not val_cases:
            raise ValueError("Split manifest must contain at least one train case and one val case")
        return train_cases, val_cases

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")

    shuffled = list(cases)
    random.Random(seed).shuffle(shuffled)

    val_count = max(1, int(round(len(shuffled) * validation_ratio)))
    if val_count >= len(shuffled):
        raise ValueError("Validation split is too large; training split would become empty")

    val_cases = shuffled[:val_count]
    train_cases = shuffled[val_count:]
    return train_cases, val_cases


def write_split_manifest(
    train_cases: Sequence[BraTSCase],
    val_cases: Sequence[BraTSCase],
    output_dir: str | Path,
) -> Path:
    """Persist the case IDs used for the current experiment."""

    output_path = Path(output_dir).expanduser().resolve() / "splits.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: SplitManifest = {
        "train_case_ids": [case["case_id"] for case in train_cases],
        "val_case_ids": [case["case_id"] for case in val_cases],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def load_split_manifest(split_manifest_path: str | Path) -> SplitManifest:
    """Load a train/validation split manifest from disk."""

    path = Path(split_manifest_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Split manifest not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in split manifest: {path}")

    train_case_ids = payload.get("train_case_ids")
    val_case_ids = payload.get("val_case_ids")
    if not isinstance(train_case_ids, list) or not all(isinstance(item, str) for item in train_case_ids):
        raise TypeError(f"Split manifest has invalid train_case_ids: {path}")
    if not isinstance(val_case_ids, list) or not all(isinstance(item, str) for item in val_case_ids):
        raise TypeError(f"Split manifest has invalid val_case_ids: {path}")

    return {
        "train_case_ids": cast(list[str], train_case_ids),
        "val_case_ids": cast(list[str], val_case_ids),
    }


def load_case_arrays(case: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Load one case into NumPy arrays for notebooks and lightweight inspection."""

    modality_items = case.get("modalities")
    if not isinstance(modality_items, Mapping):
        raise TypeError("Case dictionary must include a 'modalities' mapping")

    arrays = {
        str(modality): nib.load(str(path)).get_fdata().astype(np.float32)
        for modality, path in modality_items.items()
    }
    arrays["label"] = nib.load(str(case["label"])).get_fdata().astype(np.int16)
    return arrays


def inspect_case(case: Mapping[str, Any]) -> dict[str, Any]:
    """Return a lightweight summary for one BraTS-style case."""

    modality_items = case.get("modalities")
    if not isinstance(modality_items, Mapping):
        raise TypeError("Case dictionary must include a 'modalities' mapping")

    label_image = nib.load(str(case["label"]))
    label_array = label_image.get_fdata()
    modality_summaries: dict[str, dict[str, Any]] = {}

    for modality, path in modality_items.items():
        image = nib.load(str(path))
        modality_summaries[str(modality)] = {
            "path": str(path),
            "shape": tuple(int(dim) for dim in image.shape[:3]),
            "spacing": tuple(float(value) for value in image.header.get_zooms()[:3]),
        }

    return {
        "case_id": str(case["case_id"]),
        "modalities": modality_summaries,
        "label_path": str(case["label"]),
        "label_shape": tuple(int(dim) for dim in label_image.shape[:3]),
        "label_values": [int(value) for value in np.unique(label_array)],
    }
