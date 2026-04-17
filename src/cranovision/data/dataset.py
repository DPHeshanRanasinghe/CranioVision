"""
CranioVision — BraTS 2024 dataset scanner and splitter.

Scans the BraTS folder structure, validates all 5 files per patient,
and produces MONAI-style data dicts: {'image': [4 paths], 'label': seg_path}.

Also provides deterministic train/val/test splitting with persistence
to JSON — so splits are reproducible across all 3 model branches.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from ..config import RAW_DATA_DIR, SPLITS_DIR, MODALITIES, SEED


# ══════════════════════════════════════════════════════════════════════════════
# SCANNING
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_file(base_path: Path) -> Optional[Path]:
    """Return base_path if it exists, else base_path.nii.gz variant, else None."""
    if base_path.exists():
        return base_path
    gz = base_path.with_suffix(base_path.suffix + ".gz")
    if gz.exists():
        return gz
    return None


def scan_brats_dataset(data_dir: Path = RAW_DATA_DIR,
                       verbose: bool = True) -> List[Dict]:
    """
    Walk the BraTS folder and return a list of MONAI data dicts.

    Each dict: {
        'image'   : [t1n_path, t1c_path, t2w_path, t2f_path],  # ORDER MATTERS
        'label'   : seg_path,
        'case_id' : 'BraTS-GLI-XXXXX-XXX',
    }

    Cases with any missing file are skipped (with warning).
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    data_list: List[Dict] = []
    missing_cases: List[Tuple[str, List[str]]] = []

    patient_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if verbose:
        print(f"Scanning {len(patient_dirs)} patient folders in {data_dir.name}/")

    for patient_dir in patient_dirs:
        case_id = patient_dir.name

        # Build expected file paths for all 4 modalities + segmentation
        file_paths = {}
        for modality in MODALITIES:
            base = patient_dir / f"{case_id}-{modality}.nii"
            resolved = _resolve_file(base)
            file_paths[modality] = resolved

        seg_base = patient_dir / f"{case_id}-seg.nii"
        file_paths["seg"] = _resolve_file(seg_base)

        # Check completeness
        missing = [k for k, v in file_paths.items() if v is None]
        if missing:
            missing_cases.append((case_id, missing))
            continue

        data_list.append({
            "image"  : [str(file_paths[m]) for m in MODALITIES],
            "label"  : str(file_paths["seg"]),
            "case_id": case_id,
        })

    if verbose:
        print(f"  Valid cases: {len(data_list)}")
        if missing_cases:
            print(f"  Skipped {len(missing_cases)} cases with missing files:")
            for case, miss in missing_cases[:5]:
                print(f"    {case}: missing {miss}")
            if len(missing_cases) > 5:
                print(f"    ... and {len(missing_cases) - 5} more")

    return data_list


# ══════════════════════════════════════════════════════════════════════════════
# SPLITTING — train / val / test
# ══════════════════════════════════════════════════════════════════════════════

def split_dataset(data_list: List[Dict],
                  train_ratio: float = 0.70,
                  val_ratio: float = 0.15,
                  seed: int = SEED) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Shuffle and split into train / val / test.
    test_ratio = 1 - train_ratio - val_ratio.
    """
    assert 0 < train_ratio < 1
    assert 0 < val_ratio < 1
    assert train_ratio + val_ratio < 1, "train + val must be < 1"

    rng = random.Random(seed)
    shuffled = data_list.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]

    return train, val, test


# ══════════════════════════════════════════════════════════════════════════════
# PERSIST / LOAD SPLITS
# ══════════════════════════════════════════════════════════════════════════════

def save_split(train: List[Dict], val: List[Dict], test: List[Dict],
               split_file: Path = None) -> Path:
    """Save split as JSON so all branches/notebooks use identical splits."""
    if split_file is None:
        split_file = SPLITS_DIR / "data_split.json"
    split_file = Path(split_file)
    split_file.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "seed" : SEED,
        "train": [d["case_id"] for d in train],
        "val"  : [d["case_id"] for d in val],
        "test" : [d["case_id"] for d in test],
    }
    with open(split_file, "w") as f:
        json.dump(payload, f, indent=2)
    return split_file


def load_split(all_cases: List[Dict],
               split_file: Path = None
               ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load a previously saved split and return (train, val, test) lists
    filtered from the provided all_cases list.
    """
    if split_file is None:
        split_file = SPLITS_DIR / "data_split.json"
    split_file = Path(split_file)
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with open(split_file) as f:
        payload = json.load(f)

    case_lookup = {d["case_id"]: d for d in all_cases}

    def _pick(ids: List[str]) -> List[Dict]:
        out = []
        for cid in ids:
            if cid in case_lookup:
                out.append(case_lookup[cid])
        return out

    return _pick(payload["train"]), _pick(payload["val"]), _pick(payload["test"])


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY: get-or-create split
# ══════════════════════════════════════════════════════════════════════════════

def get_splits(data_dir: Path = RAW_DATA_DIR,
               split_file: Path = None,
               force_new: bool = False,
               verbose: bool = True
               ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    One-stop helper: scans dataset, loads existing split if present,
    else creates + saves a new split.

    Use this from notebooks — no need to call scan_brats_dataset directly.
    """
    if split_file is None:
        split_file = SPLITS_DIR / "data_split.json"

    all_cases = scan_brats_dataset(data_dir, verbose=verbose)

    if split_file.exists() and not force_new:
        if verbose:
            print(f"Loading existing split from {split_file.name}")
        train, val, test = load_split(all_cases, split_file)
    else:
        if verbose:
            print(f"Creating new split (seed={SEED})")
        train, val, test = split_dataset(all_cases)
        save_split(train, val, test, split_file)
        if verbose:
            print(f"Split saved to {split_file}")

    if verbose:
        n = len(all_cases)
        print(f"  Train: {len(train)} ({len(train)/n*100:.0f}%) | "
              f"Val: {len(val)} ({len(val)/n*100:.0f}%) | "
              f"Test: {len(test)} ({len(test)/n*100:.0f}%)")

    return train, val, test


# ══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Quick standalone test — run: python -m src.cranovision.data.dataset
    print("=" * 60)
    print("CranioVision — dataset.py smoke test")
    print("=" * 60)

    train, val, test = get_splits()
    print("\nFirst train case keys:", list(train[0].keys()))
    print("First train case_id :", train[0]["case_id"])
    print("Image paths (count) :", len(train[0]["image"]))
    print("\n✅ dataset.py works.")