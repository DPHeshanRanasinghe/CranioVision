"""
CranioVision — Batch atlas registration.

Runs ANTs registration on every case in a split (default: test set) and
caches all artifacts. Designed to be run overnight as a background task.

Usage
-----
From the project root:
    python -m src.cranovision.atlas.batch_register

Or with options:
    python -m src.cranovision.atlas.batch_register --split test --skip-existing

Resumable: if you stop it mid-run and restart, it skips cases already cached.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import List

from ..config import OUTPUTS_DIR
from ..data import get_splits
from .download import download_atlas_data, ensure_atlas_aligned
from .registration import register_patient, _is_cached
from .anatomy import analyze_tumor_anatomy
from .eloquent import compute_eloquent_distance


def _resolve_paths(case: dict) -> tuple:
    """Extract (t1n_path, mask_path) from a case dict.

    Handles BOTH dataset schemas seen so far:
      - case['image'] is a list of 4 paths in BraTS order (t1n first)
      - case['t1n'] is a single path (older schema)
    """
    if "image" in case and isinstance(case["image"], (list, tuple)):
        # Standard BraTS order: [t1n, t1c, t2w, t2f]
        t1n_path = case["image"][0]
    elif "t1n" in case:
        t1n_path = case["t1n"]
    else:
        raise KeyError(f"Could not find T1n path in case: {list(case.keys())}")

    mask_path = case.get("label") or case.get("mask")
    if mask_path is None:
        raise KeyError(f"Could not find label/mask path in case: {list(case.keys())}")

    return str(t1n_path), str(mask_path)


def _summarize_case(case_id: str, anatomy: dict, eloquent: dict) -> dict:
    """Build a small per-case summary for the master report."""
    high_risk = [
        name for name, info in eloquent.items()
        if info.get("risk_level") == "high"
    ]
    return {
        "case_id": case_id,
        "primary_lobe": anatomy.get("primary_region", "Unknown"),
        "primary_pct": anatomy.get("primary_pct", 0.0),
        "lateralization": anatomy.get("lateralization", "n/a"),
        "left_pct": anatomy.get("left_hemisphere_pct", 0.0),
        "right_pct": anatomy.get("right_hemisphere_pct", 0.0),
        "total_volume_cm3": anatomy.get("total_volume_cm3", 0.0),
        "n_high_risk_regions": len(high_risk),
        "high_risk_regions": high_risk,
    }


def run_batch(
    split: str = "test",
    skip_existing: bool = True,
    save_per_case_report: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Register every case in the chosen split and save per-case reports.

    Parameters
    ----------
    split        : 'train', 'val', or 'test'
    skip_existing: if True, skip cases that already have a complete cache
    save_per_case_report: also save anatomy + eloquent JSON per case
    verbose      : print progress

    Returns
    -------
    dict with run statistics and a per-case summary
    """
    # Make sure templates exist + are aligned
    if verbose:
        print("=" * 72)
        print("CranioVision — Batch atlas registration")
        print("=" * 72)
        print(f"Split: {split}")
        print(f"Skip existing: {skip_existing}\n")

    download_atlas_data(verbose=verbose)
    ensure_atlas_aligned(verbose=verbose)

    train_cases, val_cases, test_cases = get_splits(verbose=False)
    splits_map = {"train": train_cases, "val": val_cases, "test": test_cases}
    if split not in splits_map:
        raise ValueError(f"split must be one of {list(splits_map.keys())}")
    cases = splits_map[split]

    if verbose:
        print(f"\nProcessing {len(cases)} cases from {split} split\n")

    results: List[dict] = []
    failed: List[dict] = []
    start = time.time()
    n_cached = 0
    n_processed = 0

    for i, case in enumerate(cases, start=1):
        case_id = case["case_id"]

        if verbose:
            print(f"\n{'=' * 72}")
            print(f"[{i:3d}/{len(cases)}]  {case_id}")
            print(f"{'=' * 72}")

        # Skip if already cached
        if skip_existing and _is_cached(case_id):
            if verbose:
                print("[skip] already cached")
            n_cached += 1
            # Still load & summarize so the master report is complete
            try:
                from .registration import load_cached_registration
                cached = load_cached_registration(case_id)
                anatomy = analyze_tumor_anatomy(
                    warped_mask=cached["warped_mask"], verbose=False
                )
                eloquent = compute_eloquent_distance(
                    warped_mask=cached["warped_mask"], verbose=False
                )
                results.append(_summarize_case(case_id, anatomy, eloquent))
            except Exception as e:
                if verbose:
                    print(f"  [warn] couldn't summarize cached case: {e}")
            continue

        try:
            t1n_path, mask_path = _resolve_paths(case)

            # Register patient -> MNI
            result = register_patient(
                case_id=case_id,
                t1_path=t1n_path,
                tumor_mask_path=mask_path,
                use_cache=True,
                verbose=verbose,
            )

            # Anatomy + eloquent (fast post-processing)
            anatomy = analyze_tumor_anatomy(
                warped_mask=result["warped_mask"], verbose=False,
            )
            eloquent = compute_eloquent_distance(
                warped_mask=result["warped_mask"], verbose=False,
            )

            # Save per-case JSON if requested
            if save_per_case_report:
                report = {
                    "case_id": case_id,
                    "anatomy": anatomy,
                    "eloquent": {
                        name: {
                            "distance_mm": (
                                info["distance_mm"]
                                if info["distance_mm"] != float("inf")
                                else None
                            ),
                            "risk_level": info["risk_level"],
                            "involved": info["involved"],
                            "function": info["function"],
                            "deficit_if_damaged": info["deficit_if_damaged"],
                        }
                        for name, info in eloquent.items()
                    },
                    "registration_metadata": result["metadata"],
                }
                report_path = OUTPUTS_DIR / f"atlas_report_{case_id}.json"
                with open(report_path, "w") as f:
                    json.dump(report, f, indent=2, default=str)

            summary = _summarize_case(case_id, anatomy, eloquent)
            results.append(summary)
            n_processed += 1

            if verbose:
                print(
                    f"  [OK] {summary['primary_lobe']:<22} | "
                    f"{summary['lateralization']:<10} | "
                    f"{summary['n_high_risk_regions']} HIGH-risk | "
                    f"{summary['total_volume_cm3']:.1f} cm^3"
                )

        except Exception as e:
            failed.append({
                "case_id": case_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })
            if verbose:
                print(f"  [FAIL] {type(e).__name__}: {e}")

    elapsed = time.time() - start

    # Save master report
    master_report = {
        "split": split,
        "total_cases": len(cases),
        "newly_processed": n_processed,
        "previously_cached": n_cached,
        "failed": len(failed),
        "elapsed_minutes": round(elapsed / 60, 1),
        "results": results,
        "failures": failed,
    }
    master_path = OUTPUTS_DIR / f"atlas_batch_{split}.json"
    with open(master_path, "w") as f:
        json.dump(master_report, f, indent=2, default=str)

    if verbose:
        print("\n" + "=" * 72)
        print("BATCH COMPLETE")
        print("=" * 72)
        print(f"  Total cases       : {len(cases)}")
        print(f"  Newly processed   : {n_processed}")
        print(f"  Previously cached : {n_cached}")
        print(f"  Failed            : {len(failed)}")
        print(f"  Elapsed           : {elapsed/60:.1f} min")
        if failed:
            print("\n  Failures:")
            for f_info in failed:
                print(f"    - {f_info['case_id']}: {f_info['error']}")
        print(f"\n  Master report saved: {master_path}")
        print("=" * 72)

    return master_report


# --- CLI -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch atlas registration for CranioVision",
    )
    parser.add_argument(
        "--split", default="test",
        choices=["train", "val", "test"],
        help="which dataset split to process (default: test)",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="skip cases that already have a complete cache (default: True)",
    )
    parser.add_argument(
        "--no-skip-existing", dest="skip_existing", action="store_false",
        help="re-run all cases regardless of cache status",
    )
    parser.add_argument(
        "--no-per-case-report", dest="save_per_case_report", action="store_false",
        default=True, help="don't save per-case JSON reports",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="reduce output verbosity",
    )
    args = parser.parse_args()

    run_batch(
        split=args.split,
        skip_existing=args.skip_existing,
        save_per_case_report=args.save_per_case_report,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()