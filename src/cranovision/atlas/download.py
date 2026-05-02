"""
CranioVision — Atlas template downloader.

Downloads the standard neuroimaging atlases used for anatomical analysis:

1) MNI152NLin2009cAsym (brain only, 1mm isotropic)
   - The "common space" used in modern neuroscience papers
   - Non-linear average of 152 healthy adult brains
   - Asymmetric variant preserves left/right anatomy
   - Skull-stripped: only brain tissue, intensity 0 elsewhere
   - Hosted by TemplateFlow (https://www.templateflow.org/)

2) Harvard-Oxford Cortical + Subcortical Atlas
   - Maximum-probability thresholded at 25%
   - 48 cortical + 21 subcortical regions
   - Same coordinate space as MNI152 -> direct lookup after registration
   - Bundled with FSL; we fetch via nilearn (the standard mechanism in the
     neuroimaging community), with raw-URL fallbacks if nilearn is unavailable.

After running download_atlas_data() once, all templates live in ATLAS_DIR
and never need to be downloaded again.
"""
from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

from ..config import ATLAS_DIR


# --- File registry ----------------------------------------------------------

# Each entry has either a `url` (direct download) or `nilearn_atlas_name`
# (use nilearn.datasets.fetch_atlas_harvard_oxford). nilearn is preferred
# for Harvard-Oxford because the FSL repo URLs change every few years.
ATLAS_FILES: Dict[str, dict] = {
    "mni152_t1_brain": {
        "filename": "MNI152NLin2009cAsym_1mm_T1_brain.nii.gz",
        "urls": [
            (
                "https://templateflow.s3.amazonaws.com/"
                "tpl-MNI152NLin2009cAsym/"
                "tpl-MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz"
            ),
        ],
        "description": "MNI152 1mm T1, brain-only (skull-stripped)",
    },
    "mni152_brain_mask": {
        "filename": "MNI152NLin2009cAsym_1mm_brain_mask.nii.gz",
        "urls": [
            (
                "https://templateflow.s3.amazonaws.com/"
                "tpl-MNI152NLin2009cAsym/"
                "tpl-MNI152NLin2009cAsym_res-01_desc-brain_mask.nii.gz"
            ),
        ],
        "description": "MNI152 1mm brain mask (binary)",
    },
    "harvard_oxford_cortical": {
        "filename": "HarvardOxford-cort-maxprob-thr25-1mm.nii.gz",
        "nilearn_atlas_name": "cort-maxprob-thr25-1mm",
        "urls": [
            # Primary fallback: dmascali/mni2atlas mirror (active and stable)
            "https://github.com/dmascali/mni2atlas/raw/master/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz",
            # Secondary fallback: FSL official GitLab
            "https://git.fmrib.ox.ac.uk/fsl/data_atlases/-/raw/2103.0/HarvardOxford/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz",
        ],
        "description": "Harvard-Oxford cortical atlas, max-prob 25% threshold",
    },
    "harvard_oxford_subcortical": {
        "filename": "HarvardOxford-sub-maxprob-thr25-1mm.nii.gz",
        "nilearn_atlas_name": "sub-maxprob-thr25-1mm",
        "urls": [
            "https://github.com/dmascali/mni2atlas/raw/master/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz",
            "https://git.fmrib.ox.ac.uk/fsl/data_atlases/-/raw/2103.0/HarvardOxford/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz",
        ],
        "description": "Harvard-Oxford subcortical atlas, max-prob 25% threshold",
    },
}


# --- Region label tables ----------------------------------------------------

# Harvard-Oxford cortical labels (1-48). Index 0 is background.
HO_CORTICAL_LABELS = {
    0: "Background",
    1: "Frontal Pole",
    2: "Insular Cortex",
    3: "Superior Frontal Gyrus",
    4: "Middle Frontal Gyrus",
    5: "Inferior Frontal Gyrus, pars triangularis",
    6: "Inferior Frontal Gyrus, pars opercularis",
    7: "Precentral Gyrus",
    8: "Temporal Pole",
    9: "Superior Temporal Gyrus, anterior division",
    10: "Superior Temporal Gyrus, posterior division",
    11: "Middle Temporal Gyrus, anterior division",
    12: "Middle Temporal Gyrus, posterior division",
    13: "Middle Temporal Gyrus, temporooccipital part",
    14: "Inferior Temporal Gyrus, anterior division",
    15: "Inferior Temporal Gyrus, posterior division",
    16: "Inferior Temporal Gyrus, temporooccipital part",
    17: "Postcentral Gyrus",
    18: "Superior Parietal Lobule",
    19: "Supramarginal Gyrus, anterior division",
    20: "Supramarginal Gyrus, posterior division",
    21: "Angular Gyrus",
    22: "Lateral Occipital Cortex, superior division",
    23: "Lateral Occipital Cortex, inferior division",
    24: "Intracalcarine Cortex",
    25: "Frontal Medial Cortex",
    26: "Juxtapositional Lobule Cortex (Supplementary Motor)",
    27: "Subcallosal Cortex",
    28: "Paracingulate Gyrus",
    29: "Cingulate Gyrus, anterior division",
    30: "Cingulate Gyrus, posterior division",
    31: "Precuneus Cortex",
    32: "Cuneal Cortex",
    33: "Frontal Orbital Cortex",
    34: "Parahippocampal Gyrus, anterior division",
    35: "Parahippocampal Gyrus, posterior division",
    36: "Lingual Gyrus",
    37: "Temporal Fusiform Cortex, anterior division",
    38: "Temporal Fusiform Cortex, posterior division",
    39: "Temporal Occipital Fusiform Cortex",
    40: "Occipital Fusiform Gyrus",
    41: "Frontal Operculum Cortex",
    42: "Central Opercular Cortex",
    43: "Parietal Operculum Cortex",
    44: "Planum Polare",
    45: "Heschl's Gyrus",
    46: "Planum Temporale",
    47: "Supracalcarine Cortex",
    48: "Occipital Pole",
}

HO_SUBCORTICAL_LABELS = {
    0: "Background",
    1: "Left Cerebral White Matter",
    2: "Left Cerebral Cortex",
    3: "Left Lateral Ventricle",
    4: "Left Thalamus",
    5: "Left Caudate",
    6: "Left Putamen",
    7: "Left Pallidum",
    8: "Brain-Stem",
    9: "Left Hippocampus",
    10: "Left Amygdala",
    11: "Left Accumbens",
    12: "Right Cerebral White Matter",
    13: "Right Cerebral Cortex",
    14: "Right Lateral Ventricle",
    15: "Right Thalamus",
    16: "Right Caudate",
    17: "Right Putamen",
    18: "Right Pallidum",
    19: "Right Hippocampus",
    20: "Right Amygdala",
    21: "Right Accumbens",
}


# --- Download helpers -------------------------------------------------------

def _download_one_url(url: str, dest: Path, description: str) -> bool:
    """
    Try to download from a single URL. Returns True on success, False on
    HTTP errors so callers can try the next mirror.
    Other errors propagate.
    """
    print(f"    URL : {url}")
    print(f"    Dest: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (CranioVision atlas downloader)"},
        )
        with urllib.request.urlopen(req, timeout=120) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            block_size = 1024 * 64
            downloaded = 0

            with open(dest, "wb") as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        size_mb = downloaded / 1e6
                        total_mb = total_size / 1e6
                        print(
                            f"\r    {pct:5.1f}%  ({size_mb:6.1f} / {total_mb:.1f} MB)",
                            end="", flush=True,
                        )
        print()  # newline after progress
        size_mb = dest.stat().st_size / 1e6
        # Any sub-100KB result is suspect for a NIfTI (probably an HTML error page)
        if size_mb < 0.1:
            print(f"    [WARN] suspiciously small file ({size_mb:.3f} MB) — treating as failed")
            dest.unlink(missing_ok=True)
            return False
        print(f"    [OK] saved ({size_mb:.1f} MB)")
        return True

    except urllib.error.HTTPError as e:
        print(f"\n    [FAIL] HTTP {e.code}: {e.reason}")
        if dest.exists():
            dest.unlink()
        return False

    except Exception as e:
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"Failed to download {description}: {e}") from e


def _try_nilearn_fetch(atlas_name: str, dest: Path, description: str) -> bool:
    """
    Try to fetch a Harvard-Oxford atlas via nilearn.

    nilearn manages its own download cache and handles FSL atlas mirrors
    transparently. After it downloads, we just copy the file to our location.
    Returns True on success, False if nilearn isn't available or fails.
    """
    try:
        from nilearn import datasets as nl_datasets
    except ImportError:
        print("    [skip-nilearn] nilearn not installed (pip install nilearn)")
        return False

    try:
        print(f"    Trying nilearn.fetch_atlas_harvard_oxford('{atlas_name}')...")
        atlas = nl_datasets.fetch_atlas_harvard_oxford(atlas_name)
        # nilearn returns an atlas object with a .filename or .maps attribute
        src = getattr(atlas, "filename", None) or getattr(atlas, "maps", None)
        if src is None:
            print("    [skip-nilearn] couldn't locate downloaded file in nilearn result")
            return False

        src_path = Path(src)
        if not src_path.exists():
            print(f"    [skip-nilearn] nilearn reported {src_path} but file missing")
            return False

        shutil.copy(src_path, dest)
        size_mb = dest.stat().st_size / 1e6
        print(f"    [OK] copied from nilearn cache ({size_mb:.1f} MB)")
        return True

    except Exception as e:
        print(f"    [fail-nilearn] {type(e).__name__}: {e}")
        return False


def _download_with_fallback(info: dict, dest: Path) -> None:
    """
    Try every download method in order: nilearn first (if applicable),
    then each URL until one works.
    """
    description = info["description"]

    # 1. Try nilearn first (only for Harvard-Oxford entries)
    if "nilearn_atlas_name" in info:
        if _try_nilearn_fetch(info["nilearn_atlas_name"], dest, description):
            return

    # 2. Try each URL in order
    urls: List[str] = info.get("urls", [])
    for i, url in enumerate(urls):
        print(f"  Attempt {i + 1}/{len(urls)}:")
        if _download_one_url(url, dest, description):
            return

    raise RuntimeError(
        f"All sources failed for {description}.\n"
        f"You can manually download the file and place it at:\n"
        f"  {dest}\n"
        f"Try one of:\n"
        + "\n".join(f"  {u}" for u in urls)
    )


# --- Public API -------------------------------------------------------------

def download_atlas_data(force: bool = False, verbose: bool = True) -> Dict[str, Path]:
    """
    Download all atlas templates needed by CranioVision.

    Templates are downloaded into ATLAS_DIR.
    Existing files are skipped unless force=True.

    For each file, multiple sources are tried in order:
      1. nilearn (if applicable, for FSL atlases)
      2. Direct URL mirrors (in priority order)

    Parameters
    ----------
    force   : if True, re-download files even if they already exist.
    verbose : if True, print progress.

    Returns
    -------
    dict mapping template key -> Path to the downloaded file.
    """
    if verbose:
        print("=" * 64)
        print("CranioVision — atlas template downloader")
        print("=" * 64)
        print(f"Atlas directory: {ATLAS_DIR}")
    ATLAS_DIR.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}

    for key, info in ATLAS_FILES.items():
        dest = ATLAS_DIR / info["filename"]
        paths[key] = dest

        if dest.exists() and not force:
            if verbose:
                size_mb = dest.stat().st_size / 1e6
                print(f"\n[skip] {info['description']}")
                print(f"       (already present, {size_mb:.1f} MB)")
            continue

        if verbose:
            print(f"\n[fetch] {key}")
            print(f"  Downloading: {info['description']}")

        _download_with_fallback(info, dest)

    if verbose:
        print("\n" + "=" * 64)
        print("All atlas templates ready.")
        print("=" * 64)

    return paths


def get_atlas_paths() -> Dict[str, Path]:
    """Return paths to all atlas files. Raises if any are missing.

    Returns the RESAMPLED Harvard-Oxford atlases (aligned to MNI152NLin2009cAsym)
    if available; otherwise the originals. Run ensure_atlas_aligned() once
    after first download to create the resampled versions.
    """
    paths = {}
    missing = []
    for key, info in ATLAS_FILES.items():
        p = ATLAS_DIR / info["filename"]
        if not p.exists():
            missing.append(info["filename"])
            paths[key] = p
            continue

        # For Harvard-Oxford, prefer the _resampled version if present
        if key.startswith('harvard_oxford'):
            resampled = p.with_name(p.stem.replace('.nii', '') + '_resampled.nii.gz')
            if resampled.exists():
                paths[key] = resampled
                continue

        paths[key] = p

    if missing:
        raise FileNotFoundError(
            f"Atlas files missing: {missing}\n"
            f"Run: from src.cranovision.atlas import download_atlas_data; download_atlas_data()"
        )
    return paths


def ensure_atlas_aligned(verbose: bool = True) -> Dict[str, Path]:
    """
    Ensure Harvard-Oxford atlases are on the same grid as MNI152NLin2009cAsym.

    The FSL Harvard-Oxford atlas uses MNI152NLin6Asym (193x229x193 isn't its
    grid either — it's typically 182x218x182). To use it for tumor lookup
    after registration to our MNI152NLin2009cAsym reference, we need to
    resample HO onto the reference grid using nearest-neighbor interpolation
    (preserves discrete labels — never interpolate labels with linear!).

    Resampled files are saved with the suffix '_resampled' and used everywhere
    downstream. Idempotent — skips if already done.
    """
    import ants

    paths = get_atlas_paths()
    reference = ants.image_read(str(paths['mni152_t1_brain']))

    aligned_paths = dict(paths)  # shallow copy

    for atlas_key in ('harvard_oxford_cortical', 'harvard_oxford_subcortical'):
        src = paths[atlas_key]
        dst = src.with_name(src.stem.replace('.nii', '') + '_resampled.nii.gz')

        if dst.exists():
            if verbose:
                print(f'[skip] {dst.name} (already resampled)')
            aligned_paths[atlas_key] = dst
            continue

        if verbose:
            print(f'[resample] {atlas_key}')
            print(f'  source: {src.name}')

        ho = ants.image_read(str(src))
        if verbose:
            print(f'  source shape: {ho.shape}')
            print(f'  target shape: {reference.shape}')

        # nearest-neighbor — critical for label maps. Linear/cubic would blend
        # label IDs and produce nonsense (e.g., label "between 7 and 8").
        resampled = ants.resample_image_to_target(
            image=ho,
            target=reference,
            interp_type='nearestNeighbor',
        )
        ants.image_write(resampled, str(dst))
        aligned_paths[atlas_key] = dst

        if verbose:
            print(f'  saved: {dst.name}')

    return aligned_paths


# --- CLI / smoke test -------------------------------------------------------

if __name__ == "__main__":
    paths = download_atlas_data(verbose=True)
    print("\nFile paths:")
    for k, p in paths.items():
        print(f"  {k:<28}: {p}")