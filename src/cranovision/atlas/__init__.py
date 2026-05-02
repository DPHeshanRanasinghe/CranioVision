"""
CranioVision — Atlas registration and anatomical lookup.

Tools for aligning patient MRI to standard MNI152 space and looking up
anatomical regions for tumor voxels using the Harvard-Oxford parcellation.

Submodules:
- download:     downloads MNI152 + Harvard-Oxford templates (one-time setup)
- registration: ANTs-based patient T1 -> MNI152 registration with caching
- anatomy:      tumor -> anatomical region statistics
- eloquent:     distance-to-eloquent-cortex computation

Public API
----------
download_atlas_data()           ensure templates are present
register_patient(case, ...)     register one patient T1 to MNI152 (cached)
analyze_tumor_anatomy(...)      get anatomical region distribution of tumor
compute_eloquent_distance(...)  distance from tumor to motor / speech areas
"""

from .registration import register_patient, load_cached_registration
from .anatomy import analyze_tumor_anatomy, get_anatomical_summary
from .eloquent import compute_eloquent_distance, ELOQUENT_REGIONS
from .download import download_atlas_data, ensure_atlas_aligned, ATLAS_FILES

__all__ = [
    "download_atlas_data",
    "ATLAS_FILES",
    "register_patient",
    "load_cached_registration",
    "analyze_tumor_anatomy",
    "get_anatomical_summary",
    "compute_eloquent_distance",
    "ELOQUENT_REGIONS",
]