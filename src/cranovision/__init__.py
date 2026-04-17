"""
CranioVision — AI-assisted 3D brain tumor segmentation pipeline.
"""
__version__ = "0.1.0"
__author__ = "DP Heshan Ranasinghe"

from .config import (
    PROJECT_ROOT, RAW_DATA_DIR, MODELS_DIR, OUTPUTS_DIR,
    LABEL_MAP, CLASS_NAMES, NUM_CLASSES, MODALITIES,
    PATCH_SIZE, DEVICE, SEED, print_config,
)

__all__ = [
    "PROJECT_ROOT", "RAW_DATA_DIR", "MODELS_DIR", "OUTPUTS_DIR",
    "LABEL_MAP", "CLASS_NAMES", "NUM_CLASSES", "MODALITIES",
    "PATCH_SIZE", "DEVICE", "SEED", "print_config",
]
