"""
CranioVision — Central configuration.

All paths and hyperparameters are defined here so every notebook and
script imports the same values. Change here once, propagates everywhere.
"""
from pathlib import Path
import torch


# ══════════════════════════════════════════════════════════════════════════════
# PROJECT PATHS
# ══════════════════════════════════════════════════════════════════════════════

# Project root = 2 levels up from this file (src/cranovision/config.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR     = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "BraTS2024_small_dataset"
SPLITS_DIR   = DATA_DIR / "splits"

MODELS_DIR    = PROJECT_ROOT / "models"
OUTPUTS_DIR   = PROJECT_ROOT / "outputs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
CONFIGS_DIR   = PROJECT_ROOT / "configs"

# Create output directories if missing
for _d in (MODELS_DIR, OUTPUTS_DIR, SPLITS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# BRATS LABEL MAPPING
# ══════════════════════════════════════════════════════════════════════════════
# BraTS 2024 raw labels → internal class indices
# Raw:    0=background, 2=edema, 3=enhancing, 4=necrotic
# Mapped: 0=background, 1=edema, 2=enhancing, 3=necrotic
LABEL_MAP    = {0: 0, 2: 1, 3: 2, 4: 3}
CLASS_NAMES  = ["Background", "Edema", "Enhancing tumor", "Necrotic core"]
CLASS_COLORS = ["#111111", "#FFD700", "#FF3322", "#4488FF"]
NUM_CLASSES  = 4

# MRI modality channel order (CRITICAL — models expect this exact order)
MODALITIES = ["t1n", "t1c", "t2w", "t2f"]
NUM_CHANNELS = len(MODALITIES)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# Spatial
PATCH_SIZE         = (128, 128, 128)    # random crop during training
VAL_SPATIAL_SIZE   = (160, 192, 160)    # center crop during validation
ORIENTATION_AXCODES = "RAS"             # Right-Anterior-Superior

# Batch
BATCH_SIZE   = 1        # 3D volumes are huge, keep at 1
NUM_WORKERS  = 2

# Optimizer
LEARNING_RATE = 2e-4
WEIGHT_DECAY  = 1e-5

# Schedule
MAX_EPOCHS   = 100
VAL_INTERVAL = 5

# Reproducibility
SEED = 42


# ══════════════════════════════════════════════════════════════════════════════
# DEVICE
# ══════════════════════════════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()   # Mixed precision only works on CUDA
PIN_MEMORY = torch.cuda.is_available()


# ══════════════════════════════════════════════════════════════════════════════
# MODEL CHECKPOINT PATHS (for each branch)
# ══════════════════════════════════════════════════════════════════════════════

CHECKPOINT_PATHS = {
    "attention_unet": MODELS_DIR / "attention_unet_best.pth",
    "swin_unetr"   : MODELS_DIR / "swin_unetr_best.pth",
    "nnunet"       : MODELS_DIR / "nnunet_best.pth",
}


def print_config():
    """Pretty-print the current config — call this at the top of every notebook."""
    print("═" * 60)
    print("CranioVision Configuration")
    print("═" * 60)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Data dir     : {RAW_DATA_DIR}")
    print(f"Models dir   : {MODELS_DIR}")
    print(f"Outputs dir  : {OUTPUTS_DIR}")
    print(f"Device       : {DEVICE}")
    print(f"Mixed precision: {USE_AMP}")
    print(f"Patch size   : {PATCH_SIZE}")
    print(f"Num classes  : {NUM_CLASSES}")
    print(f"Modalities   : {MODALITIES}")
    print("═" * 60)


if __name__ == "__main__":
    print_config()
