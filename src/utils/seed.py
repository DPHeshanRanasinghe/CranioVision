"""Reproducibility helpers."""

from __future__ import annotations

import random

import numpy as np
import torch
from monai.utils import set_determinism


def set_global_seed(seed: int) -> None:
    """Set seeds for Python, NumPy, PyTorch, and MONAI determinism hooks."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)
