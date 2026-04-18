"""
CranioVision inference utilities.
"""
from .predict import (
    load_model,
    preprocess_case,
    predict_case,
    make_inferer,
    compute_region_volumes,
)
from .mc_dropout import (
    mc_dropout_predict,
    summarize_confidence,
    enable_dropout,
    count_dropout_layers,
)

__all__ = [
    # predict
    "load_model",
    "preprocess_case",
    "predict_case",
    "make_inferer",
    "compute_region_volumes",
    # mc_dropout
    "mc_dropout_predict",
    "summarize_confidence",
    "enable_dropout",
    "count_dropout_layers",
]