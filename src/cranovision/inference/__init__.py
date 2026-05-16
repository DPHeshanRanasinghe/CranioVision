"""CranioVision inference utilities."""
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
from .grad_cam import (
    GradCAM3D,
    compute_grad_cam,
    find_target_layer,
)
from .ensemble import (
    load_ensemble,
    ensemble_predict,
    compute_agreement,
    weights_from_val_dice,
    MODEL_REGISTRY,
)

__all__ = [
    # predict
    "load_model", "preprocess_case", "predict_case",
    "make_inferer", "compute_region_volumes",
    # mc_dropout
    "mc_dropout_predict", "summarize_confidence",
    "enable_dropout", "count_dropout_layers",
    # grad_cam
    "GradCAM3D", "compute_grad_cam", "find_target_layer",
    # ensemble
    "load_ensemble", "ensemble_predict", "compute_agreement",
    "weights_from_val_dice", "MODEL_REGISTRY",
]