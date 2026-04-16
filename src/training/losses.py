"""Loss builders for segmentation experiments."""

from __future__ import annotations

from typing import Any, Mapping

from monai.losses import DiceCELoss, DiceLoss


def _common_dice_kwargs(loss_options: Mapping[str, Any]) -> dict[str, Any]:
    """Collect config-driven options shared across Dice-based losses."""

    softmax = bool(loss_options.get("softmax", True))
    sigmoid = bool(loss_options.get("sigmoid", False))
    if softmax and sigmoid:
        raise ValueError("Loss options cannot enable both softmax and sigmoid at the same time.")

    return {
        "include_background": bool(loss_options.get("include_background", True)),
        "to_onehot_y": bool(loss_options.get("to_onehot_y", True)),
        "softmax": softmax,
        "sigmoid": sigmoid,
        "squared_pred": bool(loss_options.get("squared_pred", False)),
        "jaccard": bool(loss_options.get("jaccard", False)),
        "reduction": str(loss_options.get("reduction", "mean")),
        "smooth_nr": float(loss_options.get("smooth_nr", 1e-5)),
        "smooth_dr": float(loss_options.get("smooth_dr", 1e-5)),
        "batch": bool(loss_options.get("batch", False)),
    }


def build_loss(config: Mapping[str, Any]):
    """Create the training loss specified in the YAML config."""

    training_cfg = config["training"]
    loss_name = str(training_cfg.get("loss", "dice_ce")).lower()
    loss_options = training_cfg.get("loss_options", {})
    common_kwargs = _common_dice_kwargs(loss_options)

    if loss_name == "dice":
        return DiceLoss(**common_kwargs)

    if loss_name == "dice_ce":
        dice_ce_kwargs = dict(common_kwargs)
        if "lambda_dice" in loss_options:
            dice_ce_kwargs["lambda_dice"] = float(loss_options["lambda_dice"])
        if "lambda_ce" in loss_options:
            dice_ce_kwargs["lambda_ce"] = float(loss_options["lambda_ce"])
        if "ce_weight" in loss_options:
            dice_ce_kwargs["ce_weight"] = loss_options["ce_weight"]
        return DiceCELoss(**dice_ce_kwargs)

    raise ValueError(
        f"Unsupported loss '{loss_name}'. Expected one of: 'dice', 'dice_ce'."
    )
