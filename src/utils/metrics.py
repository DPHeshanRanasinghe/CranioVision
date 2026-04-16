"""Loss and metric helpers for segmentation experiments."""

from __future__ import annotations

from typing import Any

import torch
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete


def build_loss(config: dict[str, Any]):
    """Create the training loss specified in config."""

    loss_name = config["training"].get("loss", "dice_ce").lower()
    if loss_name == "dice_ce":
        return DiceCELoss(to_onehot_y=True, softmax=True)
    if loss_name == "dice":
        return DiceLoss(to_onehot_y=True, softmax=True)
    raise ValueError(f"Unsupported loss '{loss_name}'. Expected 'dice_ce' or 'dice'.")


def get_dice_metric() -> DiceMetric:
    """Return a Dice metric that is ready for per-class reporting."""

    return DiceMetric(include_background=False, reduction="mean_batch")


def get_post_transforms(num_classes: int) -> tuple[AsDiscrete, AsDiscrete]:
    """Post-processing used before metric computation."""

    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
    post_label = AsDiscrete(to_onehot=num_classes)
    return post_pred, post_label


def dice_scores_to_dict(
    dice_scores: torch.Tensor,
    class_names: list[str],
) -> dict[str, float]:
    """Convert aggregated Dice scores to a readable metrics dictionary."""

    values = dice_scores.detach().cpu().float().flatten().tolist()
    foreground_classes = class_names[1:]
    per_class = {
        f"dice_{class_name}": float(score)
        for class_name, score in zip(foreground_classes, values)
    }
    mean_dice = float(sum(values) / max(len(values), 1))
    per_class["mean_dice"] = mean_dice
    return per_class
