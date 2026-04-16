"""Metric helpers for segmentation validation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from monai.data import decollate_batch
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete


def build_dice_metric(config: Mapping[str, Any]) -> DiceMetric:
    """Return a Dice metric configured from the experiment config."""

    metric_cfg = config.get("metrics", {})
    return DiceMetric(
        include_background=bool(metric_cfg.get("include_background", False)),
        reduction=str(metric_cfg.get("reduction", "mean_batch")),
        get_not_nans=bool(metric_cfg.get("get_not_nans", False)),
    )


def build_metric_post_transforms(config: Mapping[str, Any]) -> tuple[AsDiscrete, AsDiscrete]:
    """Create post-processing transforms for prediction/label Dice evaluation."""

    num_classes = int(config["model"]["out_channels"])
    return AsDiscrete(argmax=True, to_onehot=num_classes), AsDiscrete(to_onehot=num_classes)


def prepare_metric_inputs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    post_pred: AsDiscrete,
    post_label: AsDiscrete,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Convert batched logits and labels into decollated one-hot tensors."""

    predictions = [post_pred(item) for item in decollate_batch(logits)]
    references = [post_label(item) for item in decollate_batch(labels)]
    return predictions, references


def dice_scores_to_dict(
    dice_scores: torch.Tensor,
    class_names: Sequence[str],
    include_background: bool = False,
) -> dict[str, float]:
    """Convert Dice tensors into readable scalar metrics."""

    values = torch.nan_to_num(dice_scores.detach().cpu().float(), nan=0.0).flatten().tolist()

    if include_background:
        metric_class_names = list(class_names)
    else:
        metric_class_names = list(class_names[1:])

    if len(metric_class_names) != len(values):
        metric_class_names = [f"class_{index}" for index in range(len(values))]

    metrics: dict[str, float] = {"mean_dice": float(sum(values) / max(len(values), 1))}
    for class_name, score in zip(metric_class_names, values):
        metrics[f"dice_{class_name}"] = float(score)
    return metrics
