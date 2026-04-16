"""Validation helpers for sliding-window 3D segmentation evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from monai.inferers import sliding_window_inference

from src.training.metrics import (
    build_dice_metric,
    build_metric_post_transforms,
    dice_scores_to_dict,
    prepare_metric_inputs,
)
from src.utils.visualization import save_prediction_panel


def _save_validation_samples(
    batch: Mapping[str, Any],
    logits: torch.Tensor,
    config: Mapping[str, Any],
    visual_dir: Path,
    epoch: int,
    start_index: int,
) -> int:
    """Save a small set of qualitative validation panels for debugging."""

    modality_name = str(config["visualization"].get("modality_for_panels", "t1ce"))
    modality_names = list(config["data"]["expected_modalities"])
    modality_index = modality_names.index(modality_name) if modality_name in modality_names else 0

    predictions = torch.argmax(logits, dim=1).detach().cpu()
    images = batch["image"].detach().cpu()
    labels = batch["label"].detach().cpu()
    case_ids = batch["case_id"]

    saved = 0
    for batch_index, case_id in enumerate(case_ids):
        output_path = visual_dir / f"epoch_{epoch:03d}_{start_index + batch_index:02d}_{case_id}.png"
        save_prediction_panel(
            image_volume=images[batch_index, modality_index].numpy(),
            label_volume=labels[batch_index, 0].numpy(),
            prediction_volume=predictions[batch_index].numpy(),
            output_path=output_path,
            title=f"{case_id} | {modality_name}",
            axis=int(config["visualization"].get("slice_axis", 2)),
        )
        saved += 1
    return saved


@torch.no_grad()
def run_validation(
    model: torch.nn.Module,
    val_loader,
    config: Mapping[str, Any],
    device: torch.device,
    loss_fn=None,
    epoch: int = 0,
    visual_dir: str | Path | None = None,
) -> dict[str, float]:
    """Run full-volume sliding-window validation and return scalar metrics."""

    model.eval()
    dice_metric = build_dice_metric(config)
    post_pred, post_label = build_metric_post_transforms(config)
    metric_cfg = config.get("metrics", {})

    running_val_loss = 0.0
    num_loss_batches = 0
    sample_limit = int(config.get("visualization", {}).get("prediction_sample_count", 0))
    saved_samples = 0

    resolved_visual_dir: Path | None = None
    if visual_dir is not None and sample_limit > 0:
        resolved_visual_dir = Path(visual_dir).expanduser().resolve()
        resolved_visual_dir.mkdir(parents=True, exist_ok=True)

    for batch in val_loader:
        images = batch["image"].to(device)
        labels = batch["label"].long().to(device)

        logits = sliding_window_inference(
            inputs=images,
            roi_size=tuple(config["preprocessing"]["roi_size"]),
            sw_batch_size=int(config["inference"]["sw_batch_size"]),
            predictor=model,
            overlap=float(config["inference"]["overlap"]),
        )

        if loss_fn is not None:
            running_val_loss += float(loss_fn(logits, labels).item())
            num_loss_batches += 1

        predictions, references = prepare_metric_inputs(
            logits=logits,
            labels=labels,
            post_pred=post_pred,
            post_label=post_label,
        )
        dice_metric(y_pred=predictions, y=references)

        if resolved_visual_dir is not None and saved_samples < sample_limit:
            batch_budget = min(int(images.shape[0]), sample_limit - saved_samples)
            trimmed_batch = {
                "image": batch["image"][:batch_budget],
                "label": batch["label"][:batch_budget],
                "case_id": batch["case_id"][:batch_budget],
            }
            saved_samples += _save_validation_samples(
                batch=trimmed_batch,
                logits=logits[:batch_budget],
                config=config,
                visual_dir=resolved_visual_dir,
                epoch=epoch,
                start_index=saved_samples,
            )

    aggregated = dice_metric.aggregate()
    dice_metric.reset()

    metrics = dice_scores_to_dict(
        dice_scores=aggregated,
        class_names=config["labels"]["class_names"],
        include_background=bool(metric_cfg.get("include_background", False)),
    )

    if num_loss_batches > 0:
        metrics["val_loss"] = running_val_loss / num_loss_batches

    return metrics
