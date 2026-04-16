"""Training loop for 3D medical image segmentation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference

from src.utils.logging import get_logger
from src.utils.metrics import (
    build_loss,
    dice_scores_to_dict,
    get_dice_metric,
    get_post_transforms,
)
from src.utils.visualization import save_prediction_panel


class SegmentationTrainer:
    """Small trainer class that keeps the first version explicit and readable."""

    def __init__(self, config: dict[str, Any], output_dir: str | Path) -> None:
        self.config = config
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.visual_dir = self.output_dir / "visualizations"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.visual_dir.mkdir(parents=True, exist_ok=True)

        requested_device = config["training"].get("device", "cuda")
        if requested_device == "cuda" and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        self.logger = get_logger(__name__)

        self.loss_fn = build_loss(config)
        self.dice_metric = get_dice_metric()
        self.post_pred, self.post_label = get_post_transforms(config["model"]["out_channels"])

        amp_requested = config["training"].get("amp", True)
        self.amp_enabled = amp_requested and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

    def train(
        self,
        model: torch.nn.Module,
        train_loader,
        val_loader,
    ) -> list[dict[str, float]]:
        """Run the full training and validation loop."""

        model = model.to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
        )

        epochs = self.config["training"]["epochs"]
        val_interval = self.config["training"].get("val_interval", 1)
        history: list[dict[str, float]] = []
        best_metric = -1.0

        for epoch in range(1, epochs + 1):
            train_loss = self._run_train_epoch(model, train_loader, optimizer)
            epoch_metrics: dict[str, float] = {
                "epoch": float(epoch),
                "train_loss": train_loss,
            }

            if epoch % val_interval == 0:
                val_metrics = self.validate(model, val_loader, epoch)
                epoch_metrics.update(val_metrics)

                if val_metrics["mean_dice"] > best_metric:
                    best_metric = val_metrics["mean_dice"]
                    self._save_checkpoint(model, optimizer, epoch, best_metric, is_best=True)

            self._save_checkpoint(model, optimizer, epoch, best_metric, is_best=False)
            history.append(epoch_metrics)

            metric_summary = ", ".join(
                f"{name}={value:.4f}" for name, value in epoch_metrics.items() if name != "epoch"
            )
            self.logger.info("Epoch %s | %s", epoch, metric_summary)

        history_path = self.output_dir / "metrics_history.json"
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        self.logger.info("Saved training history to %s", history_path)
        return history

    def _run_train_epoch(self, model: torch.nn.Module, train_loader, optimizer) -> float:
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].long().to(self.device)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                logits = model(images)
                loss = self.loss_fn(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            running_loss += float(loss.item())

        return running_loss / max(len(train_loader), 1)

    @torch.no_grad()
    def validate(self, model: torch.nn.Module, val_loader, epoch: int) -> dict[str, float]:
        """Run full-volume validation using sliding-window inference."""

        model.eval()
        self.dice_metric.reset()
        sample_limit = self.config["visualization"].get("prediction_sample_count", 3)
        saved_samples = 0

        for batch in val_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].long().to(self.device)

            logits = sliding_window_inference(
                inputs=images,
                roi_size=tuple(self.config["preprocessing"]["roi_size"]),
                sw_batch_size=self.config["inference"]["sw_batch_size"],
                predictor=model,
                overlap=self.config["inference"]["overlap"],
            )

            preds = [self.post_pred(item) for item in decollate_batch(logits)]
            refs = [self.post_label(item) for item in decollate_batch(labels)]
            self.dice_metric(y_pred=preds, y=refs)

            if saved_samples < sample_limit:
                self._save_visual_samples(
                    batch=batch,
                    logits=logits,
                    epoch=epoch,
                    start_index=saved_samples,
                )
                saved_samples += min(images.shape[0], sample_limit - saved_samples)

        aggregated = self.dice_metric.aggregate()
        self.dice_metric.reset()
        metric_dict = dice_scores_to_dict(
            aggregated,
            self.config["labels"]["class_names"],
        )
        return metric_dict

    def _save_visual_samples(
        self,
        batch: dict[str, Any],
        logits: torch.Tensor,
        epoch: int,
        start_index: int,
    ) -> None:
        modality_name = self.config["visualization"].get("modality_for_panels", "t1ce")
        modality_names = self.config["data"]["expected_modalities"]
        modality_index = modality_names.index(modality_name) if modality_name in modality_names else 0

        predictions = torch.argmax(logits, dim=1).detach().cpu()
        images = batch["image"].detach().cpu()
        labels = batch["label"].detach().cpu()
        case_ids = batch["case_id"]

        for batch_index, case_id in enumerate(case_ids):
            image_volume = images[batch_index, modality_index].numpy()
            label_volume = labels[batch_index, 0].numpy()
            pred_volume = predictions[batch_index].numpy()
            output_path = self.visual_dir / f"epoch_{epoch:03d}_{start_index + batch_index:02d}_{case_id}.png"
            save_prediction_panel(
                image_volume=image_volume,
                label_volume=label_volume,
                prediction_volume=pred_volume,
                output_path=output_path,
                title=f"{case_id} | {modality_name}",
                axis=self.config["visualization"].get("slice_axis", 2),
            )

    def _save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        best_metric: float,
        is_best: bool,
    ) -> None:
        filename = "best_model.pt" if is_best else "last_model.pt"
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "best_metric": best_metric,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": self.config,
            },
            checkpoint_path,
        )
