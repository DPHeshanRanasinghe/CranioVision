"""Config-driven training entrypoint for baseline 3D segmentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from monai.data import CacheDataset, DataLoader

from src.data.dataset import discover_brats_cases, split_cases, write_split_manifest
from src.data.transforms import build_train_transforms, build_val_transforms
from src.models.factory import build_model
from src.training.losses import build_loss
from src.training.validate import run_validation
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the training entrypoint."""

    parser = argparse.ArgumentParser(description="Train a baseline 3D BraTS-style segmentation model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the YAML experiment config.",
    )
    return parser.parse_args()


def resolve_device(config: Mapping[str, Any]) -> torch.device:
    """Resolve the requested torch device, falling back to CPU when needed."""

    requested_device = str(config["training"].get("device", "cuda")).lower()
    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"
    return torch.device(requested_device)


def build_datasets(config: Mapping[str, Any]):
    """Discover cases, create the split, and build MONAI datasets."""

    data_cfg = config["data"]
    output_dir = Path(config["project"]["output_dir"]).expanduser().resolve()

    cases = discover_brats_cases(
        root_dir=data_cfg["root_dir"],
        modality_aliases=data_cfg["modality_aliases"],
        label_aliases=data_cfg["label_aliases"],
        file_extensions=data_cfg["file_extensions"],
        expected_modalities=data_cfg["expected_modalities"],
        validate_shapes=data_cfg.get("validate_shapes", True),
    )
    train_cases, val_cases = split_cases(
        cases=cases,
        validation_ratio=float(data_cfg["validation_ratio"]),
        seed=int(data_cfg["seed"]),
        split_manifest_path=data_cfg.get("split_manifest_path"),
    )
    split_path = write_split_manifest(train_cases, val_cases, output_dir)

    train_dataset = CacheDataset(
        data=train_cases,
        transform=build_train_transforms(config),
        cache_rate=float(data_cfg.get("cache_rate", 0.0)),
        num_workers=int(data_cfg["num_workers"]),
    )
    val_dataset = CacheDataset(
        data=val_cases,
        transform=build_val_transforms(config),
        cache_rate=float(data_cfg.get("cache_rate", 0.0)),
        num_workers=int(data_cfg["num_workers"]),
    )

    return train_dataset, val_dataset, cases, train_cases, val_cases, split_path


def build_dataloaders(config: Mapping[str, Any], train_dataset, val_dataset, device: torch.device):
    """Create train and validation dataloaders from the config."""

    data_cfg = config["data"]
    num_workers = int(data_cfg["num_workers"])
    use_persistent_workers = bool(data_cfg.get("persistent_workers", num_workers > 0)) and num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=use_persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(data_cfg["val_batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=use_persistent_workers,
    )
    return train_loader, val_loader


def build_optimizer(model: torch.nn.Module, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    """Build the optimizer for the segmentation baseline."""

    training_cfg = config["training"]
    optimizer_name = str(training_cfg.get("optimizer", "adamw")).lower()
    learning_rate = float(training_cfg["learning_rate"])
    weight_decay = float(training_cfg.get("weight_decay", 0.0))

    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Expected 'adam' or 'adamw'.")


def train_one_epoch(
    model: torch.nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    amp_enabled: bool,
) -> float:
    """Run one training epoch and return the average loss."""

    model.train()
    running_loss = 0.0

    for batch in train_loader:
        images = batch["image"].to(device)
        labels = batch["label"].long().to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(images)
            loss = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += float(loss.item())

    return running_loss / max(len(train_loader), 1)


def save_checkpoint(
    checkpoint_dir: Path,
    filename: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    config: Mapping[str, Any],
) -> Path:
    """Serialize a training checkpoint to disk."""

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / filename
    torch.save(
        {
            "epoch": epoch,
            "best_metric": best_metric,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": dict(config),
        },
        checkpoint_path,
    )
    return checkpoint_path


def main() -> None:
    """Launch model training from the configured dataset and experiment settings."""

    args = parse_args()
    config = load_config(args.config)
    logger = get_logger(__name__)

    set_global_seed(int(config["training"]["seed"]))
    device = resolve_device(config)
    output_dir = Path(config["project"]["output_dir"]).expanduser().resolve()
    checkpoint_dir = output_dir / "checkpoints"
    visual_dir = output_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, cases, train_cases, val_cases, split_path = build_datasets(config)
    train_loader, val_loader = build_dataloaders(config, train_dataset, val_dataset, device)

    model = build_model(config).to(device)
    loss_fn = build_loss(config)
    optimizer = build_optimizer(model, config)

    amp_requested = bool(config["training"].get("amp", True))
    amp_enabled = amp_requested and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    logger.info("Resolved device: %s", device)
    logger.info("Discovered %s cases | train=%s | val=%s", len(cases), len(train_cases), len(val_cases))
    logger.info("Saved split manifest to %s", split_path)
    logger.info("Training model: %s", config["model"]["name"])

    epochs = int(config["training"]["epochs"])
    val_interval = int(config["training"].get("val_interval", 1))
    history: list[dict[str, float]] = []
    best_metric = float("-inf")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )

        epoch_metrics: dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": train_loss,
        }

        if epoch % val_interval == 0:
            validation_metrics = run_validation(
                model=model,
                val_loader=val_loader,
                config=config,
                device=device,
                loss_fn=loss_fn,
                epoch=epoch,
                visual_dir=visual_dir,
            )
            epoch_metrics.update(validation_metrics)

            current_metric = float(validation_metrics["mean_dice"])
            if current_metric > best_metric:
                best_metric = current_metric
                best_path = save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    filename="best_model.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_metric=best_metric,
                    config=config,
                )
                logger.info("Saved new best checkpoint to %s", best_path)

        last_path = save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            filename="last_model.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_metric=best_metric,
            config=config,
        )

        history.append(epoch_metrics)
        metric_summary = ", ".join(
            f"{name}={value:.4f}" for name, value in epoch_metrics.items() if name != "epoch"
        )
        logger.info("Epoch %s | %s", epoch, metric_summary)
        logger.debug("Updated last checkpoint at %s", last_path)

    history_path = output_dir / "metrics_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    logger.info("Saved training history to %s", history_path)


if __name__ == "__main__":
    main()
