"""Config-driven training entrypoint for CranioVision."""

from __future__ import annotations

import argparse
from pathlib import Path

from monai.data import CacheDataset, DataLoader

from src.data.brats_dataset import discover_brats_cases, split_cases, write_split_manifest
from src.data.transforms import build_train_transforms, build_val_transforms
from src.models.factory import build_model
from src.training.trainer import SegmentationTrainer
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 3D BraTS-style segmentation model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the YAML experiment config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logger = get_logger(__name__)

    set_global_seed(config["training"]["seed"])

    output_dir = Path(config["project"]["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = discover_brats_cases(
        root_dir=config["data"]["root_dir"],
        modality_aliases=config["data"]["modality_aliases"],
        label_aliases=config["data"]["label_aliases"],
        file_extensions=config["data"]["file_extensions"],
        expected_modalities=config["data"]["expected_modalities"],
        validate_shapes=config["data"].get("validate_shapes", True),
    )
    train_cases, val_cases = split_cases(
        cases=cases,
        validation_ratio=config["data"]["validation_ratio"],
        seed=config["data"]["seed"],
    )
    split_path = write_split_manifest(train_cases, val_cases, output_dir)
    logger.info("Discovered %s cases | train=%s | val=%s", len(cases), len(train_cases), len(val_cases))
    logger.info("Saved split manifest to %s", split_path)

    train_dataset = CacheDataset(
        data=train_cases,
        transform=build_train_transforms(config),
        cache_rate=config["data"]["cache_rate"],
        num_workers=config["data"]["num_workers"],
    )
    val_dataset = CacheDataset(
        data=val_cases,
        transform=build_val_transforms(config),
        cache_rate=config["data"]["cache_rate"],
        num_workers=config["data"]["num_workers"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["val_batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    model = build_model(config)
    trainer = SegmentationTrainer(config=config, output_dir=output_dir)
    trainer.train(model=model, train_loader=train_loader, val_loader=val_loader)


if __name__ == "__main__":
    main()
