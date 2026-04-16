"""Run sliding-window inference on the validation split and save predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.data import CacheDataset, DataLoader
from monai.inferers import sliding_window_inference

from src.data.brats_dataset import discover_brats_cases, split_cases
from src.data.transforms import build_val_transforms
from src.models.factory import build_model
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.visualization import save_prediction_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on BraTS-style validation cases.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    return parser.parse_args()


def restore_original_labels(
    prediction: np.ndarray,
    mapped_values: list[int],
    original_values: list[int],
) -> np.ndarray:
    restored = np.zeros_like(prediction, dtype=np.int16)
    for mapped_value, original_value in zip(mapped_values, original_values):
        restored[prediction == mapped_value] = original_value
    return restored


def save_prediction_nifti(
    prediction: np.ndarray,
    reference_path: str,
    output_path: Path,
) -> None:
    reference = nib.load(reference_path)
    image = nib.Nifti1Image(prediction.astype(np.int16), affine=reference.affine, header=reference.header)
    nib.save(image, str(output_path))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logger = get_logger(__name__)

    device_name = config["training"].get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    cases = discover_brats_cases(
        root_dir=config["data"]["root_dir"],
        modality_aliases=config["data"]["modality_aliases"],
        label_aliases=config["data"]["label_aliases"],
        file_extensions=config["data"]["file_extensions"],
        expected_modalities=config["data"]["expected_modalities"],
        validate_shapes=config["data"].get("validate_shapes", True),
    )
    _, val_cases = split_cases(
        cases=cases,
        validation_ratio=config["data"]["validation_ratio"],
        seed=config["data"]["seed"],
    )

    dataset = CacheDataset(
        data=val_cases,
        transform=build_val_transforms(config),
        cache_rate=config["data"]["cache_rate"],
        num_workers=config["data"]["num_workers"],
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    output_dir = Path(config["project"]["output_dir"]).expanduser().resolve() / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    modality_name = config["visualization"].get("modality_for_panels", "t1ce")
    modality_names = config["data"]["expected_modalities"]
    modality_index = modality_names.index(modality_name) if modality_name in modality_names else 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            logits = sliding_window_inference(
                inputs=images,
                roi_size=tuple(config["preprocessing"]["roi_size"]),
                sw_batch_size=config["inference"]["sw_batch_size"],
                predictor=model,
                overlap=config["inference"]["overlap"],
            )
            prediction = torch.argmax(logits, dim=1)[0].detach().cpu().numpy()
            restored = restore_original_labels(
                prediction=prediction,
                mapped_values=config["labels"]["mapped_values"],
                original_values=config["labels"]["original_values"],
            )

            case_id = batch["case_id"][0]
            case_output_dir = output_dir / case_id
            case_output_dir.mkdir(parents=True, exist_ok=True)

            reference_path = batch["label_meta_dict"]["filename_or_obj"][0]
            save_prediction_nifti(
                prediction=restored,
                reference_path=reference_path,
                output_path=case_output_dir / f"{case_id}-pred.nii.gz",
            )

            save_prediction_panel(
                image_volume=batch["image"][0, modality_index].detach().cpu().numpy(),
                label_volume=batch["label"][0, 0].detach().cpu().numpy(),
                prediction_volume=prediction,
                output_path=case_output_dir / f"{case_id}-panel.png",
                title=f"{case_id} | {modality_name}",
                axis=config["visualization"].get("slice_axis", 2),
            )
            logger.info("Saved prediction outputs for %s", case_id)


if __name__ == "__main__":
    main()
