"""Plotting helpers for dataset exploration and prediction checks."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np


def _normalize_for_display(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32)
    minimum = float(volume.min())
    maximum = float(volume.max())
    return (volume - minimum) / (maximum - minimum + 1e-8)


def _select_slice(volume: np.ndarray, axis: int, slice_index: int | None) -> np.ndarray:
    if slice_index is None:
        slice_index = volume.shape[axis] // 2
    return np.take(volume, indices=slice_index, axis=axis)


def _mask_to_rgb(mask_slice: np.ndarray) -> np.ndarray:
    """Color map for remapped labels {0, 1, 2, 3}."""

    overlay = np.zeros((*mask_slice.shape, 4), dtype=np.float32)
    overlay[mask_slice == 1] = [1.0, 0.9, 0.0, 0.6]
    overlay[mask_slice == 2] = [1.0, 0.2, 0.1, 0.8]
    overlay[mask_slice == 3] = [0.1, 0.4, 1.0, 0.8]
    return overlay


def plot_modalities_and_label(
    modality_arrays: Mapping[str, np.ndarray],
    label_array: np.ndarray,
    case_id: str,
    axis: int = 2,
    slice_index: int | None = None,
    output_path: str | Path | None = None,
) -> None:
    """Plot all modalities plus the mask for a single case."""

    figure, axes = plt.subplots(1, len(modality_arrays) + 1, figsize=(20, 5))

    for subplot, (modality_name, volume) in zip(axes[:-1], modality_arrays.items()):
        image_slice = _normalize_for_display(_select_slice(volume, axis=axis, slice_index=slice_index))
        subplot.imshow(np.rot90(image_slice), cmap="gray")
        subplot.set_title(modality_name)
        subplot.axis("off")

    label_slice = _select_slice(label_array, axis=axis, slice_index=slice_index)
    axes[-1].imshow(np.rot90(label_slice), cmap="viridis")
    axes[-1].set_title("label")
    axes[-1].axis("off")

    figure.suptitle(f"{case_id} | modality and label overview")
    figure.tight_layout()

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.show()


def save_prediction_panel(
    image_volume: np.ndarray,
    label_volume: np.ndarray,
    prediction_volume: np.ndarray,
    output_path: str | Path,
    title: str,
    axis: int = 2,
    slice_index: int | None = None,
) -> None:
    """Save a 3-panel view of raw MRI, ground truth, and model prediction."""

    image_slice = _normalize_for_display(_select_slice(image_volume, axis=axis, slice_index=slice_index))
    label_slice = _select_slice(label_volume, axis=axis, slice_index=slice_index)
    prediction_slice = _select_slice(prediction_volume, axis=axis, slice_index=slice_index)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(np.rot90(image_slice), cmap="gray")
    axes[0].set_title("MRI")
    axes[1].imshow(np.rot90(image_slice), cmap="gray")
    axes[1].imshow(np.rot90(_mask_to_rgb(label_slice)))
    axes[1].set_title("Ground Truth")
    axes[2].imshow(np.rot90(image_slice), cmap="gray")
    axes[2].imshow(np.rot90(_mask_to_rgb(prediction_slice)))
    axes[2].set_title("Prediction")

    for axis_handle in axes:
        axis_handle.axis("off")

    figure.suptitle(title)
    figure.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
