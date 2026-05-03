"""
CranioVision — Figure generation for the clinical PDF report.

All matplotlib figures rendered for the report live here. They use a clinical
palette (white background, dark text) so they integrate cleanly into the PDF.
Medical figures preserve meaningful color coding:

  - Edema      : yellow
  - Enhancing  : red
  - Necrotic   : blue
  - Heatmaps   : red-orange (warm spectrum)
  - Risk levels: red (high) / orange (moderate) / green (low/minimal)
"""
from __future__ import annotations

from io import BytesIO
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safer for batch / server use)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch


# -----------------------------------------------------------------------------
# COLOR DEFINITIONS — consistent across all report figures
# -----------------------------------------------------------------------------

# Clinical tumor colors (used in segmentation overlays everywhere)
CLINICAL_COLORS = {
    1: (1.00, 0.85, 0.00),  # Edema      - golden yellow
    2: (0.92, 0.20, 0.13),  # Enhancing  - red
    3: (0.20, 0.45, 0.85),  # Necrotic   - blue
}

# Risk level colors (used in eloquent cortex badges)
RISK_COLORS = {
    "high":     "#D73027",  # red
    "moderate": "#FC8D59",  # orange
    "low":      "#91BFDB",  # light blue
    "minimal":  "#4575B4",  # dark blue
    "n/a":      "#888888",
    "unknown":  "#888888",
}

# Grad-CAM heatmap colormap (warm spectrum on white background)
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "heatmap",
    ["#FFFFFF", "#FFEDA0", "#FEB24C", "#FD8D3C", "#FC4E2A", "#BD0026"],
)


def _format_axis(ax) -> None:
    """Strip ticks/labels from an image axis (for medical slices)."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize an array to [0, 1]."""
    a = arr.astype(np.float32)
    mn, mx = a.min(), a.max()
    if mx > mn:
        return (a - mn) / (mx - mn)
    return np.zeros_like(a)


def _overlay_segmentation(
    mri_slice: np.ndarray,
    seg_slice: np.ndarray,
    alpha: float = 0.55,
) -> np.ndarray:
    """Build an RGB image: grayscale MRI with colored segmentation labels."""
    base = _normalize(mri_slice)
    rgb = np.stack([base, base, base], axis=-1)

    for label, color in CLINICAL_COLORS.items():
        mask = seg_slice == label
        if mask.any():
            for c in range(3):
                rgb[..., c] = np.where(
                    mask,
                    (1 - alpha) * rgb[..., c] + alpha * color[c],
                    rgb[..., c],
                )
    return np.clip(rgb, 0, 1)


def _overlay_heatmap(
    mri_slice: np.ndarray,
    heatmap: np.ndarray,
    threshold: float = 0.05,
    alpha: float = 0.6,
) -> np.ndarray:
    """Build an RGB image: grayscale MRI with warm heatmap overlay."""
    base = _normalize(mri_slice)
    rgb = np.stack([base, base, base], axis=-1)

    hm_n = _normalize(heatmap)
    colored = HEATMAP_CMAP(hm_n)[..., :3]
    mask = hm_n > threshold

    for c in range(3):
        rgb[..., c] = np.where(
            mask,
            (1 - alpha * hm_n) * rgb[..., c] + alpha * hm_n * colored[..., c],
            rgb[..., c],
        )
    return np.clip(rgb, 0, 1)


def _to_numpy(t):
    """Tensor -> numpy, leaves arrays alone."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _save_to_buffer(fig) -> BytesIO:
    """Render figure to an in-memory PNG buffer for ReportLab embedding."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf


# -----------------------------------------------------------------------------
# FIGURE 1 — Page 1 hero: 3-view anatomical projection of the chosen prediction
# -----------------------------------------------------------------------------

def render_hero_segmentation(
    image: torch.Tensor,
    prediction: torch.Tensor,
) -> BytesIO:
    """
    Render the page-1 hero figure: axial / coronal / sagittal slices of the
    T1c MRI with the chosen prediction overlaid.

    image      : (4, D, H, W) preprocessed input. Channel 1 = T1c.
    prediction : (D, H, W) class indices.

    Returns
    -------
    BytesIO containing PNG bytes.
    """
    img = _to_numpy(image)
    pred = _to_numpy(prediction)
    t1c = img[1]   # T1c modality

    # Find the most informative slice in each plane (most tumor)
    z_axial = int(((pred > 0).sum(axis=(0, 1))).argmax())
    y_coronal = int(((pred > 0).sum(axis=(0, 2))).argmax())
    x_sagittal = int(((pred > 0).sum(axis=(1, 2))).argmax())

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    fig.patch.set_facecolor("white")

    # Axial (looking from above)
    axes[0].imshow(_overlay_segmentation(t1c[:, :, z_axial].T,
                                          pred[:, :, z_axial].T),
                    origin="lower")
    axes[0].set_title(f"Axial  (z={z_axial})", color="black", fontsize=11)
    _format_axis(axes[0])

    # Coronal (looking from front)
    axes[1].imshow(_overlay_segmentation(t1c[:, y_coronal, :].T,
                                          pred[:, y_coronal, :].T),
                    origin="lower")
    axes[1].set_title(f"Coronal  (y={y_coronal})", color="black", fontsize=11)
    _format_axis(axes[1])

    # Sagittal (looking from side)
    axes[2].imshow(_overlay_segmentation(t1c[x_sagittal, :, :].T,
                                          pred[x_sagittal, :, :].T),
                    origin="lower")
    axes[2].set_title(f"Sagittal  (x={x_sagittal})", color="black", fontsize=11)
    _format_axis(axes[2])

    # Legend
    handles = [
        mpatches.Patch(color=CLINICAL_COLORS[1], label="Edema"),
        mpatches.Patch(color=CLINICAL_COLORS[2], label="Enhancing tumor"),
        mpatches.Patch(color=CLINICAL_COLORS[3], label="Necrotic core"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    return _save_to_buffer(fig)


# -----------------------------------------------------------------------------
# FIGURE 2 — Page 2: side-by-side per-model predictions
# -----------------------------------------------------------------------------

def render_model_comparison(
    image: torch.Tensor,
    predictions: Dict[str, torch.Tensor],
    display_names: Dict[str, str],
) -> BytesIO:
    """
    Render a row of 4 thumbnail axial slices, one per prediction
    (3 models + ensemble). Same slice across all four for visual comparison.

    predictions   : dict[model_name, (D, H, W) tensor]
    display_names : dict[model_name, human-readable label]
    """
    img = _to_numpy(image)
    t1c = img[1]

    # Find an axial slice that has tumor in (using any prediction)
    pick = next(iter(predictions.values()))
    pick = _to_numpy(pick)
    z = int(((pick > 0).sum(axis=(0, 1))).argmax())

    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.5))
    fig.patch.set_facecolor("white")
    if n == 1:
        axes = [axes]

    for ax, (name, pred) in zip(axes, predictions.items()):
        pred_np = _to_numpy(pred)
        ax.imshow(_overlay_segmentation(t1c[:, :, z].T, pred_np[:, :, z].T),
                   origin="lower")
        title = display_names.get(name, name)
        ax.set_title(title, color="black", fontsize=10, fontweight="bold")
        _format_axis(ax)

    plt.tight_layout()
    return _save_to_buffer(fig)


# -----------------------------------------------------------------------------
# FIGURE 3 — Page 3: anatomical lobe pie chart
# -----------------------------------------------------------------------------

def render_lobe_pie(anatomy: Dict) -> Optional[BytesIO]:
    """
    Render a pie chart of tumor distribution across brain lobes.
    Returns None if no anatomical data is available.
    """
    lobes = anatomy.get("lobes", {})
    if not lobes:
        return None

    labels = list(lobes.keys())
    sizes = [lobes[l]["pct_of_tumor"] for l in labels]

    # Sensible color palette for lobes (medical-themed but distinct)
    palette = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
               "#8C564B", "#E377C2", "#7F7F7F"]
    colors = palette[:len(labels)]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    fig.patch.set_facecolor("white")

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=90,
        textprops={"color": "black", "fontsize": 9},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
        at.set_fontsize(9)

    ax.set_title("Tumor distribution by brain lobe",
                  color="black", fontsize=11, pad=10)
    plt.tight_layout()
    return _save_to_buffer(fig)


# -----------------------------------------------------------------------------
# FIGURE 4 — Page 3: eloquent-cortex distance bar chart
# -----------------------------------------------------------------------------

def render_eloquent_distances(eloquent: Dict) -> BytesIO:
    """
    Horizontal bar chart of distances from tumor to each eloquent region.
    Color-coded by risk level. Threshold lines at 5/10/20mm.
    """
    names = list(eloquent.keys())
    distances = []
    colors = []
    for name in names:
        info = eloquent[name]
        d = info.get("distance_mm")
        if d is None or d == float("inf"):
            d = 100.0
        distances.append(min(float(d), 100.0))
        colors.append(RISK_COLORS.get(info.get("risk_level", "unknown"), "#888"))

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("white")
    y_pos = np.arange(len(names))

    ax.barh(y_pos, distances, color=colors, edgecolor="black", linewidth=0.5)

    # Threshold reference lines
    ax.axvline(x=5, color="#D73027", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(x=10, color="#FC8D59", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(x=20, color="#91BFDB", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, color="black", fontsize=9)
    ax.set_xlabel("Min distance from tumor edge (mm)", color="black", fontsize=10)
    ax.set_xlim(0, max(distances) * 1.15 + 5)
    ax.invert_yaxis()
    ax.tick_params(axis="x", colors="black")
    ax.set_facecolor("white")
    for s in ax.spines.values():
        s.set_edgecolor("#888")

    # Annotate bars with their distance values
    for i, (bar_d, info_name) in enumerate(zip(distances, names)):
        info = eloquent[info_name]
        if info.get("involved"):
            label_text = "INVOLVED"
        elif info.get("distance_mm") is None or info.get("distance_mm") == float("inf"):
            label_text = "n/a"
        else:
            label_text = f"{bar_d:.1f} mm"
        ax.text(bar_d + 1, i, label_text, va="center",
                color="black", fontsize=8.5)

    ax.grid(alpha=0.15, axis="x")
    ax.set_title("Eloquent cortex proximity",
                 color="black", fontsize=11, pad=8)
    plt.tight_layout()
    return _save_to_buffer(fig)


# -----------------------------------------------------------------------------
# FIGURE 5 — Page 4: Grad-CAM heatmaps for each tumor class
# -----------------------------------------------------------------------------

def render_xai_heatmaps(
    image: torch.Tensor,
    heatmaps: Dict[int, torch.Tensor],
    prediction: torch.Tensor,
    class_names: Dict[int, str],
) -> BytesIO:
    """
    Render a 1xN grid of Grad-CAM heatmaps overlaid on the T1c MRI,
    one panel per tumor class.

    heatmaps    : dict[class_id, (D, H, W) tensor], normalized to [0, 1]
    prediction  : (D, H, W) — used to find the best slice
    class_names : dict[class_id, name]
    """
    img = _to_numpy(image)
    pred = _to_numpy(prediction)
    t1c = img[1]

    z = int(((pred > 0).sum(axis=(0, 1))).argmax())
    classes = sorted(heatmaps.keys())

    n = len(classes)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 4))
    fig.patch.set_facecolor("white")
    if n == 1:
        axes = [axes]

    for ax, cls in zip(axes, classes):
        hm = _to_numpy(heatmaps[cls])
        rgb = _overlay_heatmap(t1c[:, :, z].T, hm[:, :, z].T)
        ax.imshow(rgb, origin="lower")
        ax.set_title(class_names.get(cls, f"Class {cls}"),
                     color="black", fontsize=10)
        _format_axis(ax)

    fig.suptitle("Feature attention (Grad-CAM)", color="black", fontsize=11,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    return _save_to_buffer(fig)