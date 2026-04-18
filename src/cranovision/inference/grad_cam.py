"""
CranioVision — 3D Grad-CAM for segmentation networks (patch-based).

Memory-efficient Grad-CAM for 4GB GPUs:
  - Finds the tumor center from an initial prediction
  - Crops a 128³ patch around it
  - Computes Grad-CAM on that patch only
  - Stitches heatmap back into full-volume coordinates

Gradients take ~3× more memory than forward passes. On 4GB cards, full-volume
Grad-CAM over 160³+ crashes. Patch-based computation is standard practice for
high-resolution 3D XAI.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DEVICE, USE_AMP, PATCH_SIZE, NUM_CLASSES, CLASS_NAMES
from ..data import get_val_transforms


# ══════════════════════════════════════════════════════════════════════════════
# LAYER SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def find_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """Pick a sensible deep Conv3d layer for Grad-CAM extraction."""
    convs = [m for m in model.modules() if isinstance(m, nn.Conv3d)]
    if not convs:
        raise RuntimeError("No Conv3d layer found in model.")
    return convs[-2] if len(convs) >= 2 else convs[-1]


# ══════════════════════════════════════════════════════════════════════════════
# CORE GRAD-CAM
# ══════════════════════════════════════════════════════════════════════════════

class GradCAM3D:
    """3D Grad-CAM with forward/backward hooks on a target layer."""
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fwd = target_layer.register_forward_hook(self._save_act)
        self._bwd = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, input, output):
        self.activations = output.detach()

    def _save_grad(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self._fwd.remove()
        self._bwd.remove()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        target_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.model.eval()
        self.model.zero_grad()
        input_tensor = input_tensor.clone().detach().requires_grad_(True)

        logits = self.model(input_tensor)
        class_logits = logits[0, target_class]

        if target_mask is not None:
            target_mask = target_mask.to(class_logits.device)
            if target_mask.sum() > 0:
                score = class_logits[target_mask].sum()
            else:
                score = class_logits.sum()
        else:
            score = class_logits.sum()

        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hooks did not fire.")

        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:],
                            mode="trilinear", align_corners=False)
        cam = cam.squeeze(0).squeeze(0)

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = torch.zeros_like(cam)

        del logits, class_logits, score, weights
        torch.cuda.empty_cache()
        return cam.cpu()


# ══════════════════════════════════════════════════════════════════════════════
# PATCH EXTRACTION AROUND TUMOR
# ══════════════════════════════════════════════════════════════════════════════

def _find_tumor_centroid(mask: torch.Tensor) -> Tuple[int, int, int]:
    nz = torch.nonzero(mask > 0)
    if nz.numel() == 0:
        D, H, W = mask.shape
        return D // 2, H // 2, W // 2
    cz = int(nz[:, 0].float().mean().item())
    cy = int(nz[:, 1].float().mean().item())
    cx = int(nz[:, 2].float().mean().item())
    return cz, cy, cx


def _extract_patch(volume: torch.Tensor,
                   center: Tuple[int, int, int],
                   patch: Tuple[int, int, int]) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    _, D, H, W = volume.shape
    pd, ph, pw = patch
    cz, cy, cx = center
    z0 = max(0, min(D - pd, cz - pd // 2))
    y0 = max(0, min(H - ph, cy - ph // 2))
    x0 = max(0, min(W - pw, cx - pw // 2))
    crop = volume[:, z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]
    return crop, (z0, y0, x0)


# ══════════════════════════════════════════════════════════════════════════════
# HIGH-LEVEL API
# ══════════════════════════════════════════════════════════════════════════════

def compute_grad_cam(
    model: nn.Module,
    case_dict: Dict,
    model_name: str = "attention_unet",
    target_classes: Tuple[int, ...] = (1, 2, 3),
    use_predicted_mask: bool = True,
    patch_size: Tuple[int, int, int] = PATCH_SIZE,
    verbose: bool = True,
) -> Dict:
    """
    Patch-based Grad-CAM — runs on a single 128³ crop centered on the tumor.
    """
    transforms = get_val_transforms()
    sample = transforms(case_dict)
    full_image = sample["image"]

    # Initial prediction
    from monai.inferers import SlidingWindowInferer
    inferer = SlidingWindowInferer(roi_size=patch_size, sw_batch_size=1,
                                    overlap=0.5, mode="gaussian")
    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            full_logits = inferer(full_image.unsqueeze(0).to(DEVICE), model)
        full_pred = full_logits.argmax(dim=1).squeeze(0).cpu()
    del full_logits
    torch.cuda.empty_cache()

    center = _find_tumor_centroid(full_pred)
    if verbose:
        print(f"Tumor centroid (z, y, x): {center}")

    img_patch, (z0, y0, x0) = _extract_patch(full_image, center, patch_size)
    pred_patch = full_pred[z0:z0 + patch_size[0],
                            y0:y0 + patch_size[1],
                            x0:x0 + patch_size[2]]

    if verbose:
        print(f"Patch offset (z0, y0, x0): ({z0}, {y0}, {x0})")
        print(f"Patch shape             : {tuple(img_patch.shape)}")

    target_layer = find_target_layer(model, model_name)
    layer_name = type(target_layer).__name__
    if verbose:
        print(f"Target layer: {layer_name}")

    cam_engine = GradCAM3D(model, target_layer)
    heatmaps_patch = {}
    heatmaps_full  = {}

    try:
        img_batch = img_patch.unsqueeze(0).to(DEVICE)
        for cls in target_classes:
            if verbose:
                print(f"  → Generating heatmap for class {cls} ({CLASS_NAMES[cls]})")
            mask = (pred_patch == cls) if use_predicted_mask else None
            hm_patch = cam_engine.generate(
                input_tensor=img_batch,
                target_class=cls,
                target_mask=mask,
            )
            heatmaps_patch[cls] = hm_patch

            hm_full = torch.zeros(full_pred.shape, dtype=torch.float32)
            hm_full[z0:z0 + patch_size[0],
                    y0:y0 + patch_size[1],
                    x0:x0 + patch_size[2]] = hm_patch
            heatmaps_full[cls] = hm_full
    finally:
        cam_engine.remove_hooks()
        torch.cuda.empty_cache()

    return {
        "heatmaps_patch": heatmaps_patch,
        "heatmaps"      : heatmaps_full,
        "image"         : full_image,
        "pred"          : full_pred,
        "label"         : sample["label"].cpu() if "label" in sample else None,
        "case_id"       : case_dict.get("case_id", "unknown"),
        "target_layer"  : layer_name,
        "patch_offset"  : (z0, y0, x0),
        "patch_size"    : patch_size,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from ..config import MODELS_DIR
    from ..data import get_splits
    from .predict import load_model

    print("=" * 60)
    print("CranioVision — grad_cam.py smoke test (patch-based)")
    print("=" * 60)

    ckpt = MODELS_DIR / "attention_unet_best.pth"
    model = load_model("attention_unet", ckpt)

    _, _, test_cases = get_splits(verbose=False)
    case = test_cases[0]

    result = compute_grad_cam(
        model=model,
        case_dict=case,
        model_name="attention_unet",
        target_classes=(1, 2, 3),
        use_predicted_mask=True,
        verbose=True,
    )

    print(f"\n─── Result ───")
    print(f"Case ID     : {result['case_id']}")
    print(f"Target layer: {result['target_layer']}")
    print(f"Patch offset: {result['patch_offset']}")
    print(f"Patch size  : {result['patch_size']}")
    print(f"Heatmaps    :")
    for cls, hm in result["heatmaps_patch"].items():
        hm_full = result["heatmaps"][cls]
        print(f"  Class {cls} ({CLASS_NAMES[cls]:20s}): "
              f"patch {tuple(hm.shape)} range [{hm.min():.3f}, {hm.max():.3f}]")

    print("\n✅ grad_cam.py works.")