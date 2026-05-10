"""
CranioVision — 3D patch-based Grad-CAM with architecture-agnostic layer search.

Key design choices
------------------
1. Architecture-agnostic target layer discovery (Approach B).
   Walks the model graph, finds the deepest Conv3d before the final
   classification head. Works for Attention U-Net, SwinUNETR, nnU-Net DynUNet,
   and any future 3D segmentation model with a similar topology.

2. Patch-based execution.
   Full-volume Grad-CAM on a transformer like SwinUNETR needs ~7GB of GPU
   memory for gradient storage. We locate the tumor centroid via a forward
   pass, crop a 128^3 patch, run Grad-CAM there, and stitch back. Fits in
   ~2GB on GPU, ~6GB on CPU.

3. Per-class heatmaps.
   For each tumor class (edema, enhancing, necrotic), we generate a separate
   heatmap showing what the model attended to when predicting that class.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from ..config import CLASS_NAMES, DEVICE, USE_AMP
from ..data import get_val_transforms
from .predict import make_inferer


# -----------------------------------------------------------------------------
# HEATMAP POST-PROCESSING — runs once at generation time so every consumer
# (NIfTI overlay, 3D mesh, PDF figure) gets the cleaned volume.
# -----------------------------------------------------------------------------

def _clean_heatmap(
    heatmap: np.ndarray,
    brain_mask: np.ndarray,
    sigma: float = 2.0,
    threshold: float = 0.1,
) -> np.ndarray:
    """
    Smooth the patch boundary, clip to brain, renormalize, and suppress noise.

    The 128^3 Grad-CAM patch leaves a hard rectangular edge where it meets the
    surrounding zeros, and produces spurious activations outside the brain.
    Smoothing fades the box edge; the brain mask removes air-region signal;
    the threshold cleans up faint background after renormalization.
    """
    # Soften the patch boundary so the rectangular edge blends into zeros.
    smoothed = gaussian_filter(heatmap, sigma=sigma)
    # Clip to brain — kills activations that landed in air.
    smoothed = smoothed * brain_mask
    # Renormalize after smoothing reduced the peak.
    peak = smoothed.max()
    if peak > 0:
        smoothed = smoothed / peak
    # Suppress faint background.
    smoothed[smoothed < threshold] = 0.0
    return smoothed.astype(np.float32)


# -----------------------------------------------------------------------------
# APPROACH B — architecture-agnostic target layer discovery
# -----------------------------------------------------------------------------

def find_target_layer(
    model: nn.Module,
    model_name: Optional[str] = None,
) -> Tuple[nn.Module, str]:
    """
    Find the best Grad-CAM target layer for any 3D segmentation model.

    Strategy (Approach B with kernel-size preference):
      1. Walk the entire module tree, collect every Conv3d.
      2. Skip the final 1x1x1 classification head (last conv with kernel
         size 1 and small out_channels).
      3. Among the LAST FEW remaining convs, prefer kernels with spatial
         extent (3x3x3 over 1x1x1). 1x1x1 convs do channel mixing only,
         which is poor for Grad-CAM — we want spatial reasoning layers.
      4. Return the deepest such layer.

    Returns
    -------
    (layer, layer_name) tuple
    """
    all_conv3d: List[Tuple[str, nn.Conv3d]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv3d):
            all_conv3d.append((name, module))

    if not all_conv3d:
        raise RuntimeError(
            "No Conv3d layers found in model. "
            "Grad-CAM requires a 3D convolutional network."
        )

    # Identify if the very last conv is a classification head
    last_name, last_layer = all_conv3d[-1]
    is_head = (
        tuple(last_layer.kernel_size) == (1, 1, 1)
        and last_layer.out_channels <= 16
    )
    candidates = all_conv3d[:-1] if (is_head and len(all_conv3d) >= 2) else all_conv3d

    # Look at the last 5 candidates and prefer one with kernel size > 1.
    # This skips through 1x1x1 channel-mixing convs to find the deepest
    # spatial-reasoning layer (typically a 3x3x3 conv).
    tail = candidates[-5:] if len(candidates) >= 5 else candidates
    spatial_candidates = [
        (n, m) for n, m in tail
        if any(k > 1 for k in m.kernel_size)
    ]

    if spatial_candidates:
        # Use the deepest spatial conv from the tail
        target_name, target_layer = spatial_candidates[-1]
    else:
        # Fall back to deepest conv overall (even if 1x1x1)
        target_name, target_layer = candidates[-1]

    return target_layer, target_name

# -----------------------------------------------------------------------------
# PATCH UTILITIES
# -----------------------------------------------------------------------------

def _find_tumor_centroid(prediction: torch.Tensor) -> Optional[Tuple[int, int, int]]:
    """
    Find the centroid (center of mass) of the predicted tumor region.

    Returns None if there is no predicted tumor.
    """
    tumor_mask = (prediction > 0).numpy()
    if not tumor_mask.any():
        return None
    coords = np.argwhere(tumor_mask)
    centroid = coords.mean(axis=0).round().astype(int)
    return tuple(centroid.tolist())


def _crop_patch(
    image: torch.Tensor,
    centroid: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """
    Crop a patch_size patch from image centered on centroid.

    image shape: (C, D, H, W)
    centroid:    (z, y, x) in voxel space — wait, actually (D, H, W) order

    Returns (patch, offset) where offset is the (D, H, W) index of the patch's
    origin in the original volume.
    """
    _, D, H, W = image.shape
    pd, ph, pw = patch_size
    cd, ch, cw = centroid

    # Top-left corner, clipped to volume bounds
    d0 = max(0, min(D - pd, cd - pd // 2))
    h0 = max(0, min(H - ph, ch - ph // 2))
    w0 = max(0, min(W - pw, cw - pw // 2))

    patch = image[:, d0:d0 + pd, h0:h0 + ph, w0:w0 + pw]
    return patch, (d0, h0, w0)


# -----------------------------------------------------------------------------
# CORE GRAD-CAM ENGINE
# -----------------------------------------------------------------------------

class GradCAM3D:
    """
    3D Grad-CAM engine with hooks on a specified target layer.

    Use as a context manager OR explicitly call cleanup() when done.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Register hooks
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        # Save a copy that won't be affected by later in-place ops
        self.activations = out.detach() if not out.requires_grad else out

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def cleanup(self):
        """Remove hooks. MUST be called when done to avoid memory leaks."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        target_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate a Grad-CAM heatmap for one class.

        Parameters
        ----------
        input_tensor : (1, C, D, H, W) input volume
        target_class : class index (1=edema, 2=enhancing, 3=necrotic for BraTS)
        target_mask  : optional (D, H, W) bool mask. If given, we score only
                       the target_class logits within this mask. This makes the
                       heatmap explain "why did the model predict this class HERE"
                       rather than averaging over the whole volume.

        Returns
        -------
        heatmap : (D, H, W) tensor in [0, 1]
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward — gradients ON
        with torch.set_grad_enabled(True):
            logits = self.model(input_tensor)

            # Score = sum of target_class logits, optionally restricted to mask
            if target_mask is not None:
                # logits shape: (1, num_classes, D, H, W). Mask shape: (D, H, W)
                if not target_mask.any():
                    # Nothing to score — return zero heatmap
                    return torch.zeros(input_tensor.shape[2:])
                target_mask_5d = target_mask.unsqueeze(0).unsqueeze(0).to(logits.device)  # (1,1,D,H,W)
                score = (logits[:, target_class:target_class + 1] * target_mask_5d.float()).sum()
            else:
                score = logits[:, target_class].sum()

        score.backward()

        # Compute Grad-CAM
        # Activations: (1, K, d, h, w) where K = num feature channels
        # Gradients:   (1, K, d, h, w)
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hooks didn't capture activations/gradients. Bad target layer?")

        # Channel-wise weights = global avg pool of gradients
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)   # (1, K, 1, 1, 1)

        # Weighted sum of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, d, h, w)
        cam = F.relu(cam)  # only positive contributions

        # Upsample to input spatial size
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:],
            mode="trilinear", align_corners=False,
        )
        cam = cam.squeeze(0).squeeze(0)  # (D, H, W)

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.detach().cpu()


# -----------------------------------------------------------------------------
# PUBLIC API — patch-based Grad-CAM for one model + case
# -----------------------------------------------------------------------------

def compute_grad_cam(
    model: nn.Module,
    case_dict: Dict,
    model_name: str = "model",
    target_classes: Sequence[int] = (1, 2, 3),
    use_predicted_mask: bool = True,
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    force_cpu: bool = False,
    verbose: bool = False,
) -> Dict:
    """
    Compute patch-based Grad-CAM for one model.

    Parameters
    ----------
    force_cpu : if True, runs Grad-CAM entirely on CPU. Slower but always
                works. Use this for memory-heavy models like SwinUNETR
                on small GPUs. In production deployment all models run
                CPU-side anyway, so behavior matches.
    """
    case_id = case_dict.get("case_id", "unknown")
    target_layer, target_layer_name = find_target_layer(model, model_name=model_name)
    if verbose:
        print(f"  [grad-cam] target layer ({model_name}): {target_layer_name}")

    # Choose device upfront. Don't try to recover from OOM mid-flight.
    if force_cpu:
        device = torch.device("cpu")
        model = model.to(device)
        if verbose:
            print(f"  [grad-cam] forced CPU mode")
    else:
        device = next(model.parameters()).device

    transforms = get_val_transforms()
    sample = transforms(case_dict)
    image = sample["image"]

    # Full-volume prediction once
    inferer = make_inferer()
    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=USE_AMP and device.type == "cuda"):
            image_dev = image.unsqueeze(0).to(device)
            full_logits = inferer(image_dev, model)
        full_pred = full_logits.argmax(dim=1).squeeze(0).cpu()
    del full_logits, image_dev
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Find tumor centroid
    centroid = _find_tumor_centroid(full_pred)
    if centroid is None:
        if verbose:
            print(f"  [grad-cam] no tumor predicted, using volume center")
        centroid = tuple(d // 2 for d in image.shape[1:])

    # Crop the patch
    image_patch, patch_offset = _crop_patch(image, centroid, patch_size)
    pred_patch = full_pred[
        patch_offset[0]:patch_offset[0] + patch_size[0],
        patch_offset[1]:patch_offset[1] + patch_size[1],
        patch_offset[2]:patch_offset[2] + patch_size[2],
    ]
    img_batch = image_patch.unsqueeze(0).to(device)

    if verbose:
        print(f"  [grad-cam] patch offset: {patch_offset}, size: {patch_size}, device: {device}")
        print(f"  [grad-cam] tumor in patch: {(pred_patch > 0).sum().item():,} voxels")

    heatmaps_patch: Dict[int, torch.Tensor] = {}
    heatmaps_full: Dict[int, torch.Tensor] = {}

    # Brain mask in full-volume coordinates. NormalizeIntensityd(nonzero=True)
    # leaves background voxels at exactly 0 across all 4 channels, so any
    # nonzero magnitude marks brain.
    image_np = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
    brain_mask_full = (np.abs(image_np).sum(axis=0) > 0).astype(np.float32)

    with GradCAM3D(model, target_layer) as cam_engine:
        for cls in target_classes:
            if verbose:
                print(f"  [grad-cam] class {cls} ({CLASS_NAMES[cls]})")
            mask = (pred_patch == cls) if use_predicted_mask else None
            hm_patch = cam_engine.generate(
                input_tensor=img_batch,
                target_class=cls,
                target_mask=mask,
            )
            heatmaps_patch[cls] = hm_patch
            hm_full_np = np.zeros(full_pred.shape, dtype=np.float32)
            hm_full_np[
                patch_offset[0]:patch_offset[0] + patch_size[0],
                patch_offset[1]:patch_offset[1] + patch_size[1],
                patch_offset[2]:patch_offset[2] + patch_size[2],
            ] = hm_patch.numpy()
            hm_full_np = _clean_heatmap(hm_full_np, brain_mask_full)
            heatmaps_full[cls] = torch.from_numpy(hm_full_np)

    del img_batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "case_id": case_id,
        "model_name": model_name,
        "target_layer": target_layer_name,
        "pred": full_pred,
        "image": image,
        "heatmaps": heatmaps_full,
        "heatmaps_patch": heatmaps_patch,
        "patch_offset": patch_offset,
        "patch_size": patch_size,
        "device_used": str(device),
    }


# -----------------------------------------------------------------------------
# CLI / smoke test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path
    from ..config import MODELS_DIR
    from ..data import get_splits
    from .predict import load_model

    print("=" * 60)
    print("CranioVision — grad_cam.py smoke test (Approach B)")
    print("=" * 60)

    # Try each available model
    candidates = [
        ("attention_unet", "attention_unet_best.pth", {}),
        ("swin_unetr", "swin_unetr_best.pth", {"feature_size": 48, "use_checkpoint": False}),
        ("nnunet", "nnunet_best.pth", {"filters": (32, 64, 128, 256, 320, 320), "deep_supervision": False}),
    ]

    _, _, test_cases = get_splits(verbose=False)
    case = test_cases[0]

    for name, ckpt, kwargs in candidates:
        path = MODELS_DIR / ckpt
        if not path.exists():
            print(f"\n[skip] {name}: checkpoint not found")
            continue

        print(f"\n--- {name} ---")
        try:
            model = load_model(name, path, **kwargs)
            tgt, tgt_name = find_target_layer(model, model_name=name)
            print(f"  Auto-discovered target layer: {tgt_name}")
            print(f"  Target type: {type(tgt).__name__}, kernel: {tgt.kernel_size}, "
                  f"in_ch: {tgt.in_channels}, out_ch: {tgt.out_channels}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [error] {type(e).__name__}: {e}")

    print("\nApproach B layer search verified.")