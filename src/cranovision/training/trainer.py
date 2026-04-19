"""
CranioVision — Unified training loop.

Model-agnostic trainer used by all branches (attention-unet, SwinUNETR, nnU-Net).
Each branch only needs to pass its model + a name; training logic is shared.

Design principles:
  - Framework code here, NOT model code.
  - Training loop, validation loop, checkpointing, AMP, history tracking.
  - Callers inject their own model + loss + optimizer + (optional) scheduler.
  - Best model is saved by validation mean Dice.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, List

import torch
from torch.utils.data import DataLoader
from monai.data import decollate_batch
from monai.inferers import SlidingWindowInferer

from ..config import (
    DEVICE,
    USE_AMP,
    MAX_EPOCHS,
    VAL_INTERVAL,
    PATCH_SIZE,
    MODELS_DIR,
    OUTPUTS_DIR,
)
from .metrics import make_dice_metric, post_pred, post_label, format_per_class_dice


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING CONFIG (lightweight dataclass)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    """Everything a training run needs beyond the model itself."""
    model_name   : str                        # e.g. "attention_unet"
    max_epochs   : int = MAX_EPOCHS
    val_interval : int = VAL_INTERVAL
    patch_size   : tuple = PATCH_SIZE
    use_amp      : bool = USE_AMP
    sw_overlap   : float = 0.5                # sliding-window overlap at val
    sw_batch     : int = 2                    # sliding-window batch at val
    ckpt_dir     : Path = MODELS_DIR
    history_dir  : Path = OUTPUTS_DIR


@dataclass
class TrainHistory:
    train_loss : List[float] = field(default_factory=list)
    val_epochs : List[int]   = field(default_factory=list)
    val_dice   : List[float] = field(default_factory=list)
    best_epoch : int = 0
    best_dice  : float = -1.0

    def to_dict(self):
        return {
            "train_loss": self.train_loss,
            "val_epochs": self.val_epochs,
            "val_dice"  : self.val_dice,
            "best_epoch": self.best_epoch,
            "best_dice" : self.best_dice,
        }


# ══════════════════════════════════════════════════════════════════════════════
# INFERER — sliding window for validation
# ══════════════════════════════════════════════════════════════════════════════

def build_inferer(patch_size=PATCH_SIZE, sw_batch=2, overlap=0.5):
    return SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=sw_batch,
        overlap=overlap,
        mode="gaussian",
    )


# ══════════════════════════════════════════════════════════════════════════════
# ONE EPOCH — training
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, loss_fn, optimizer, scaler, use_amp: bool):
    model.train()
    running_loss = 0.0
    n_batches = 0

    for batch in loader:
        images = batch["image"].to(DEVICE, non_blocking=True)
        if hasattr(images, "as_tensor"): images = images.as_tensor()
        labels = batch["label"].to(DEVICE, non_blocking=True)
        if hasattr(labels, "as_tensor"): labels = labels.as_tensor()

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = loss_fn(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(n_batches, 1)


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION — mean Dice with sliding-window inference
# ══════════════════════════════════════════════════════════════════════════════

def validate(model, loader, inferer, use_amp: bool):
    """Returns (mean_dice, per_class_dice_tensor)."""
    model.eval()
    dice = make_dice_metric()
    dice.reset()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(DEVICE, non_blocking=True)
            if hasattr(images, "as_tensor"): images = images.as_tensor()
            labels = batch["label"].to(DEVICE, non_blocking=True)
            if hasattr(labels, "as_tensor"): labels = labels.as_tensor()

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = inferer(images, model)

            # Post-process for metric
            outputs_list = decollate_batch(outputs)
            labels_list  = decollate_batch(labels)
            pred_onehot  = [post_pred(o)  for o in outputs_list]
            label_onehot = [post_label(l) for l in labels_list]

            dice(y_pred=pred_onehot, y=label_onehot)

    per_class = dice.aggregate()            # shape: (num_classes-1,)
    mean_dice = per_class.mean().item()
    dice.reset()
    return mean_dice, per_class


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TRAIN FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    config: Optional[TrainConfig] = None,
    verbose: bool = True,
) -> TrainHistory:
    """
    Full training loop with validation + best-model checkpointing.

    Returns a TrainHistory with loss/Dice curves and best epoch info.
    """
    if config is None:
        raise ValueError("TrainConfig required — pass config=TrainConfig(model_name='...').")

    config.ckpt_dir.mkdir(parents=True, exist_ok=True)
    config.history_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(DEVICE)
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)
    inferer = build_inferer(config.patch_size, config.sw_batch, config.sw_overlap)

    history = TrainHistory()
    best_ckpt = config.ckpt_dir / f"{config.model_name}_best.pth"
    hist_file = config.history_dir / f"{config.model_name}_history.json"

    if verbose:
        print("═" * 70)
        print(f"Training: {config.model_name}")
        print(f"Device  : {DEVICE}  |  AMP: {config.use_amp}")
        print(f"Epochs  : {config.max_epochs}  (validate every {config.val_interval})")
        print(f"Batches : {len(train_loader)} train, {len(val_loader)} val")
        print(f"Best ckpt → {best_ckpt}")
        print("═" * 70)

    start = time.time()

    for epoch in range(1, config.max_epochs + 1):

        # ─── Train ────────────────────────────────────────────────────────
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler, config.use_amp
        )
        history.train_loss.append(train_loss)

        if scheduler is not None:
            scheduler.step()

        # ─── Validate ─────────────────────────────────────────────────────
        do_val = (epoch % config.val_interval == 0) or (epoch == 1) or (epoch == config.max_epochs)

        if do_val:
            mean_dice, per_class = validate(model, val_loader, inferer, config.use_amp)
            history.val_epochs.append(epoch)
            history.val_dice.append(mean_dice)

            is_best = mean_dice > history.best_dice
            if is_best:
                history.best_dice = mean_dice
                history.best_epoch = epoch
                torch.save(model.state_dict(), best_ckpt)
                marker = " ← BEST"
            else:
                marker = ""

            if verbose:
                elapsed = (time.time() - start) / 60
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"Ep {epoch:3d}/{config.max_epochs} | "
                    f"Loss: {train_loss:.4f} | "
                    f"Val Dice: {mean_dice:.4f}  [{format_per_class_dice(per_class)}] | "
                    f"LR: {lr_now:.2e} | {elapsed:.1f}m{marker}"
                )
        else:
            if verbose and (epoch % 10 == 0):
                elapsed = (time.time() - start) / 60
                print(f"Ep {epoch:3d}/{config.max_epochs} | Loss: {train_loss:.4f} | {elapsed:.1f}m")

        # Save history every epoch (cheap, small JSON)
        with open(hist_file, "w") as f:
            json.dump(history.to_dict(), f, indent=2)

    total_min = (time.time() - start) / 60
    if verbose:
        print("═" * 70)
        print(f"Done in {total_min:.1f} minutes.")
        print(f"Best Val Dice: {history.best_dice:.4f} at epoch {history.best_epoch}")
        print(f"Checkpoint   : {best_ckpt}")
        print(f"History JSON : {hist_file}")
        print("═" * 70)

    return history


# ══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST — train a tiny dummy model for 2 epochs
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Run: python -m src.cranovision.training.trainer
    # Uses a throwaway 1-layer model on a few real BraTS cases.
    import torch.nn as nn
    from monai.data import Dataset, DataLoader as MonaiLoader
    from monai.losses import DiceCELoss

    from ..data import get_splits, get_train_transforms, get_val_transforms
    from ..config import NUM_CLASSES, NUM_CHANNELS

    print("=" * 60)
    print("CranioVision — trainer.py smoke test (2 epochs, tiny model)")
    print("=" * 60)

    train_files, val_files, _ = get_splits(verbose=False)
    # Limit to 2 train, 1 val for speed
    train_files = train_files[:2]
    val_files   = val_files[:1]
    print(f"Using {len(train_files)} train / {len(val_files)} val cases for smoke test")

    train_ds = Dataset(data=train_files, transform=get_train_transforms())
    val_ds   = Dataset(data=val_files,   transform=get_val_transforms())

    train_loader = MonaiLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    val_loader   = MonaiLoader(val_ds,   batch_size=1, shuffle=False, num_workers=0)

    # Dummy single-conv model — just to test the loop runs
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(NUM_CHANNELS, NUM_CLASSES, kernel_size=3, padding=1)
        def forward(self, x):
            return self.conv(x)

    model = DummyModel()
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=None,
        config=TrainConfig(model_name="smoke_test", max_epochs=2, val_interval=1),
        verbose=True,
    )

    print(f"\n✅ trainer.py works.  Smoke history: {history.to_dict()}")