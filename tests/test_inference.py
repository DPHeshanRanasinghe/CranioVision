"""
CranioVision — Inference test suite.

Covers the inference stack (predict, mc_dropout, grad_cam, ensemble):
  - Model loading + shape contracts
  - Volume computation unit math
  - MC Dropout dropout toggling + output shapes
  - Grad-CAM layer discovery + output shapes
  - Ensemble graceful handling of missing checkpoints

Run with:  pytest -v tests/
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES — runtime setup
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def ckpt_path():
    """Path to the Attention U-Net checkpoint (skips entire file if missing)."""
    from src.cranovision.config import MODELS_DIR
    p = MODELS_DIR / "attention_unet_best.pth"
    if not p.exists():
        pytest.skip(f"Checkpoint not found at {p} — run training first")
    return p


@pytest.fixture(scope="session")
def test_cases():
    """List of test cases for real inference runs."""
    from src.cranovision.data import get_splits
    _, _, cases = get_splits(verbose=False)
    if len(cases) == 0:
        pytest.skip("No test cases found — check data split")
    return cases


@pytest.fixture(scope="session")
def loaded_model(ckpt_path):
    """Load Attention U-Net once — expensive, reused across tests."""
    from src.cranovision.inference import load_model
    return load_model("attention_unet", ckpt_path)


# ══════════════════════════════════════════════════════════════════════════════
# VOLUME COMPUTATION — pure unit math, no GPU needed
# ══════════════════════════════════════════════════════════════════════════════

class TestVolumeComputation:
    """compute_region_volumes must produce correct cm³ values."""

    def test_all_background_zero_volume(self):
        """A mask of all zeros (no tumor) should report zero volume everywhere."""
        from src.cranovision.inference import compute_region_volumes
        mask = torch.zeros(10, 10, 10, dtype=torch.long)
        vols = compute_region_volumes(mask)
        assert vols["Edema"] == 0.0
        assert vols["Enhancing tumor"] == 0.0
        assert vols["Necrotic core"] == 0.0
        assert vols["Total tumor"] == 0.0

    def test_unit_conversion_mm3_to_cm3(self):
        """1000 voxels @ 1mm³ each = 1.00 cm³."""
        from src.cranovision.inference import compute_region_volumes
        mask = torch.zeros(10, 10, 10, dtype=torch.long)
        mask[:10, :10, :10] = 1  # 1000 voxels of class 1 (edema)
        vols = compute_region_volumes(mask, voxel_volume_mm3=1.0)
        assert abs(vols["Edema"] - 1.0) < 1e-6, f"Expected 1.00 cm³, got {vols['Edema']}"

    def test_total_is_sum_of_parts(self):
        """Total tumor == sum of all foreground class volumes."""
        from src.cranovision.inference import compute_region_volumes
        mask = torch.zeros(20, 20, 20, dtype=torch.long)
        mask[0:5,   :, :] = 1    # edema
        mask[5:10,  :, :] = 2    # enhancing
        mask[10:15, :, :] = 3    # necrotic
        vols = compute_region_volumes(mask)
        parts = vols["Edema"] + vols["Enhancing tumor"] + vols["Necrotic core"]
        assert abs(vols["Total tumor"] - parts) < 1e-6

    def test_accepts_4d_mask(self):
        """Should handle (1, D, H, W) channel-first masks too."""
        from src.cranovision.inference import compute_region_volumes
        mask = torch.ones(1, 10, 10, 10, dtype=torch.long)  # all edema
        vols = compute_region_volumes(mask)
        assert vols["Edema"] == 1.0
        assert vols["Total tumor"] == 1.0


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

class TestModelLoading:

    def test_loads_in_eval_mode(self, loaded_model):
        """Loaded model must be in eval mode by default."""
        assert not loaded_model.training, "Model should be in eval mode after load_model()"

    def test_model_is_on_correct_device(self, loaded_model):
            """Model must be on the configured DEVICE."""
            from src.cranovision.config import DEVICE
            param = next(loaded_model.parameters())
            # DEVICE can be a torch.device or a string — handle both
            expected_type = DEVICE.type if hasattr(DEVICE, "type") else str(DEVICE).split(":")[0]
            assert param.device.type == expected_type

    def test_unknown_model_name_raises(self, ckpt_path):
        """Unknown model_name should raise ValueError."""
        from src.cranovision.inference import load_model
        with pytest.raises(ValueError):
            load_model("nonexistent_model_xyz", ckpt_path)

    def test_missing_checkpoint_raises(self):
        """Nonexistent checkpoint path should raise FileNotFoundError."""
        from src.cranovision.inference import load_model
        with pytest.raises(FileNotFoundError):
            load_model("attention_unet", Path("/nonexistent/path/model.pth"))


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT_CASE — full pipeline smoke test
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictCase:

    @pytest.fixture(scope="class")
    def prediction(self, loaded_model, test_cases):
        """Run one inference — expensive but many tests can share it."""
        from src.cranovision.inference import predict_case
        return predict_case(loaded_model, test_cases[0])

    def test_prediction_shape(self, prediction):
        """Prediction should be 3D — no channel dim, spatial H×W×D."""
        assert prediction["pred"].ndim == 3

    def test_prediction_dtype_is_integer(self, prediction):
        """Class predictions must be integer indices, not floats."""
        from src.cranovision.config import NUM_CLASSES
        assert prediction["pred"].dtype in (torch.int64, torch.int32, torch.long)

    def test_prediction_values_in_valid_range(self, prediction):
        """All class indices must be in [0, NUM_CLASSES)."""
        from src.cranovision.config import NUM_CLASSES
        unique = prediction["pred"].unique().tolist()
        assert all(0 <= v < NUM_CLASSES for v in unique), \
            f"Prediction contains class indices outside [0, {NUM_CLASSES}): {unique}"

    def test_case_id_preserved(self, prediction, test_cases):
        """case_id should round-trip through the pipeline."""
        assert prediction["case_id"] == test_cases[0]["case_id"]


# ══════════════════════════════════════════════════════════════════════════════
# MC DROPOUT
# ══════════════════════════════════════════════════════════════════════════════

class TestMCDropout:

    def test_count_dropout_layers(self, loaded_model):
        """Attention U-Net should have dropout layers (built with dropout=0.1)."""
        from src.cranovision.inference import count_dropout_layers
        n = count_dropout_layers(loaded_model)
        assert n > 0, "Model must have Dropout layers for MC Dropout to work"

    def test_enable_dropout_toggles_training_mode(self, loaded_model):
        """After enable_dropout, dropout layers must be in training mode."""
        from src.cranovision.inference import enable_dropout
        loaded_model.eval()
        enable_dropout(loaded_model)
        dropout_states = [
            m.training for m in loaded_model.modules()
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d))
        ]
        assert all(dropout_states), "All Dropout layers should be in training mode"
        loaded_model.eval()  # restore

    def test_batchnorm_stays_in_eval(self, loaded_model):
        """After enable_dropout, BatchNorm must stay in eval mode (key MC Dropout invariant)."""
        from src.cranovision.inference import enable_dropout
        loaded_model.eval()
        enable_dropout(loaded_model)
        bn_states = [
            m.training for m in loaded_model.modules()
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
        ]
        if len(bn_states) > 0:
            assert not any(bn_states), "BatchNorm layers must stay in eval — this is the MC Dropout invariant"
        loaded_model.eval()

    @pytest.mark.slow
    def test_mc_dropout_output_shapes(self, loaded_model, test_cases):
        """MC Dropout output should have correct shapes."""
        from src.cranovision.inference import mc_dropout_predict
        result = mc_dropout_predict(loaded_model, test_cases[0], n_samples=2, verbose=False)
        pred = result["pred"]
        unc = result["uncertainty"]
        mean_p = result["mean_probs"]

        assert pred.ndim == 3
        assert unc.shape == pred.shape
        assert mean_p.ndim == 4
        assert mean_p.shape[0] == 4  # 4 classes

    @pytest.mark.slow
    def test_mc_dropout_confidence_in_valid_range(self, loaded_model, test_cases):
        """Mean confidence must be in [0, 1]."""
        from src.cranovision.inference import mc_dropout_predict, summarize_confidence
        result = mc_dropout_predict(loaded_model, test_cases[0], n_samples=2, verbose=False)
        conf = summarize_confidence(result)
        assert 0 <= conf["mean_confidence"] <= 1
        assert 0 <= conf["uncertain_fraction"] <= 1


# ══════════════════════════════════════════════════════════════════════════════
# GRAD-CAM
# ══════════════════════════════════════════════════════════════════════════════

class TestGradCAM:

    def test_find_target_layer_returns_conv3d(self, loaded_model):
        """The auto-selected target layer must be a Conv3d."""
        from src.cranovision.inference import find_target_layer
        layer = find_target_layer(loaded_model, "attention_unet")
        assert isinstance(layer, nn.Conv3d), \
            f"Target layer should be Conv3d, got {type(layer).__name__}"

    @pytest.mark.slow
    def test_gradcam_heatmaps_in_valid_range(self, loaded_model, test_cases):
        """Heatmaps must be in [0, 1] after normalization."""
        from src.cranovision.inference import compute_grad_cam
        result = compute_grad_cam(
            model=loaded_model, case_dict=test_cases[0],
            model_name="attention_unet", target_classes=(1, 2, 3),
            verbose=False,
        )
        for cls, hm in result["heatmaps_patch"].items():
            assert hm.min() >= 0 and hm.max() <= 1, \
                f"Class {cls} heatmap out of [0, 1]: [{hm.min()}, {hm.max()}]"


# ══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════

class TestEnsemble:

    def test_load_ensemble_handles_missing_checkpoints(self):
        """Should load only the models whose checkpoints exist, skip others gracefully."""
        from src.cranovision.inference import load_ensemble
        models = load_ensemble(verbose=False)
        assert isinstance(models, dict)
        # At least 1 model expected (Attention U-Net)
        assert len(models) >= 1, "Expected at least Attention U-Net to be loaded"

    def test_weights_from_val_dice_normalizes_to_1(self):
        """Weight computation should sum to 1."""
        from src.cranovision.inference import weights_from_val_dice
        dice = {"attention_unet": 0.76, "swin_unetr": 0.84, "nnunet": 0.78}
        weights = weights_from_val_dice(dice)
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_weights_proportional_to_dice(self):
        """Model with higher val Dice should get higher weight."""
        from src.cranovision.inference import weights_from_val_dice
        dice = {"a": 0.5, "b": 0.9}
        w = weights_from_val_dice(dice)
        assert w["b"] > w["a"], "Higher Dice should yield higher ensemble weight"

    @pytest.mark.slow
    def test_ensemble_predict_with_single_model(self, loaded_model, test_cases):
        """Ensemble of 1 model == that model with weight 1.0."""
        from src.cranovision.inference import ensemble_predict
        models = {"attention_unet": loaded_model}
        result = ensemble_predict(models, test_cases[0], verbose=False)

        assert result["n_models"] == 1
        assert result["weights_used"]["attention_unet"] == 1.0
        assert result["pred"].ndim == 3


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

class TestMetrics:

    def test_perfect_prediction_gives_dice_1(self):
        """Dice of identical masks should be 1.0."""
        from src.cranovision.training.metrics import compute_case_dice
        mask = torch.zeros(10, 10, 10, dtype=torch.long)
        mask[:5, :5, :5] = 1
        mask[5:, 5:, :] = 2
        dice = compute_case_dice(mask, mask)
        assert all(abs(d - 1.0) < 1e-5 for d in dice), \
            f"Perfect overlap should give Dice 1.0, got {dice}"

    def test_zero_overlap_gives_dice_0(self):
        """Dice of completely disjoint masks should be 0.0."""
        from src.cranovision.training.metrics import compute_case_dice
        a = torch.zeros(10, 10, 10, dtype=torch.long)
        b = torch.zeros(10, 10, 10, dtype=torch.long)
        a[:5, :, :] = 1
        b[5:, :, :] = 1
        dice = compute_case_dice(a, b)
        assert abs(dice[0]) < 1e-5, f"Disjoint masks should give Dice 0, got {dice[0]}"

    def test_brats_region_dice_includes_all_regions(self):
        """compute_brats_region_dice should return WT, TC, ET."""
        from src.cranovision.training.metrics import compute_brats_region_dice
        pred = torch.zeros(10, 10, 10, dtype=torch.long)
        pred[:3, :, :] = 1  # edema
        pred[3:6, :, :] = 2  # enhancing
        result = compute_brats_region_dice(pred, pred)
        assert set(result.keys()) == {"WT", "TC", "ET"}
        # Self-overlap = perfect
        for r, d in result.items():
            assert abs(d - 1.0) < 1e-5


# ══════════════════════════════════════════════════════════════════════════════
# PYTEST MARKERS
# ══════════════════════════════════════════════════════════════════════════════

# Register the 'slow' marker — tests that run actual GPU inference
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: test runs actual GPU inference (~1-2 min)")