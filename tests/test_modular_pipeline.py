"""
Test Suite for Modular Deep Learning Components and CLI Runners.
Validates:
1. ModularCropClassifier architecture, forward pass, and weight loading from model.pt.
2. CropDiseaseDataset loading from flat image directories (sample_real_leaves) and manifests.
3. ModelEvaluator and TemperatureScalingCalibrator robustness with normal and empty batches.
4. Evaluation and Calibration CLI runners (evaluate.py and calibrate.py).
"""

import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

from src.models.factory import build_model
from src.data.dataset import CropDiseaseDataset
from src.data.transforms import build_val_transforms
from src.data.dataloader import build_dataloader
from src.engine.evaluator import ModelEvaluator
from src.engine.calibrator import TemperatureScalingCalibrator
from src.utils.metrics import compute_classification_metrics, compute_expected_calibration_error
from src.utils.config import load_config


def test_modular_classifier_architecture_and_weights():
    """Validates that ModularCropClassifier correctly loads model.pt without key errors."""
    cfg = load_config("configs/base_config.yaml")
    device = torch.device("cpu")
    model = build_model(cfg.model).to(device)

    # 1. Check forward pass on dummy input
    dummy_input = torch.randn(2, 3, 224, 224, device=device)
    logits = model(dummy_input)
    assert logits.shape == (2, 39), f"Expected logits shape (2, 39), got {logits.shape}"

    # 2. Check loading model.pt
    model_path = "ml_service/model/model.pt"
    assert os.path.exists(model_path), f"Checkpoint missing at {model_path}"
    model.load_weights(model_path, map_location=device)

    # 3. Parameter count sanity check
    params = model.count_parameters()
    assert params["total"] > 1_000_000, "Parameter count unexpectedly low"
    assert params["trainable"] == params["total"]


def test_dataset_flat_directory_loading():
    """Validates that CropDiseaseDataset successfully loads from flat directory of real leaves."""
    with open("ml_service/model/labels.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    classes = meta["classes"] if isinstance(meta, dict) else meta

    val_tf = build_val_transforms(img_size=224)
    dataset = CropDiseaseDataset("data/sample_real_leaves", transform=val_tf, classes=classes)

    assert len(dataset) > 0, "Flat sample_real_leaves directory loaded 0 samples!"
    tensor, label_idx = dataset[0]
    assert tensor.shape == (3, 224, 224)
    assert 0 <= label_idx < len(classes)


def test_dataset_manifest_loading():
    """Validates that CropDiseaseDataset loads CSV manifests containing 'path' column."""
    manifest_path = "data/manifests/val_manifest.csv"
    if not os.path.exists(manifest_path):
        pytest.skip("val_manifest.csv not present")

    val_tf = build_val_transforms(img_size=224)
    dataset = CropDiseaseDataset(manifest_path, transform=val_tf)
    assert len(dataset) > 0
    tensor, label_idx = dataset[0]
    assert tensor.shape == (3, 224, 224)


def test_metrics_empty_array_guards():
    """Validates that evaluation and calibration metrics do not raise on empty input."""
    empty_res = compute_classification_metrics(np.array([]), np.array([]))
    assert empty_res["accuracy"] == 0.0
    assert empty_res["macro_f1"] == 0.0

    empty_ece = compute_expected_calibration_error(np.array([]), np.array([]))
    assert empty_ece == 0.0


def test_evaluator_and_calibrator_on_samples():
    """Validates ModelEvaluator and TemperatureScalingCalibrator execution."""
    cfg = load_config("configs/base_config.yaml")
    device = torch.device("cpu")
    model = build_model(cfg.model).to(device)
    model.load_weights("ml_service/model/model.pt", map_location=device)

    with open("ml_service/model/labels.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    classes = meta["classes"] if isinstance(meta, dict) else meta

    val_tf = build_val_transforms(img_size=224)
    dataset = CropDiseaseDataset("data/sample_real_leaves", transform=val_tf, classes=classes)
    loader = build_dataloader(dataset, batch_size=8, shuffle=False, num_workers=0)

    # Evaluator
    evaluator = ModelEvaluator(model, device, classes=classes)
    metrics, y_pred, y_true, y_probs = evaluator.evaluate(loader)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "ece" in metrics
    assert len(y_pred) == len(dataset)

    # Calibrator
    calibrator = TemperatureScalingCalibrator(model, device)
    calib_res = calibrator.calibrate(loader, max_iter=5)
    assert "temperature" in calib_res
    assert calib_res["temperature"] > 0.0


def test_evaluate_cli():
    """Verifies that evaluate.py CLI runs to completion with exit code 0."""
    python_exe = sys.executable
    cmd = [
        python_exe, "evaluate.py",
        "--config", "configs/base_config.yaml",
        "--data", "data/sample_real_leaves"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f"evaluate.py failed with stderr: {proc.stderr}\nstdout: {proc.stdout}"
    assert "Summary Metrics" in proc.stderr or "Summary Metrics" in proc.stdout


def test_calibrate_cli():
    """Verifies that calibrate.py CLI runs to completion with exit code 0."""
    python_exe = sys.executable
    cmd = [
        python_exe, "calibrate.py",
        "--config", "configs/base_config.yaml",
        "--data", "data/sample_real_leaves",
        "--max-iter", "5"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f"calibrate.py failed with stderr: {proc.stderr}\nstdout: {proc.stdout}"
    assert "Calibration Result" in proc.stderr or "Calibration Result" in proc.stdout
