"""
Unified Training CLI Runner.
Usage:
    python train.py --config configs/base_config.yaml
"""

import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.data.dataset import CropDiseaseDataset
from src.data.transforms import build_train_transforms, build_val_transforms
from src.data.dataloader import build_dataloader
from src.models.factory import build_model
from src.engine.losses import build_criterion
from src.engine.trainer import ModelTrainer
from src.engine.evaluator import ModelEvaluator
from src.engine.calibrator import TemperatureScalingCalibrator


def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seed across frameworks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop Disease Deep Learning Training Pipeline")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to YAML configuration")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("TrainCLI")
    set_seed(cfg.training.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Initializing experiment on device: %s", device)

    # 1. Dataset & DataLoader Construction
    data_dir = Path(cfg.data.train_dir)
    val_data_dir = Path(cfg.data.val_dir)
    train_manifest = Path("data/manifests/train_manifest.csv")
    val_manifest = Path("data/manifests/val_manifest.csv")

    train_tf = build_train_transforms(img_size=cfg.data.img_size)
    val_tf = build_val_transforms(img_size=cfg.data.img_size)

    # Check for pre-existing labels to preserve exact class order if available
    labels_path = Path(cfg.artifacts.labels_path)
    initial_classes = None
    if labels_path.exists():
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            initial_classes = meta["classes"] if (isinstance(meta, dict) and "classes" in meta) else meta
        except Exception as e:
            logger.warning("Could not read existing labels from %s: %s", labels_path, e)

    # Load separate train and val splits when available to prevent split leakage
    if data_dir.exists() and val_data_dir.exists():
        logger.info("Using directory-based splits: train=%s, val=%s", data_dir, val_data_dir)
        train_dataset = CropDiseaseDataset(data_dir, transform=train_tf, classes=initial_classes)
        classes = train_dataset.classes
        val_dataset = CropDiseaseDataset(val_data_dir, transform=val_tf, classes=classes)
    elif train_manifest.exists() and val_manifest.exists():
        logger.info("Using manifest-based splits: train=%s, val=%s", train_manifest, val_manifest)
        train_dataset = CropDiseaseDataset(train_manifest, transform=train_tf, classes=initial_classes)
        classes = train_dataset.classes
        val_dataset = CropDiseaseDataset(val_manifest, transform=val_tf, classes=classes)
    else:
        # Fallback to single directory/manifest with stratified split
        if data_dir.exists():
            dataset_source = data_dir
        elif train_manifest.exists():
            dataset_source = train_manifest
        else:
            dataset_source = Path(cfg.data.sample_leaves_dir)
            logger.warning("Train directory/manifest not found. Falling back to '%s'", dataset_source)

        full_dataset = CropDiseaseDataset(dataset_source, transform=train_tf, classes=initial_classes)
        classes = full_dataset.classes
        val_size = int(len(full_dataset) * cfg.data.val_split)
        train_size = len(full_dataset) - val_size
        if val_size == 0 and len(full_dataset) > 1:
            val_size = 1
            train_size = len(full_dataset) - 1

        train_subset, val_subset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(cfg.training.seed)
        )
        val_dataset_base = CropDiseaseDataset(dataset_source, transform=val_tf, classes=classes)
        val_dataset = torch.utils.data.Subset(val_dataset_base, val_subset.indices)
        train_dataset = train_subset

    num_classes = len(classes)
    logger.info("Dataset loaded across %d classes.", num_classes)

    train_loader = build_dataloader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers)
    val_loader = build_dataloader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers)

    # 2. Model Initialization
    cfg.model.num_classes = num_classes
    model = build_model(cfg.model).to(device)
    params_info = model.count_parameters()
    logger.info("Model: %s | Total Params: %d | Trainable: %d", cfg.model.architecture, params_info["total"], params_info["trainable"])

    # Compute balanced class weights to prevent minority class neglect
    class_counts = np.zeros(num_classes, dtype=np.float32)
    sample_items = getattr(train_dataset, "samples", None)
    if sample_items is None and hasattr(train_dataset, "dataset"):
        # Subset
        underlying = train_dataset.dataset
        sample_items = [underlying.samples[idx] for idx in train_dataset.indices]
    if sample_items:
        for _, lbl_idx in sample_items:
            if 0 <= lbl_idx < num_classes:
                class_counts[lbl_idx] += 1
    class_counts = np.maximum(class_counts, 1.0)
    w = len(train_dataset) / (num_classes * class_counts)
    w = np.power(w, 0.85)  # Label smoothing exponent
    w = w / np.mean(w)
    class_weights_t = torch.tensor(w, dtype=torch.float32).to(device)

    criterion = build_criterion(label_smoothing=cfg.training.label_smoothing, class_weights=class_weights_t)

    # 3. Stage 1: Frozen Backbone Warm-Up
    if cfg.training.stage1.epochs > 0:
        logger.info("=== STAGE 1: Warm-up Classification Head (%d epochs) ===", cfg.training.stage1.epochs)
        model.set_backbone_frozen(freeze=True)
        opt_stage1 = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.training.stage1.lr, weight_decay=cfg.training.weight_decay)
        trainer_stage1 = ModelTrainer(model, criterion, opt_stage1, device, use_amp=cfg.training.use_amp, logger=logger)

        for epoch in range(1, cfg.training.stage1.epochs + 1):
            train_m = trainer_stage1.train_one_epoch(train_loader)
            val_m = trainer_stage1.validate(val_loader)
            logger.info("Stage 1 Epoch %d/%d - Train Loss: %.4f Acc: %.2f%% | Val Loss: %.4f Acc: %.2f%%",
                        epoch, cfg.training.stage1.epochs, train_m.loss, train_m.accuracy * 100, val_m.loss, val_m.accuracy * 100)

    # 4. Stage 2: Full Backbone Fine-Tuning
    if cfg.training.stage2.epochs > 0:
        logger.info("=== STAGE 2: Full Fine-Tuning (%d epochs) ===", cfg.training.stage2.epochs)
        model.set_backbone_frozen(freeze=False)
        opt_stage2 = AdamW(model.parameters(), lr=cfg.training.stage2.lr, weight_decay=cfg.training.weight_decay)
        scheduler = CosineAnnealingLR(opt_stage2, T_max=cfg.training.stage2.epochs, eta_min=1e-6)
        trainer_stage2 = ModelTrainer(model, criterion, opt_stage2, device, scheduler=scheduler, use_amp=cfg.training.use_amp, logger=logger)

        best_val_loss = float("inf")
        best_weights = model.state_dict()

        for epoch in range(1, cfg.training.stage2.epochs + 1):
            train_m = trainer_stage2.train_one_epoch(train_loader)
            val_m = trainer_stage2.validate(val_loader)
            logger.info("Stage 2 Epoch %d/%d - Train Loss: %.4f Acc: %.2f%% | Val Loss: %.4f Acc: %.2f%%",
                        epoch, cfg.training.stage2.epochs, train_m.loss, train_m.accuracy * 100, val_m.loss, val_m.accuracy * 100)

            if val_m.loss < best_val_loss:
                best_val_loss = val_m.loss
                best_weights = model.state_dict()

        model.load_state_dict(best_weights)

    # 5. Final Evaluation
    evaluator = ModelEvaluator(model, device, classes=classes)
    eval_metrics, y_pred, y_true, y_probs = evaluator.evaluate(val_loader)
    logger.info("Validation Results -> Accuracy: %.2f%% | Macro-F1: %.4f | Pre-Calib ECE: %.4f",
                eval_metrics["accuracy"] * 100, eval_metrics["macro_f1"], eval_metrics["ece"])

    # 6. Post-Hoc Temperature Scaling Calibration
    logger.info("Optimizing Temperature Scaling for Uncertainty Calibration...")
    calibrator = TemperatureScalingCalibrator(model, device)
    calib_res = calibrator.calibrate(val_loader)
    logger.info("Calibration Complete: T = %.4f | ECE: %.4f -> %.4f",
                calib_res["temperature"], calib_res["pre_ece"], calib_res["post_ece"])

    # 7. Save Production Artifacts
    model_path = Path(cfg.artifacts.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "architecture": cfg.model.architecture,
        "num_classes": num_classes,
        "classes": classes,
        "best_val_f1": eval_metrics.get("macro_f1", 0.0),
    }, model_path)
    logger.info("Saved model weights to %s", model_path)

    calibrator.save_calibration(calib_res, cfg.artifacts.calibration_path)
    logger.info("Saved calibration parameters to %s", cfg.artifacts.calibration_path)

    labels_path = Path(cfg.artifacts.labels_path)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_payload = {
        "classes": classes,
        "label_to_idx": {cls_name: i for i, cls_name in enumerate(classes)}
    }
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels_payload, f, indent=2)
    logger.info("Saved class labels (%d classes) to %s", len(classes), labels_path)


if __name__ == "__main__":
    main()
