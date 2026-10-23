"""
Standalone Temperature Calibration Runner.
Usage:
    python calibrate.py --config configs/base_config.yaml
"""

import argparse
import json
from pathlib import Path
import torch

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.data.dataset import CropDiseaseDataset
from src.data.transforms import build_val_transforms
from src.data.dataloader import build_dataloader
from src.models.factory import build_model
from src.engine.calibrator import TemperatureScalingCalibrator


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate Model Temperature for Calibrated Probabilities")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to YAML configuration")
    parser.add_argument("--weights", type=str, default="ml_service/model/model.pt", help="Path to model weights checkpoint")
    parser.add_argument("--data", type=str, default=None, help="Path to evaluation data directory or manifest CSV (overrides config)")
    parser.add_argument("--max-iter", type=int, default=None, help="Maximum L-BFGS iterations (default: from config)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("CalibCLI")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels_path = Path(cfg.artifacts.labels_path)
    with open(labels_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if isinstance(meta, dict) and "classes" in meta:
        classes = meta["classes"]
    elif isinstance(meta, list):
        classes = meta
    else:
        classes = list(meta.keys())

    cfg.model.num_classes = len(classes)
    model = build_model(cfg.model).to(device)
    model.load_weights(args.weights, map_location=device)
    logger.info("Loaded weights from %s (%d classes)", args.weights, len(classes))

    # Resolve validation data source: --data -> val_dir -> val_manifest -> sample_leaves_dir
    if args.data is not None:
        dataset_source = Path(args.data)
        logger.info("Using specified data source: %s", dataset_source)
    else:
        data_dir = Path(cfg.data.val_dir)
        val_manifest = Path("data/manifests/val_manifest.csv")
        if data_dir.exists():
            dataset_source = data_dir
        elif val_manifest.exists():
            dataset_source = val_manifest
            logger.info("Val dir not found; using validation manifest: %s", val_manifest)
        else:
            dataset_source = Path(cfg.data.sample_leaves_dir)
            logger.warning("Val dir and manifest not found, falling back to '%s'", dataset_source)

    val_tf = build_val_transforms(img_size=cfg.data.img_size)
    dataset = CropDiseaseDataset(dataset_source, transform=val_tf, classes=classes)
    logger.info("Loaded calibration dataset with %d samples.", len(dataset))
    loader = build_dataloader(dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers)

    calibrator = TemperatureScalingCalibrator(model, device)
    max_iter = args.max_iter if args.max_iter is not None else cfg.calibration.max_iter
    res = calibrator.calibrate(loader, max_iter=max_iter, lr=cfg.calibration.lr)
    logger.info("Calibration Result: Temperature = %.4f | ECE: %.4f -> %.4f", res["temperature"], res["pre_ece"], res["post_ece"])

    calibrator.save_calibration(res, cfg.artifacts.calibration_path)
    logger.info("Saved calibration parameters to %s", cfg.artifacts.calibration_path)


if __name__ == "__main__":
    main()
