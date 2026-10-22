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
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("CalibCLI")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels_path = Path(cfg.artifacts.labels_path)
    with open(labels_path, "r", encoding="utf-8") as f:
        classes = json.load(f)

    cfg.model.num_classes = len(classes)
    model = build_model(cfg.model).to(device)
    model.load_weights(args.weights, map_location=device)

    data_dir = Path(cfg.data.val_dir)
    if not data_dir.exists():
        data_dir = Path(cfg.data.sample_leaves_dir)

    val_tf = build_val_transforms(img_size=cfg.data.img_size)
    dataset = CropDiseaseDataset(data_dir, transform=val_tf, classes=classes)
    loader = build_dataloader(dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers)

    calibrator = TemperatureScalingCalibrator(model, device)
    res = calibrator.calibrate(loader, max_iter=cfg.calibration.max_iter, lr=cfg.calibration.lr)
    logger.info("Calibration Result: Temperature = %.4f | ECE: %.4f -> %.4f", res["temperature"], res["pre_ece"], res["post_ece"])

    calibrator.save_calibration(res, cfg.artifacts.calibration_path)
    logger.info("Saved calibration parameters to %s", cfg.artifacts.calibration_path)


if __name__ == "__main__":
    main()
