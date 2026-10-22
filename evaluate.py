"""
Standalone Model Evaluation Runner.
Usage:
    python evaluate.py --config configs/base_config.yaml --weights ml_service/model/model.pt
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
from src.engine.evaluator import ModelEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Crop Disease Classifier")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to YAML configuration")
    parser.add_argument("--weights", type=str, default="ml_service/model/model.pt", help="Path to model weights checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("EvalCLI")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load labels
    labels_path = Path(cfg.artifacts.labels_path)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found at {labels_path}")
    with open(labels_path, "r", encoding="utf-8") as f:
        classes = json.load(f)

    cfg.model.num_classes = len(classes)
    model = build_model(cfg.model).to(device)
    model.load_weights(args.weights, map_location=device)
    logger.info("Loaded weights from %s", args.weights)

    data_dir = Path(cfg.data.test_dir)
    if not data_dir.exists():
        data_dir = Path(cfg.data.sample_leaves_dir)
        logger.warning("Test dir '%s' not found, using '%s'", cfg.data.test_dir, data_dir)

    val_tf = build_val_transforms(img_size=cfg.data.img_size)
    dataset = CropDiseaseDataset(data_dir, transform=val_tf, classes=classes)
    loader = build_dataloader(dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers)

    evaluator = ModelEvaluator(model, device, classes=classes)
    metrics, y_pred, y_true, y_probs = evaluator.evaluate(loader)

    report_str = evaluator.generate_report_string(y_true, y_pred)
    logger.info("Classification Report:\n%s", report_str)
    logger.info("Summary Metrics: Accuracy = %.2f%% | Macro-F1 = %.4f | ECE = %.4f",
                metrics["accuracy"] * 100, metrics["macro_f1"], metrics["ece"])

    out_report = Path(cfg.artifacts.metrics_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", out_report)


if __name__ == "__main__":
    main()
