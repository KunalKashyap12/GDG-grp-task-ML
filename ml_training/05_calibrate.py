"""
Script 05: Post-Hoc Model Calibration via Temperature Scaling
1. Solves for optimal scalar temperature T on validation logits minimizing NLL.
2. Computes Expected Calibration Error (ECE) before and after calibration.
3. Generates Reliability Diagrams.
4. Calibrates Energy Score threshold for Out-Of-Distribution (OOD / Not-a-leaf) detection.
5. Exports calibration.json to ml_service/model/calibration.json.
"""

import os
import json
import yaml
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import LBFGS
from torchvision.models import mobilenet_v3_large
import matplotlib.pyplot as plt
from typing import Tuple

from utils.augmentations import get_eval_transforms
from utils.dataset import CropDiseaseDataset
from utils.metrics import compute_ece

def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate model confidence with Temperature Scaling.")
    parser.add_argument("--config", type=str, default="ml_training/config.yaml", help="Path to config.yaml")
    return parser.parse_args()


class ModelWithTemperature(nn.Module):
    """Decorator model that applies learned temperature scaling to raw logits."""
    def __init__(self, model: nn.Module):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        # Expand temperature to match logits size
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_temperature(self, val_loader, device):
        """Tunes temperature of model using validation set logits via L-BFGS."""
        self.to(device)
        self.eval()
        nll_criterion = nn.CrossEntropyLoss().to(device)

        # Collect all logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for images, targets, _, _ in val_loader:
                images = images.to(device)
                logits = self.model(images)
                logits_list.append(logits)
                labels_list.append(targets)
            logits = torch.cat(logits_list).to(device)
            labels = torch.cat(labels_list).to(device)

        # Optimize temperature
        optimizer = LBFGS([self.temperature], lr=0.01, max_iter=100)

        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return logits, labels


def plot_reliability_diagrams(
    pre_diag: dict,
    post_diag: dict,
    pre_ece: float,
    post_ece: float,
    save_path: str
):
    """Plots comparative reliability diagrams before and after temperature scaling."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Before calibration
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")
    ax.bar(
        pre_diag["bin_confidences"],
        pre_diag["bin_accuracies"],
        width=0.05,
        alpha=0.6,
        color="crimson",
        edgecolor="black",
        label=f"Outputs (ECE={pre_ece*100:.2f}%)"
    )
    ax.set_title("Before Calibration (Uncalibrated)", fontsize=14)
    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_ylabel("Empirical Accuracy", fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=11)

    # After calibration
    ax = axes[1]
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")
    ax.bar(
        post_diag["bin_confidences"],
        post_diag["bin_accuracies"],
        width=0.05,
        alpha=0.6,
        color="forestgreen",
        edgecolor="black",
        label=f"Calibrated (ECE={post_ece*100:.2f}%)"
    )
    ax.set_title("After Temperature Scaling", fontsize=14)
    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_ylabel("Empirical Accuracy", fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[+] Saved reliability diagram to: {save_path}")


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load labels
    labels_path = config["artifacts"]["labels_path"]
    with open(labels_path, "r") as f:
        labels_meta = json.load(f)
    class_names = labels_meta["classes"]
    label_to_idx = labels_meta["label_to_idx"]

    # Load base model
    model_path = config["artifacts"]["model_path"]
    base_model = mobilenet_v3_large(weights=None)
    in_features = base_model.classifier[3].in_features
    base_model.classifier[3] = nn.Linear(in_features, len(class_names))

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    base_model.load_state_dict(state_dict)

    manifest_dir = config["data"].get("manifest_dir", "data/manifests")
    val_csv = os.path.join(manifest_dir, "val_manifest.csv")
    if not os.path.exists(val_csv):
        val_csv = os.path.join(config["data"]["processed_dir"], "val_manifest.csv")
    val_df = pd.read_csv(val_csv)
    eval_transform = get_eval_transforms(config["model"].get("img_size", 224))
    dataset = CropDiseaseDataset(val_df, label_to_idx, transform=eval_transform)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    print("[*] Calibrating temperature on validation logits...")
    calibrated_model = ModelWithTemperature(base_model)
    raw_logits, labels = calibrated_model.set_temperature(val_loader, device)

    learned_temp = float(calibrated_model.temperature.item())
    print(f"[+] Optimized Temperature Parameter (T): {learned_temp:.4f}")

    # Uncalibrated probabilities & ECE
    uncalibrated_probs = torch.softmax(raw_logits, dim=1).detach().cpu().numpy()
    pre_confs = np.max(uncalibrated_probs, axis=1)
    pre_preds = np.argmax(uncalibrated_probs, axis=1)
    y_true = labels.detach().cpu().numpy()

    pre_ece, pre_diag = compute_ece(pre_confs, pre_preds, y_true)

    # Calibrated probabilities & ECE
    calibrated_logits = raw_logits / learned_temp
    calibrated_probs = torch.softmax(calibrated_logits, dim=1).detach().cpu().numpy()
    post_confs = np.max(calibrated_probs, axis=1)
    post_preds = np.argmax(calibrated_probs, axis=1)

    post_ece, post_diag = compute_ece(post_confs, post_preds, y_true)

    print(f"  - Pre-calibration ECE:  {pre_ece * 100:.2f}%")
    print(f"  - Post-calibration ECE: {post_ece * 100:.2f}% (Reduction: {(pre_ece - post_ece)*100:.2f}%)")

    # Energy score calibration: E(x) = -T * logsumexp(z / T)
    # Lower energy indicates in-distribution; higher energy indicates out-of-distribution (OOD)
    with torch.no_grad():
        energies = -learned_temp * torch.logsumexp(raw_logits / learned_temp, dim=1)
        energy_arr = energies.cpu().numpy()
        # Set threshold at 95th percentile of in-distribution energy
        calibrated_energy_threshold = float(np.percentile(energy_arr, 95))
        print(f"[+] Calibrated OOD Energy Threshold (95th percentile): {calibrated_energy_threshold:.4f}")

    # Plot reliability diagrams
    diag_path = "ml_training/plots/reliability_diagram.png"
    plot_reliability_diagrams(pre_diag, post_diag, pre_ece, post_ece, diag_path)

    # Save calibration metadata
    calib_export_path = config["artifacts"]["calibration_path"]
    os.makedirs(os.path.dirname(calib_export_path), exist_ok=True)
    calibration_data = {
        "temperature": round(learned_temp, 4),
        "pre_ece": round(pre_ece, 4),
        "post_ece": round(post_ece, 4),
        "energy_threshold": round(calibrated_energy_threshold, 4),
        "confidence_thresholds": {
            "high": float(config["uncertainty"]["confidence_thresholds"]["high"]),
            "medium": float(config["uncertainty"]["confidence_thresholds"]["medium"])
        }
    }

    with open(calib_export_path, "w") as f:
        json.dump(calibration_data, f, indent=2)
    print(f"[+] Calibration parameters written to: {calib_export_path}")

if __name__ == "__main__":
    main()
