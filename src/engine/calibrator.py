"""
Temperature Scaling & Uncertainty Calibration Engine.
Learns post-hoc scalar parameter T to minimize Expected Calibration Error (ECE).
"""

import json
from pathlib import Path
from typing import Dict, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.optim import LBFGS
from torch.utils.data import DataLoader
from src.utils.metrics import compute_expected_calibration_error


class TemperatureScaler(nn.Module):
    """
    A single-parameter layer that scales logits by 1/T:
    q_i = softmax(z_i / T)
    """

    def __init__(self, init_temp: float = 1.0) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # Temperature clamp prevents numerical collapse
        temperature = torch.clamp(self.temperature, min=0.01, max=10.0)
        return logits / temperature


class TemperatureScalingCalibrator:
    """
    Manages logit extraction and optimization of temperature parameter T.
    """

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model.to(device)
        self.device = device
        self.scaler = TemperatureScaler().to(device)

    def calibrate(self, val_loader: DataLoader, max_iter: int = 50, lr: float = 0.01) -> Dict[str, float]:
        """
        Collects validation logits and optimizes T using L-BFGS on NLL loss.
        """
        self.model.eval()
        logits_list = []
        labels_list = []

        with torch.inference_mode():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                logits = self.model(inputs)
                logits_list.append(logits)
                labels_list.append(targets.to(self.device))

        logits_all = torch.cat(logits_list, dim=0)
        labels_all = torch.cat(labels_list, dim=0)

        # 1. Pre-calibration ECE
        probs_pre = torch.softmax(logits_all, dim=-1).cpu().numpy()
        pre_ece = compute_expected_calibration_error(probs_pre, labels_all.cpu().numpy())

        # 2. Optimize Temperature via L-BFGS
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = LBFGS([self.scaler.temperature], lr=lr, max_iter=max_iter)

        def eval_step():
            optimizer.zero_grad()
            scaled_logits = self.scaler(logits_all)
            loss = nll_criterion(scaled_logits, labels_all)
            loss.backward()
            return loss

        optimizer.step(eval_step)

        # 3. Post-calibration ECE
        with torch.no_grad():
            scaled_logits = self.scaler(logits_all)
            probs_post = torch.softmax(scaled_logits, dim=-1).cpu().numpy()

        post_ece = compute_expected_calibration_error(probs_post, labels_all.cpu().numpy())
        final_temp = float(self.scaler.temperature.item())

        return {
            "temperature": round(final_temp, 4),
            "pre_ece": round(pre_ece, 4),
            "post_ece": round(post_ece, 4),
        }

    def save_calibration(self, results: Dict[str, float], output_path: Union[str, Path]) -> None:
        """Saves calibration parameters into structured JSON artifact."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "temperature": results["temperature"],
            "pre_ece": results["pre_ece"],
            "post_ece": results["post_ece"],
            "energy_threshold": -4.0,
            "max_prob_threshold": 0.10,
            "confidence_thresholds": {
                "high": 0.75,
                "medium": 0.50,
            },
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
