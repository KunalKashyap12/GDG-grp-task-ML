"""
Model Evaluation and Diagnostic Benchmark Engine.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from src.utils.metrics import compute_classification_metrics, compute_expected_calibration_error


class ModelEvaluator:
    """
    Computes rigorous classification reports, confusion arrays,
    and Expected Calibration Error across test sets.
    """

    def __init__(self, model: nn.Module, device: torch.device, classes: Optional[List[str]] = None) -> None:
        self.model = model.to(device)
        self.device = device
        self.classes = classes

    @torch.inference_mode()
    def evaluate(self, dataloader: DataLoader) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs comprehensive test pass.

        Returns:
            metrics: Dict containing accuracy, macro_f1, weighted_f1, ece
            all_preds: 1D array of predicted class indices
            all_targets: 1D array of true class indices
            all_probs: 2D array of predicted class probabilities [N, Num_Classes]
        """
        self.model.eval()
        preds_list: List[np.ndarray] = []
        targets_list: List[np.ndarray] = []
        probs_list: List[np.ndarray] = []

        for inputs, targets in dataloader:
            inputs = inputs.to(self.device, non_blocking=True)
            logits = self.model(inputs)
            probs = torch.softmax(logits, dim=-1)

            preds = torch.argmax(probs, dim=-1).cpu().numpy()
            preds_list.append(preds)
            targets_list.append(targets.numpy())
            probs_list.append(probs.cpu().numpy())

        if not preds_list:
            empty_metrics = {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0, "ece": 0.0}
            return empty_metrics, np.array([], dtype=int), np.array([], dtype=int), np.array([])

        all_preds = np.concatenate(preds_list, axis=0)
        all_targets = np.concatenate(targets_list, axis=0)
        all_probs = np.concatenate(probs_list, axis=0)

        metrics = compute_classification_metrics(all_targets, all_preds)
        metrics["ece"] = compute_expected_calibration_error(all_probs, all_targets)

        return metrics, all_preds, all_targets, all_probs

    def generate_report_string(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Generates detailed text summary via sklearn classification_report."""
        if len(y_true) == 0:
            return "No samples evaluated (dataset was empty)."
        zero_div: Any = 0
        labels_param = list(range(len(self.classes))) if self.classes else None
        report = classification_report(
            y_true,
            y_pred,
            labels=labels_param,
            target_names=self.classes,
            zero_division=zero_div,
            digits=4,
        )
        return str(report)
