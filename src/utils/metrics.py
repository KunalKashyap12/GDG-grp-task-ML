"""
Evaluation and Uncertainty Calibration Metrics.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Computes top-level classification metrics: accuracy, macro-F1, weighted-F1.
    """
    if len(y_true) == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
        }
    zero_div: Any = 0
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=zero_div))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=zero_div))

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
    }


def compute_expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 15,
) -> float:
    """
    Computes Expected Calibration Error (ECE) across confidence bins:
    ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    if len(labels) == 0 or len(probabilities) == 0:
        return 0.0

    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece: float = 0.0
    n_samples = len(labels)

    for i in range(num_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(round(ece, 4))
