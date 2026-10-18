import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Computes overall accuracy, macro/weighted F1, and per-class / per-crop statistics.
    """
    overall_acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    # Per-class metrics
    p_per_class, r_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names)), zero_division=0
    )

    per_class = {}
    per_crop_data: Dict[str, Dict[str, List[float]]] = {}

    for idx, name in enumerate(class_names):
        crop = name.split("___")[0]
        per_class[name] = {
            "precision": float(p_per_class[idx]),
            "recall": float(r_per_class[idx]),
            "f1": float(f1_per_class[idx]),
            "support": int(support[idx])
        }

        if crop not in per_crop_data:
            per_crop_data[crop] = {"f1": [], "support": []}
        per_crop_data[crop]["f1"].append(float(f1_per_class[idx]))
        per_crop_data[crop]["support"].append(int(support[idx]))

    # Average metrics per crop
    per_crop = {}
    for crop, vals in per_crop_data.items():
        per_crop[crop] = {
            "mean_f1": float(np.mean(vals["f1"])),
            "total_support": int(np.sum(vals["support"]))
        }

    return {
        "accuracy": float(overall_acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "per_crop": per_crop
    }


def compute_source_breakdown(
    y_true: List[int],
    y_pred: List[int],
    sources: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Computes accuracy per source dataset (e.g. PlantVillage vs PlantDoc vs Kaggle)
    to diagnose shortcut learning or domain shifts.
    """
    df = pd.DataFrame({"true": y_true, "pred": y_pred, "source": sources})
    breakdown = {}
    for source_name, group in df.groupby("source"):
        acc = accuracy_score(group["true"], group["pred"])
        breakdown[str(source_name)] = {
            "accuracy": float(acc),
            "samples": len(group)
        }
    return breakdown


def find_weakest_classes(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Finds the top_k weakest classes by F1 score along with their primary confusion partners.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    _, _, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names)), zero_division=0
    )

    weak_indices = np.argsort(f1_per_class)[:top_k]
    results = []

    for idx in weak_indices:
        target_name = class_names[idx]
        target_f1 = float(f1_per_class[idx])
        target_support = int(support[idx])

        # Identify confusion partners (predictions when true class was idx)
        row = cm[idx].copy()
        row[idx] = 0  # ignore correct predictions
        top_confused_indices = np.argsort(row)[::-1][:3]

        partners = []
        for c_idx in top_confused_indices:
            if row[c_idx] > 0:
                partners.append({
                    "confused_with": class_names[c_idx],
                    "count": int(row[c_idx])
                })

        results.append({
            "class": target_name,
            "f1_score": round(target_f1, 4),
            "support": target_support,
            "confusion_partners": partners
        })

    return results


def compute_ece(
    confidences: np.ndarray,
    predictions: np.ndarray,
    ground_truths: np.ndarray,
    n_bins: int = 15
) -> Tuple[float, Dict[str, List[float]]]:
    """
    Computes Expected Calibration Error (ECE) and returns calibration bin stats
    for plotting reliability diagrams.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_props = []

    total_samples = len(confidences)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop = np.mean(in_bin)
        bin_props.append(float(prop))

        if np.sum(in_bin) > 0:
            bin_acc = np.mean(predictions[in_bin] == ground_truths[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += np.abs(bin_conf - bin_acc) * prop
            bin_accs.append(float(bin_acc))
            bin_confs.append(float(bin_conf))
        else:
            bin_accs.append(0.0)
            bin_confs.append(float((bin_lower + bin_upper) / 2))

    diagram_data = {
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_proportions": bin_props,
        "bin_boundaries": [float(b) for b in bin_boundaries]
    }

    return float(ece), diagram_data
