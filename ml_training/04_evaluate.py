"""
Script 04: Evaluation & Diagnostic Suite
1. Evaluates model across Val, Test_Lab (PlantVillage), and Test_Field (PlantDoc/Kaggle).
2. Leave-One-Source-Out (LOSO) cross-domain evaluation mode.
3. Computes Accuracy, Precision, Recall, and F1 per class and per crop.
4. Diagnoses shortcut learning by computing accuracy per source dataset.
5. Generates per-source and per-class CSV report, flagging shortcut-learning risks (>90% single source).
6. Lists the 5 weakest classes with their primary confusion partners.
7. Saves confusion matrix heatmaps and a gallery of misclassified images.
"""

import os
import json
import yaml
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
import sys
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from utils.augmentations import get_eval_transforms
from utils.dataset import CropDiseaseDataset, create_dataloader
from utils.metrics import compute_classification_metrics, compute_source_breakdown, find_weakest_classes


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate crop disease model on lab and field benchmarks.")
    parser.add_argument("--config", type=str, default="ml_training/config.yaml", help="Path to config.yaml")
    parser.add_argument("--leave-out-source", type=str, default=None, help="Evaluate on a specific held-out source (e.g. PlantDoc, Kaggle_Rice) for LOSO evaluation")
    parser.add_argument("--loso", action="store_true", help="Run comprehensive Leave-One-Source-Out evaluation across every dataset source")
    parser.add_argument("--report-csv", type=str, default="ml_training/source_class_report.csv", help="Path to output per-source and per-class shortcut-learning audit CSV")
    return parser.parse_args()


def load_model_for_eval(checkpoint_path: str, num_classes: int, device: torch.device) -> nn.Module:
    """Loads trained checkpoint into MobileNetV3-Large."""
    model = mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def run_evaluation_on_split(
    model: nn.Module,
    df: pd.DataFrame,
    label_to_idx: Dict[str, int],
    class_names: List[str],
    img_size: int,
    device: torch.device
) -> Tuple[Dict[str, Any], List[int], List[int], List[float], List[str], List[str]]:
    """Runs inference across a split manifest and aggregates detailed diagnostics."""
    if len(df) == 0:
        return {}, [], [], [], [], []

    eval_transform = get_eval_transforms(img_size)
    dataset = CropDiseaseDataset(df, label_to_idx, transform=eval_transform)
    # Use num_workers=0 on Windows; pin_memory speeds up GPU eval significantly.
    _num_workers = 0 if os.name == "nt" else 4
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=_num_workers,
        pin_memory=torch.cuda.is_available()
    )

    all_targets = []
    all_preds = []
    all_confs = []
    all_sources = []
    all_paths = []

    for images, targets, sources, paths in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        confs, preds = torch.max(probs, dim=1)

        all_targets.extend(targets.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_confs.extend(confs.cpu().numpy().tolist())
        all_sources.extend(list(sources))
        all_paths.extend(list(paths))

    metrics = compute_classification_metrics(all_targets, all_preds, class_names)
    metrics["source_breakdown"] = compute_source_breakdown(all_targets, all_preds, all_sources)
    metrics["weakest_classes"] = find_weakest_classes(all_targets, all_preds, class_names, top_k=5)

    return metrics, all_targets, all_preds, all_confs, all_sources, all_paths


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], class_names: List[str], save_path: str, title: str):
    """Generates and saves a confusion matrix visualization."""
    from sklearn.metrics import confusion_matrix
    if len(y_true) == 0:
        return

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    plt.figure(figsize=(18, 14))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=16)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def create_misclassified_gallery(
    targets: List[int],
    preds: List[int],
    confs: List[float],
    paths: List[str],
    class_names: List[str],
    save_path: str,
    max_images: int = 12
):
    """Creates a visual collage of misclassified examples with true/predicted labels."""
    misclassified = [
        (paths[i], targets[i], preds[i], confs[i])
        for i in range(len(targets))
        if targets[i] != preds[i]
    ]

    if not misclassified:
        print("[*] No misclassified samples to plot in gallery!")
        return

    selected = misclassified[:max_images]
    rows = int(np.ceil(len(selected) / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for i, (path, true_idx, pred_idx, conf) in enumerate(selected):
        ax = axes[i]
        try:
            img = Image.open(path).convert("RGB")
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "Image Load Error", ha="center")
        true_name = class_names[true_idx]
        pred_name = class_names[pred_idx]
        ax.set_title(f"True: {true_name}\nPred: {pred_name}\nConf: {conf:.2f}", fontsize=8, color="red")
        ax.axis("off")

    for j in range(len(selected), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[+] Saved misclassified gallery to: {save_path}")


def generate_source_class_report(
    manifest_df: pd.DataFrame,
    class_names: List[str],
    eval_metrics: Optional[Dict[str, Any]] = None,
    output_csv_path: str = "ml_training/source_class_report.csv"
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Generates a per-source and per-class CSV report.
    Flags classes where a single source makes up >= 90% of images (shortcut-learning risks).
    """
    if "source" not in manifest_df.columns:
        manifest_df["source"] = "unknown"

    all_sources = sorted(list(manifest_df["source"].unique()))
    rows = []
    flagged = []

    per_class_metrics = eval_metrics.get("per_class", {}) if eval_metrics else {}

    for cls in class_names:
        sub = manifest_df[manifest_df["label"] == cls]
        total_samples = len(sub)
        crop = cls.split("___")[0]
        disease = cls.split("___")[1] if "___" in cls else "Unknown"

        row = {
            "class_name": cls,
            "crop": crop,
            "disease": disease,
            "total_images": total_samples
        }

        source_counts = sub["source"].value_counts()
        dom_source = "None"
        dom_count = 0
        dom_pct = 0.0

        if total_samples > 0 and len(source_counts) > 0:
            dom_source = str(source_counts.index[0])
            dom_count = int(source_counts.iloc[0])
            dom_pct = float(dom_count) / float(total_samples)

        row["dominant_source"] = dom_source
        row["dominant_source_count"] = dom_count
        row["dominant_source_pct"] = round(dom_pct * 100.0, 2)

        # Flag shortcut-learning risk when a single source contributes >= 90%
        is_shortcut_risk = bool(dom_pct >= 0.90 and total_samples > 0)
        row["shortcut_risk"] = is_shortcut_risk
        row["risk_status"] = f"FLAGGED ({dom_pct*100:.1f}% from {dom_source})" if is_shortcut_risk else "BALANCED"

        for s in all_sources:
            cnt = int(source_counts.get(s, 0))
            pct = round((cnt / total_samples * 100.0), 2) if total_samples > 0 else 0.0
            row[f"count_{s}"] = cnt
            row[f"pct_{s}"] = pct

        cls_eval = per_class_metrics.get(cls, {})
        row["eval_precision"] = round(cls_eval.get("precision", 0.0), 4)
        row["eval_recall"] = round(cls_eval.get("recall", 0.0), 4)
        row["eval_f1"] = round(cls_eval.get("f1", 0.0), 4)
        row["eval_support"] = cls_eval.get("support", 0)

        rows.append(row)
        if is_shortcut_risk:
            flagged.append({
                "class": cls,
                "crop": crop,
                "dominant_source": dom_source,
                "dominant_pct": round(dom_pct * 100.0, 1),
                "total_images": total_samples
            })

    report_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    report_df.to_csv(output_csv_path, index=False)
    return report_df, flagged


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

    model_path = config["artifacts"]["model_path"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Run 03_train.py first.")

    print(f"[*] Loading model for evaluation from {model_path}...")
    model = load_model_for_eval(model_path, len(class_names), device)

    manifest_dir = config["data"].get("manifest_dir", "data/manifests")

    def _find_csv(name):
        p = os.path.join(manifest_dir, name)
        return p if os.path.exists(p) else os.path.join(config["data"]["processed_dir"], name)

    splits = {
        "validation": pd.read_csv(_find_csv("val_manifest.csv")),
        "test_lab": pd.read_csv(_find_csv("test_lab_manifest.csv")),
        "test_field": pd.read_csv(_find_csv("test_field_manifest.csv"))
    }

    # Load full manifest for shortcut learning analysis and LOSO mode
    main_manifest_path = config["data"].get("manifest_path", os.path.join(manifest_dir, "manifest.csv"))
    if os.path.exists(main_manifest_path):
        full_manifest_df = pd.read_csv(main_manifest_path)
    else:
        full_manifest_df = pd.concat(list(splits.values()), ignore_index=True)

    full_report: Dict[str, Any] = {}
    plot_dir = "ml_training/plots"
    img_size = config["model"].get("img_size", 224)

    # 1. Standard Splits Evaluation
    primary_eval_metrics = None
    for split_name, df_split in splits.items():
        print(f"\n{'='*65}\nEvaluating Split: {split_name.upper()} ({len(df_split)} samples)\n{'='*65}")
        metrics, targets, preds, confs, sources, paths = run_evaluation_on_split(
            model, df_split, label_to_idx, class_names, img_size, device
        )
        full_report[split_name] = metrics

        if not metrics:
            print(f"[!] Split {split_name} is empty. Skipping.")
            continue

        if split_name == "validation":
            primary_eval_metrics = metrics

        print(f"  - Overall Accuracy: {metrics['accuracy'] * 100:.2f}%")
        print(f"  - Macro F1-Score:   {metrics['macro_f1'] * 100:.2f}%")
        print(f"  - Weighted F1-Score:{metrics['weighted_f1'] * 100:.2f}%")

        print("\n  [Source Breakdown (Shortcut Learning Detection)]:")
        for src, stat in metrics["source_breakdown"].items():
            print(f"    * {src:<18}: {stat['accuracy'] * 100:>6.2f}% (N={stat['samples']})")

        print("\n  [Top 5 Weakest Classes & Confusion Partners]:")
        for weak in metrics["weakest_classes"]:
            partners_str = ", ".join([f"{p['confused_with']} ({p['count']})" for p in weak["confusion_partners"]]) or "None"
            print(f"    * {weak['class']:<35} (F1: {weak['f1_score']:.3f}, N={weak['support']}) -> Confused with: {partners_str}")

        cm_path = os.path.join(plot_dir, f"confusion_matrix_{split_name}.png")
        plot_confusion_matrix(targets, preds, class_names, cm_path, f"Confusion Matrix - {split_name.capitalize()}")

        if split_name in ["test_field", "validation"]:
            gallery_path = os.path.join(plot_dir, f"misclassified_gallery_{split_name}.png")
            create_misclassified_gallery(targets, preds, confs, paths, class_names, gallery_path)

    # 2. Leave-One-Source-Out (LOSO) Evaluation Mode
    loso_sources_to_run = []
    if args.leave_out_source:
        loso_sources_to_run = [args.leave_out_source]
    elif args.loso:
        if "source" in full_manifest_df.columns:
            loso_sources_to_run = sorted(list(full_manifest_df["source"].dropna().unique()))

    if loso_sources_to_run:
        print(f"\n{'='*70}\nLEAVE-ONE-SOURCE-OUT (LOSO) CROSS-DOMAIN BENCHMARK\n{'='*70}")
        loso_results = {}
        for src_name in loso_sources_to_run:
            sub_src = full_manifest_df[full_manifest_df["source"].astype(str).str.lower() == src_name.lower()]
            if len(sub_src) == 0:
                print(f"[!] Warning: No images found for source '{src_name}' in manifest. Skipping.")
                continue

            metrics_src, _, _, _, _, _ = run_evaluation_on_split(
                model, sub_src, label_to_idx, class_names, img_size, device
            )
            loso_results[src_name] = {
                "samples": len(sub_src),
                "classes": int(sub_src["label"].nunique()),
                "accuracy": round(metrics_src.get("accuracy", 0.0), 4),
                "macro_f1": round(metrics_src.get("macro_f1", 0.0), 4),
                "weighted_f1": round(metrics_src.get("weighted_f1", 0.0), 4)
            }
            print(f"  * Held-Out Source: {src_name:<18} | Samples: {len(sub_src):>4} | Accuracy: {metrics_src['accuracy']*100:>6.2f}% | Macro-F1: {metrics_src['macro_f1']:>6.4f}")

        full_report["leave_one_source_out"] = loso_results

    # 3. Per-Source and Per-Class Shortcut-Learning Audit
    report_df, flagged_classes = generate_source_class_report(
        full_manifest_df,
        class_names,
        eval_metrics=primary_eval_metrics,
        output_csv_path=args.report_csv
    )
    full_report["shortcut_risks"] = flagged_classes

    print(f"\n{'='*70}")
    print(f"SHORTCUT LEARNING RISK AUDIT (Single Source > 90%)")
    print(f"{'='*70}")
    print(f"  - Total Classes Analyzed: {len(class_names)}")
    print(f"  - Flagged Shortcut Risks: {len(flagged_classes)} classes where a single source makes up >= 90% of data")
    print(f"  - Full Audit CSV Exported: {args.report_csv}")
    if flagged_classes:
        print("\n  [Flagged Shortcut Risk Classes (Top 10)]:")
        for fc in flagged_classes[:10]:
            print(f"    * {fc['class']:<35} : {fc['dominant_pct']}% from {fc['dominant_source']} (N={fc['total_images']})")
    print(f"{'='*70}")

    # Export report JSON
    report_path = config["artifacts"].get("metrics_report", "ml_training/evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"\n[+] Full evaluation report exported to: {report_path}")


if __name__ == "__main__":
    main()
