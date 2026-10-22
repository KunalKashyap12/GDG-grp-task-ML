"""
Script 03: Two-Stage Model Training
Performs transfer learning with MobileNetV3-Large on ImageNet weights:
- Stage 1: Freeze backbone, train classification head (LR: 1e-3).
- Stage 2: Unfreeze entire backbone, fine-tune with lower LR (LR: 1e-4).
- Optimizations: AdamW, Cosine Annealing, Label Smoothing, Mixed Precision (AMP),
  Class Weighting, and Early Stopping.
"""

import os
import sys
import copy
import json
import yaml
import argparse
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from PIL import Image

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from utils.augmentations import get_train_transforms, get_eval_transforms
from utils.dataset import CropDiseaseDataset, compute_class_weights, create_dataloader
from utils.metrics import compute_classification_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Train MobileNetV3 Crop Disease Classifier.")
    parser.add_argument("--config", type=str, default="ml_training/config.yaml", help="Path to config.yaml")
    parser.add_argument("--epochs1", type=int, default=None, help="Override Stage 1 epochs")
    parser.add_argument("--epochs2", type=int, default=None, help="Override Stage 2 epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--leave-out-source", type=str, default=None, help="Hold out a dataset source completely from training/validation (LOSO mode)")
    return parser.parse_args()


def run_manifest_preflight_check(manifest_df: pd.DataFrame, split_name: str, max_reported: int = 10) -> List[Tuple[str, str]]:
    """
    Pre-flight integrity verification:
    Attempts to open every image path in the manifest once before training begins.
    Returns list of (path, error_reason) for any broken images.
    """
    print(f"[*] Running pre-flight integrity check on {split_name} manifest ({len(manifest_df)} paths)...")
    broken_entries: List[Tuple[str, str]] = []

    for idx, row in manifest_df.iterrows():
        img_path = row["path"]
        if not os.path.exists(img_path):
            broken_entries.append((img_path, "File does not exist on disk"))
            continue
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception as e:
            broken_entries.append((img_path, str(e)))

    if broken_entries:
        print(f"\n{'!'*70}\n[!] PRE-FLIGHT INTEGRITY CHECK FAILED FOR {split_name.upper()} MANIFEST!\n{'!'*70}")
        print(f"[!] Total broken image paths found: {len(broken_entries)} / {len(manifest_df)} ({len(broken_entries)/len(manifest_df):.2%})")
        print("[!] Sample broken paths:")
        for path_str, reason in broken_entries[:max_reported]:
            print(f"    - {path_str} -> {reason}")
        if len(broken_entries) > max_reported:
            print(f"    ... and {len(broken_entries) - max_reported} more unreadable paths.")
        print(f"{'!'*70}\n")
    else:
        print(f"[+] Pre-flight check PASSED for {split_name} ({len(manifest_df)}/{len(manifest_df)} image paths verified).")

    return broken_entries


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Builds MobileNetV3-Large model and replaces final classification layer."""
    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_large(weights=weights)

    # The classifier structure in MobileNetV3-Large is:
    # (0): Linear(in_features=960, out_features=1280)
    # (1): Hardswish()
    # (2): Dropout(p=0.2)
    # (3): Linear(in_features=1280, out_features=1000)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool
) -> Tuple[float, float]:
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets, _, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: List[str]
) -> Tuple[float, Dict]:
    """Evaluates model on validation loader."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets, _, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(targets.cpu().numpy().tolist())

    total = len(all_targets)
    val_loss = running_loss / max(total, 1)
    metrics = compute_classification_metrics(all_targets, all_preds, class_names)
    return val_loss, metrics


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set seeds
    seed = config["training"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # AMP only valid on CUDA — forcibly disabled on CPU to prevent GradScaler runtime errors
    use_amp = config["training"].get("use_amp", True) and (device.type == "cuda")
    print(f"[*] Training on device: {device} (Mixed Precision AMP: {use_amp})")

    # Load labels
    labels_path = config["artifacts"]["labels_path"]
    with open(labels_path, "r") as f:
        labels_meta = json.load(f)
    class_names = labels_meta["classes"]
    label_to_idx = labels_meta["label_to_idx"]
    num_classes = len(class_names)

    # Load dataset manifests
    manifest_dir = config["data"].get("manifest_dir", "data/manifests")
    train_csv = os.path.join(manifest_dir, "train_manifest.csv")
    if not os.path.exists(train_csv):
        train_csv = os.path.join(config["data"]["processed_dir"], "train_manifest.csv")
    val_csv = os.path.join(manifest_dir, "val_manifest.csv")
    if not os.path.exists(val_csv):
        val_csv = os.path.join(config["data"]["processed_dir"], "val_manifest.csv")
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    if args.leave_out_source:
        print(f"[*] Leave-One-Source-Out mode: Excluding source '{args.leave_out_source}' from training & validation...")
        if "source" in train_df.columns:
            n_tr = len(train_df)
            train_df = train_df[train_df["source"] != args.leave_out_source].reset_index(drop=True)
            print(f"    - Train set: {n_tr} -> {len(train_df)} images (removed {n_tr - len(train_df)} from {args.leave_out_source})")
        if "source" in val_df.columns:
            n_vl = len(val_df)
            val_df = val_df[val_df["source"] != args.leave_out_source].reset_index(drop=True)
            print(f"    - Val set: {n_vl} -> {len(val_df)} images (removed {n_vl - len(val_df)} from {args.leave_out_source})")

    print(f"[*] Active dataset size - Train: {len(train_df)}, Val: {len(val_df)}")

    # Pre-flight check: verify all image paths in train/val manifests before training
    train_broken = run_manifest_preflight_check(train_df, "train")
    val_broken = run_manifest_preflight_check(val_df, "val")
    if train_broken or val_broken:
        total_broken = len(train_broken) + len(val_broken)
        raise RuntimeError(
            f"Pre-flight integrity check aborted training! "
            f"Found {total_broken} broken image paths ({len(train_broken)} in train, {len(val_broken)} in val). "
            f"Please repair or regenerate dataset manifests before starting training."
        )

    # Transforms & Loaders
    img_size = config["model"].get("img_size", 224)
    batch_size = args.batch_size or config["training"].get("batch_size", 64)
    # num_workers > 0 requires spawn start method on Windows — keep 0 for safety, but
    # enable persistent_workers when workers > 0 to avoid per-epoch fork overhead.
    num_workers = 0 if os.name == "nt" else config["training"].get("num_workers", 2)
    use_persistent_workers = (num_workers > 0)
    max_failure_rate = float(config["data"].get("max_failure_rate", 0.005))
    backgrounds_dir = config["data"].get("backgrounds_dir", "data/backgrounds")

    train_transform = get_train_transforms(img_size, backgrounds_dir=backgrounds_dir)
    eval_transform = get_eval_transforms(img_size)

    train_loader = create_dataloader(
        train_df,
        label_to_idx,
        train_transform,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        max_failure_rate=max_failure_rate,
        persistent_workers=use_persistent_workers
    )
    val_loader = create_dataloader(
        val_df,
        label_to_idx,
        eval_transform,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        max_failure_rate=max_failure_rate,
        persistent_workers=use_persistent_workers
    )

    # Compute balanced class weights
    class_weights = compute_class_weights(train_df["label"].tolist(), label_to_idx).to(device)
    smoothing = config["training"].get("label_smoothing", 0.1)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=smoothing)

    # Build model
    model = build_model(num_classes=num_classes, pretrained=config["model"].get("pretrained", True)).to(device)
    # Use the new torch.amp.GradScaler API (torch.cuda.amp.GradScaler is deprecated in PyTorch 2.x)
    scaler = GradScaler(device="cuda", enabled=use_amp) if use_amp else GradScaler(device="cpu", enabled=False)

    best_model_weights = copy.deepcopy(model.state_dict())
    best_val_f1 = 0.0
    patience = config["training"]["early_stopping"].get("patience", 5)
    patience_counter = 0

    # -------------------------------------------------------------
    # STAGE 1: Train classification head only
    # -------------------------------------------------------------
    epochs_s1 = args.epochs1 or config["training"]["stage1"].get("epochs", 5)
    lr_s1 = float(config["training"]["stage1"].get("lr", 1e-3))
    print(f"\n{'='*60}\nSTAGE 1: Training Classification Head ({epochs_s1} epochs, LR={lr_s1})\n{'='*60}")

    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer_s1 = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_s1, weight_decay=1e-4)
    scheduler_s1 = CosineAnnealingLR(optimizer_s1, T_max=epochs_s1, eta_min=1e-6)

    patience_counter_s1 = 0
    for epoch in range(1, epochs_s1 + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer_s1, scaler, device, use_amp)
        val_loss, metrics = evaluate_epoch(model, val_loader, criterion, device, class_names)
        scheduler_s1.step()

        val_acc = metrics["accuracy"]
        val_f1 = metrics["macro_f1"]
        print(f"[Stage 1 | Epoch {epoch:02d}/{epochs_s1:02d}] Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_f1:.4f}")

        if val_f1 > best_val_f1 + 0.001:
            best_val_f1 = val_f1
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter_s1 = 0
        else:
            patience_counter_s1 += 1
            if patience_counter_s1 >= patience:
                print(f"[!] Stage 1 early stopping triggered after {patience} epochs without improvement.")
                break

    # -------------------------------------------------------------
    # STAGE 2: Fine-tune entire network with lower LR
    # -------------------------------------------------------------
    epochs_s2 = args.epochs2 or config["training"]["stage2"].get("epochs", 15)
    lr_s2 = float(config["training"]["stage2"].get("lr", 1e-4))
    print(f"\n{'='*60}\nSTAGE 2: Fine-tuning Full Network ({epochs_s2} epochs, LR={lr_s2})\n{'='*60}")

    model.load_state_dict(best_model_weights)
    for param in model.parameters():
        param.requires_grad = True

    # Differential learning rates: slightly lower for backbone features
    optimizer_s2 = AdamW([
        {"params": model.features.parameters(), "lr": lr_s2 * 0.2},
        {"params": model.classifier.parameters(), "lr": lr_s2}
    ], weight_decay=config["training"].get("weight_decay", 1e-4))
    scheduler_s2 = CosineAnnealingLR(optimizer_s2, T_max=epochs_s2, eta_min=1e-7)

    patience_counter = 0
    for epoch in range(1, epochs_s2 + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer_s2, scaler, device, use_amp)
        val_loss, metrics = evaluate_epoch(model, val_loader, criterion, device, class_names)
        scheduler_s2.step()

        val_acc = metrics["accuracy"]
        val_f1 = metrics["macro_f1"]
        print(f"[Stage 2 | Epoch {epoch:02d}/{epochs_s2:02d}] Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_f1:.4f}")

        if val_f1 > best_val_f1 + 0.001:
            best_val_f1 = val_f1
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[!] Early stopping triggered after {patience} epochs without improvement.")
                break

    # Save final best model
    output_model_path = config["artifacts"]["model_path"]
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    model.load_state_dict(best_model_weights)

    # Save checkpoint with metadata
    torch.save({
        "state_dict": model.state_dict(),
        "architecture": config["model"]["architecture"],
        "num_classes": num_classes,
        "classes": class_names,
        "best_val_f1": best_val_f1
    }, output_model_path)

    print(f"\n[+] Training complete! Best Validation Macro-F1: {best_val_f1:.4f}")
    print(f"[+] Saved model checkpoint to: {output_model_path}")

if __name__ == "__main__":
    main()
