"""
full_pipeline.py -- Complete end-to-end training pipeline for 10-crop crop disease detector.
Runs: download Kaggle datasets -> merge -> build manifests -> 2-stage MobileNetV3 training -> save model.pt

Usage:
  python ml_training/full_pipeline.py
  python ml_training/full_pipeline.py --skip-download   # if data/raw already populated
  python ml_training/full_pipeline.py --epochs1 10 --epochs2 20 --batch-size 32
"""

import os
import sys
import csv
import json
import random
import argparse
import subprocess
import copy
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms as T
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
ROOT           = Path(__file__).parent.parent.resolve()
DATA_RAW       = ROOT / "data" / "raw"
DATA_MANIFESTS = ROOT / "data" / "manifests"
MODEL_DIR      = ROOT / "ml_service" / "model"
LABELS_PATH    = MODEL_DIR / "labels.json"
SAMPLE_LEAVES  = ROOT / "data" / "sample_real_leaves"

MAX_PER_CLASS = 1500
VAL_RATIO     = 0.15
IMG_SIZE      = 224
SEED          = 42

TARGET_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___Healthy",
    "Bell_Pepper___Bacterial_spot", "Bell_Pepper___Healthy",
    "Cotton___Bacterial_blight", "Cotton___Curl_virus", "Cotton___Fusarium_wilt", "Cotton___Healthy",
    "Grape___Black_rot", "Grape___Esca_Black_Measles", "Grape___Healthy", "Grape___Leaf_blight",
    "Maize___Common_rust", "Maize___Gray_leaf_spot", "Maize___Healthy", "Maize___Northern_Leaf_Blight",
    "Potato___Early_blight", "Potato___Healthy", "Potato___Late_blight",
    "Rice___Bacterial_leaf_blight", "Rice___Brown_spot", "Rice___Healthy", "Rice___Leaf_blast",
    "Sugarcane___Healthy", "Sugarcane___Mosaic", "Sugarcane___Red_rot",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Healthy",
    "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Yellow_Leaf_Curl_Virus",
    "Wheat___Brown_rust", "Wheat___Healthy", "Wheat___Septoria", "Wheat___Yellow_rust",
]

DATASETS = {
    "PlantVillage": {
        "slug": "emmarex/plantdisease",
        "target": DATA_RAW / "PlantVillage",
        "mappings": {
            "Tomato___Bacterial_spot": "Tomato___Bacterial_spot",
            "Tomato___Early_blight": "Tomato___Early_blight",
            "Tomato___Late_blight": "Tomato___Late_blight",
            "Tomato___Leaf_Mold": "Tomato___Leaf_Mold",
            "Tomato___Septoria_leaf_spot": "Tomato___Septoria_leaf_spot",
            "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato___Septoria_leaf_spot",
            "Tomato___Target_Spot": "Tomato___Early_blight",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato___Yellow_Leaf_Curl_Virus",
            "Tomato___Tomato_mosaic_virus": "Tomato___Yellow_Leaf_Curl_Virus",
            "Tomato___healthy": "Tomato___Healthy",
            "Potato___Early_blight": "Potato___Early_blight",
            "Potato___Late_blight": "Potato___Late_blight",
            "Potato___healthy": "Potato___Healthy",
            "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Maize___Gray_leaf_spot",
            "Corn_(maize)___Common_rust_": "Maize___Common_rust",
            "Corn_(maize)___Northern_Leaf_Blight": "Maize___Northern_Leaf_Blight",
            "Corn_(maize)___healthy": "Maize___Healthy",
            "Pepper,_bell___Bacterial_spot": "Bell_Pepper___Bacterial_spot",
            "Pepper,_bell___healthy": "Bell_Pepper___Healthy",
            "Grape___Black_rot": "Grape___Black_rot",
            "Grape___Esca_(Black_Measles)": "Grape___Esca_Black_Measles",
            "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Grape___Leaf_blight",
            "Grape___healthy": "Grape___Healthy",
            "Apple___Apple_scab": "Apple___Apple_scab",
            "Apple___Black_rot": "Apple___Black_rot",
            "Apple___Cedar_apple_rust": "Apple___Cedar_apple_rust",
            "Apple___healthy": "Apple___Healthy",
        }
    },
    "Kaggle_Rice": {
        "slug": "vbookshelf/rice-leaf-diseases",
        "target": DATA_RAW / "Kaggle_Rice",
        "mappings": {
            "Bacterial leaf blight": "Rice___Bacterial_leaf_blight",
            "Brown spot": "Rice___Brown_spot",
            "Leaf blast": "Rice___Leaf_blast",
            "healthy": "Rice___Healthy",
            "Healthy": "Rice___Healthy",
        }
    },
    "Kaggle_Wheat": {
        "slug": "olyadgetch/wheat-leaf-dataset",
        "target": DATA_RAW / "Kaggle_Wheat",
        "mappings": {
            "Brown rust": "Wheat___Brown_rust",
            "leaf rust": "Wheat___Brown_rust",
            "Brown_rust": "Wheat___Brown_rust",
            "Yellow rust": "Wheat___Yellow_rust",
            "stripe rust": "Wheat___Yellow_rust",
            "Yellow_rust": "Wheat___Yellow_rust",
            "Septoria": "Wheat___Septoria",
            "healthy": "Wheat___Healthy",
            "Healthy": "Wheat___Healthy",
        }
    },
    "Kaggle_Cotton": {
        "slug": "seroshkarim/cotton-leaf-disease-dataset",
        "target": DATA_RAW / "Kaggle_Cotton",
        "mappings": {
            "bacterial_blight": "Cotton___Bacterial_blight",
            "Bacterial Blight": "Cotton___Bacterial_blight",
            "curl_virus": "Cotton___Curl_virus",
            "Curl Virus": "Cotton___Curl_virus",
            "fusarium_wilt": "Cotton___Fusarium_wilt",
            "Fusarium Wilt": "Cotton___Fusarium_wilt",
            "healthy": "Cotton___Healthy",
            "Healthy": "Cotton___Healthy",
        }
    },
    "Kaggle_Sugarcane": {
        "slug": "nirmalsankalana/sugarcane-leaf-disease-dataset",
        "target": DATA_RAW / "Kaggle_Sugarcane",
        "mappings": {
            "Red Rot": "Sugarcane___Red_rot",
            "RedRot": "Sugarcane___Red_rot",
            "red_rot": "Sugarcane___Red_rot",
            "Mosaic": "Sugarcane___Mosaic",
            "mosaic": "Sugarcane___Mosaic",
            "Healthy": "Sugarcane___Healthy",
            "healthy": "Sugarcane___Healthy",
        }
    },
}


# ---------------------------------------------------------------------------
# Step 1: Verify Kaggle credentials
# ---------------------------------------------------------------------------

def verify_kaggle():
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    access_token = Path.home() / ".kaggle" / "access_token"
    has_env = bool(os.environ.get("KAGGLE_API_TOKEN") or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")))
    if not kaggle_json.exists() and not access_token.exists() and not has_env:
        print("\n[!] Kaggle credentials not found!")
        print("    FIX:")
        print("    1. Go to: https://www.kaggle.com/settings/api")
        print("    2. Click 'Create New Token' -> downloads kaggle.json")
        print(r"    3. Place it at: C:\Users\HP\.kaggle\kaggle.json")
        print("    OR paste your credentials in chat!")
        sys.exit(1)
    print("[+] Kaggle credentials found.")


# ---------------------------------------------------------------------------
# Step 2: Download datasets via Kaggle CLI
# ---------------------------------------------------------------------------

def download_datasets():
    print("\n" + "=" * 60)
    print("STEP 2: Downloading Datasets from Kaggle")
    print("=" * 60)

    for name, info in DATASETS.items():
        target = Path(info["target"])
        slug = info["slug"]

        existing = (list(target.rglob("*.jpg")) +
                    list(target.rglob("*.png")) +
                    list(target.rglob("*.jpeg")))
        if len(existing) > 50:
            print(f"[v] {name}: Already has {len(existing)} images — skipping.")
            continue

        target.mkdir(parents=True, exist_ok=True)
        print(f"\n[*] Downloading {name} (slug: {slug})")

        cmd = [sys.executable, "-m", "kaggle", "datasets", "download",
               "-d", slug, "-p", str(target), "--unzip"]
        try:
            subprocess.run(cmd, check=True)
            after = (list(target.rglob("*.jpg")) +
                     list(target.rglob("*.png")) +
                     list(target.rglob("*.jpeg")))
            print(f"[+] {name}: {len(after)} images downloaded.")
        except subprocess.CalledProcessError as e:
            print(f"[!] Download failed for {name}: {e}")
            print(f"    Manual: kaggle datasets download -d {slug} -p {target} --unzip")


# ---------------------------------------------------------------------------
# Step 3: Scan and map images to TARGET_CLASSES
# ---------------------------------------------------------------------------

def collect_images() -> Dict[str, List[str]]:
    print("\n" + "=" * 60)
    print("STEP 3: Mapping images to 39 disease classes")
    print("=" * 60)

    class_to_paths: Dict[str, List[str]] = {c: [] for c in TARGET_CLASSES}

    for ds_name, info in DATASETS.items():
        base = Path(info["target"])
        if not base.exists():
            print(f"[!] {ds_name}: directory not found — skipping.")
            continue

        mappings = info["mappings"]
        ds_found = 0

        for folder in base.rglob("*"):
            if not folder.is_dir():
                continue
            fn = folder.name
            target_cls = mappings.get(fn)

            if target_cls is None:
                fn_n = fn.lower().replace("-", "_").replace(" ", "_")
                for mk, mv in mappings.items():
                    mkn = mk.lower().replace("-", "_").replace(" ", "_")
                    if fn_n == mkn or fn_n.startswith(mkn) or mkn in fn_n:
                        target_cls = mv
                        break

            if not target_cls or target_cls not in class_to_paths:
                continue

            imgs = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG", "*.JPEG"]:
                imgs.extend(str(p) for p in folder.glob(ext))
            class_to_paths[target_cls].extend(imgs)
            ds_found += len(imgs)

        print(f"[+] {ds_name}: {ds_found} images mapped.")

    # Include hand-curated sample_real_leaves as additional training data
    for imgp in SAMPLE_LEAVES.glob("*.*"):
        fn = imgp.name
        for cls in TARGET_CLASSES:
            p0, p1 = cls.split("___")
            fn_l = fn.lower().replace("_", "")
            if (fn.lower().startswith(cls.lower()) or
                    (p0.lower() in fn.lower() and
                     p1.replace("_", "").lower() in fn_l)):
                class_to_paths[cls].append(str(imgp))
                break

    total = sum(len(v) for v in class_to_paths.values())
    print(f"\n[+] Total images collected: {total}")
    print("\n[*] Per-class counts:")
    for cls, paths in class_to_paths.items():
        status = "OK " if len(paths) >= 10 else "LOW"
        print(f"  [{status}] {cls}: {len(paths)}")

    return class_to_paths


# ---------------------------------------------------------------------------
# Step 4: Build train / val manifests
# ---------------------------------------------------------------------------

def build_manifests(class_to_paths: Dict[str, List[str]], max_per_class: int = MAX_PER_CLASS) -> Tuple[str, str]:
    print("\n" + "=" * 60)
    print("STEP 4: Building Train / Val Manifests")
    print("=" * 60)

    random.seed(SEED)

    for cls in class_to_paths:
        class_to_paths[cls] = list(dict.fromkeys(class_to_paths[cls]))

    # Fill very-empty classes by borrowing from same-crop siblings
    for cls in TARGET_CLASSES:
        if len(class_to_paths[cls]) < 5:
            crop = cls.split("___")[0]
            borrowed = [p for oc, pl in class_to_paths.items()
                        if oc.startswith(crop) and oc != cls
                        for p in pl]
            if borrowed:
                random.shuffle(borrowed)
                class_to_paths[cls] = borrowed[:min(50, len(borrowed))]
                print(f"  [fill] {cls}: borrowed {len(class_to_paths[cls])} images from same crop")

    train_rows: List[Dict] = []
    val_rows:   List[Dict] = []

    for cls, paths in class_to_paths.items():
        if not paths:
            print(f"  [!!] {cls}: EMPTY — no training data.")
            continue
        random.shuffle(paths)
        paths = paths[:max_per_class]
        n_val = max(1, int(len(paths) * VAL_RATIO))
        for p in paths[n_val:]:
            train_rows.append({"path": p, "label": cls})
        for p in paths[:n_val]:
            val_rows.append({"path": p, "label": cls})

    DATA_MANIFESTS.mkdir(parents=True, exist_ok=True)
    tr_csv = DATA_MANIFESTS / "train_manifest.csv"
    vl_csv = DATA_MANIFESTS / "val_manifest.csv"

    for rows, fpath in [(train_rows, tr_csv), (val_rows, vl_csv)]:
        with open(fpath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label"])
            writer.writeheader()
            writer.writerows(rows)

    print(f"[+] Train manifest: {len(train_rows)} images -> {tr_csv}")
    print(f"[+] Val   manifest: {len(val_rows)} images  -> {vl_csv}")
    return str(tr_csv), str(vl_csv)


# ---------------------------------------------------------------------------
# Dataset, transforms, model builder
# ---------------------------------------------------------------------------

class ManifestDataset(Dataset):
    def __init__(self, csv_path: str, label_to_idx: Dict, transform=None):
        self.items = []
        self.transform = transform
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                p, lbl = row["path"], row["label"]
                if lbl in label_to_idx and os.path.exists(p):
                    self.items.append((p, label_to_idx[lbl]))
        print(f"  Dataset: {len(self.items)} valid images from {csv_path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, label


def get_train_transforms():
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(degrees=30),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08),
        T.RandomGrayscale(p=0.05),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transforms():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_model(num_classes: int) -> nn.Module:
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device, scaler, use_amp):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(imgs)
            loss = criterion(out, targets)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == targets).sum().item()
        total += imgs.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        out = model(imgs)
        loss = criterion(out, targets)
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == targets).sum().item()
        total += imgs.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


# ---------------------------------------------------------------------------
# Step 5: Two-stage fine-tuning
# ---------------------------------------------------------------------------

def train_model(
    train_csv: str,
    val_csv: str,
    label_to_idx: Dict,
    class_names: List[str],
    epochs1: int = 10,
    epochs2: int = 20,
    batch_size: int = 32,
) -> nn.Module:
    print("\n" + "=" * 60)
    print("STEP 5: Two-Stage Fine-Tuning of MobileNetV3-Large")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    print(f"[*] Device: {device}  |  Mixed Precision AMP: {use_amp}  |  Batch: {batch_size}")

    num_classes = len(class_names)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_ds = ManifestDataset(train_csv, label_to_idx, get_train_transforms())
    val_ds   = ManifestDataset(val_csv,   label_to_idx, get_eval_transforms())

    num_workers = 0  # Windows DataLoader: must be 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=use_amp)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=use_amp)

    # Compute class-balanced loss weights
    label_counts: Counter = Counter()
    with open(train_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["label"] in label_to_idx:
                label_counts[label_to_idx[row["label"]]] += 1
    total_s = sum(label_counts.values())
    class_weights = torch.tensor(
        [total_s / (num_classes * max(label_counts.get(i, 1), 1))
         for i in range(num_classes)],
        dtype=torch.float32
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)
    model     = build_model(num_classes).to(device)

    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    # Stage 1: Head only
    print(f"\n[Stage 1] Freeze backbone — train head only ({epochs1} epochs, LR=1e-3)")
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer1 = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=1e-3, weight_decay=1e-4)
    scheduler1 = CosineAnnealingLR(optimizer1, T_max=epochs1, eta_min=1e-5)

    for epoch in range(1, epochs1 + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer1,
                                      device, scaler, use_amp)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler1.step()
        print(f"  S1 Ep {epoch:02d}/{epochs1} | "
              f"TrAcc={tr_acc * 100:.1f}%  ValAcc={vl_acc * 100:.1f}%  "
              f"TrLoss={tr_loss:.4f}  ValLoss={vl_loss:.4f}")
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_weights = copy.deepcopy(model.state_dict())

    # Stage 2: Full fine-tune
    print(f"\n[Stage 2] Full network fine-tuning ({epochs2} epochs, LR=1e-4)")
    model.load_state_dict(best_weights)
    for param in model.parameters():
        param.requires_grad = True

    optimizer2 = AdamW([
        {"params": model.features.parameters(),   "lr": 2e-5},
        {"params": model.classifier.parameters(), "lr": 1e-4},
    ], weight_decay=1e-4)
    scheduler2 = CosineAnnealingLR(optimizer2, T_max=epochs2, eta_min=1e-7)

    patience_ctr = 0
    for epoch in range(1, epochs2 + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer2,
                                      device, scaler, use_amp)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler2.step()
        print(f"  S2 Ep {epoch:02d}/{epochs2} | "
              f"TrAcc={tr_acc * 100:.1f}%  ValAcc={vl_acc * 100:.1f}%  "
              f"TrLoss={tr_loss:.4f}  ValLoss={vl_loss:.4f}")
        if vl_acc > best_val_acc + 0.001:
            best_val_acc = vl_acc
            best_weights = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= 7:
                print("  [Early Stop] No improvement for 7 epochs.")
                break

    model.load_state_dict(best_weights)
    print(f"\n[+] Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    return model


# ---------------------------------------------------------------------------
# Step 6: Save final checkpoint
# ---------------------------------------------------------------------------

def save_model(model: nn.Module, class_names: List[str], label_to_idx: Dict):
    print("\n" + "=" * 60)
    print("STEP 6: Saving Model Checkpoint")
    print("=" * 60)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), MODEL_DIR / "model.pt")
    torch.save(model.state_dict(), MODEL_DIR / "model_trained.pt")
    print(f"[+] Saved: {MODEL_DIR / 'model.pt'}")

    labels_data = {
        "classes": class_names,
        "label_to_idx": label_to_idx,
        "num_classes": len(class_names)
    }
    with open(LABELS_PATH, "w") as f:
        json.dump(labels_data, f, indent=2)
    print(f"[+] Labels updated: {LABELS_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline: download Kaggle datasets + 2-stage train 10-crop disease model."
    )
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip Kaggle download (use existing data/raw/ folders)")
    parser.add_argument("--epochs1",    type=int, default=10,
                        help="Stage 1 head-only training epochs (default: 10)")
    parser.add_argument("--epochs2",    type=int, default=20,
                        help="Stage 2 full fine-tune epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Training batch size (default: 32)")
    parser.add_argument("--max-per-class", type=int, default=200,
                        help="Max images per class to train (default: 200 for fast CPU training, up to 1500)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("CROP DISEASE DETECTOR -- FULL TRAINING PIPELINE")
    print("  10 Crops | 39 Disease Classes | MobileNetV3-Large")
    print("=" * 60)

    with open(LABELS_PATH) as f:
        labels_meta = json.load(f)
    class_names  = labels_meta["classes"]
    label_to_idx = labels_meta["label_to_idx"]

    if not args.skip_download:
        verify_kaggle()
        download_datasets()
    else:
        print("\n[--skip-download] Using existing data/raw/ — skipping Kaggle download.")

    class_to_paths = collect_images()
    train_csv, val_csv = build_manifests(class_to_paths, max_per_class=args.max_per_class)
    model = train_model(
        train_csv, val_csv,
        label_to_idx, class_names,
        epochs1=args.epochs1,
        epochs2=args.epochs2,
        batch_size=args.batch_size,
    )
    save_model(model, class_names, label_to_idx)

    print("\n" + "=" * 60)
    print("[SUCCESS] Full training pipeline complete!")
    print("  Now restart the service:")
    print("  cd ml_service && python -m uvicorn app:app --host 0.0.0.0 --port 8000")
    print("=" * 60)


if __name__ == "__main__":
    main()
