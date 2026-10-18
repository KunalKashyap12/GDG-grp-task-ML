"""
Script 02: Filter and Split
1. Verifies images, removes corrupt files.
2. Deduplicates near-identical images using perceptual hashing (pHash)
   to prevent data leakage between train, val, and test partitions.
3. Splits into train, val, test_lab, and test_field sets.
4. Analyzes class distribution, prints imbalance statistics, and exports labels.json.
"""

import os
import shutil
import json
import yaml
import argparse
import pandas as pd
import numpy as np
from PIL import Image
import imagehash
from typing import Dict, List, Set

def parse_args():
    parser = argparse.ArgumentParser(description="Filter corrupted/duplicate images and partition dataset splits.")
    parser.add_argument("--config", type=str, default="ml_training/config.yaml", help="Path to config.yaml")
    return parser.parse_args()


def compute_perceptual_hash(image_path: str):
    """Calculates perceptual hash of image using pHash."""
    try:
        with Image.open(image_path) as img:
            return imagehash.phash(img)
    except Exception:
        return None


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    manifest_path = config["data"]["manifest_path"]
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at {manifest_path}. Run 01_merge_datasets.py first.")

    df = pd.read_csv(manifest_path)
    print(f"[*] Loaded manifest with {len(df)} initial image records.")

    # 1. Filter corrupt images and deduplicate with perceptual hashing
    print("[*] Performing integrity check and perceptual hash deduplication...")
    valid_records = []
    seen_hashes: Dict[str, List] = {} # per-class hash dictionary
    phash_threshold = config["data"].get("phash_threshold", 4)
    corrupted_count = 0
    duplicate_count = 0

    for _, row in df.iterrows():
        path = row["path"]
        label = row["label"]

        # Check if file exists and can be opened
        if not os.path.exists(path):
            corrupted_count += 1
            continue

        h = compute_perceptual_hash(path)
        if h is None:
            corrupted_count += 1
            continue

        # Check against previously seen hashes in this class
        is_dup = False
        if label not in seen_hashes:
            seen_hashes[label] = []

        for existing_h in seen_hashes[label]:
            if (h - existing_h) <= phash_threshold: # Hamming distance check
                is_dup = True
                break

        if is_dup:
            duplicate_count += 1
            continue

        seen_hashes[label].append(h)
        valid_records.append(row.to_dict())

    clean_df = pd.DataFrame(valid_records)
    print(f"[+] Integrity check complete:")
    print(f"    - Valid images retained: {len(clean_df)}")
    print(f"    - Corrupt images removed: {corrupted_count}")
    print(f"    - Near-duplicates removed: {duplicate_count}")

    # 2. Extract and save standardized label mapping
    unique_classes = sorted(clean_df["label"].unique().tolist())
    label_to_idx = {cls_name: idx for idx, cls_name in enumerate(unique_classes)}

    labels_export_path = config["artifacts"]["labels_path"]
    os.makedirs(os.path.dirname(labels_export_path), exist_ok=True)
    with open(labels_export_path, "w") as f:
        json.dump({
            "classes": unique_classes,
            "label_to_idx": label_to_idx,
            "num_classes": len(unique_classes)
        }, f, indent=2)
    print(f"[+] Saved {len(unique_classes)} class labels to: {labels_export_path}")

    # 3. Create partitions: train, val, test_lab, test_field
    print("[*] Partitioning datasets into Train, Val, Test_Lab, and Test_Field...")
    val_ratio = config["data"].get("val_split", 0.15)
    test_ratio = config["data"].get("test_split", 0.15)

    train_rows = []
    val_rows = []
    test_lab_rows = []
    test_field_rows = []

    # Stratified partitioning per class and source type
    np.random.seed(config["training"].get("seed", 42))

    for label, group in clean_df.groupby("label"):
        # Partition lab images
        lab_group = group[group["type"] == "lab"].sample(frac=1.0, random_state=42)
        field_group = group[group["type"] == "field"].sample(frac=1.0, random_state=42)

        # For lab images: portion to test_lab, rest split into train/val
        n_lab = len(lab_group)
        n_lab_test = max(1, int(n_lab * test_ratio)) if n_lab > 3 else 0
        test_lab_rows.extend(lab_group.iloc[:n_lab_test].to_dict("records"))
        remaining_lab = lab_group.iloc[n_lab_test:]

        # For field images: portion to test_field, rest split into train/val
        n_field = len(field_group)
        n_field_test = max(1, int(n_field * test_ratio)) if n_field > 3 else 0
        test_field_rows.extend(field_group.iloc[:n_field_test].to_dict("records"))
        remaining_field = field_group.iloc[n_field_test:]

        # Combine remaining into pool for train and val
        combined_pool = pd.concat([remaining_lab, remaining_field]).sample(frac=1.0, random_state=42)
        n_pool = len(combined_pool)
        n_val = max(1, int(n_pool * val_ratio)) if n_pool > 3 else 0

        val_rows.extend(combined_pool.iloc[:n_val].to_dict("records"))
        train_rows.extend(combined_pool.iloc[n_val:].to_dict("records"))

    df_train = pd.DataFrame(train_rows)
    df_val = pd.DataFrame(val_rows)
    df_test_lab = pd.DataFrame(test_lab_rows)
    df_test_field = pd.DataFrame(test_field_rows)

    # Save partition manifests
    manifest_dir = config["data"].get("manifest_dir", "data/manifests")
    os.makedirs(manifest_dir, exist_ok=True)
    df_train.to_csv(os.path.join(manifest_dir, "train_manifest.csv"), index=False)
    df_val.to_csv(os.path.join(manifest_dir, "val_manifest.csv"), index=False)
    df_test_lab.to_csv(os.path.join(manifest_dir, "test_lab_manifest.csv"), index=False)
    df_test_field.to_csv(os.path.join(manifest_dir, "test_field_manifest.csv"), index=False)

    # 4. Copy / Link files into folder hierarchy for standard PyTorch ImageFolder compatibility
    splits = {
        "train": (df_train, config["data"]["train_dir"]),
        "val": (df_val, config["data"]["val_dir"]),
        "test_lab": (df_test_lab, config["data"]["test_lab_dir"]),
        "test_field": (df_test_field, config["data"]["test_field_dir"])
    }

    print("\n[*] Populating dataset directory structure...")
    for split_name, (split_df, target_dir) in splits.items():
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.makedirs(target_dir, exist_ok=True)

        for _, row in split_df.iterrows():
            cls_dir = os.path.join(target_dir, row["label"])
            os.makedirs(cls_dir, exist_ok=True)
            fname = os.path.basename(row["path"])
            dst = os.path.join(cls_dir, f"{row['source']}_{fname}")
            try:
                shutil.copy2(row["path"], dst)
            except Exception:
                pass
        print(f"    - {split_name:10s}: {len(split_df):5d} images -> {target_dir}")

    # 5. Class imbalance analysis
    print("\n" + "="*50)
    print("CLASS DISTRIBUTION SUMMARY (TRAIN SET)")
    print("="*50)
    train_dist = df_train["label"].value_counts()
    for cls_name, count in train_dist.items():
        print(f"  {cls_name:<38}: {count:>4} images")

    min_count = train_dist.min()
    max_count = train_dist.max()
    imbalance_ratio = max_count / max(min_count, 1)
    print(f"\n[!] Imbalance Ratio: {imbalance_ratio:.2f}:1 (Max: {max_count}, Min: {min_count})")
    print(f"[!] Class imbalance will be balanced via cost-sensitive smoothed weights in 03_train.py")
    print("="*50)

if __name__ == "__main__":
    main()
