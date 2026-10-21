import os
import json
import pandas as pd
import numpy as np
from collections import Counter
from PIL import Image
import imagehash

print("="*60)
print("AUDIT CHECK 3: DATASET INTEGRITY & SPLITS")
print("="*60)

manifest_dir = "data/manifests"
manifest_files = {
    "train": os.path.join(manifest_dir, "train_manifest.csv"),
    "val": os.path.join(manifest_dir, "val_manifest.csv"),
    "test_lab": os.path.join(manifest_dir, "test_lab_manifest.csv"),
    "test_field": os.path.join(manifest_dir, "test_field_manifest.csv"),
    "full": os.path.join(manifest_dir, "manifest.csv")
}

dfs = {}
for name, p in manifest_files.items():
    if os.path.exists(p):
        df = pd.read_csv(p)
        dfs[name] = df
        print(f"\nManifest '{name}': {len(df)} rows, columns: {list(df.columns)}")
        if "label" in df.columns:
            n_classes = df["label"].nunique()
            print(f"  Unique classes in {name}: {n_classes}")
        if "source" in df.columns:
            print(f"  Sources in {name}: {dict(df['source'].value_counts())}")
        if "is_field" in df.columns:
            print(f"  Field vs Lab in {name}: {dict(df['is_field'].value_counts())}")

# Check disk existence of sample paths
if "train" in dfs:
    train_df = dfs["train"]
    missing = 0
    for p in train_df["path"]:
        if not os.path.exists(p):
            missing += 1
    print(f"\nTrain set disk check: {len(train_df) - missing}/{len(train_df)} exist on disk. ({missing} missing)")

if "val" in dfs:
    val_df = dfs["val"]
    missing = 0
    for p in val_df["path"]:
        if not os.path.exists(p):
            missing += 1
    print(f"Val set disk check: {len(val_df) - missing}/{len(val_df)} exist on disk. ({missing} missing)")

# Check raw directories
print("\nScanning data/raw directory:")
for root, dirs, files in os.walk("data/raw"):
    if files:
        exts = set(os.path.splitext(f)[1].lower() for f in files)
        img_count = sum(1 for f in files if os.path.splitext(f)[1].lower() in [".jpg", ".jpeg", ".png", ".webp"])
        rel_path = os.path.relpath(root, "data/raw")
        parts = rel_path.split(os.sep)
        if len(parts) <= 2:
            print(f"  {rel_path}: {img_count} images")

# Check sample_real_leaves
sample_leaves_dir = "data/sample_real_leaves"
sample_files = [f for f in os.listdir(sample_leaves_dir) if os.path.isfile(os.path.join(sample_leaves_dir, f))]
print(f"\ndata/sample_real_leaves contains {len(sample_files)} images.")

# Check for data leakage / duplicates across train, val, and test_lab
print("\nChecking for near-duplicates / hash collisions across train and val...")
if "train" in dfs and "val" in dfs:
    # Sample 200 train and all val images if available
    train_paths = [p for p in dfs["train"]["path"] if os.path.exists(p)]
    val_paths = [p for p in dfs["val"]["path"] if os.path.exists(p)]
    
    # Path collision check
    overlap = set(train_paths).intersection(set(val_paths))
    print(f"Direct file path overlap between train and val: {len(overlap)}")
    
    # Check duplicate filenames
    train_basenames = [os.path.basename(p) for p in train_paths]
    val_basenames = [os.path.basename(p) for p in val_paths]
    name_overlap = set(train_basenames).intersection(set(val_basenames))
    print(f"Filename overlap between train and val: {len(name_overlap)}")
    if name_overlap:
        print(f"  Sample overlapping filenames: {list(name_overlap)[:5]}")

    # Compute pHash for a subset to detect data leakage
    print("Computing perceptual hashes on val set vs train set (sample of 100 val images)...")
    val_hashes = {}
    for p in val_paths[:100]:
        try:
            with Image.open(p) as img:
                val_hashes[p] = imagehash.phash(img)
        except Exception:
            pass
            
    train_hashes = {}
    for p in train_paths[:500]:
        try:
            with Image.open(p) as img:
                train_hashes[p] = imagehash.phash(img)
        except Exception:
            pass
            
    near_dupes = 0
    for vp, vh in val_hashes.items():
        for tp, th in train_hashes.items():
            if vh - th <= 2:  # very close
                near_dupes += 1
                break
    print(f"Near-duplicate val images found in train sample (Hamming dist <= 2): {near_dupes}/{len(val_hashes)} ({near_dupes/len(val_hashes):.1%})")
