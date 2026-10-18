"""
Script 01: Merge Datasets
Reads raw datasets, maps folder names to standardized Crop___Disease labels
via mapping.yaml, enforces class capping (<=1500 images/class),
and writes data/manifest.csv.
"""

import os
import glob
import yaml
import argparse
import pandas as pd
from PIL import Image, ImageDraw
import numpy as np
from typing import Dict, List

def parse_args():
    parser = argparse.ArgumentParser(description="Merge diverse crop disease datasets into a unified manifest.")
    parser.add_argument("--config", type=str, default="ml_training/config.yaml", help="Path to config.yaml")
    parser.add_argument("--mapping", type=str, default="ml_training/mapping.yaml", help="Path to mapping.yaml")
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate synthetic samples for missing classes to ensure complete runnable pipeline.")
    return parser.parse_args()


def create_synthetic_image(output_path: str, label: str, source: str, idx: int):
    """Generates synthetic leaf image with realistic variation for instant local testing."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Determine base color based on crop and disease
    is_healthy = "Healthy" in label
    base_color = (34, 139, 34) if is_healthy else (100, 110, 30) # Forest green or diseased olive

    # Add background: lab (gray/black) or field (brown/greenish soil)
    bg_color = (230, 230, 230) if source == "PlantVillage" else (80, 60, 40)
    img = Image.new("RGB", (256, 256), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Draw leaf shape
    draw.ellipse([40, 40, 216, 216], fill=base_color, outline=(20, 80, 20))
    if not is_healthy:
        # Draw disease spots/lesions
        for _ in range(8):
            rx = int(np.random.randint(60, 190))
            ry = int(np.random.randint(60, 190))
            draw.ellipse([rx, ry, rx + 14, ry + 14], fill=(139, 69, 19)) # Rust/blight brown spots

    img.save(output_path)


def main():
    args = parse_args()

    # Load configurations
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    with open(args.mapping, "r") as f:
        mapping = yaml.safe_load(f)

    target_classes = set(mapping["target_classes"])
    raw_dir = config["data"]["raw_dir"]
    manifest_path = config["data"]["manifest_path"]
    max_per_class = config["data"].get("max_images_per_class", 1500)

    print(f"[*] Starting dataset merge process...")
    print(f"[*] Target Classes: {len(target_classes)}")
    print(f"[*] Max images per class cap: {max_per_class}")

    records = []
    class_counts = {c: 0 for c in target_classes}

    # Iterate over sources defined in mapping.yaml
    sources_config = mapping.get("sources", {})
    for source_name, source_info in sources_config.items():
        source_type = source_info.get("type", "lab")
        folder_mappings = source_info.get("folder_mappings", {})
        source_dir = os.path.join(raw_dir, source_name)

        if not os.path.exists(source_dir):
            print(f"[!] Source folder not found: {source_dir} (Skipping or awaiting synthetic generation)")
            continue

        print(f"[*] Scanning source: {source_name} ({source_type})")
        for folder_name, target_class in folder_mappings.items():
            if target_class not in target_classes:
                continue

            folder_path = os.path.join(source_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            # Find all image files
            images = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                images.extend(glob.glob(os.path.join(folder_path, ext)))

            for img_path in images:
                if class_counts[target_class] < max_per_class:
                    records.append({
                        "path": os.path.abspath(img_path),
                        "label": target_class,
                        "source": source_name,
                        "type": source_type
                    })
                    class_counts[target_class] += 1

    # Check if sample real leaves are available
    sample_dir = os.path.join(os.path.dirname(raw_dir), "sample_real_leaves")
    if os.path.isdir(sample_dir):
        sample_images = glob.glob(os.path.join(sample_dir, "*.jpg")) + glob.glob(os.path.join(sample_dir, "*.png"))
        if sample_images:
            print(f"[*] Found {len(sample_images)} real photographic leaf sample(s) in {sample_dir}")

    # If records are empty and synthetic is not explicitly requested, warn the user
    if len(records) == 0:
        if args.generate_synthetic:
            print("[!] WARNING: --generate-synthetic requested. Generating geometric mock images (green circles with spots).")
            print("[!] NOTE: Real CNN model training requires actual leaf photographs from PlantVillage, PlantDoc, Kaggle!")
            synthetic_dir = os.path.join(config["data"]["processed_dir"], "synthetic_raw")
            for cls in target_classes:
                crop = cls.split("___")[0]
                if crop in ["Tomato", "Potato", "Maize", "Bell_Pepper", "Grape", "Apple"]:
                    source_choices = [("PlantVillage", "lab"), ("PlantDoc", "field")]
                else:
                    source_choices = [(f"Kaggle_{crop}", "field")]

                needed = max(0, 10 - class_counts[cls])
                for i in range(needed):
                    s_name, s_type = source_choices[i % len(source_choices)]
                    out_file = os.path.join(synthetic_dir, s_name, cls, f"sample_{i:03d}.jpg")
                    create_synthetic_image(out_file, cls, s_name, i)
                    records.append({
                        "path": os.path.abspath(out_file),
                        "label": cls,
                        "source": s_name,
                        "type": s_type
                    })
                    class_counts[cls] += 1
        else:
            print("\n" + "=" * 70)
            print("[!] No raw dataset images found in data/raw/.")
            print("[!] Real leaf photographs are required to train the CNN model.")
            print("[!] Please download the real datasets with:")
            print("      python ml_training/download_datasets.py --download-all")
            print("[!] Or place downloaded folders in data/raw/ (PlantVillage, PlantDoc, Kaggle_Rice, etc.)")
            print("=" * 70 + "\n")

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    df.to_csv(manifest_path, index=False)

    print(f"\n[+] Manifest written to: {manifest_path}")
    print(f"[+] Total images merged: {len(df)}")
    print(f"[+] Distribution across sources:\n{df['source'].value_counts().to_string()}")
    print(f"[+] Number of active classes represented: {df['label'].nunique()} / {len(target_classes)}")

if __name__ == "__main__":
    main()
