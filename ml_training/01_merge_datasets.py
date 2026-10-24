"""
Script 01: Merge Datasets
Reads raw datasets, maps folder names to standardized Crop___Disease labels
via mapping.yaml, enforces class capping (<=1500 images/class),
and writes data/manifests/manifest.csv.
"""

import os
import re
import glob
import yaml
import argparse
import pandas as pd
from PIL import Image, ImageDraw
import numpy as np
from typing import Dict, List, Set, Tuple


def parse_args():
    parser = argparse.ArgumentParser(description="Merge diverse crop disease datasets into a unified manifest.")
    parser.add_argument("--config", type=str, default="ml_training/config.yaml", help="Path to config.yaml")
    parser.add_argument("--mapping", type=str, default="ml_training/mapping.yaml", help="Path to mapping.yaml")
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate synthetic samples for missing classes to ensure complete runnable pipeline.")
    return parser.parse_args()


def normalize_name(s: str) -> str:
    """Normalize folder names by stripping non-alphanumeric chars and converting to lowercase."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def create_synthetic_image(output_path: str, label: str, source: str, idx: int):
    """Generates synthetic leaf image with realistic variation for instant local testing."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    is_healthy = "Healthy" in label
    base_color = (34, 139, 34) if is_healthy else (100, 110, 30)

    bg_color = (230, 230, 230) if source == "PlantVillage" else (80, 60, 40)
    img = Image.new("RGB", (256, 256), color=bg_color)
    draw = ImageDraw.Draw(img)

    draw.ellipse([40, 40, 216, 216], fill=base_color, outline=(20, 80, 20))
    if not is_healthy:
        for _ in range(8):
            rx = int(np.random.randint(60, 190))
            ry = int(np.random.randint(60, 190))
            draw.ellipse([rx, ry, rx + 14, ry + 14], fill=(139, 69, 19))

    img.save(output_path)


def main():
    args = parse_args()

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

    # Build normalized folder -> (target_class, source_name, source_type) lookup
    folder_lookup: Dict[Tuple[str, str], Tuple[str, str, str]] = {}
    sources_config = mapping.get("sources", {})
    for source_name, source_info in sources_config.items():
        source_type = source_info.get("type", "lab")
        for f_name, t_cls in source_info.get("folder_mappings", {}).items():
            if t_cls in target_classes:
                folder_lookup[(source_name, normalize_name(f_name))] = (t_cls, source_name, source_type)
        for tc in target_classes:
            folder_lookup[(source_name, normalize_name(tc))] = (tc, source_name, source_type)

    records: List[Dict[str, str]] = []
    seen_filenames: Set[Tuple[str, str]] = set()
    class_counts = {c: 0 for c in target_classes}

    for source_name, source_info in sources_config.items():
        source_type = source_info.get("type", "lab")
        source_dir = os.path.join(raw_dir, source_name)
        if not os.path.exists(source_dir):
            print(f"[!] Source folder not found: {source_dir} (Skipping)")
            continue

        print(f"[*] Scanning source: {source_name} ({source_type}) recursively...")
        source_added = 0
        for r, dirs, files in os.walk(source_dir):
            bname = normalize_name(os.path.basename(r))
            key = (source_name, bname)
            if key in folder_lookup:
                t_cls, src, stype = folder_lookup[key]
                for f in files:
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        dedup_key = (t_cls, f.lower())
                        if dedup_key in seen_filenames:
                            continue
                        seen_filenames.add(dedup_key)

                        if class_counts[t_cls] < max_per_class:
                            raw_file = os.path.join(r, f)
                            img_path = os.path.normpath(os.path.relpath(raw_file, start=os.getcwd()))
                            records.append({
                                "path": img_path,
                                "label": t_cls,
                                "source": src,
                                "type": stype
                            })
                            class_counts[t_cls] += 1
                            source_added += 1
        print(f"    -> Added {source_added} valid unique images from {source_name}")

    # Generate synthetic samples for missing classes if requested
    missing_classes = [c for c in target_classes if class_counts[c] == 0]
    if missing_classes and args.generate_synthetic:
        print(f"[*] Generating synthetic samples for {len(missing_classes)} unrepresented classes...")
        synthetic_dir = os.path.join(config["data"]["processed_dir"], "synthetic_raw")
        for cls in missing_classes:
            crop = cls.split("___")[0]
            s_name = "PlantVillage" if crop in ["Tomato", "Potato", "Maize", "Bell_Pepper", "Grape", "Apple"] else f"Kaggle_{crop}"
            s_type = "lab" if s_name == "PlantVillage" else "field"
            for i in range(10):
                out_file = os.path.join(synthetic_dir, s_name, cls, f"sample_{i:03d}.jpg")
                create_synthetic_image(out_file, cls, s_name, i)
                records.append({
                    "path": os.path.normpath(os.path.relpath(out_file, start=os.getcwd())),
                    "label": cls,
                    "source": s_name,
                    "type": s_type
                })
                class_counts[cls] += 1

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    df.to_csv(manifest_path, index=False)

    print(f"\n[+] Manifest written to: {manifest_path}")
    print(f"[+] Total images merged: {len(df)}")
    if len(df) > 0:
        print(f"[+] Distribution across sources:\n{df['source'].value_counts().to_string()}")
        print(f"[+] Active classes represented: {df['label'].nunique()} / {len(target_classes)}")


if __name__ == "__main__":
    main()
