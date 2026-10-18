"""
Dataset Downloader Helper Script
Provides automated download commands and links for the 6 primary source datasets:
1. PlantVillage (Lab: Apple, Bell Pepper, Grape, Maize, Potato, Tomato)
2. PlantDoc (Field: Real-world benchmark)
3. Kaggle Rice Leaf Disease (Field)
4. Kaggle Wheat Disease (Field)
5. Kaggle Cotton Disease (Field)
6. Kaggle Sugarcane Disease (Field)
"""

import os
import argparse
import subprocess

DATASET_SOURCES = {
    "PlantVillage": {
        "description": "PlantVillage laboratory dataset (Apple, Pepper, Grape, Maize, Potato, Tomato)",
        "type": "lab",
        "kaggle_slug": "emmarex/plantdisease",
        "alt_url": "https://github.com/spMohanty/PlantVillage-Dataset/archive/master.zip",
        "target_dir": "data/raw/PlantVillage"
    },
    "PlantDoc": {
        "description": "PlantDoc in-field real-world leaf dataset",
        "type": "field",
        "kaggle_slug": "daeyun/plantdoc-dataset",
        "alt_url": "https://github.com/pratikkayal/PlantDoc-Dataset/archive/master.zip",
        "target_dir": "data/raw/PlantDoc"
    },
    "Kaggle_Rice": {
        "description": "Rice Leaf Disease Dataset (Bacterial blight, Brown spot, Leaf blast)",
        "type": "field",
        "kaggle_slug": "vbookshelf/rice-leaf-diseases",
        "target_dir": "data/raw/Kaggle_Rice"
    },
    "Kaggle_Wheat": {
        "description": "Wheat Rust and Disease Dataset",
        "type": "field",
        "kaggle_slug": "olyadivotkey/wheat-leaf-dataset",
        "target_dir": "data/raw/Kaggle_Wheat"
    },
    "Kaggle_Cotton": {
        "description": "Cotton Disease Dataset (Bacterial blight, Curl virus, Fusarium wilt)",
        "type": "field",
        "kaggle_slug": "dhanushnarayananr/cotton-plant-disease",
        "target_dir": "data/raw/Kaggle_Cotton"
    },
    "Kaggle_Sugarcane": {
        "description": "Sugarcane Disease Dataset (Red rot, Mosaic, Healthy)",
        "type": "field",
        "kaggle_slug": "nirmalsankalana/sugarcane-leaf-disease-dataset",
        "target_dir": "data/raw/Kaggle_Sugarcane"
    }
}

def print_dataset_instructions():
    print("=" * 80)
    print("DATASET ACQUISITION GUIDE FOR LOCAL CROP DISEASE DETECTOR")
    print("=" * 80)
    print("\nThe full datasets total ~4-8 GB. You can acquire them via Kaggle CLI or direct URL:\n")

    for name, info in DATASET_SOURCES.items():
        print(f"[{name}] ({info['description']})")
        print(f"  Destination directory: {info['target_dir']}")
        if "kaggle_slug" in info:
            print(f"  Kaggle CLI command:    kaggle datasets download -d {info['kaggle_slug']} -p {info['target_dir']} --unzip")
        if "alt_url" in info:
            print(f"  Direct Download Link:  {info['alt_url']}")
        print()

    print("-" * 80)
    print("QUICK SETUP ON GOOGLE COLAB:")
    print("  !pip install kaggle")
    print("  # Upload your kaggle.json API token to Colab")
    print("  !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json")
    print("  !python ml_training/download_datasets.py --download-all")
    print("=" * 80)

def download_via_kaggle(name, info):
    if "kaggle_slug" not in info:
        print(f"[!] No Kaggle slug for {name}, skipping automated download.")
        return
    target = info["target_dir"]
    os.makedirs(target, exist_ok=True)
    slug = info["kaggle_slug"]
    print(f"[*] Downloading {name} from Kaggle ({slug}) into {target}...")
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", target, "--unzip"]
    try:
        subprocess.run(cmd, check=True)
        print(f"[+] Successfully downloaded and unzipped {name}")
    except Exception as e:
        print(f"[!] Failed to download {name} via Kaggle CLI: {e}")
        print(f"    Please download manually from: https://www.kaggle.com/datasets/{slug}")

def main():
    parser = argparse.ArgumentParser(description="Download datasets for training.")
    parser.add_argument("--download-all", action="store_true", help="Download all datasets via Kaggle API")
    parser.add_argument("--source", type=str, choices=list(DATASET_SOURCES.keys()), help="Download specific dataset")
    args = parser.parse_args()

    if args.download_all:
        for name, info in DATASET_SOURCES.items():
            download_via_kaggle(name, info)
    elif args.source:
        download_via_kaggle(args.source, DATASET_SOURCES[args.source])
    else:
        print_dataset_instructions()

if __name__ == "__main__":
    main()
