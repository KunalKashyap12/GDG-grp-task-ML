"""
Google Colab Execution Script
Runs the end-to-end training pipeline on a free Colab T4 GPU:
1. Environment verification (CUDA, AMP support, GPU memory).
2. Dataset merge and mapping across 10 crops (39 classes).
3. Deduplication with perceptual hashing & four-way partition.
4. Two-stage MobileNetV3 transfer learning with mixed precision (AMP).
5. Diagnostic evaluation on Val, Test_Lab, and Test_Field benchmarks.
6. Temperature scaling confidence calibration and artifact packaging.
"""

import os
import sys
import subprocess
import torch

def run_cmd(cmd_list, description):
    print(f"\n{'='*70}\n[STEP] {description}\nCOMMAND: {' '.join(cmd_list)}\n{'='*70}")
    res = subprocess.run(cmd_list, check=True)
    if res.returncode != 0:
        print(f"[!] Error in step: {description}")
        sys.exit(res.returncode)

def main():
    print("=" * 70)
    print("LOCAL CROP DISEASE DETECTOR - GOOGLE COLAB T4 PIPELINE")
    print("=" * 70)

    # 1. Hardware Check
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[+] GPU Detected: {gpu_name} ({gpu_mem:.2f} GB VRAM)")
        print(f"[+] Mixed Precision (AMP) Enabled: True (Recommended for fast T4 convergence)")
    else:
        print("[!] No GPU detected. Running on CPU. (Ensure Colab runtime is set to T4 GPU!)")

    # 2. Install dependencies
    print("\n[*] Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "torch", "torchvision", "imagehash", "opencv-python", "pyyaml", "seaborn"], check=True)

    # 3. Pipeline Execution
    # Step 1: Merge raw datasets into manifest
    run_cmd([sys.executable, "ml_training/01_merge_datasets.py"],
            "Merging raw source datasets with class capping at 1500 images/class")

    # Step 2: Perceptual hash deduplication & 4-way split
    run_cmd([sys.executable, "ml_training/02_filter_and_split.py"],
            "Filtering corrupt images, removing near-duplicates (pHash), and building splits")

    # Step 3: Two-stage transfer learning with AMP
    run_cmd([sys.executable, "ml_training/03_train.py"],
            "Training MobileNetV3-Large (Stage 1: Head, Stage 2: Full network fine-tuning)")

    # Step 4: Comprehensive Lab vs Field diagnostic evaluation
    run_cmd([sys.executable, "ml_training/04_evaluate.py"],
            "Evaluating model on Val, Test_Lab (PlantVillage), and Test_Field (PlantDoc/Kaggle)")

    # Step 5: Temperature scaling calibration & OOD energy threshold
    run_cmd([sys.executable, "ml_training/05_calibrate.py"],
            "Calibrating temperature scaling parameter T and calculating ECE")

    # 4. Package Artifacts
    print("\n[*] Packaging trained artifacts into model_bundle.zip...")
    import shutil
    shutil.make_archive("model_bundle", "zip", "ml_service/model")
    print("\n[+] SUCCESS! model_bundle.zip created.")
    print("[+] Download model_bundle.zip and place contents into ml_service/model/ for production deployment.")

if __name__ == "__main__":
    main()
