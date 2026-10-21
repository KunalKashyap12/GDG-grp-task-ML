import os
import sys
import json
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import pandas as pd

print("="*60)
print("AUDIT CHECK 1: PREPROCESSING & LABEL MAPPING")
print("="*60)

# Check labels.json
labels_path = "ml_service/model/labels.json"
with open(labels_path, "r") as f:
    legacy_meta = json.load(f)
legacy_classes = legacy_meta.get("classes", [])
print(f"Legacy model (labels.json) class count: {len(legacy_classes)}")
print(f"Legacy sample classes: {legacy_classes[:5]}")

# Check Hugging Face model
try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    hf_model_name = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
    hf_model = AutoModelForImageClassification.from_pretrained(hf_model_name)
    hf_processor = AutoImageProcessor.from_pretrained(hf_model_name)
    print(f"HuggingFace model num_classes: {len(hf_model.config.id2label)}")
    print(f"HuggingFace processor config: {hf_processor}")
except Exception as e:
    print(f"HF loading error: {e}")

# Check PyTorch checkpoint model.pt and model_trained.pt
for pt_name in ["model.pt", "model_trained.pt"]:
    pt_path = os.path.join("ml_service/model", pt_name)
    if os.path.exists(pt_path):
        ckpt = torch.load(pt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            sd = ckpt["state_dict"]
            print(f"{pt_name} is dict with keys: {list(ckpt.keys())}")
        elif isinstance(ckpt, dict):
            sd = ckpt
            print(f"{pt_name} is state_dict directly with {len(sd)} keys")
        else:
            print(f"{pt_name} is {type(ckpt)}")
            sd = ckpt.state_dict() if hasattr(ckpt, "state_dict") else None
        
        # Check classifier weights shape
        for k in sd.keys():
            if "classifier" in k and "weight" in k:
                print(f"  {pt_name} classifier layer '{k}' shape: {sd[k].shape}")

print("\n" + "="*60)
print("AUDIT CHECK 2: ONNX MODEL VERIFICATION")
print("="*60)
onnx_files = [f for f in os.listdir("ml_service/model") if f.endswith(".onnx")]
print(f"ONNX files in ml_service/model: {onnx_files}")
