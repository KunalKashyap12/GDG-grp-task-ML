import sys
import os

workspace_root = r"c:\Users\HP\OneDrive\Desktop\gdg-grp-task(ml)"
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

import pandas as pd
import torch
import importlib
try:
    from ml_training.utils.dataset import CropDiseaseDataset  # type: ignore
except ImportError:
    CropDiseaseDataset = importlib.import_module("ml_training.utils.dataset").CropDiseaseDataset

train_module = importlib.import_module("ml_training.03_train")
run_manifest_preflight_check = train_module.run_manifest_preflight_check

print("[*] Testing Task 3 implementations...")

# 1. Test preflight check on a dummy dataframe with valid and broken paths
dummy_df = pd.DataFrame([
    {"path": "data/sample_real_leaves/Bell_Pepper___Bacterial_spot_0.jpg", "label": "Bell_Pepper___Bacterial_spot"},
    {"path": "data/sample_real_leaves/Apple___Apple_scab_0.jpg", "label": "Apple___Apple_scab"},
    {"path": "non_existent/fake_image.jpg", "label": "Fake___Disease"}
])

print("\n--- Testing run_manifest_preflight_check ---")
broken = run_manifest_preflight_check(dummy_df, "test_split")
print(f"Preflight identified {len(broken)} broken paths: {broken}")
assert len(broken) == 1, f"Expected 1 broken path, got {len(broken)}"
assert broken[0][0] == "non_existent/fake_image.jpg"
print("[+] Preflight check properly identified broken path.")

# 2. Test CropDiseaseDataset failure rate threshold
print("\n--- Testing CropDiseaseDataset Failure Handling ---")
label_to_idx = {
    "Bell_Pepper___Bacterial_spot": 0,
    "Apple___Apple_scab": 1,
    "Fake___Disease": 2
}

# Threshold = 0.5 (50%), with 1 broken out of 3 (33%) -> should succeed with warning and black fallback
ds_tolerant = CropDiseaseDataset(dummy_df, label_to_idx, max_failure_rate=0.50)
img, label, source, p = ds_tolerant[0] # valid
assert isinstance(img, object)
img_fail, label_fail, source_fail, p_fail = ds_tolerant[2] # broken
assert ds_tolerant.failure_count == 1
print(f"[+] Loaded broken image with fallback. Failure count = {ds_tolerant.failure_count}/{len(ds_tolerant)}")

# Threshold = 0.1 (10%), with 1 broken out of 3 (33%) -> should raise RuntimeError
ds_strict = CropDiseaseDataset(dummy_df, label_to_idx, max_failure_rate=0.10)
try:
    ds_strict[2]
    raise AssertionError("Expected RuntimeError when failure rate exceeds threshold, but none was raised!")
except RuntimeError as e:
    print(f"[+] Successfully caught expected failure limit error:\n    {e}")

print("\n[+] All Task 3 tests PASSED successfully!")
