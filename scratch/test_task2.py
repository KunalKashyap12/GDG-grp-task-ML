import sys
import os

workspace_root = r"c:\Users\HP\OneDrive\Desktop\gdg-grp-task(ml)"
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

import glob
import cv2
import numpy as np
import importlib

try:
    from ml_service.predict import CropDiseasePredictor  # type: ignore
except ImportError:
    CropDiseasePredictor = importlib.import_module("ml_service.predict").CropDiseasePredictor

p = CropDiseasePredictor("ml_service/config.yaml")
print("[+] Predictor successfully initialized.")

leaf_path = glob.glob("data/sample_real_leaves/Bell_Pepper*.jpg")[0]
with open(leaf_path, "rb") as f:
    leaf_bytes = f.read()

# 1. No hint
r1 = p.predict(leaf_bytes)
print("1. No hint -> crop:", r1["crop"], "| disease:", r1["disease"], "| conf:", r1["confidence"], "| certainty:", r1["certainty"], "| status:", r1["status"])

# 2. Synonym hint 'capsicum'
r2 = p.predict(leaf_bytes, crop_hint="capsicum")
print("2. Hint 'capsicum' -> crop:", r2["crop"], "| conf:", r2["confidence"], "| status:", r2["status"])

# 3. Synonym hint 'corn' on pepper leaf (mismatch)
r3 = p.predict(leaf_bytes, crop_hint="corn")
print("3. Hint 'corn' (mismatch) -> crop:", r3["crop"], "| conf:", r3["confidence"], "| status:", r3["status"], "| warn:", r3["mismatch_warning"])

# 4. Unknown hint 'banana'
r4 = p.predict(leaf_bytes, crop_hint="banana")
print("4. Unknown hint 'banana' -> crop:", r4["crop"], "| conf:", r4["confidence"], "| status:", r4["status"], "| warn:", r4["mismatch_warning"])

# 5. Poor quality (blurred leaf)
img = cv2.imread(leaf_path)
blurred = cv2.GaussianBlur(img, (51, 51), 0)
_, buf = cv2.imencode(".jpg", blurred)
r5 = p.predict(buf.tobytes())
print("5. Blurry image -> status:", r5["status"], "| msg:", r5["message"][:60])

# 6. Non-leaf screenshot
user_nl = r"C:\Users\HP\.gemini\antigravity-ide\brain\8bbc2a29-e5d0-4751-9629-35039465d8d4\.user_uploaded\media_1792384994281.png"
with open(user_nl, "rb") as f:
    nl_bytes = f.read()
r6 = p.predict(nl_bytes)
print("6. Non-leaf screenshot -> status:", r6["status"], "| msg:", r6["message"][:60])
