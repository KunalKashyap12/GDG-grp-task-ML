import requests
import json

base_url = "http://127.0.0.1:8000"

# 1. Health
r_health = requests.get(f"{base_url}/health")
print("=== 1. GET /health ===")
print(r_health.status_code, r_health.json())

# 2. Classes
r_classes = requests.get(f"{base_url}/classes")
print("\n=== 2. GET /classes ===")
data_cls = r_classes.json()
print("Total classes:", data_cls["total_classes"])
print("Crops count:", len(data_cls["crops"]))
print("Sample crops:", list(data_cls["crops"].keys())[:6])

# 3. Predict on real Tomato leaf
img_path = "data/sample_real_leaves/Tomato___Early_blight_0.jpg"
with open(img_path, "rb") as f:
    files = {"image": ("tomato_leaf.jpg", f.read(), "image/jpeg")}
r_pred = requests.post(f"{base_url}/predict", files=files, data={"crop": "Tomato"})
print("\n=== 3. POST /predict (Tomato Early Blight) ===")
print(f"HTTP {r_pred.status_code}")
print(json.dumps(r_pred.json(), indent=2))

# 4. Predict with Unsupported Crop Hint (Wheat)
with open(img_path, "rb") as f:
    files = {"image": ("tomato_leaf.jpg", f.read(), "image/jpeg")}
r_unsupp = requests.post(f"{base_url}/predict", files=files, data={"crop": "Wheat"})
print("\n=== 4. POST /predict with Unsupported Crop (Wheat) ===")
print(f"HTTP {r_unsupp.status_code}")
print(json.dumps(r_unsupp.json(), indent=2))

# 5. Predict with Corrupted Bytes (Verify clean 4xx)
files_bad = {"image": ("bad.jpg", b"corrupted_non_image_payload_bytes", "image/jpeg")}
r_bad = requests.post(f"{base_url}/predict", files=files_bad)
print("\n=== 5. POST /predict with Corrupted Image (Verify clean 4xx) ===")
print(f"HTTP {r_bad.status_code}: {r_bad.text}")
