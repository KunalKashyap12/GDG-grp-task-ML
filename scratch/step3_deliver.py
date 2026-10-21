"""
Step 3 Deliverable:
  Export ONNX from model.pt, verify parity,
  then run before/after evaluation on real photos
  and generate the final evaluation report.
"""
import os
import sys
import json
import torch
import torchvision.models as models
import numpy as np
import onnxruntime as ort
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "ml_service")
from predict import CropDiseasePredictor
from eval_real import KNOWN_SAMPLE_ALIASES, parse_label_from_filename

# ─────────────────────────────────────────────
# A: Export ONNX from legacy model.pt
# ─────────────────────────────────────────────
print("="*60)
print("A. ONNX EXPORT")
print("="*60)

pt_path = "ml_service/model/model.pt"
onnx_path = "ml_service/model/model.onnx"
labels_path = "ml_service/model/labels.json"

with open(labels_path, "r") as f:
    meta = json.load(f)
num_classes = len(meta["classes"])

model = models.mobilenet_v3_large(weights=None)
in_features = model.classifier[3].in_features
model.classifier[3] = torch.nn.Linear(in_features, num_classes)
ckpt = torch.load(pt_path, map_location="cpu")
sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
model.load_state_dict(sd)
model.eval()

dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy, onnx_path,
    export_params=True, opset_version=13, do_constant_folding=True,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    dynamo=False
)
size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
print(f"[+] Exported: {onnx_path} ({size_mb:.2f} MB)")

session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
max_errs = []
for _ in range(5):
    t = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        pt_out = model(t).numpy()
    onnx_out = session.run(["output"], {"input": t.numpy()})[0]
    max_errs.append(np.max(np.abs(pt_out - onnx_out)))

print(f"[+] ONNX/PT parity: max_abs_err={max(max_errs):.2e}, mean={np.mean(max_errs):.2e}")

# ─────────────────────────────────────────────
# B: Before/After Evaluation on real photos
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("B. BEFORE / AFTER EVALUATION ON REAL PHOTOS")
print("="*60)

sample_dir = "data/sample_real_leaves"
real_images = sorted([os.path.join(sample_dir, f)
                      for f in os.listdir(sample_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

report = {}

for backend in ["huggingface", "legacy"]:
    os.environ["MODEL_BACKEND"] = backend
    predictor = CropDiseasePredictor(config_path="ml_service/config.yaml")

    n_total = 0
    n_crop_ok = 0
    n_class_ok = 0
    crop_conf_sum = 0.0
    disease_conf_sum = 0.0
    confusions = Counter()
    per_crop = {}

    for img_p in real_images:
        true_label = parse_label_from_filename(img_p, predictor.classes)
        if not true_label:
            continue
        true_crop = true_label.split("___")[0].replace("_", " ")
        true_crop_std = None
        for std_c in predictor.standard_crops:
            if true_crop.lower() in std_c.lower() or std_c.lower() in true_crop.lower():
                true_crop_std = std_c
                break
        if not true_crop_std:
            continue

        with open(img_p, "rb") as f:
            img_bytes = f.read()
        res = predictor.predict(img_bytes, crop_hint=None, use_priors=False)
        pred_crop = res.get("crop", "Unknown")
        pred_disease = res.get("disease", "Unknown")
        pred_class = f"{pred_crop}___{pred_disease}".replace(" ", "_")
        
        # NEW fields
        crop_conf = res.get("crop_confidence", 0.0) or 0.0
        dis_conf  = res.get("disease_confidence", 0.0) or 0.0

        n_total += 1
        crop_conf_sum += crop_conf
        disease_conf_sum += dis_conf

        crop_match = (true_crop_std.lower() == pred_crop.lower()
                      or true_crop_std.lower() in pred_crop.lower())
        class_match = (true_label.lower() == pred_class.lower())

        if crop_match:
            n_crop_ok += 1
        if class_match:
            n_class_ok += 1
        else:
            confusions[(true_label, pred_class)] += 1

        if true_crop_std not in per_crop:
            per_crop[true_crop_std] = {"n": 0, "crop_ok": 0, "class_ok": 0}
        per_crop[true_crop_std]["n"] += 1
        per_crop[true_crop_std]["crop_ok"] += int(crop_match)
        per_crop[true_crop_std]["class_ok"] += int(class_match)

    crop_acc = n_crop_ok / max(n_total, 1)
    class_acc = n_class_ok / max(n_total, 1)
    avg_crop_conf = crop_conf_sum / max(n_total, 1)
    avg_dis_conf  = disease_conf_sum / max(n_total, 1)

    print(f"\n[{backend.upper()} BACKEND] — {n_total} images")
    print(f"  Crop Accuracy:         {n_crop_ok}/{n_total} ({crop_acc:.2%})")
    print(f"  Full Class Accuracy:   {n_class_ok}/{n_total} ({class_acc:.2%})")
    print(f"  Avg crop_confidence:   {avg_crop_conf:.4f}")
    print(f"  Avg disease_confidence:{avg_dis_conf:.4f}")

    print("\n  Per-Crop Breakdown:")
    for c, s in sorted(per_crop.items()):
        print(f"    {c:18s}: Crop {s['crop_ok']}/{s['n']} ({s['crop_ok']/s['n']:.0%}) | Disease {s['class_ok']}/{s['n']} ({s['class_ok']/s['n']:.0%})")

    print("\n  Top Confusions:")
    for (t, p), cnt in confusions.most_common(8):
        t_crop = t.split("___")[0]
        p_crop = p.split("___")[0]
        tag = "[CROSS-CROP]" if t_crop.lower() != p_crop.lower() else "[SAME-CROP ]"
        print(f"    {tag} {t:35s} -> {p:35s} ({cnt}x)")

    report[backend] = {
        "n_total": n_total,
        "crop_accuracy": round(crop_acc, 4),
        "full_class_accuracy": round(class_acc, 4),
        "avg_crop_confidence": round(avg_crop_conf, 4),
        "avg_disease_confidence": round(avg_dis_conf, 4),
        "top_confusions": [{"true": t, "pred": p, "count": cnt}
                           for (t, p), cnt in confusions.most_common(10)],
    }

# Save report
out_path = "ml_training/evaluation_report.json"
try:
    with open(out_path, "r") as f:
        existing = json.load(f)
except Exception:
    existing = {}
existing.update({"step3_before_after": report})
with open(out_path, "w") as f:
    json.dump(existing, f, indent=2)
print(f"\n[+] Evaluation report saved to {out_path}")
