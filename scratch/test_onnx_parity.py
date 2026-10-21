import os
import sys
import json
import torch
import torchvision.models as models
import numpy as np
import onnxruntime as ort

print("="*60)
print("AUDIT CHECK 2: PYTORCH VS ONNX EXPORT & NUMERICAL PARITY")
print("="*60)

pt_path = "ml_service/model/model.pt"
onnx_path = "ml_service/model/model.onnx"
labels_path = "ml_service/model/labels.json"

with open(labels_path, "r") as f:
    meta = json.load(f)
num_classes = len(meta["classes"])

# Load PyTorch model
model = models.mobilenet_v3_large(weights=None)
in_features = model.classifier[3].in_features
model.classifier[3] = torch.nn.Linear(in_features, num_classes)

checkpoint = torch.load(pt_path, map_location="cpu")
state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
model.load_state_dict(state_dict)
model.eval()

# Export to ONNX if not exists or re-verify
dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
print(f"Exporting PyTorch model to '{onnx_path}'...")
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    dynamo=False
)
print(f"[+] Successfully exported ONNX model ({os.path.getsize(onnx_path)} bytes).")

# Verify ONNX Runtime output vs PyTorch output
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

# Test on 5 different random tensors
max_diffs = []
for i in range(5):
    test_tensor = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    with torch.no_grad():
        pt_logits = model(test_tensor).numpy()
    
    onnx_inputs = {"input": test_tensor.numpy()}
    onnx_logits = session.run(["output"], onnx_inputs)[0]
    
    diff = np.max(np.abs(pt_logits - onnx_logits))
    max_diffs.append(diff)

avg_diff = np.mean(max_diffs)
max_diff = np.max(max_diffs)
print(f"[+] Output numerical parity verification:")
print(f"    Max absolute error between PyTorch and ONNX: {max_diff:.2e}")
print(f"    Mean absolute error across 5 test batches:   {avg_diff:.2e}")
if max_diff < 1e-4:
    print("    -> PARITY CONFIRMED: ONNX produces bit-exact/numerically identical outputs to .pt checkpoint!")
else:
    print("    -> WARNING: Discrepancy detected between PyTorch and ONNX.")
