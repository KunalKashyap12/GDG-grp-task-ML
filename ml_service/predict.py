"""
Prediction Pipeline for Local Crop Disease Detector
- Fast CPU-optimized inference with MobileNetV3 (< 1 second).
- Image quality validation (blur via Laplacian variance, brightness limits).
- Temperature-scaled confidence calibration.
- Out-of-Distribution (OOD / Not-a-leaf) detection using energy scores.
- Dynamic Crop-Hinting: restricts softmax distribution to requested crop if specified.
"""

import os
import io
import json
import yaml
import time
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_large

class CropDiseasePredictor:
    def __init__(self, config_path: str = "config.yaml"):
        if not os.path.exists(config_path):
            dir_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
            if os.path.exists(dir_cfg):
                config_path = dir_cfg
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.base_dir = os.path.dirname(os.path.abspath(config_path))
        self.device = torch.device(self.config["model"].get("device", "cpu"))
        self._load_labels()
        self._load_calibration()
        self._load_guidance()
        self._build_and_load_model()
        self._build_transforms()

        # Build lookup table mapping crop -> class indices
        self.crop_to_indices: Dict[str, List[int]] = {}
        for idx, cls_name in enumerate(self.classes):
            crop_name = cls_name.split("___")[0].lower()
            if crop_name not in self.crop_to_indices:
                self.crop_to_indices[crop_name] = []
            self.crop_to_indices[crop_name].append(idx)

    def _resolve_path(self, rel_path: str) -> str:
        if os.path.isabs(rel_path):
            return rel_path
        return os.path.normpath(os.path.join(self.base_dir, rel_path))

    def _load_labels(self):
        labels_path = self._resolve_path(self.config["model"]["labels_path"])
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file missing at {labels_path}")
        with open(labels_path, "r") as f:
            meta = json.load(f)
        self.classes = meta["classes"]
        self.label_to_idx = meta["label_to_idx"]
        self.num_classes = len(self.classes)

    def _load_calibration(self):
        calib_path = self._resolve_path(self.config["model"]["calibration_path"])
        if os.path.exists(calib_path):
            with open(calib_path, "r") as f:
                calib = json.load(f)
            self.temperature = float(calib.get("temperature", 1.0))
            self.energy_threshold = float(calib.get("energy_threshold", -5.0))
            conf_thresh = calib.get("confidence_thresholds", {})
            self.thresh_high = float(conf_thresh.get("high", 0.85))
            self.thresh_medium = float(conf_thresh.get("medium", 0.60))
        else:
            self.temperature = 1.0
            self.energy_threshold = float(self.config["uncertainty"]["ood"].get("energy_threshold", -5.0))
            self.thresh_high = float(self.config["uncertainty"]["confidence_thresholds"].get("high", 0.85))
            self.thresh_medium = float(self.config["uncertainty"]["confidence_thresholds"].get("medium", 0.60))

    def _load_guidance(self):
        guidance_path = self._resolve_path("model/guidance.json")
        self.guidance = {}
        if os.path.exists(guidance_path):
            try:
                with open(guidance_path, "r", encoding="utf-8") as f:
                    self.guidance = json.load(f)
            except Exception as e:
                print(f"[!] Warning: failed to load guidance.json: {e}")

    def _build_and_load_model(self):
        model_path = self._resolve_path(self.config["model"]["path"])
        model = mobilenet_v3_large(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, self.num_classes)

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
            model.load_state_dict(state_dict)
        else:
            print(f"[!] Warning: Model file {model_path} not found. Running with initialized weights for scaffolding.")

        model.to(self.device)
        model.eval()
        self.model = model

    def _build_transforms(self):
        img_size = self.config["model"].get("img_size", 224)
        self.transform = T.Compose([
            T.Resize(int(img_size * 1.14)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def assess_quality(self, image_cv: np.ndarray) -> Tuple[bool, float, float, Optional[str]]:
        """Validates blur and illumination brightness."""
        blur_thresh = float(self.config["quality"]["blur_threshold"])
        min_bright = float(self.config["quality"]["min_brightness"])
        max_bright = float(self.config["quality"]["max_brightness"])

        if len(image_cv.shape) == 3:
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_cv

        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))

        if blur_score < blur_thresh:
            return False, round(blur_score, 2), round(brightness, 2), f"CLEAR PHOTO NAHI HAI: Photo bahut dhundhli (blurry) hai (sharpness: {blur_score:.1f} < {blur_thresh:.1f}). Kripya camera saaf karke clear photo dubara upload karein."

        if brightness < min_bright:
            return False, round(blur_score, 2), round(brightness, 2), f"CLEAR PHOTO NAHI HAI: Photo bahut andhere mein hai (brightness: {brightness:.1f} < {min_bright:.1f}). Kripya acchi roshni mein photo dubara upload karein."

        if brightness > max_bright:
            return False, round(blur_score, 2), round(brightness, 2), f"CLEAR PHOTO NAHI HAI: Photo par dhoop/glare bahut zyada hai. Kripya saaf photo dubara upload karein."

        return True, round(blur_score, 2), round(brightness, 2), None

    def is_leaf_present(self, image_cv: np.ndarray) -> Tuple[bool, float]:
        """
        Validates vegetation presence using the Excess Green Index (ExG)
        and plant foliage spectral constraints to reject non-leaf objects (shoes, cars, faces, walls).
        """
        if len(image_cv.shape) != 3:
            return False, 0.0
        b, g, r = [image_cv[:, :, i].astype(np.float32) for i in range(3)]
        exg = 2 * g - r - b
        foliage = (exg > 7.0) | ((g >= r - 10) & (g > b + 14))
        ratio = float(np.count_nonzero(foliage)) / float(image_cv.shape[0] * image_cv.shape[1])
        return ratio >= 0.10, round(ratio, 4)

    def predict(self, image_bytes: bytes, crop_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes end-to-end prediction:
        1. Decodes and checks image quality (blur & brightness).
        2. Validates leaf presence (rejects non-leaf objects).
        3. Computes calibrated logits.
        4. Applies crop-hint conditioning and re-normalizes softmax.
        5. Computes crop-level confidence comparison breakdown.
        6. Formats standardized output response.
        """
        start_time = time.time()

        # Decode image
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            # Convert to BGR numpy array for OpenCV checks
            np_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            return {
                "status": "error",
                "crop": "unknown",
                "disease": "unknown",
                "confidence": 0.0,
                "certainty": "low",
                "top3": [],
                "crop_comparison": [],
                "quality": {"blur_score": 0.0, "brightness": 0.0},
                "guidance": None,
                "message": f"Corrupted or invalid image format: {str(e)}"
            }

        # 1. Quality Assessment (Blur & Brightness)
        is_quality_ok, blur_score, brightness, quality_reason = self.assess_quality(np_img)
        quality_info = {"blur_score": blur_score, "brightness": brightness}

        if not is_quality_ok:
            return {
                "status": "poor_image_quality",
                "crop": "unknown",
                "disease": "unknown",
                "confidence": 0.0,
                "certainty": "low",
                "top3": [],
                "crop_comparison": [],
                "quality": quality_info,
                "guidance": None,
                "message": quality_reason or "CLEAR PHOTO NAHI HAI: Photo bahut dhundhli (blurry) hai. Kripya camera saaf karke clear photo dubara upload karein."
            }

        # 2. Leaf Structure & Foliage Verification (Reject non-leaf items)
        is_leaf, foliage_ratio = self.is_leaf_present(np_img)
        if not is_leaf:
            return {
                "status": "not_a_leaf",
                "crop": "unknown",
                "disease": "unknown",
                "confidence": 0.0,
                "certainty": "low",
                "top3": [],
                "crop_comparison": [],
                "quality": quality_info,
                "guidance": None,
                "message": "NOT A LEAF: Yeh kisi fasal ki patti nahi hai! Kripya kisi fasal (Crop Leaf) ki photo upload karein."
            }

        # 2. PyTorch Preprocessing & Forward pass
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            raw_logits = self.model(input_tensor) # shape: (1, num_classes)
            # Apply learned temperature scaling
            scaled_logits = raw_logits / self.temperature

            # 3. Out-Of-Distribution (Not a leaf) Detection
            # Free energy: E(x) = -T * log(sum(exp(z_i / T)))
            energy = float(-self.temperature * torch.logsumexp(scaled_logits, dim=1).item())
            all_probs = torch.softmax(scaled_logits, dim=1).squeeze(0)
            max_unconstrained_prob = float(torch.max(all_probs).item())

            min_leaf_prob = float(self.config["uncertainty"]["ood"].get("min_leaf_max_prob", 0.35))
            if (energy > self.energy_threshold and max_unconstrained_prob < min_leaf_prob) or max_unconstrained_prob < 0.25:
                return {
                    "status": "not_a_leaf",
                    "crop": "unknown",
                    "disease": "unknown",
                    "confidence": round(max_unconstrained_prob, 4),
                    "certainty": "low",
                    "top3": [],
                    "crop_comparison": [],
                    "quality": quality_info,
                    "guidance": None,
                    "message": "NOT A LEAF: Yeh kisi fasal ki patti nahi hai! Kripya kisi fasal (Crop Leaf) ki photo upload karein."
                }

            # 4. Optional Crop Hint Conditioning & Softmax Renormalization
            effective_crop = None
            if crop_hint and crop_hint.strip():
                clean_hint = crop_hint.strip().lower().replace(" ", "_")
                # Look up matching class indices
                matching_indices = self.crop_to_indices.get(clean_hint, [])
                if not matching_indices:
                    # Try partial match (e.g. "pepper" for "bell_pepper")
                    for c_name, indices in self.crop_to_indices.items():
                        if clean_hint in c_name or c_name in clean_hint:
                            matching_indices = indices
                            clean_hint = c_name
                            break

                if matching_indices:
                    effective_crop = self.classes[matching_indices[0]].split("___")[0]
                    # Restrict logits to only matching classes and renormalize softmax
                    sub_logits = scaled_logits[0, matching_indices]
                    sub_probs = torch.softmax(sub_logits, dim=0).cpu().numpy()

                    top_sub_order = np.argsort(sub_probs)[::-1]
                    top_prob = float(sub_probs[top_sub_order[0]])
                    top_class_name = self.classes[matching_indices[top_sub_order[0]]]

                    # Construct top3
                    top3 = []
                    for rank in range(min(3, len(matching_indices))):
                        idx = matching_indices[top_sub_order[rank]]
                        top3.append({
                            "label": self.classes[idx],
                            "score": round(float(sub_probs[top_sub_order[rank]]), 4)
                        })
                else:
                    # Hint provided was unknown, fallback to unconstrained
                    pass

            # If no crop hint applied, use full 36-class softmax
            if effective_crop is None:
                probs = all_probs.cpu().numpy()
                top_order = np.argsort(probs)[::-1]
                top_prob = float(probs[top_order[0]])
                top_class_name = self.classes[top_order[0]]
                effective_crop = top_class_name.split("___")[0]

                top3 = []
                for rank in range(min(3, len(self.classes))):
                    idx = top_order[rank]
                    top3.append({
                        "label": self.classes[idx],
                        "score": round(float(probs[idx]), 4)
                    })

        # 5. Extract disease portion and assign certainty level
        disease_name = top_class_name.split("___")[1]
        confidence = round(top_prob, 4)

        # 6. Compute crop-level confidence comparison breakdown across all 10 crops
        probs_all = all_probs.cpu().numpy()
        crop_scores: Dict[str, float] = {}
        for c_idx, c_prob in enumerate(probs_all):
            c_name = self.classes[c_idx].split("___")[0].replace("_", " ")
            crop_scores[c_name] = crop_scores.get(c_name, 0.0) + float(c_prob)

        crop_comparison = [
            {"crop": c, "confidence": round(score, 4), "percent": round(score * 100, 1)}
            for c, score in sorted(crop_scores.items(), key=lambda x: x[1], reverse=True)
        ]

        if confidence >= self.thresh_high:
            certainty = "high"
            status = "ok"
            msg = "High confidence prediction."
        elif confidence >= self.thresh_medium:
            certainty = "medium"
            status = "ok"
            msg = "Moderate confidence prediction. Inspect field symptoms carefully."
        else:
            certainty = "low"
            status = "low_confidence"
            msg = "Low confidence prediction. Symptoms are ambiguous or lighting is suboptimal."

        latency_ms = round((time.time() - start_time) * 1000, 1)

        # Look up further agronomic guidance
        full_key = f"{effective_crop}___{disease_name}"
        guidance_info = self.guidance.get(full_key, None)

        return {
            "status": status,
            "crop": effective_crop,
            "disease": disease_name,
            "confidence": confidence,
            "certainty": certainty,
            "top3": top3,
            "crop_comparison": crop_comparison,
            "quality": quality_info,
            "guidance": guidance_info,
            "message": msg
        }


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Run Crop Disease ML Prediction on an image.")
    parser.add_argument("image", nargs="?", default=None, help="Path to input crop leaf image (jpg/png)")
    parser.add_argument("--crop", type=str, default=None, help="Optional crop hint (e.g. potato, apple, maize)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    predictor = CropDiseasePredictor(config_path=args.config)
    print(f"\n[+] MobileNetV3 Model loaded successfully ({predictor.num_classes} classes on device: {predictor.device}).\n")

    images_to_run = []
    if args.image:
        images_to_run.append((args.image, args.crop))
    else:
        samples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "samples")
        if os.path.exists(samples_dir):
            for fname in os.listdir(samples_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    images_to_run.append((os.path.join(samples_dir, fname), None))

    if not images_to_run:
        print("[!] No image provided and no sample images found in samples/ directory.")
        sys.exit(1)

    print(f"Executing ML Predictions on {len(images_to_run)} image(s):")
    print("=" * 65)
    for img_path, crop_hint in images_to_run:
        print(f"\n[>] Image: {img_path}")
        if crop_hint:
            print(f"    Crop Hint: {crop_hint}")
        if not os.path.exists(img_path):
            print(f"    [!] File not found: {img_path}")
            continue

        with open(img_path, "rb") as f:
            img_bytes = f.read()

        res = predictor.predict(img_bytes, crop_hint=crop_hint)
        print(f"    Status:     {res['status']}")
        print(f"    Crop:       {res['crop']}")
        print(f"    Disease:    {res['disease']}")
        print(f"    Confidence: {res['confidence'] * 100:.2f}% ({res['certainty']} certainty)")
        print(f"    Quality:    blur={res['quality']['blur_score']}, brightness={res['quality']['brightness']}")
        print(f"    Message:    {res['message']}")
        print("    Top-3 Predictions:")
        for t in res.get("top3", []):
            print(f"      - {t['label']}: {t['score'] * 100:.2f}%")
        if res.get("guidance"):
            g = res["guidance"]
            print(f"    Guidance:")
            print(f"      Category:  {g.get('category')}")
            print(f"      Urgency:   {g.get('urgency')}")
            print(f"      Treatment: {g.get('treatment_summary')}")
    print("\n" + "=" * 65)
