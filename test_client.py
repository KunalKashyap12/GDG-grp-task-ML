"""
Script: test_client.py
Automated validation script that sends 20 unseen synthetic/field test images
to the running Crop Disease Detector API (Spring Boot backend or FastAPI ML service)
and displays diagnostic predictions in a clean tabular report.
"""

import os
import io
import time
import argparse
import requests
import numpy as np
from PIL import Image, ImageDraw
from typing import Tuple

def parse_args():
    parser = argparse.ArgumentParser(description="Send 20 test images to Crop Disease API.")
    parser.add_argument("--url", type=str, default="http://localhost:8080/api/predict",
                        help="Prediction endpoint URL (default: Spring Boot http://localhost:8080/api/predict)")
    parser.add_argument("--ml-url", type=str, default="http://localhost:8000/predict",
                        help="Direct ML service endpoint URL")
    parser.add_argument("--direct-ml", action="store_true",
                        help="Send directly to ML service instead of Spring Boot backend")
    parser.add_argument("--real-leaves", action="store_true",
                        help="Test using genuine photographic leaf images from data/sample_real_leaves")
    return parser.parse_args()


def generate_test_image(scenario: str) -> Tuple[bytes, str]:
    """Generates synthetic test images representing diverse edge-case scenarios."""
    img = Image.new("RGB", (256, 256), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)

    if scenario == "healthy_leaf":
        # Clear, textured green leaf
        arr = np.random.randint(20, 60, (256, 256, 3), dtype=np.uint8)
        arr[:, :, 1] = np.random.randint(120, 190, (256, 256), dtype=np.uint8) # green channel
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        draw.ellipse([40, 40, 216, 216], fill=(34, 139, 34), outline=(10, 60, 10))
        # Add high-frequency vein texture to ensure sharp blur score
        for i in range(50, 200, 15):
            draw.line([(128, 50), (i, i)], fill=(50, 205, 50), width=2)

    elif scenario == "diseased_leaf":
        # Leaf with lesion spots
        arr = np.random.randint(20, 50, (256, 256, 3), dtype=np.uint8)
        arr[:, :, 1] = np.random.randint(100, 160, (256, 256), dtype=np.uint8)
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        draw.ellipse([30, 30, 226, 226], fill=(85, 107, 47), outline=(20, 40, 10))
        for _ in range(12):
            rx = int(np.random.randint(50, 200))
            ry = int(np.random.randint(50, 200))
            draw.ellipse([rx, ry, rx + 16, ry + 16], fill=(139, 69, 19), outline=(0, 0, 0))

    elif scenario == "blurry_image":
        # Uniform solid circle with zero edge contrast
        draw.ellipse([50, 50, 200, 200], fill=(40, 140, 40))

    elif scenario == "too_dark":
        # Underexposed image (average brightness < 30)
        img = Image.new("RGB", (256, 256), color=(15, 20, 15))

    elif scenario == "too_bright":
        # Overexposed washed-out image (average brightness > 230)
        img = Image.new("RGB", (256, 256), color=(245, 245, 245))

    elif scenario == "random_noise_not_leaf":
        # High frequency uniform noise (no leaf characteristics)
        arr = np.random.randint(60, 200, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(arr)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue(), f"{scenario}.jpg"


def run_benchmark():
    args = parse_args()
    target_url = args.ml_url if args.direct_ml else args.url
    print(f"[*] Dispatching 20 test images to target endpoint: {target_url}\n")

    # Define test suite of 20 diverse cases
    test_cases = [
        # 1-10: Sharp leaf photos across supported crops with and without hints
        {"id": 1,  "scenario": "diseased_leaf", "crop_hint": "Tomato",      "desc": "Tomato leaf with spots (hint: Tomato)"},
        {"id": 2,  "scenario": "diseased_leaf", "crop_hint": "Potato",      "desc": "Potato leaf with spots (hint: Potato)"},
        {"id": 3,  "scenario": "diseased_leaf", "crop_hint": "Rice",        "desc": "Rice leaf lesion (hint: Rice)"},
        {"id": 4,  "scenario": "diseased_leaf", "crop_hint": "Wheat",       "desc": "Wheat rust pustules (hint: Wheat)"},
        {"id": 5,  "scenario": "diseased_leaf", "crop_hint": "Maize",       "desc": "Maize leaf blight (hint: Maize)"},
        {"id": 6,  "scenario": "diseased_leaf", "crop_hint": "Cotton",      "desc": "Cotton bacterial blight (hint: Cotton)"},
        {"id": 7,  "scenario": "diseased_leaf", "crop_hint": "Apple",       "desc": "Apple scab lesion (hint: Apple)"},
        {"id": 8,  "scenario": "diseased_leaf", "crop_hint": "Grape",       "desc": "Grape black rot (hint: Grape)"},
        {"id": 9,  "scenario": "diseased_leaf", "crop_hint": "Sugarcane",   "desc": "Sugarcane red rot (hint: Sugarcane)"},
        {"id": 10, "scenario": "healthy_leaf",  "crop_hint": "Bell_Pepper", "desc": "Bell pepper healthy (hint: Bell_Pepper)"},
        # 11-13: Unconstrained crop predictions (no crop hint)
        {"id": 11, "scenario": "healthy_leaf",  "crop_hint": None,          "desc": "Healthy leaf unconstrained (no hint)"},
        {"id": 12, "scenario": "diseased_leaf", "crop_hint": None,          "desc": "Diseased leaf unconstrained (no hint)"},
        {"id": 13, "scenario": "diseased_leaf", "crop_hint": None,          "desc": "Diseased leaf unconstrained (no hint)"},
        # 14-16: Image quality failure modes
        {"id": 14, "scenario": "blurry_image",  "crop_hint": "Tomato",      "desc": "Out-of-focus camera (blur detection)"},
        {"id": 15, "scenario": "too_dark",      "crop_hint": "Rice",        "desc": "Night/underexposed capture (dark)"},
        {"id": 16, "scenario": "too_bright",    "crop_hint": "Wheat",       "desc": "Direct sun glare/overexposed (bright)"},
        # 17-18: Out-of-Distribution / Non-leaf images
        {"id": 17, "scenario": "random_noise_not_leaf", "crop_hint": None,   "desc": "Random texture noise (not a leaf)"},
        {"id": 18, "scenario": "random_noise_not_leaf", "crop_hint": "Maize","desc": "Desk/wall noise texture (not a leaf)"},
        # 19-20: Edge crop hints
        {"id": 19, "scenario": "healthy_leaf",  "crop_hint": "cotton",      "desc": "Lowercase crop hint check ('cotton')"},
        {"id": 20, "scenario": "diseased_leaf", "crop_hint": "tomato",      "desc": "Lowercase crop hint check ('tomato')"}
    ]

    header_fmt = "{:<3} | {:<25} | {:<10} | {:<16} | {:<18} | {:<8} | {:<7} | {:<9} | {:<8}"
    print(header_fmt.format(
        "#", "Scenario", "Crop Hint", "Status", "Predicted Class", "Conf", "Certain", "Guidance", "Latency"
    ))
    print("-" * 125)

    success_count = 0
    file_key = "file" if args.direct_ml or "8000" in target_url else "image"

    if args.real_leaves:
        import glob
        real_files = sorted(glob.glob("data/sample_real_leaves/*.jpg") + glob.glob("data/sample_real_leaves/*.png"))
        if not real_files:
            print("[!] No real leaf images found in data/sample_real_leaves/")
            return
        print(f"[*] Testing with {len(real_files)} real photographic leaf images from data/sample_real_leaves/:\n")
        print(header_fmt.format(
            "#", "Image Name", "Crop Hint", "Status", "Predicted Class", "Conf", "Certain", "Guidance", "Latency"
        ))
        print("-" * 125)
        for idx, r_path in enumerate(real_files, 1):
            fname = os.path.basename(r_path)
            crop_hint = None
            for c in ["Tomato", "Potato", "Maize", "Apple", "Grape", "Pepper"]:
                if c.lower() in fname.lower():
                    crop_hint = c
                    break
            with open(r_path, "rb") as f:
                img_bytes = f.read()
            files = {file_key: (fname, img_bytes, "image/jpeg")}
            data = {"crop": crop_hint} if crop_hint else {}
            t0 = time.time()
            try:
                resp = requests.post(target_url, files=files, data=data, timeout=10)
                latency = (time.time() - t0) * 1000
                if resp.status_code == 200:
                    success_count += 1
                    res = resp.json()
                    status_str = res.get("status", "unknown")
                    crop = res.get("crop", "N/A")
                    disease = res.get("disease", "N/A")
                    pred_label = f"{crop}___{disease}" if crop != "unknown" else "None"
                    conf = res.get("confidence", 0.0)
                    certainty = res.get("certainty", "N/A")
                    has_guidance = "Yes" if res.get("guidance") is not None else "No"
                    print(header_fmt.format(
                        idx, fname[:25], str(crop_hint or "-")[:10], status_str[:16],
                        pred_label[:18], f"{conf:.2f}", certainty[:7], has_guidance, f"{latency:.1f}ms"
                    ))
                else:
                    print(header_fmt.format(
                        idx, fname[:25], str(crop_hint or "-")[:10], f"HTTP {resp.status_code}",
                        "Error", "-", "-", "-", f"{latency:.1f}ms"
                    ))
            except Exception as e:
                print(f"[{idx:02d}] Error: {e}")
        print("-" * 125)
        print(f"[+] Real Leaf Benchmark: {success_count}/{len(real_files)} tests succeeded.")
        return

    for case in test_cases:
        img_bytes, filename = generate_test_image(case["scenario"])
        files = {file_key: (filename, img_bytes, "image/jpeg")}
        data = {}
        if case["crop_hint"]:
            data["crop"] = case["crop_hint"]

        t0 = time.time()
        try:
            resp = requests.post(target_url, files=files, data=data, timeout=10)
            latency = (time.time() - t0) * 1000

            if resp.status_code == 200:
                success_count += 1
                res = resp.json()
                status_str = res.get("status", "unknown")
                crop = res.get("crop", "N/A")
                disease = res.get("disease", "N/A")
                pred_label = f"{crop}___{disease}" if crop != "unknown" else "None"
                conf = res.get("confidence", 0.0)
                certainty = res.get("certainty", "N/A")
                has_guidance = "Yes" if res.get("guidance") is not None else "No"

                print(header_fmt.format(
                    case["id"],
                    case["desc"][:25],
                    str(case["crop_hint"] or "-")[:10],
                    status_str[:16],
                    pred_label[:18],
                    f"{conf:.2f}",
                    certainty[:7],
                    has_guidance,
                    f"{latency:.1f}ms"
                ))
            else:
                print(header_fmt.format(
                    case["id"],
                    case["desc"][:25],
                    str(case["crop_hint"] or "-")[:10],
                    f"HTTP {resp.status_code}",
                    "Error",
                    "-",
                    "-",
                    "-",
                    f"{latency:.1f}ms"
                ))
        except Exception as e:
            print(f"[{case['id']:02d}] Connection failed to {target_url}: {str(e)}")

    print("-" * 125)
    print(f"[+] Completed: {success_count}/{len(test_cases)} requests responded successfully.")


if __name__ == "__main__":
    from typing import Tuple
    run_benchmark()
