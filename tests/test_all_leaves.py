import os
import sys
import glob
sys.path.insert(0, os.path.abspath("."))
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from ml_service.predict import CropDiseasePredictor

def test_all_sample_leaves_and_perturbations():
    predictor = CropDiseasePredictor(config_path="ml_service/config.yaml")
    
    leaf_files = glob.glob("data/sample_real_leaves/*.*")
    print(f"[*] Testing {len(leaf_files)} leaf images from sample_real_leaves...")
    
    rejections = []
    successes = 0
    
    for fpath in leaf_files:
        with open(fpath, "rb") as f:
            img_bytes = f.read()
            
        res = predictor.predict(img_bytes)
        status = res.get("status")
        
        # Verify that it is NEVER rejected as not_a_leaf or poor_image_quality
        if status in ["not_a_leaf", "poor_image_quality", "error"]:
            rejections.append((fpath, status, res.get("message")))
        else:
            successes += 1
            # Verify report fields
            assert res.get("crop") is not None and res.get("crop") != "unknown"
            assert res.get("disease") is not None and res.get("disease") != "unknown"
            assert len(res.get("crop_comparison", [])) == 10
            assert res.get("morphology") is not None
            # Verify no crop is 100%
            for c in res["crop_comparison"]:
                assert c["confidence"] < 0.999, f"Crop {c['crop']} was >= 100%!"

    print(f"[+] Direct Leaf Test: {successes}/{len(leaf_files)} passed with valid reports!")
    assert len(rejections) == 0, f"Found unexpected rejections: {rejections}"
    
    # Now test perturbed unseen images (bright, dim, blurred, rotated, white-padded)
    print("\n[*] Testing unseen perturbed leaf conditions (dim, bright, blurred, padded)...")
    base_img = Image.open("data/sample_real_leaves/a-bellpepper-leaf.png").convert("RGB")
    
    perturbations = {
        "dim_lighting": ImageEnhance.Brightness(base_img).enhance(0.4),
        "bright_lighting": ImageEnhance.Brightness(base_img).enhance(1.8),
        "low_contrast": ImageEnhance.Contrast(base_img).enhance(0.5),
        "blurred_leaf": base_img.filter(ImageFilter.GaussianBlur(radius=3)),
        "rotated_45": base_img.rotate(45, expand=True, fillcolor=(240, 240, 240)),
        "white_background_paper": Image.new("RGB", (400, 400), (250, 250, 250))
    }
    # Paste leaf into white paper
    perturbations["white_background_paper"].paste(base_img.resize((200, 200)), (100, 100))
    
    for name, p_img in perturbations.items():
        import io
        buf = io.BytesIO()
        p_img.save(buf, format="JPEG")
        p_bytes = buf.getvalue()
        
        res = predictor.predict(p_bytes)
        status = res.get("status")
        print(f"  [Perturbation: {name}] -> Status: {status}, Crop: {res.get('crop')}, Disease: {res.get('disease')}, Conf: {res.get('confidence')}")
        assert status not in ["not_a_leaf", "poor_image_quality", "error"], f"Failed on {name}: {res}"
        assert res.get("crop") is not None
        assert res.get("morphology") is not None

    print("\n[SUCCESS] All leaves and perturbed images generate full matched diagnostic reports!")

if __name__ == "__main__":
    test_all_sample_leaves_and_perturbations()
