"""
Test Suite for Non-Leaf Image Rejection (Black, Random Noise, Skin Tone)
Asserts that out-of-distribution / non-leaf images are rejected with status='unknown'.
"""

import io
import os
import sys
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml_service")))

from ml_service.predict import CropDiseasePredictor


@pytest.fixture(scope="module")
def predictor():
    config_path = os.path.abspath("ml_service/config.yaml")
    return CropDiseasePredictor(config_path=config_path)


def _to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_black_image_rejected(predictor):
    """Asserts that a completely black image is rejected with status in ['unknown', 'not_a_leaf']."""
    black_img = Image.new("RGB", (224, 224), (0, 0, 0))
    res = predictor.predict(_to_bytes(black_img))
    assert res["status"] in ["unknown", "not_a_leaf"]
    assert res["disease"] == "unknown"


def test_noise_image_rejected(predictor):
    """Asserts that a random noise image is rejected with status in ['unknown', 'not_a_leaf']."""
    np.random.seed(42)
    noise_arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    noise_img = Image.fromarray(noise_arr)
    res = predictor.predict(_to_bytes(noise_img))
    assert res["status"] in ["unknown", "not_a_leaf"]
    assert res["disease"] == "unknown"


def test_skin_tone_image_rejected(predictor):
    """Asserts that a uniform human skin tone image is rejected with status in ['unknown', 'not_a_leaf']."""
    skin_img = Image.new("RGB", (224, 224), (210, 160, 130))
    res = predictor.predict(_to_bytes(skin_img))
    assert res["status"] in ["unknown", "not_a_leaf"]
    assert res["disease"] == "unknown"
