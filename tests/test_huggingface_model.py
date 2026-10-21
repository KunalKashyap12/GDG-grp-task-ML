"""
Test Suite for Hugging Face MobileNetV2 Plant Disease Integration
Covers:
1. Label parser splitting on '___', cleaning formatting, normalizing 'Healthy',
   and parsing natural language labels from linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification.
2. Model initialization using Hugging Face Transformers and dynamic class derivation.
3. Verification of strict compatibility with Spring Boot API contract (MlPredictionResponse DTO).
4. Confidence policy enforcement (threshold = 0.60).
5. Graceful handling of unsupported crops (Rice, Wheat, Cotton, Sugarcane).
6. Clean error handling on corrupted / non-image files.
"""

import os
import io
import sys
import pytest
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml_service")))

from ml_service.predict import parse_label, clean_token, CropDiseasePredictor


# -----------------------------------------------------------------------------
# 1. Label Parser Tests
# -----------------------------------------------------------------------------
def test_label_parser_folder_format():
    """Validates parser on '___' folder-style labels with various edge cases."""
    # Standard underscore format
    c, d = parse_label("Tomato___Late_blight")
    assert c == "Tomato"
    assert d == "Late blight"

    # Trailing underscore and parentheses: Corn_(maize)___Common_rust_ -> Corn (maize)
    c, d = parse_label("Corn_(maize)___Common_rust_")
    assert c == "Corn (maize)"
    assert d == "Common rust"

    # Pepper with comma and underscore: Pepper,_bell___Bacterial_spot -> Pepper, bell
    c, d = parse_label("Pepper,_bell___Bacterial_spot")
    assert c == "Pepper, bell"
    assert d == "Bacterial spot"

    # Lowercase 'healthy' should normalize to 'Healthy'
    c, d = parse_label("Tomato___healthy")
    assert c == "Tomato"
    assert d == "Healthy"

    # Parentheses in crop
    c, d = parse_label("Cherry_(including_sour)___Powdery_mildew")
    assert c == "Cherry (including sour)"
    assert d == "Powdery mildew"


def test_label_parser_natural_language_format():
    """Validates parser on natural language labels from linkanjarad's 38 classes."""
    cases = [
        ("Healthy Apple", "Apple", "Healthy"),
        ("Healthy Blueberry Plant", "Blueberry", "Healthy"),
        ("Healthy Cherry Plant", "Cherry", "Healthy"),
        ("Cherry with Powdery Mildew", "Cherry", "Powdery Mildew"),
        ("Corn (Maize) with Common Rust", "Corn (Maize)", "Common Rust"),
        ("Corn (Maize) with Northern Leaf Blight", "Corn (Maize)", "Northern Leaf Blight"),
        ("Grape with Black Rot", "Grape", "Black Rot"),
        ("Grape with Esca (Black Measles)", "Grape", "Esca (Black Measles)"),
        ("Bell Pepper with Bacterial Spot", "Bell Pepper", "Bacterial Spot"),
        ("Healthy Bell Pepper Plant", "Bell Pepper", "Healthy"),
        ("Potato with Early Blight", "Potato", "Early Blight"),
        ("Tomato with Late Blight", "Tomato", "Late Blight"),
        ("Tomato Yellow Leaf Curl Virus", "Tomato", "Yellow Leaf Curl Virus"),
        ("Tomato Mosaic Virus", "Tomato", "Mosaic Virus"),
        ("Orange with Citrus Greening", "Orange", "Citrus Greening"),
        ("Squash with Powdery Mildew", "Squash", "Powdery Mildew"),
        ("Strawberry with Leaf Scorch", "Strawberry", "Leaf Scorch"),
    ]
    for raw, expected_crop, expected_disease in cases:
        c, d = parse_label(raw)
        assert c == expected_crop, f"Expected crop '{expected_crop}' for '{raw}', got '{c}'"
        assert d == expected_disease, f"Expected disease '{expected_disease}' for '{raw}', got '{d}'"


# -----------------------------------------------------------------------------
# 2. Model Loading & Dynamic Class Derivation Tests
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def predictor():
    old_backend = os.environ.get("MODEL_BACKEND")
    os.environ["MODEL_BACKEND"] = "huggingface"
    config_path = os.path.abspath("ml_service/config.yaml")
    p = CropDiseasePredictor(config_path=config_path)
    yield p
    if old_backend is not None:
        os.environ["MODEL_BACKEND"] = old_backend
    else:
        os.environ.pop("MODEL_BACKEND", None)


def test_model_loads_hf_and_classes_dynamically(predictor):
    """Verifies that model is loaded with Hugging Face and classes are derived from config."""
    assert predictor.backend == "huggingface"
    assert predictor.num_classes == 38
    assert len(predictor.classes) == 38
    assert len(predictor.parsed_classes) == 38
    # Assert standard crops derived from id2label
    assert "Tomato" in predictor.standard_crops
    assert "Apple" in predictor.standard_crops
    assert "Potato" in predictor.standard_crops
    assert len(predictor.standard_crops) >= 12


# -----------------------------------------------------------------------------
# 3. Contract Matching with Spring Boot Backend DTOs
# -----------------------------------------------------------------------------
def test_prediction_contract_matches_spring_boot(predictor):
    """
    Asserts that the prediction output dictionary contains all required fields
    and types expected by Spring Boot's MlPredictionResponse.java:
    - status: String
    - crop: String
    - disease: String
    - confidence: Double (0.0 to 1.0)
    - certainty: String ("high", "medium", "low")
    - top3: List[dict(label=str, score=float)]
    - quality: dict(blur_score=float, brightness=float)
    - message: String
    """
    sample_path = "data/sample_real_leaves/Tomato___Early_blight_0.jpg"
    assert os.path.exists(sample_path), f"Sample image missing at {sample_path}"

    with open(sample_path, "rb") as f:
        img_bytes = f.read()

    res = predictor.predict(img_bytes, crop_hint="Tomato")

    # Contract assertions
    assert "status" in res and isinstance(res["status"], str)
    assert res["status"] in ["ok", "low_confidence", "poor_image_quality", "not_a_leaf", "error", "species_mismatch", "unknown"]

    assert "crop" in res and isinstance(res["crop"], str)
    assert res["crop"] == "Tomato"

    assert "disease" in res and isinstance(res["disease"], str)
    assert "Early" in res["disease"]

    assert "confidence" in res and isinstance(res["confidence"], float)
    assert 0.0 <= res["confidence"] <= 1.0

    assert "certainty" in res and isinstance(res["certainty"], str)
    assert res["certainty"] in ["high", "medium", "low"]

    assert "top3" in res and isinstance(res["top3"], list)
    assert len(res["top3"]) <= 3
    for item in res["top3"]:
        assert "label" in item and isinstance(item["label"], str)
        assert "___" in item["label"]  # Spring Boot expects Crop___Disease labels
        assert "score" in item and isinstance(item["score"], float)

    assert "quality" in res and isinstance(res["quality"], dict)
    assert "blur_score" in res["quality"] and isinstance(res["quality"]["blur_score"], float)
    assert "brightness" in res["quality"] and isinstance(res["quality"]["brightness"], float)

    assert "message" in res and isinstance(res["message"], str)

    # Optional / additive fields
    assert "guidance" in res
    if res["guidance"] is not None:
        assert "disease_name" in res["guidance"]
        assert "preventive_measures" in res["guidance"]
        assert "treatment_summary" in res["guidance"]


# -----------------------------------------------------------------------------
# 4. Confidence Policy Tests
# -----------------------------------------------------------------------------
def test_confidence_policy_threshold(predictor):
    """
    Asserts that predictions with confidence below CONFIDENCE_THRESHOLD (0.60)
    are marked with status='low_confidence' and certainty='low'.
    """
    assert predictor.confidence_threshold == 0.60

    # Low confidence handling test
    sample_path = "data/sample_real_leaves/Tomato___Early_blight_0.jpg"
    with open(sample_path, "rb") as f:
        img_bytes = f.read()

    # Artificially set a high threshold to test that the policy triggers
    original_thresh = predictor.confidence_threshold
    try:
        predictor.confidence_threshold = 0.99
        res = predictor.predict(img_bytes)
        assert res["status"] == "low_confidence"
        assert res["certainty"] == "low"
        assert "Low confidence" in res["message"]
    finally:
        predictor.confidence_threshold = original_thresh


# -----------------------------------------------------------------------------
# 5. Unsupported Crop Graceful Handling Tests
# -----------------------------------------------------------------------------
def test_unsupported_crop_fails_gracefully(predictor):
    """
    Asserts that crops NOT covered by PlantVillage (Rice, Wheat, Cotton, Sugarcane)
    fail gracefully with status='low_confidence'.
    """
    sample_path = "data/sample_real_leaves/Tomato___Early_blight_0.jpg"
    with open(sample_path, "rb") as f:
        img_bytes = f.read()

    for unsupported in ["Wheat", "Rice", "Cotton", "Sugarcane"]:
        res = predictor.predict(img_bytes, crop_hint=unsupported)
        assert res["status"] == "low_confidence"
        assert res["certainty"] == "low"
        assert "Unsupported crop" in res["message"]


# -----------------------------------------------------------------------------
# 6. Bad Input Error Handling Tests
# -----------------------------------------------------------------------------
def test_corrupted_image_returns_error_dict(predictor):
    """Asserts that corrupted bytes return clean error dictionary without raising 500 exceptions."""
    corrupted_bytes = b"NOT_A_VALID_JPEG_IMAGE_DATA_123456"
    res = predictor.predict(corrupted_bytes)
    assert res["status"] == "error"
    assert "Corrupted or invalid" in res["message"]
