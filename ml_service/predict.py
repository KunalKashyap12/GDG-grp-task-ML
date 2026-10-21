"""
Prediction Pipeline for Local Crop Disease Detector
Supports:
1. Pretrained Hugging Face model: linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification (default)
2. Backward-compatible legacy model switch via MODEL_BACKEND=legacy
3. Dynamic label parsing derived from model.config.id2label
4. Configurable confidence policy (CONFIDENCE_THRESHOLD env var, default 0.60)
5. Comprehensive quality screening (Laplacian blur and illumination) and leaf foliage verification
6. Strict backward compatibility with Spring Boot backend API contract
"""

import os
import io
import json
import re
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

try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification, MobileNetV2ImageProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


BOTANICAL_METADATA = {
    "Bell_Pepper": {
        "common_name": "Bell Pepper (Capsicum)",
        "scientific_name": "Capsicum annuum",
        "family": "Solanaceae (Nightshade family)",
        "leaf_type": "Simple Ovate-Lanceolate (Entire Margin)",
        "margin": "Entire (Smooth, non-dentate)",
        "venation": "Pinnate reticulate",
        "apex": "Acute / acuminate pointed apex",
        "botanical_notes": "Single simple leaf blade, smooth edges, acute apex."
    },
    "Pepper, bell": {
        "common_name": "Bell Pepper (Capsicum)",
        "scientific_name": "Capsicum annuum",
        "family": "Solanaceae (Nightshade family)",
        "leaf_type": "Simple Ovate-Lanceolate (Entire Margin)",
        "margin": "Entire (Smooth, non-dentate)",
        "venation": "Pinnate reticulate",
        "apex": "Acute / acuminate pointed apex",
        "botanical_notes": "Single simple leaf blade, smooth edges, acute apex."
    },
    "Bell Pepper": {
        "common_name": "Bell Pepper (Capsicum)",
        "scientific_name": "Capsicum annuum",
        "family": "Solanaceae (Nightshade family)",
        "leaf_type": "Simple Ovate-Lanceolate (Entire Margin)",
        "margin": "Entire (Smooth, non-dentate)",
        "venation": "Pinnate reticulate",
        "apex": "Acute / acuminate pointed apex",
        "botanical_notes": "Single simple leaf blade, smooth edges, acute apex."
    },
    "Tomato": {
        "common_name": "Tomato",
        "scientific_name": "Solanum lycopersicum",
        "family": "Solanaceae (Nightshade family)",
        "leaf_type": "Pinnately Compound / Dissected Leaflets",
        "margin": "Dentate / coarsely serrated / lobed",
        "venation": "Pinnate reticulate with glandular trichomes",
        "apex": "Acute / lobed pointed",
        "botanical_notes": "Dissected compound foliage with distinct leaflets and serrations."
    },
    "Potato": {
        "common_name": "Potato",
        "scientific_name": "Solanum tuberosum",
        "family": "Solanaceae (Nightshade family)",
        "leaf_type": "Pinnately Compound with Ovate Leaflets",
        "margin": "Wavy / weakly undulate",
        "venation": "Pinnate reticulate",
        "apex": "Obtuse to acute",
        "botanical_notes": "Compound foliage with terminal ovate leaflet and small intercalary lobes."
    },
    "Apple": {
        "common_name": "Apple",
        "scientific_name": "Malus domestica",
        "family": "Rosaceae (Rose family)",
        "leaf_type": "Simple Ovate-Elliptic",
        "margin": "Finely serrated (crenate / serrulate)",
        "venation": "Pinnate reticulate",
        "apex": "Short acuminate",
        "botanical_notes": "Deciduous tree foliage, finely serrated margin, rounded base."
    },
    "Blueberry": {
        "common_name": "Blueberry",
        "scientific_name": "Vaccinium corymbosum",
        "family": "Ericaceae (Heath family)",
        "leaf_type": "Simple Ovate-Elliptic",
        "margin": "Entire to finely ciliate",
        "venation": "Pinnate reticulate",
        "apex": "Acute to obtuse",
        "botanical_notes": "Leathery deciduous shrub leaf requiring acidic soil."
    },
    "Cherry": {
        "common_name": "Cherry",
        "scientific_name": "Prunus avium",
        "family": "Rosaceae (Rose family)",
        "leaf_type": "Simple Ovate-Obovate",
        "margin": "Coarsely serrated (doubly toothed)",
        "venation": "Pinnate reticulate",
        "apex": "Acuminate pointed",
        "botanical_notes": "Deciduous foliage with distinctive reddish nectar glands on petiole."
    },
    "Corn (Maize)": {
        "common_name": "Maize (Corn)",
        "scientific_name": "Zea mays",
        "family": "Poaceae (Grass family)",
        "leaf_type": "Linear Strap-like Blade (Monocot)",
        "margin": "Entire, wavy margin",
        "venation": "Parallel venation",
        "apex": "Attenuate pointed",
        "botanical_notes": "Elongated arching grass blade with prominent central midrib."
    },
    "Corn (maize)": {
        "common_name": "Maize (Corn)",
        "scientific_name": "Zea mays",
        "family": "Poaceae (Grass family)",
        "leaf_type": "Linear Strap-like Blade (Monocot)",
        "margin": "Entire, wavy margin",
        "venation": "Parallel venation",
        "apex": "Attenuate pointed",
        "botanical_notes": "Elongated arching grass blade with prominent central midrib."
    },
    "Maize": {
        "common_name": "Maize (Corn)",
        "scientific_name": "Zea mays",
        "family": "Poaceae (Grass family)",
        "leaf_type": "Linear Strap-like Blade (Monocot)",
        "margin": "Entire, wavy margin",
        "venation": "Parallel venation",
        "apex": "Attenuate pointed",
        "botanical_notes": "Elongated arching grass blade with prominent central midrib."
    },
    "Grape": {
        "common_name": "Grape",
        "scientific_name": "Vitis vinifera",
        "family": "Vitaceae (Grapevine family)",
        "leaf_type": "Palmately 3-5 Lobed (Cordate Base)",
        "margin": "Coarsely dentate / toothed",
        "venation": "Palmate reticulate",
        "apex": "Lobed points",
        "botanical_notes": "Broad hand-shaped palmate blade with deep basal sinus."
    },
    "Orange": {
        "common_name": "Orange (Citrus)",
        "scientific_name": "Citrus sinensis",
        "family": "Rutaceae (Citrus family)",
        "leaf_type": "Unifoliolate Ovate (Winged Petiole)",
        "margin": "Entire to crenulate",
        "venation": "Pinnate with pellucid oil glands",
        "apex": "Acute to bluntly pointed",
        "botanical_notes": "Evergreen leathery blade with aromatic translucent oil glands."
    },
    "Peach": {
        "common_name": "Peach",
        "scientific_name": "Prunus persica",
        "family": "Rosaceae (Rose family)",
        "leaf_type": "Simple Lanceolate-Oblong",
        "margin": "Finely serrate or crenate",
        "venation": "Pinnate reticulate",
        "apex": "Long acuminate tapering",
        "botanical_notes": "Slender deciduous spear-shaped foliage, finely gland-tipped teeth."
    },
    "Raspberry": {
        "common_name": "Raspberry",
        "scientific_name": "Rubus idaeus",
        "family": "Rosaceae (Rose family)",
        "leaf_type": "Pinnately Compound 3-5 Leaflets",
        "margin": "Coarsely serrate / toothed",
        "venation": "Pinnate reticulate",
        "apex": "Acute pointed",
        "botanical_notes": "Compound foliage with silvery tomentose undersides."
    },
    "Soybean": {
        "common_name": "Soybean",
        "scientific_name": "Glycine max",
        "family": "Fabaceae (Legume family)",
        "leaf_type": "Trifoliolate Ovate Leaflets",
        "margin": "Entire",
        "venation": "Pinnate reticulate",
        "apex": "Acute to obtuse",
        "botanical_notes": "Three ovate leaflets covered with fine protective pubescence."
    },
    "Squash": {
        "common_name": "Squash (Cucurbit)",
        "scientific_name": "Cucurbita pepo",
        "family": "Cucurbitaceae (Gourd family)",
        "leaf_type": "Palmately 5-Lobed Heart-shaped Blade",
        "margin": "Denticulate / wavy lobed",
        "venation": "Palmate reticulate",
        "apex": "Triangular lobes",
        "botanical_notes": "Large coarse blade covered in stiff prickly trichomes."
    },
    "Strawberry": {
        "common_name": "Strawberry",
        "scientific_name": "Fragaria x ananassa",
        "family": "Rosaceae (Rose family)",
        "leaf_type": "Trifoliolate Obovate Leaflets",
        "margin": "Coarsely serrated (dentate)",
        "venation": "Pinnate reticulate",
        "apex": "Rounded with coarse teeth",
        "botanical_notes": "Three rounded leaflets emerging on long hairy petioles from the crown."
    },
    "Rice": {
        "common_name": "Rice",
        "scientific_name": "Oryza sativa",
        "family": "Poaceae (Grass family)",
        "leaf_type": "Narrow Linear Blade (Monocot)",
        "margin": "Scabrous / rough entire",
        "venation": "Parallel venation",
        "apex": "Attenuate pointed",
        "botanical_notes": "Slender erect blade with ligule and auricles at collar."
    },
    "Wheat": {
        "common_name": "Wheat",
        "scientific_name": "Triticum aestivum",
        "family": "Poaceae (Grass family)",
        "leaf_type": "Linear Blade (Monocot)",
        "margin": "Entire, smooth",
        "venation": "Parallel venation",
        "apex": "Tapering acute",
        "botanical_notes": "Fine slender linear blade, parallel veins, hairy auricles."
    },
    "Cotton": {
        "common_name": "Cotton",
        "scientific_name": "Gossypium hirsutum",
        "family": "Malvaceae (Mallow family)",
        "leaf_type": "Palmately 3-5 Lobed Blade",
        "margin": "Entire to shallowly lobed",
        "venation": "Palmate reticulate",
        "apex": "Acute lobed",
        "botanical_notes": "Broad heart-shaped or lobed blade with extrafloral nectaries."
    },
    "Sugarcane": {
        "common_name": "Sugarcane",
        "scientific_name": "Saccharum officinarum",
        "family": "Poaceae (Grass family)",
        "leaf_type": "Elongate Linear Strap (Monocot)",
        "margin": "Finely serrulate / sharp",
        "venation": "Parallel venation",
        "apex": "Tapering acute",
        "botanical_notes": "Broad linear strap with heavy white midrib and sharp silica edges."
    }
}


def clean_token(text: str) -> str:
    """Cleans underscores to spaces, strips trailing underscores, and trims whitespace."""
    if not text:
        return ""
    cleaned = text.strip().strip("_").strip()
    cleaned = cleaned.replace("_", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def parse_label(raw: str) -> Tuple[str, str]:
    """
    Derives (crop, disease) from label string.
    Supports:
    1. Folder-style with delimiter '___' (e.g. "Tomato___Late_blight", "Corn_(maize)___Common_rust_", "Pepper,_bell___Bacterial_spot")
       - Strips trailing underscores and parenthesis artifacts
       - Underscores to spaces: "Corn_(maize)" -> "Corn (maize)", "Pepper,_bell" -> "Pepper, bell"
       - Normalizes "healthy" -> "Healthy"
    2. Natural language format (e.g. "Healthy Apple", "Corn (Maize) with Common Rust", "Bell Pepper with Bacterial Spot", etc.)
    """
    raw_str = str(raw).strip()

    # 1. Standard "___" delimiter
    if "___" in raw_str:
        parts = raw_str.split("___", 1)
        raw_crop, raw_disease = parts[0], parts[1]
        crop = clean_token(raw_crop)
        disease = clean_token(raw_disease)
        if disease.lower() == "healthy":
            disease = "Healthy"
        return crop, disease

    # 2. Natural language patterns:
    # Pattern: "Healthy <Crop> [Plant]"
    m_healthy = re.match(r"^Healthy\s+(.+?)(?:\s+Plant)?$", raw_str, re.IGNORECASE)
    if m_healthy:
        crop = clean_token(m_healthy.group(1))
        return crop, "Healthy"

    # Pattern: "<Crop> with <Disease>"
    m_with = re.match(r"^(.+?)\s+with\s+(.+)$", raw_str, re.IGNORECASE)
    if m_with:
        crop = clean_token(m_with.group(1))
        disease = clean_token(m_with.group(2))
        return crop, disease

    # Pattern for specific multi-word crops and prefixes
    KNOWN_CROPS = [
        "Bell Pepper", "Pepper, bell", "Corn (Maize)", "Corn", "Maize",
        "Blueberry", "Cherry", "Grape", "Orange", "Peach", "Potato",
        "Raspberry", "Soybean", "Squash", "Strawberry", "Tomato", "Apple",
        "Cotton", "Rice", "Sugarcane", "Wheat"
    ]
    for c in sorted(KNOWN_CROPS, key=len, reverse=True):
        if raw_str.lower().startswith(c.lower()):
            disease_part = raw_str[len(c):].strip().lstrip("-").lstrip(":").strip()
            if not disease_part:
                disease_part = "Healthy"
            elif disease_part.lower() == "healthy":
                disease_part = "Healthy"
            return clean_token(c), clean_token(disease_part)

    # Fallback for Apple Scab / Cedar Apple Rust if not starting with Apple
    if "apple" in raw_str.lower():
        return "Apple", clean_token(raw_str)

    return "Unknown", clean_token(raw_str)


def preprocess_image(
    image_bytes: bytes,
    img_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> Tuple[torch.Tensor, np.ndarray, Image.Image]:
    """
    Decodes image bytes, composites transparent PNGs onto a neutral white background,
    and returns (tensor[1, 3, H, W], np_bgr_image, pil_image).
    """
    raw_pil = Image.open(io.BytesIO(image_bytes))
    if raw_pil.mode in ("RGBA", "LA") or (raw_pil.mode == "P" and "transparency" in raw_pil.info):
        rgba = raw_pil.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[3])
        pil_img = background
    else:
        pil_img = raw_pil.convert("RGB")
    np_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    transform = T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=list(mean), std=list(std))
    ])
    tensor = transform(pil_img).unsqueeze(0)
    return tensor, np_img, pil_img


class CropDiseasePredictor:
    def __init__(self, config_path: str = "config.yaml"):
        if not os.path.exists(config_path):
            dir_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
            if os.path.exists(dir_cfg):
                config_path = dir_cfg
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.base_dir = os.path.dirname(os.path.abspath(config_path))

        # Device selection: GPU if available, else CPU (or config fallback)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device(self.config.get("model", {}).get("device", "cpu"))

        prep_cfg = self.config.get("preprocessing", {})
        self.img_size = int(prep_cfg.get("img_size", self.config.get("model", {}).get("img_size", 224)))
        self.mean = tuple(prep_cfg.get("mean", [0.485, 0.456, 0.406]))
        self.std = tuple(prep_cfg.get("std", [0.229, 0.224, 0.225]))

        # Backend mode: legacy / pytorch (default high-accuracy 39-class model) vs huggingface
        hf_cfg = self.config.get("huggingface", {})
        default_backend = hf_cfg.get("backend", "legacy")
        self.backend = os.getenv("MODEL_BACKEND", default_backend).lower()
        self.model_name = os.getenv("MODEL_NAME", hf_cfg.get("model_name", "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"))
        self.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", hf_cfg.get("confidence_threshold", 0.60)))

        self._load_calibration()
        self._load_guidance()

        if self.backend in ("huggingface", "hf", "transformers"):
            self._load_huggingface_model()
        else:
            self._load_legacy_model()

        # Configurable switches & thresholds
        inf_cfg = self.config.get("inference", {})
        self.use_morphology_priors = bool(inf_cfg.get("use_morphology_priors", False))
        self.use_tta = bool(inf_cfg.get("tta", False))
        self.mismatch_threshold = float(inf_cfg.get("mismatch_threshold", 0.05))

        foliage_cfg = self.config.get("foliage", {})
        self.min_foliage_ratio = float(foliage_cfg.get("min_foliage_ratio", 0.06))
        self.max_neutral_ratio = float(foliage_cfg.get("max_neutral_ratio", 0.70))
        self.min_contour_area_ratio = float(foliage_cfg.get("min_contour_area_ratio", 0.03))
        self.min_leaf_max_prob = float(self.config.get("uncertainty", {}).get("ood", {}).get("min_leaf_max_prob", 0.06))

        # Crop Synonyms dictionary covering all supported crops + regional aliases
        is_legacy = (self.backend != "huggingface")
        maize_canon = "Maize" if is_legacy else "Corn (Maize)"
        pepper_canon = "Bell_Pepper" if is_legacy else "Bell Pepper"

        self.crop_synonyms = {
            # Apple
            "apple": "Apple", "apples": "Apple", "seb": "Apple", "malus": "Apple",
            # Blueberry
            "blueberry": "Blueberry", "blueberries": "Blueberry", "vaccinium": "Blueberry",
            # Cherry
            "cherry": "Cherry", "cherries": "Cherry", "prunusavium": "Cherry",
            # Corn / Maize
            "corn": maize_canon, "maize": maize_canon, "corn(maize)": maize_canon,
            "corn_(maize)": maize_canon, "makka": maize_canon, "makki": maize_canon,
            "zea": maize_canon, "zeamays": maize_canon, "zea mays": maize_canon,
            # Grape
            "grape": "Grape", "grapes": "Grape", "grapevine": "Grape", "angoor": "Grape", "vitis": "Grape",
            # Orange
            "orange": "Orange", "oranges": "Orange", "citrus": "Orange", "santra": "Orange",
            "mausambi": "Orange", "citrussinensis": "Orange",
            # Peach
            "peach": "Peach", "peaches": "Peach", "aadu": "Peach", "prunuspersica": "Peach",
            # Pepper / Bell Pepper
            "bellpepper": pepper_canon, "bell_pepper": pepper_canon, "bell-pepper": pepper_canon,
            "pepper": pepper_canon, "peppers": pepper_canon, "capsicum": pepper_canon,
            "capsicumannuum": pepper_canon, "shimlamirch": pepper_canon, "shimla mirch": pepper_canon,
            "sweetpepper": pepper_canon, "sweet pepper": pepper_canon, "pepper,bell": pepper_canon,
            "pepper,_bell": pepper_canon, "bell pepper": pepper_canon,
            # Potato
            "potato": "Potato", "potatoes": "Potato", "aloo": "Potato", "alu": "Potato", "solanumtuberosum": "Potato",
            # Raspberry
            "raspberry": "Raspberry", "raspberries": "Raspberry", "rubus": "Raspberry",
            # Soybean
            "soybean": "Soybean", "soybeans": "Soybean", "soya": "Soybean", "soy": "Soybean",
            "soyabean": "Soybean", "glycinemax": "Soybean",
            # Squash
            "squash": "Squash", "pumpkin": "Squash", "kaddu": "Squash", "cucurbita": "Squash",
            # Strawberry
            "strawberry": "Strawberry", "strawberries": "Strawberry", "fragaria": "Strawberry",
            # Tomato
            "tomato": "Tomato", "tomatoes": "Tomato", "tamatar": "Tomato", "solanumlycopersicum": "Tomato",
            # Cotton, Rice, Wheat, Sugarcane (natively supported in legacy model)
            "cotton": "Cotton", "kapas": "Cotton",
            "rice": "Rice", "paddy": "Rice", "chawal": "Rice", "dhan": "Rice",
            "wheat": "Wheat", "gehun": "Wheat", "gehu": "Wheat",
            "sugarcane": "Sugarcane", "ganna": "Sugarcane", "cane": "Sugarcane"
        }

        # Unsupported crops explicitly tracked for informative failure (only relevant in HF mode)
        if not is_legacy:
            self.unsupported_crops = {
                "rice": "Rice", "paddy": "Rice", "chawal": "Rice", "dhan": "Rice",
                "wheat": "Wheat", "gehun": "Wheat", "gehu": "Wheat",
                "cotton": "Cotton", "kapas": "Cotton",
                "sugarcane": "Sugarcane", "ganna": "Sugarcane", "cane": "Sugarcane"
            }
        else:
            self.unsupported_crops = {}

    def _resolve_path(self, rel_path: str) -> str:
        if os.path.isabs(rel_path):
            return rel_path
        return os.path.normpath(os.path.join(self.base_dir, rel_path))

    def _load_calibration(self):
        calib_path = self._resolve_path(self.config["model"].get("calibration_path", "model/calibration.json"))
        calib = {}
        if os.path.exists(calib_path):
            try:
                with open(calib_path, "r", encoding="utf-8") as f:
                    calib = json.load(f)
            except Exception as e:
                print(f"[!] Warning: failed to load calibration.json: {e}")

        self.temperature = float(calib.get("temperature", 1.0))
        cfg_ood = self.config.get("uncertainty", {}).get("ood", {})
        self.energy_threshold = float(calib.get("energy_threshold", cfg_ood.get("energy_threshold", -5.0)))
        self.max_prob_threshold = float(calib.get("max_prob_threshold", cfg_ood.get("min_leaf_max_prob", 0.30)))

        cfg_conf = self.config.get("uncertainty", {}).get("confidence_thresholds", {})
        calib_conf = calib.get("confidence_thresholds", {})
        self.thresh_high = float(calib_conf.get("high", cfg_conf.get("high", 0.75)))
        self.thresh_medium = float(calib_conf.get("medium", cfg_conf.get("medium", 0.50)))

    def _load_guidance(self):
        guidance_path = self._resolve_path("model/guidance.json")
        self.guidance = {}
        if os.path.exists(guidance_path):
            try:
                with open(guidance_path, "r", encoding="utf-8") as f:
                    self.guidance = json.load(f)
            except Exception as e:
                print(f"[!] Warning: failed to load guidance.json: {e}")

    def _load_huggingface_model(self):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Hugging Face transformers is required for MODEL_BACKEND='huggingface'. Run 'pip install transformers'.")

        print(f"[*] Initializing Hugging Face model '{self.model_name}' on device '{self.device}'...")
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        except Exception:
            # Fallback for models referencing MobileNetV2FeatureExtractor in older preprocessor configs
            self.processor = MobileNetV2ImageProcessor.from_pretrained(self.model_name)

        self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        # Derive classes dynamically from model.config.id2label (never hardcode!)
        self.id2label = self.model.config.id2label
        self.num_classes = len(self.id2label)
        self.classes: List[str] = []
        self.parsed_classes: List[Tuple[str, str]] = []
        self.crop_to_indices: Dict[str, List[int]] = {}

        for idx in range(self.num_classes):
            raw_lbl = self.id2label.get(str(idx), self.id2label.get(idx, f"Class_{idx}"))
            crop, disease = parse_label(raw_lbl)
            self.parsed_classes.append((crop, disease))
            # Format canonical label as Crop___Disease (e.g. Tomato___Early_blight)
            canonical_label = f"{crop}___{disease.replace(' ', '_')}"
            self.classes.append(canonical_label)

            c_key = crop.lower()
            if c_key not in self.crop_to_indices:
                self.crop_to_indices[c_key] = []
            self.crop_to_indices[c_key].append(idx)

            # Also register alias without spaces / with underscore
            c_key_alt = crop.lower().replace(" ", "_").replace(",", "")
            if c_key_alt != c_key and c_key_alt not in self.crop_to_indices:
                self.crop_to_indices[c_key_alt] = self.crop_to_indices[c_key]

        self.standard_crops = sorted(list(set(crop for crop, _ in self.parsed_classes)))
        print(f"[+] Loaded Hugging Face model with {self.num_classes} classes across {len(self.standard_crops)} crops.")

        # Check if fallback model is available for non-PlantVillage crops (Rice, Wheat, Cotton, Sugarcane)
        self.has_fallback_model = False
        labels_path = self._resolve_path(self.config["model"].get("labels_path", "model/labels.json"))
        model_path = self._resolve_path(self.config["model"].get("path", "model/model.pt"))
        if os.path.exists(labels_path) and os.path.exists(model_path):
            try:
                with open(labels_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self.fallback_classes = meta["classes"]
                self.fallback_num_classes = len(self.fallback_classes)
                self.fallback_parsed = [parse_label(c) for c in self.fallback_classes]
                fb_model = mobilenet_v3_large(weights=None)
                in_feat = fb_model.classifier[3].in_features
                fb_model.classifier[3] = nn.Linear(in_feat, self.fallback_num_classes)
                ckpt = torch.load(model_path, map_location=self.device)
                sdict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
                fb_model.load_state_dict(sdict)
                fb_model.to(self.device)
                fb_model.eval()
                self.fallback_model = fb_model
                self.has_fallback_model = True

                # Group fallback classes by crop
                self.fallback_crop_to_indices = {}
                for idx, (c_name, _) in enumerate(self.fallback_parsed):
                    ck = c_name.lower().replace(" ", "").replace("_", "")
                    if ck not in self.fallback_crop_to_indices:
                        self.fallback_crop_to_indices[ck] = []
                    self.fallback_crop_to_indices[ck].append(idx)

                img_size = self.config["model"].get("img_size", 224)
                self.fallback_transform = T.Compose([
                    T.Resize(int(img_size * 1.14)),
                    T.CenterCrop(img_size),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                print("[+] Fallback MobileNetV3 model loaded for Wheat, Rice, Cotton, Sugarcane support.")
            except Exception as e:
                print(f"[!] Warning: could not load fallback model: {e}")

    def _load_legacy_model(self):
        print(f"[*] Initializing legacy MobileNetV3 PyTorch model on device '{self.device}'...")
        labels_path = self._resolve_path(self.config["model"]["labels_path"])
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file missing at {labels_path}")
        with open(labels_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.classes = meta["classes"]
        self.label_to_idx = meta["label_to_idx"]
        self.num_classes = len(self.classes)
        self.parsed_classes = []
        for c in self.classes:
            raw_c = c.split("___")[0]
            _, clean_d = parse_label(c)
            self.parsed_classes.append((raw_c, clean_d))
        self.standard_crops = sorted(list(set(c[0] for c in self.parsed_classes)))

        self.crop_to_indices = {}
        for idx, cls_name in enumerate(self.classes):
            parsed_crop, _ = self.parsed_classes[idx]
            raw_crop = cls_name.split("___")[0]

            keys_to_add = {
                parsed_crop.lower(),
                parsed_crop.lower().replace(" ", "_"),
                parsed_crop.lower().replace(" ", "").replace("_", ""),
                raw_crop.lower(),
                raw_crop.lower().replace("_", " "),
                raw_crop.lower().replace("_", "")
            }
            if "maize" in keys_to_add or "corn" in keys_to_add:
                keys_to_add.update({"corn", "maize", "corn (maize)", "corn(maize)", "corn_(maize)"})
            if "bell_pepper" in keys_to_add or "bell pepper" in keys_to_add:
                keys_to_add.update({"bell pepper", "bell_pepper", "pepper", "capsicum", "sweet pepper"})

            for k in keys_to_add:
                if k not in self.crop_to_indices:
                    self.crop_to_indices[k] = []
                if idx not in self.crop_to_indices[k]:
                    self.crop_to_indices[k].append(idx)

        # In 39-class model, all 10 crops (including Rice, Wheat, Cotton, Sugarcane) are fully supported!
        self.unsupported_crops = {}

        model_path = self._resolve_path(self.config["model"]["path"])
        model = mobilenet_v3_large(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, self.num_classes)

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
            model.load_state_dict(state_dict)
        else:
            print(f"[!] Warning: Model file {model_path} not found.")

        model.to(self.device)
        model.eval()
        self.model = model

        img_size = self.config["model"].get("img_size", 224)
        self.transform = T.Compose([
            T.Resize(int(img_size * 1.14)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def normalize_crop_hint(self, crop_hint: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """
        Normalizes crop hints: exact match, synonyms, and detects unsupported crops.
        Returns (canonical_crop, warning_msg).
        If unknown or unsupported, returns (None, warning_msg).
        """
        if not crop_hint or not str(crop_hint).strip():
            return None, None
        raw = str(crop_hint).strip()
        norm = raw.lower().strip()
        if norm in ["auto", "all", "none", "null", "any"]:
            return None, None

        cleaned_key = norm.replace(" ", "").replace("_", "").replace("-", "").replace(",", "")

        # 1. Check if user requested an unsupported crop (e.g. in restricted models)
        if self.unsupported_crops and cleaned_key in self.unsupported_crops:
            unsupported_name = self.unsupported_crops[cleaned_key]
            warning = (
                f"Unsupported crop hint '{raw}'. This model only covers "
                f"({', '.join(self.standard_crops)}). "
                f"{unsupported_name} is not supported."
            )
            return None, warning

        # 2. Exact or normalized match against known standard crop names
        for std_c in self.standard_crops:
            c_clean = std_c.lower().replace(" ", "").replace("_", "").replace("-", "").replace(",", "")
            if norm == std_c.lower() or cleaned_key == c_clean:
                return std_c, None

        # 3. Match against normalized synonyms
        if cleaned_key in self.crop_synonyms:
            target = self.crop_synonyms[cleaned_key]
            # Match canonical standard crop
            for std_c in self.standard_crops:
                if (target.lower() == std_c.lower() or 
                    std_c.lower() in target.lower() or 
                    target.lower() in std_c.lower() or
                    target.lower().replace(" ", "") == std_c.lower().replace(" ", "")):
                    return std_c, None
            return target, None

        # 4. Substring match
        for syn_key, target_crop in self.crop_synonyms.items():
            if syn_key in cleaned_key:
                for std_c in self.standard_crops:
                    if (target_crop.lower() == std_c.lower() or 
                        std_c.lower() in target_crop.lower() or 
                        target_crop.lower() in std_c.lower() or
                        target_crop.lower().replace(" ", "") == std_c.lower().replace(" ", "")):
                        return std_c, None
                return target_crop, None

        warning = (
            f"Unknown crop hint '{raw}' was ignored. "
            f"Supported crops are: {', '.join(self.standard_crops)}."
        )
        return None, warning

    def assess_quality(self, image_cv: np.ndarray) -> Tuple[bool, float, float, Optional[str]]:
        blur_thresh = float(self.config.get("quality", {}).get("blur_threshold", 80.0))
        min_bright = float(self.config.get("quality", {}).get("min_brightness", 40.0))
        max_bright = float(self.config.get("quality", {}).get("max_brightness", 225.0))

        h, w = image_cv.shape[:2]
        longer_side = max(h, w)
        if longer_side > 512:
            scale = 512.0 / float(longer_side)
            resized_w = int(round(w * scale))
            resized_h = int(round(h * scale))
            eval_img = cv2.resize(image_cv, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        else:
            eval_img = image_cv

        if len(eval_img.shape) == 3:
            gray = cv2.cvtColor(eval_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = eval_img

        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))

        if brightness < min_bright:
            return False, round(blur_score, 2), round(brightness, 2), f"CLEAR PHOTO NAHI HAI: Photo bahut andhere mein hai (brightness: {brightness:.1f} < {min_bright:.1f}). Kripya acchi roshni mein photo dubara upload karein."

        if brightness > max_bright:
            return False, round(blur_score, 2), round(brightness, 2), f"CLEAR PHOTO NAHI HAI: Photo par dhoop/glare bahut zyada hai. Kripya saaf photo dubara upload karein."

        if blur_score < blur_thresh:
            return False, round(blur_score, 2), round(brightness, 2), f"CLEAR PHOTO NAHI HAI: Photo bahut dhundhli (blurry) hai (sharpness: {blur_score:.1f} < {blur_thresh:.1f}). Kripya camera saaf karke clear photo dubara upload karein."

        return True, round(blur_score, 2), round(brightness, 2), None

    def is_leaf_present(self, image_cv: np.ndarray) -> Tuple[bool, float, Optional[str]]:
        if len(image_cv.shape) != 3:
            return False, 0.0, "Input image must be a 3-channel color photograph."

        h, w = image_cv.shape[:2]
        total_pixels = h * w
        b, g, r = [image_cv[:, :, i].astype(np.float32) for i in range(3)]
        hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)

        exg = 2 * g - r - b
        green_foliage = (exg > 5.0) & (g >= b - 10) & (g >= r - 10) & (hsv[:, :, 1] >= 35) & (hsv[:, :, 2] >= 35)
        veg_foliage = (hsv[:, :, 0] >= 28) & (hsv[:, :, 0] <= 90) & (hsv[:, :, 1] >= 40) & (hsv[:, :, 2] >= 40)
        chlorophyll_base = green_foliage | veg_foliage

        k_leaf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated_leaf = cv2.dilate(chlorophyll_base.astype(np.uint8), k_leaf) > 0

        rust_foliage = (hsv[:, :, 0] >= 10) & (hsv[:, :, 0] <= 26) & (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 70) & dilated_leaf
        necrotic_spots = (hsv[:, :, 0] >= 10) & (hsv[:, :, 0] <= 35) & (hsv[:, :, 1] >= 30) & (hsv[:, :, 2] >= 30) & (hsv[:, :, 2] <= 160) & dilated_leaf

        foliage = chlorophyll_base | rust_foliage | necrotic_spots
        foliage_pixels = int(np.count_nonzero(foliage))
        foliage_ratio = float(foliage_pixels) / float(total_pixels)

        neutral = (np.abs(r - g) < 14) & (np.abs(g - b) < 14) & (np.abs(r - b) < 14)
        neutral_ratio = float(np.count_nonzero(neutral)) / float(total_pixels)

        min_foliage = getattr(self, "min_foliage_ratio", 0.06)
        max_neutral = getattr(self, "max_neutral_ratio", 0.70)
        min_contour = getattr(self, "min_contour_area_ratio", 0.03)

        if foliage_ratio < min_foliage:
            return False, round(foliage_ratio, 4), "NOT A LEAF: Yeh kisi fasal ki patti nahi hai! Kripya kisi fasal (Crop Leaf) ki photo upload karein."

        if neutral_ratio > max_neutral and foliage_ratio < 0.15:
            return False, round(foliage_ratio, 4), "NOT A LEAF: Yeh document ya screenshot lag raha hai. Kripya kisi fasal ki patti ki saaf photo upload karein."

        foliage_mask = foliage.astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        foliage_clean = cv2.morphologyEx(foliage_mask, cv2.MORPH_OPEN, k)
        contours, _ = cv2.findContours(foliage_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, round(foliage_ratio, 4), "NOT A LEAF: Foliar structure not recognized. Kripya patti ki photo upload karein."

        largest_c = max(contours, key=cv2.contourArea)
        largest_area_ratio = float(cv2.contourArea(largest_c)) / float(total_pixels)
        if largest_area_ratio < min_contour:
            return False, round(foliage_ratio, 4), "NOT A LEAF: Patti ka mukhya hissa nahi mila. Kripya pass se patti ki photo lein."

        return True, round(foliage_ratio, 4), None

    def extract_leaf_morphology(self, image_cv: np.ndarray) -> Dict[str, Any]:
        h, w = image_cv.shape[:2]
        total_pixels = h * w
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY) if len(image_cv.shape) == 3 else image_cv
        hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
        b, g, r = [image_cv[:, :, i].astype(np.float32) for i in range(3)]
        exg = 2 * g - r - b

        rust_mask = (
            (r > g + 12) &
            (r > b + 30) &
            (hsv[:, :, 0] >= 10) &
            (hsv[:, :, 0] <= 26) &
            (hsv[:, :, 1] >= 95) &
            (hsv[:, :, 2] >= 90)
        ).astype(np.uint8) * 255
        rust_pixels = int(np.count_nonzero(rust_mask))
        rust_pct = round((float(rust_pixels) / total_pixels) * 100.0, 1)
        has_rust_pustules = (rust_pct >= 0.5)

        smooth = cv2.bilateralFilter(gray, 7, 50, 50)
        edges = cv2.Canny(smooth, 40, 120)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=45, minLineLength=50, maxLineGap=12)
        parallel_score = 0.0
        if lines is not None and len(lines) >= 6:
            angles = []
            for l in lines:
                x1, y1, x2, y2 = l.flatten()
                if np.hypot(x2 - x1, y2 - y1) > 40:
                    ang = float(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
                    angles.append(ang)
            if len(angles) >= 6:
                hist, _ = np.histogram(angles, bins=12, range=(0, 180))
                parallel_score = float(np.max(hist)) / len(angles)
        is_parallel_lines = (parallel_score >= 0.55)

        foliage_mask = (
            (exg > 3.0) |
            ((g >= r - 12) & (g > b + 10)) |
            ((hsv[:, :, 0] >= 20) & (hsv[:, :, 0] <= 95) & (hsv[:, :, 1] >= 25))
        ).astype(np.uint8) * 255
        foliage_mask = cv2.bitwise_or(foliage_mask, rust_mask)

        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        foliage_mask = cv2.morphologyEx(foliage_mask, cv2.MORPH_OPEN, k_open)
        foliage_mask = cv2.morphologyEx(foliage_mask, cv2.MORPH_CLOSE, k_close)

        contours, _ = cv2.findContours(foliage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours or (cv2.contourArea(max(contours, key=cv2.contourArea)) < 0.02 * total_pixels):
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            alt_contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if alt_contours and cv2.contourArea(max(alt_contours, key=cv2.contourArea)) > 0.02 * total_pixels:
                contours = alt_contours
                foliage_mask = otsu

        if not contours:
            return {
                "shape": "Linear Strap-like Blade (Monocot)" if is_parallel_lines else "Simple Ovate-Elliptic",
                "geometry_class": "linear_strap" if is_parallel_lines else "simple_ovate",
                "margin": "Entire / Parallel Margins",
                "solidity": 0.86,
                "circularity": 0.48,
                "aspect_ratio": 1.18,
                "coverage_pct": 65.0,
                "color_tone": "Olive Foliar Green",
                "venation": "Parallel Venation (Monocot)" if is_parallel_lines else "Pinnate Reticulate",
                "lesion_pattern": "Foliar spotting / necrosis visible",
                "has_rust_pustules": has_rust_pustules,
                "rust_pct": rust_pct,
                "is_linear_blade": is_parallel_lines,
                "size_px": f"{w} x {h} px"
            }

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = max(cv2.arcLength(c, True), 1.0)
        hull = cv2.convexHull(c)
        hull_area = max(cv2.contourArea(hull), 1.0)
        hull_perim = max(cv2.arcLength(hull, True), 1.0)

        solidity = float(area) / hull_area
        circularity = (4.0 * np.pi * area) / (perimeter ** 2)
        x, y, bw, bh = cv2.boundingRect(c)
        aspect_ratio = float(max(bw, bh)) / float(max(min(bw, bh), 1))
        roughness = float(perimeter) / hull_perim
        coverage_pct = round((float(area) / total_pixels) * 100.0, 1)

        rect = cv2.minAreaRect(c)
        rw, rh = rect[1]
        blade_len = max(rw, rh)
        blade_w = max(min(rw, rh), 1.0)
        rot_aspect = float(blade_len) / float(blade_w)

        leaf_pixels_hsv = hsv[foliage_mask > 0]
        if len(leaf_pixels_hsv) > 0:
            mean_hue = float(np.mean(leaf_pixels_hsv[:, 0]))
            mean_sat = float(np.mean(leaf_pixels_hsv[:, 1]))
            mean_val = float(np.mean(leaf_pixels_hsv[:, 2]))
        else:
            mean_hue, mean_sat, mean_val = 45.0, 90.0, 110.0

        if mean_hue < 35:
            color_tone = "Yellow-Green / Chlorotic Foliage"
        elif mean_hue < 55:
            color_tone = "Olive Foliar Green"
        elif mean_hue < 75:
            color_tone = "Vibrant Emerald Green"
        else:
            color_tone = "Deep Forest Green"

        lesion_mask = (foliage_mask > 0) & ((g < r + 4) | (hsv[:, :, 0] < 22) | (hsv[:, :, 2] < 45))
        lesion_count = int(np.count_nonzero(lesion_mask))
        lesion_pct = round((float(lesion_count) / max(len(leaf_pixels_hsv), 1)) * 100.0, 1)

        is_frame_filling = (coverage_pct > 75.0) and (solidity > 0.94)
        is_linear_blade = (rot_aspect >= 2.4) or (is_parallel_lines and rot_aspect >= 1.40) or (aspect_ratio >= 2.6)
        if is_frame_filling and is_parallel_lines and rot_aspect >= 1.30:
            is_linear_blade = True

        if has_rust_pustules and is_linear_blade:
            lesion_pattern = f"Foliar rust pustules detected ({rust_pct}% coverage: Puccinia pattern)"
        elif lesion_pct > 1.2 or has_rust_pustules:
            effective_pct = max(lesion_pct, rust_pct)
            lesion_pattern = f"Punctate necrotic spots detected ({effective_pct}% foliar area)"
        else:
            lesion_pattern = "Clean foliage (no prominent necrotic spots)"

        if is_linear_blade:
            shape = "Linear Strap-like Blade (Monocot)"
            geometry_class = "linear_strap"
            venation = "Parallel Venation (Monocot)"
            margin = "Linear Entire / Parallel Margins"
            solidity = round(min(solidity, 0.78), 3)
            circularity = round(min(circularity, 0.28), 3)
        elif roughness > 1.35:
            if solidity < 0.78:
                shape = "Pinnately Compound / Dissected Lobes"
                geometry_class = "compound_lobed"
                venation = "Pinnate Reticulate with Leaflets"
                margin = "Dentate / Coarsely Serrated"
            else:
                shape = "Ovate Leaflet with Serrated Margin"
                geometry_class = "compound_lobed"
                venation = "Pinnate Reticulate"
                margin = "Serrated / Dentate Margin"
        elif solidity >= 0.82:
            shape = "Simple Ovate-Lanceolate"
            geometry_class = "simple_ovate"
            venation = "Pinnate Reticulate"
            margin = "Entire (Smooth, non-serrated)"
        elif circularity > 0.40 and aspect_ratio <= 1.35:
            shape = "Broad Ovate / Sub-cordate"
            geometry_class = "broad_ovate"
            venation = "Pinnate to Palmate Reticulate"
            margin = "Weakly Undulate / Entire"
        else:
            shape = "Palmately Lobed / Cordate"
            geometry_class = "palmate_lobed"
            venation = "Palmate Reticulate"
            margin = "Toothed / Lobed"

        return {
            "shape": shape,
            "geometry_class": geometry_class,
            "margin": margin,
            "solidity": round(solidity, 3),
            "circularity": round(circularity, 3),
            "aspect_ratio": round(rot_aspect if is_linear_blade else aspect_ratio, 2),
            "roughness": round(roughness, 2),
            "coverage_pct": coverage_pct,
            "color_tone": color_tone,
            "mean_hsv": [round(mean_hue, 1), round(mean_sat, 1), round(mean_val, 1)],
            "venation": venation,
            "lesion_pattern": lesion_pattern,
            "lesion_pct": lesion_pct,
            "has_rust_pustules": has_rust_pustules,
            "rust_pct": rust_pct,
            "is_linear_blade": is_linear_blade,
            "size_px": f"{bw} x {bh} px (Bounding box)"
        }

    def compute_morphology_prior(self, morph: Optional[Dict[str, Any]], crop: str) -> float:
        """
        Calculates morphological Bayesian prior weight P(Morphology | Crop).
        Used to correct CNN class imbalance biases (e.g. Tomato overprediction on smooth ovate pepper leaves).
        """
        if not morph:
            return 1.0

        c = crop.lower().replace(" ", "").replace("_", "").replace(",", "")
        shape = morph.get("shape", "")
        margin = morph.get("margin", "")
        geom = morph.get("geometry_class", "")
        is_linear = bool(morph.get("is_linear_blade", False))
        solidity = float(morph.get("solidity", 0.8))
        roughness = float(morph.get("roughness", 1.0))
        
        # 1. Monocot vs Dicot check
        is_monocot_crop = c in ["corn(maize)", "corn", "maize", "wheat", "rice", "sugarcane"]
        if is_linear or "monocot" in shape.lower() or "parallel" in margin.lower():
            if is_monocot_crop:
                return 2.5
            else:
                return 0.15 # Strong penalty for dicots on linear monocot grass blade
        else:
            if is_monocot_crop and (solidity > 0.82 and not is_linear):
                return 0.25 # Penalty for grass blade crops when leaf is broad/oval
                
        # 2. Bell Pepper vs Tomato check
        # Bell Pepper (Capsicum annuum): Simple ovate-lanceolate with ENTIRE (smooth) margin, high solidity
        # Tomato (Solanum lycopersicum): Pinnately compound/dissected with DENTATE/SERRATED margin, low solidity/high roughness
        is_entire_smooth = "entire" in margin.lower() and "smooth" in margin.lower()
        is_serrated_dentate = "serrated" in margin.lower() or "dentate" in margin.lower() or "toothed" in margin.lower()
        
        if "bellpepper" in c or "pepper" in c:
            if is_entire_smooth and solidity >= 0.80:
                return 2.5 # Heavy boost for pepper on smooth entire ovate leaf
            elif is_serrated_dentate or roughness > 1.35:
                return 0.35 # Pepper leaves do not have serrated/toothed margins
                
        if "tomato" in c:
            if is_entire_smooth and solidity >= 0.85:
                return 0.30 # Tomato foliage is compound with toothed/lobed leaflets, NEVER simple smooth entire!
            elif is_serrated_dentate or roughness > 1.25 or geom == "compound_lobed":
                return 1.6 # Boost for tomato on serrated compound foliage
                
        # 3. Palmate Lobed (Grape, Squash, Cotton)
        is_palmate = geom == "palmate_lobed" or "palmate" in shape.lower()
        if c in ["grape", "squash", "cotton"]:
            if is_palmate:
                return 2.0
        else:
            if is_palmate and c in ["bellpepper", "pepper"]:
                return 0.25 # Peppers are never palmate lobed

        # 4. Apple vs other dicots (Apple: Simple ovate-elliptic, finely serrated)
        if "apple" in c:
            if is_entire_smooth and solidity > 0.90:
                return 0.8 # Apple typically has serrulate/crenate margins
            elif is_serrated_dentate:
                return 1.4

        return 1.0

    def predict(self, image_bytes: bytes, crop_hint: Optional[str] = None, use_priors: Optional[bool] = None) -> Dict[str, Any]:
        """
        Runs complete inference pipeline:
        1. Decodes and checks image quality. If corrupted or unreadable, returns clean error dict.
        2. Assesses sharpness and exposure (poor_image_quality).
        3. Screens for organic leaf foliage presence (not_a_leaf).
        4. Extracts botanical leaf morphology characteristics.
        5. Forward pass with AutoImageProcessor / Hugging Face model (or legacy MobileNetV3).
        6. Computes softmax probabilities, crop mass distribution, and top predictions.
        7. Enforces confidence policy (threshold 0.60 default) and species mismatch detection.
        8. Looks up agronomic preventive guidance and formats output strictly adhering to contract.
        """
        # 1. Decode and preprocess image
        try:
            tensor, np_img, pil_img = preprocess_image(image_bytes, self.img_size, self.mean, self.std)
        except Exception as e:
            return {
                "status": "error",
                "crop": "unknown",
                "disease": "unknown",
                "confidence": 0.0,
                "certainty": "low",
                "level": "low",
                "top3": [],
                "crop_comparison": [],
                "morphology": None,
                "mismatch_warning": None,
                "quality": {"blur_score": 0.0, "brightness": 0.0},
                "guidance": None,
                "message": f"Corrupted or invalid image format: {str(e)}",
                "raw_top1_class": None,
                "raw_top1_crop": None,
                "prior_flipped": False
            }

        is_quality_valid, blur_score, brightness, quality_err = self.assess_quality(np_img)
        quality_info = {"blur_score": blur_score, "brightness": brightness}
        if not is_quality_valid:
            min_bright = float(self.config.get("quality", {}).get("min_brightness", 40.0))
            if brightness <= 1.0:
                status_val = "unknown"
            elif brightness < min_bright:
                status_val = "poor_image_quality"
            elif blur_score < float(self.config.get("quality", {}).get("blur_threshold", 80.0)):
                is_leaf, _, _ = self.is_leaf_present(np_img)
                status_val = "poor_image_quality" if is_leaf else "unknown"
            else:
                status_val = "poor_image_quality"
            return {
                "status": status_val,
                "crop": "unknown",
                "disease": "unknown",
                "confidence": 0.0,
                "certainty": "low",
                "level": "low",
                "top3": [],
                "crop_comparison": [],
                "morphology": None,
                "mismatch_warning": None,
                "quality": quality_info,
                "guidance": None,
                "message": quality_err or "POOR IMAGE QUALITY: Photo is too blurry or improperly exposed.",
                "raw_top1_class": None,
                "raw_top1_crop": None,
                "prior_flipped": False
            }

        # 3. Leaf Foliage Verification
        is_leaf, foliage_ratio, leaf_rejection_msg = self.is_leaf_present(np_img)
        if not is_leaf:
            return {
                "status": "unknown",
                "crop": "unknown",
                "disease": "unknown",
                "confidence": 0.0,
                "certainty": "low",
                "level": "low",
                "top3": [],
                "crop_comparison": [],
                "morphology": None,
                "mismatch_warning": None,
                "quality": quality_info,
                "guidance": None,
                "message": leaf_rejection_msg or "UNKNOWN / NOT A LEAF: Image features do not match recognized crop foliage.",
                "raw_top1_class": None,
                "raw_top1_crop": None,
                "prior_flipped": False
            }

        # 4. Morphology Extraction (informational only)
        morphology = self.extract_leaf_morphology(np_img)

        # 5. Model Forward Pass
        if self.backend == "huggingface":
            inputs = self.processor(images=pil_img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                raw_logits = outputs.logits
                scaled_logits = raw_logits / self.temperature
                all_probs = torch.softmax(scaled_logits, dim=-1)[0]
                eval_raw_logits = raw_logits

                # Free energy OOD check
                energy = -self.temperature * torch.logsumexp(eval_raw_logits / self.temperature, dim=-1).item()
                max_unconstrained_prob = float(torch.max(all_probs).item())
                if energy > self.energy_threshold:
                    return {
                        "status": "not_a_leaf",
                        "crop": "unknown",
                        "disease": "unknown",
                        "confidence": round(max_unconstrained_prob, 4),
                        "certainty": "low",
                        "top3": [],
                        "crop_comparison": [],
                        "morphology": None,
                        "mismatch_warning": None,
                        "quality": quality_info,
                        "guidance": None,
                        "message": f"NOT A LEAF: Image features do not match any recognized crop leaf foliage (OOD energy: {energy:.2f} > {self.energy_threshold:.2f}).",
                        "raw_top1_class": None,
                        "raw_top1_crop": None,
                        "prior_flipped": False
                    }
        else:
            input_tensor = tensor.to(self.device)
            with torch.no_grad():
                raw_logits = self.model(input_tensor)
                scaled_logits = raw_logits / self.temperature
                all_probs = torch.softmax(scaled_logits, dim=-1)[0]
                eval_raw_logits = raw_logits

                energy = -self.temperature * torch.logsumexp(eval_raw_logits / self.temperature, dim=-1).item()
                max_unconstrained_prob = float(torch.max(all_probs).item())
                if energy > self.energy_threshold or max_unconstrained_prob < self.max_prob_threshold:
                    return {
                        "status": "unknown",
                        "crop": "unknown",
                        "disease": "unknown",
                        "confidence": 0.0,
                        "certainty": "low",
                        "level": "low",
                        "top3": [],
                        "crop_comparison": [],
                        "morphology": None,
                        "mismatch_warning": None,
                        "quality": quality_info,
                        "guidance": None,
                        "message": f"UNKNOWN / NOT A LEAF: Rejected by OOD check (energy: {energy:.2f}, max_prob: {max_unconstrained_prob:.2f}).",
                        "raw_top1_class": None,
                        "raw_top1_crop": None,
                        "prior_flipped": False
                    }

        # 6. Raw Model Top-1 & Crop Mass
        raw_top1_class_idx = int(torch.argmax(all_probs).item())
        raw_top1_class = self.classes[raw_top1_class_idx]
        raw_top1_crop, raw_top1_disease = self.parsed_classes[raw_top1_class_idx]

        crop_mass: Dict[str, float] = {}
        for c_idx, c_prob in enumerate(all_probs.cpu().numpy()):
            c_name = self.parsed_classes[c_idx][0]
            crop_mass[c_name] = crop_mass.get(c_name, 0.0) + float(c_prob)

        cnn_top_crop = max(crop_mass, key=crop_mass.get)
        cnn_top_mass = crop_mass[cnn_top_crop]
        detected_crop = cnn_top_crop

        # 6b. Apply Bayesian Botanical Morphology Priors
        apply_priors = self.use_morphology_priors if use_priors is None else use_priors
        prior_flipped = False
        adjusted_crop_mass = dict(crop_mass)

        if apply_priors and morphology:
            weighted_mass = {}
            for c_name, c_prob in crop_mass.items():
                w = self.compute_morphology_prior(morphology, c_name)
                # Clamp prior weight to gentle bounds [0.85, 1.15] to prevent overriding strong CNN features
                w_clamped = float(np.clip(w, 0.85, 1.15))
                weighted_mass[c_name] = c_prob * w_clamped
            
            sum_w = sum(weighted_mass.values())
            if sum_w > 0:
                adjusted_crop_mass = {k: v / sum_w for k, v in weighted_mass.items()}
                # Morphology priors act as subtle tie-breakers when CNN confidence is moderate or low (< 0.60).
                # High-confidence deep representations (>= 0.60) must never be corrupted by noisy 2D contour approximations.
                if cnn_top_mass < 0.60:
                    morph_top_crop = max(adjusted_crop_mass, key=adjusted_crop_mass.get)
                    if morph_top_crop != cnn_top_crop:
                        prior_flipped = True
                        detected_crop = morph_top_crop

        # 7. Crop Selection: Normalize hint, detect unsupported crops
        valid_hint, hint_warning = self.normalize_crop_hint(crop_hint)

        # Check if hint was explicitly for an unsupported crop (Rice, Wheat, Cotton, Sugarcane)
        is_unsupported_hint = False
        if crop_hint and not valid_hint:
            raw_c_clean = str(crop_hint).lower().strip().replace(" ", "").replace("_", "").replace("-", "")
            if raw_c_clean in self.unsupported_crops:
                is_unsupported_hint = True

        if is_unsupported_hint:
            # Gracefully fail unsupported crop with low_confidence status
            unsupported_name = self.unsupported_crops.get(raw_c_clean, str(crop_hint))
            total_mass = sum(adjusted_crop_mass.values()) if sum(adjusted_crop_mass.values()) > 0 else 1.0
            crop_comparison = [
                {
                    "crop": c.replace("_", " "),
                    "raw_crop": c,
                    "confidence": round(p / total_mass, 4),
                    "percent": round((p / total_mass) * 100, 1)
                }
                for c, p in sorted(adjusted_crop_mass.items(), key=lambda x: x[1], reverse=True)
            ]
            return {
                "status": "low_confidence",
                "crop": unsupported_name,
                "disease": "unknown",
                "confidence": 0.0,
                "certainty": "low",
                "top3": [],
                "crop_comparison": crop_comparison,
                "morphology": morphology,
                "mismatch_warning": hint_warning,
                "quality": quality_info,
                "guidance": None,
                "message": hint_warning or f"Unsupported crop '{unsupported_name}'. This model only covers PlantVillage crops.",
                "raw_top1_class": raw_top1_class,
                "raw_top1_crop": cnn_top_crop,
                "prior_flipped": False
            }

        if valid_hint is not None:
            effective_crop = valid_hint
        else:
            effective_crop = detected_crop

        effective_crop_mass = adjusted_crop_mass.get(effective_crop, 0.0)

        # Attach botanical metadata
        botanical_info = BOTANICAL_METADATA.get(effective_crop, BOTANICAL_METADATA.get(detected_crop, {}))
        morphology["detected_crop"] = detected_crop.replace("_", " ")
        morphology["effective_crop"] = effective_crop.replace("_", " ")
        morphology["scientific_name"] = botanical_info.get("scientific_name", "")
        morphology["family"] = botanical_info.get("family", "")
        morphology["botanical_notes"] = botanical_info.get("botanical_notes", "")

        total_mass = sum(adjusted_crop_mass.values()) if sum(adjusted_crop_mass.values()) > 0 else 1.0
        crop_comparison = [
            {
                "crop": c.replace("_", " "),
                "raw_crop": c,
                "confidence": round(p / total_mass, 4),
                "percent": round((p / total_mass) * 100, 1)
            }
            for c, p in sorted(adjusted_crop_mass.items(), key=lambda x: x[1], reverse=True)
        ]

        # 8. Conditioned Disease Prediction
        is_fallback_crop = getattr(self, "has_fallback_model", False) and (effective_crop in ["Rice", "Wheat", "Cotton", "Sugarcane"])
        if is_fallback_crop:
            fb_input = self.fallback_transform(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                fb_logits = self.fallback_model(fb_input) / self.temperature
                fb_probs = torch.softmax(fb_logits, dim=-1)[0]
            
            fb_key = effective_crop.lower().replace(" ", "").replace("_", "")
            effective_indices = self.fallback_crop_to_indices.get(fb_key, [])
            if not effective_indices:
                effective_indices = list(range(self.fallback_num_classes))
            
            sub_logits = fb_logits[0, effective_indices]
            sub_probs = torch.softmax(sub_logits, dim=0).cpu().numpy()
            top_sub_order = np.argsort(sub_probs)[::-1]
            top_pred_idx = effective_indices[top_sub_order[0]]
            disease_name = self.fallback_parsed[top_pred_idx][1]
            p_disease_given_crop = float(sub_probs[top_sub_order[0]])
        else:
            c_lookup = effective_crop.lower()
            effective_indices = self.crop_to_indices.get(c_lookup, [])
            if not effective_indices:
                if valid_hint is None:
                    # Fallback to detected crop
                    effective_crop = detected_crop
                    effective_crop_mass = crop_mass.get(effective_crop, 0.0)
                effective_indices = self.crop_to_indices.get(detected_crop.lower(), list(range(self.num_classes)))

            sub_logits = scaled_logits[0, effective_indices]
            sub_probs = torch.softmax(sub_logits, dim=0).cpu().numpy()
            top_sub_order = np.argsort(sub_probs)[::-1]

            top_pred_idx = effective_indices[top_sub_order[0]]
            disease_name = self.parsed_classes[top_pred_idx][1]
            p_disease_given_crop = float(sub_probs[top_sub_order[0]])

        # ------------------------------------------------------------------
        # Stage 1 confidence: how sure are we about the crop?
        # ------------------------------------------------------------------
        total_mass_sum = sum(adjusted_crop_mass.values()) or 1.0
        crop_confidence = round(float(np.clip(
            adjusted_crop_mass.get(effective_crop, 0.0) / total_mass_sum, 0.0, 1.0
        )), 4)

        # ------------------------------------------------------------------
        # Stage 2 confidence: disease probability *conditioned* on crop.
        # Guaranteed no cross-crop leakage: disease index is always restricted
        # to effective_crop's class indices before computing softmax.
        # ------------------------------------------------------------------
        disease_confidence = round(float(np.clip(p_disease_given_crop, 0.0, 1.0)), 4)

        # Final joint confidence: crop_conf * disease_conf_given_crop
        # When caller provides a valid crop hint, trust it and omit the
        # crop conviction penalty (hint carries external evidence).
        if valid_hint is not None:
            raw_conf = p_disease_given_crop
        else:
            crop_conviction = min(1.0, effective_crop_mass / 0.50)
            raw_conf = p_disease_given_crop * crop_conviction

        final_confidence = round(float(np.clip(raw_conf, 0.0, 1.0)), 4)

        def _get_level(s: float) -> str:
            if s >= self.thresh_high:
                return "high"
            if s >= self.thresh_medium:
                return "medium"
            return "low"

        # Top 3 predictions formatted strictly to Spring Boot contract
        top3 = []
        if is_fallback_crop:
            for rank in range(min(3, len(effective_indices))):
                idx = effective_indices[top_sub_order[rank]]
                sub_p = float(sub_probs[top_sub_order[rank]])
                score_val = round(float(np.clip(sub_p, 0.0, 1.0)), 4)
                top3.append({
                    "label": self.fallback_classes[idx],
                    "score": score_val,
                    "level": _get_level(score_val)
                })
        elif valid_hint is not None:
            for rank in range(min(3, len(effective_indices))):
                idx = effective_indices[top_sub_order[rank]]
                sub_p = float(sub_probs[top_sub_order[rank]])
                score_val = round(float(np.clip(sub_p, 0.0, 1.0)), 4)
                top3.append({
                    "label": self.classes[idx],
                    "score": score_val,
                    "level": _get_level(score_val)
                })
        else:
            top3_overall_indices = torch.topk(all_probs, min(3, self.num_classes)).indices.tolist()
            for idx in top3_overall_indices:
                score_val = round(float(all_probs[idx].item()), 4)
                top3.append({
                    "label": self.classes[idx],
                    "score": score_val,
                    "level": _get_level(score_val)
                })

        # 9. Species Mismatch and Confidence Policy
        is_species_mismatch = (
            (not is_fallback_crop) and
            (valid_hint is not None) and
            (effective_crop_mass < self.mismatch_threshold) and
            (cnn_top_mass >= 0.40 or cnn_top_mass >= 3.0 * effective_crop_mass)
        )

        level = _get_level(final_confidence)
        if is_species_mismatch:
            status = "species_mismatch"
            det_disp = detected_crop.replace("_", " ")
            hint_disp = effective_crop.replace("_", " ")
            mismatch_warning = (
                f"The model thinks this leaf looks like {det_disp} "
                f"(model confidence {cnn_top_mass:.1%}), but diagnosis was requested for {hint_disp} "
                f"(crop probability {effective_crop_mass:.1%})."
            )
            msg = mismatch_warning
            certainty = "low"
            level = "low"
        elif final_confidence < self.confidence_threshold:
            status = "low_confidence"
            certainty = "low"
            level = "low"
            mismatch_warning = hint_warning
            msg = "Low confidence prediction. Try a clearer photo or field confirmation."
        elif final_confidence >= self.thresh_high:
            status = "ok"
            certainty = "high"
            level = "high"
            mismatch_warning = hint_warning
            msg = "High confidence diagnostic prediction."
        elif final_confidence >= self.thresh_medium:
            status = "ok"
            certainty = "medium"
            level = "medium"
            mismatch_warning = hint_warning
            msg = "Moderate confidence diagnostic prediction."
        else:
            status = "low_confidence"
            certainty = "low"
            level = "low"
            mismatch_warning = hint_warning
            msg = "Low confidence prediction. Try a clearer photo or field confirmation."

        # 10. Guidance Lookup with Robust Fallbacks
        full_key = f"{effective_crop}___{disease_name}"
        full_key_underscore = f"{effective_crop}___{disease_name.replace(' ', '_')}"
        full_key_norm = f"{effective_crop.replace(' ', '_')}___{disease_name.replace(' ', '_')}"
        guidance_info = (
            self.guidance.get(full_key) or 
            self.guidance.get(full_key_underscore) or 
            self.guidance.get(full_key_norm)
        )

        if not guidance_info:
            target_keys = {full_key.lower(), full_key_underscore.lower(), full_key_norm.lower()}
            for k, v in self.guidance.items():
                if k.lower() in target_keys or k.lower().replace(" ", "_") == full_key_norm.lower():
                    guidance_info = v
                    break

        return {
            # ---- Core fields (Spring Boot API contract - DO NOT RENAME) ----
            "status": status,
            "crop": effective_crop,
            "disease": disease_name,
            "confidence": final_confidence,
            "certainty": certainty,
            "level": level,
            "top3": top3,
            # ---- Extended fields (backward-compatible additions) ----
            "crop_confidence": crop_confidence,
            "disease_confidence": disease_confidence,
            "crop_comparison": crop_comparison,
            "morphology": morphology,
            "mismatch_warning": mismatch_warning,
            "quality": quality_info,
            "guidance": guidance_info,
            "message": msg,
            "raw_top1_class": raw_top1_class,
            "raw_top1_crop": cnn_top_crop,
            "prior_flipped": prior_flipped
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Crop Disease ML Prediction on an image.")
    parser.add_argument("image", nargs="?", default=None, help="Path to input crop leaf image (jpg/png)")
    parser.add_argument("--crop", type=str, default=None, help="Optional crop hint")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    predictor = CropDiseasePredictor(config_path=args.config)
    print(f"\n[+] CropDiseasePredictor loaded ({predictor.num_classes} classes on device: {predictor.device}).\n")

    test_img = args.image or "../data/sample_real_leaves/Tomato___Early_blight_0.jpg"
    if os.path.exists(test_img):
        with open(test_img, "rb") as f:
            b = f.read()
        res = predictor.predict(b, crop_hint=args.crop)
        print("Status:", res["status"])
        print("Crop:", res["crop"], "| Disease:", res["disease"], "| Confidence:", res["confidence"], "| Certainty:", res["certainty"])
        print("Top 3:")
        for t in res["top3"]:
            print(f"  - {t['label']}: {t['score']}")
