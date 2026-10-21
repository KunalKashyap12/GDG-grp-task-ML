import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("ml_service"))
from predict import CropDiseasePredictor

predictor = CropDiseasePredictor(config_path="ml_service/config.yaml")

def get_morphology_prior(morph: dict, crop: str) -> float:
    c = crop.lower().replace(" ", "").replace("_", "").replace(",", "")
    shape = morph.get("shape", "")
    margin = morph.get("margin", "")
    geom = morph.get("geometry_class", "")
    is_linear = morph.get("is_linear_blade", False)
    solidity = morph.get("solidity", 0.8)
    roughness = morph.get("roughness", 1.0)
    
    # 1. Monocot vs Dicot check
    is_monocot_crop = c in ["corn(maize)", "corn", "maize", "wheat", "rice", "sugarcane"]
    if is_linear or "monocot" in shape.lower() or "parallel" in margin.lower():
        if is_monocot_crop:
            return 2.5
        else:
            return 0.15 # Strong penalty for dicots on monocot grass blade
    else:
        if is_monocot_crop and (solidity > 0.82 and not is_linear):
            return 0.25 # Penalty for grass blade crops when leaf is broad/oval
            
    # 2. Bell Pepper vs Tomato check
    # Bell Pepper: Simple ovate-lanceolate with ENTIRE (smooth) margin, high solidity
    # Tomato: Pinnately compound/dissected with DENTATE/SERRATED margin, low solidity/high roughness
    is_entire_smooth = "entire" in margin.lower() and "smooth" in margin.lower()
    is_serrated_dentate = "serrated" in margin.lower() or "dentate" in margin.lower() or "toothed" in margin.lower()
    
    if "bellpepper" in c or "pepper" in c:
        if is_entire_smooth and solidity >= 0.82:
            return 2.2 # Heavy boost for pepper on smooth ovate leaf
        elif is_serrated_dentate or roughness > 1.35:
            return 0.35 # Pepper leaves do not have serrated/toothed margins
            
    if "tomato" in c:
        if is_entire_smooth and solidity >= 0.88:
            return 0.30 # Tomato leaves are NOT simple smooth entire leaves!
        elif is_serrated_dentate or roughness > 1.30 or geom == "compound_lobed":
            return 1.8 # Boost for tomato on serrated compound foliage
            
    # 3. Palmate Lobed (Grape, Squash, Cotton)
    is_palmate = geom == "palmate_lobed" or "palmate" in shape.lower()
    if c in ["grape", "squash", "cotton"]:
        if is_palmate:
            return 2.0
    else:
        if is_palmate and c in ["bellpepper", "pepper"]:
            return 0.25 # Peppers are never palmate lobed
            
    return 1.0

print("Testing morphology prior function logic...")
test_morph_pepper = {
    "shape": "Simple Ovate-Lanceolate",
    "geometry_class": "simple_ovate",
    "margin": "Entire (Smooth, non-serrated)",
    "solidity": 0.95,
    "circularity": 0.588,
    "roughness": 1.05,
    "is_linear_blade": False
}

p_pepper = get_morphology_prior(test_morph_pepper, "Bell Pepper")
p_tomato = get_morphology_prior(test_morph_pepper, "Tomato")
print(f"On Bell Pepper leaf: Prior(Bell Pepper) = {p_pepper}, Prior(Tomato) = {p_tomato}")
assert p_pepper > 2.0
assert p_tomato < 0.4
print("Logic verified successfully!")
