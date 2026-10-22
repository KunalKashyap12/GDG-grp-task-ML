"""
Image Quality and Botanical Foliage Screening Utilities.
Performs Laplacian sharpness screening, illumination bounds checking,
and foliar chlorophyll ratio verification.
"""

from typing import Optional, Tuple
import cv2
import numpy as np


def assess_quality(
    image_cv: np.ndarray,
    min_bright: float = 40.0,
    max_bright: float = 220.0,
    blur_thresh: float = 80.0,
) -> Tuple[bool, float, float, Optional[str]]:
    """
    Evaluates image blur score (Laplacian variance on downscaled 512px representation)
    and mean grayscale illumination.
    """
    if len(image_cv.shape) != 3 or image_cv.size == 0:
        return False, 0.0, 0.0, "Corrupt or empty image buffer."

    h, w = image_cv.shape[:2]
    longer_side = max(h, w)
    if longer_side > 512:
        scale = 512.0 / float(longer_side)
        resized_w = round(w * scale)
        resized_h = round(h * scale)
        eval_img = cv2.resize(image_cv, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    else:
        eval_img = image_cv

    gray = cv2.cvtColor(eval_img, cv2.COLOR_BGR2GRAY) if len(eval_img.shape) == 3 else eval_img
    gray_arr = np.asarray(gray, dtype=np.float32)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray_arr))

    if brightness < min_bright:
        return False, round(blur_score, 2), round(brightness, 2), f"CLEAR PHOTO NAHI HAI: Photo bahut andhere mein hai (brightness: {brightness:.1f} < {min_bright:.1f})."

    if brightness > max_bright:
        return False, round(blur_score, 2), round(brightness, 2), "CLEAR PHOTO NAHI HAI: Photo par dhoop/glare bahut zyada hai."

    if blur_score < blur_thresh:
        return False, round(blur_score, 2), round(brightness, 2), f"CLEAR PHOTO NAHI HAI: Photo bahut dhundhli (blurry) hai (sharpness: {blur_score:.1f} < {blur_thresh:.1f})."

    return True, round(blur_score, 2), round(brightness, 2), None


def verify_leaf_presence(
    image_cv: np.ndarray,
    min_foliage_ratio: float = 0.06,
    max_neutral_ratio: float = 0.70,
    min_contour_ratio: float = 0.03,
) -> Tuple[bool, float, Optional[str]]:
    """
    Verifies that the image contains valid plant foliage (chlorophyll base, rust, or necrotic foliar spots)
    and rejects document scans, paper receipts, screenshots, and non-plant objects.
    """
    if len(image_cv.shape) != 3:
        return False, 0.0, "Input image must be a 3-channel color photograph."

    h, w = image_cv.shape[:2]
    total_pixels = h * w
    b, g, r = [image_cv[:, :, i].astype(np.float32) for i in range(3)]
    hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)

    # 1. Excess Green index: ExG = 2G - R - B
    exg = 2 * g - r - b
    green_foliage = (exg > 5.0) & (g >= b - 10) & (g >= r - 10) & (hsv[:, :, 1] >= 35) & (hsv[:, :, 2] >= 35)
    veg_foliage = (hsv[:, :, 0] >= 28) & (hsv[:, :, 0] <= 90) & (hsv[:, :, 1] >= 40) & (hsv[:, :, 2] >= 40)
    chlorophyll_base = green_foliage | veg_foliage

    # 2. Rust & necrotic diseased leaf tissue
    k_leaf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated_leaf = cv2.dilate(chlorophyll_base.astype(np.uint8), k_leaf) > 0
    rust_foliage = (hsv[:, :, 0] >= 10) & (hsv[:, :, 0] <= 26) & (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 70) & dilated_leaf
    necrotic_spots = (hsv[:, :, 0] >= 10) & (hsv[:, :, 0] <= 35) & (hsv[:, :, 1] >= 30) & (hsv[:, :, 2] >= 30) & (hsv[:, :, 2] <= 160) & dilated_leaf

    foliage = chlorophyll_base | rust_foliage | necrotic_spots
    foliage_pixels = np.count_nonzero(foliage)
    foliage_ratio = float(foliage_pixels) / float(total_pixels)

    # 3. Neutral gray/white pixel detector (documents, web pages, walls)
    neutral = (np.abs(r - g) < 14) & (np.abs(g - b) < 14) & (np.abs(r - b) < 14)
    neutral_ratio = float(np.count_nonzero(neutral)) / float(total_pixels)

    if foliage_ratio < min_foliage_ratio:
        return False, round(foliage_ratio, 4), "NOT A LEAF: The uploaded image does not contain a crop leaf. Please upload a clear photo of a plant or crop leaf. (Yeh kisi fasal ki patti nahi hai! Kripya kisi fasal ki patti ki saaf photo upload karein.)"

    if neutral_ratio > max_neutral_ratio and foliage_ratio < 0.15:
        return False, round(foliage_ratio, 4), "NOT A LEAF: This appears to be a document or screenshot. Please upload a clear photo of a crop leaf. (Yeh document ya screenshot lag raha hai. Kripya kisi fasal ki patti ki photo upload karein.)"

    # 4. Contour morphology clean-up
    foliage_mask = foliage.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    foliage_clean = cv2.morphologyEx(foliage_mask, cv2.MORPH_OPEN, k)
    contours, _ = cv2.findContours(foliage_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, round(foliage_ratio, 4), "NOT A LEAF: Foliar structure not recognized. Please upload a clear photo of a crop leaf."

    largest_c = max(contours, key=cv2.contourArea)
    largest_area_ratio = float(cv2.contourArea(largest_c)) / float(total_pixels)
    if largest_area_ratio < min_contour_ratio:
        return False, round(foliage_ratio, 4), "NOT A LEAF: No dominant leaf structure detected. Please upload a close-up photo of the leaf."

    return True, round(foliage_ratio, 4), None
