import cv2
import numpy as np
from typing import Tuple, Optional

def assess_image_quality(
    image: np.ndarray,
    blur_threshold: float = 100.0,
    min_brightness: float = 40.0,
    max_brightness: float = 225.0
) -> Tuple[bool, float, float, Optional[str]]:
    """
    Evaluates image quality based on sharpness (variance of Laplacian)
    and average illumination brightness.

    Parameters:
        image: BGR or RGB numpy image array.
        blur_threshold: Minimum Laplacian variance for acceptable sharpness.
        min_brightness: Minimum mean pixel brightness (0-255).
        max_brightness: Maximum mean pixel brightness (0-255).

    Returns:
        is_acceptable (bool): True if quality checks pass.
        blur_score (float): Sharpness score (Laplacian variance).
        brightness (float): Mean grayscale intensity.
        reason (str or None): Explanation if check failed.
    """
    if image is None or image.size == 0:
        return False, 0.0, 0.0, "Empty or corrupted image data."

    # Convert to grayscale if multi-channel
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # 1. Blur Detection via Laplacian variance
    # High variance indicates crisp edges, low variance indicates blurriness
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = float(laplacian.var())

    # 2. Illumination / Brightness assessment
    brightness = float(np.mean(gray))

    # Evaluate criteria
    if blur_score < blur_threshold:
        return False, round(blur_score, 2), round(brightness, 2), f"Image is too blurry (sharpness score {blur_score:.1f} < {blur_threshold:.1f}). Please hold camera steady."

    if brightness < min_brightness:
        return False, round(blur_score, 2), round(brightness, 2), f"Image is underexposed/too dark (brightness {brightness:.1f} < {min_brightness:.1f}). Please capture in better lighting."

    if brightness > max_brightness:
        return False, round(blur_score, 2), round(brightness, 2), f"Image is overexposed/too bright (brightness {brightness:.1f} > {max_brightness:.1f}). Please reduce direct glare."

    return True, round(blur_score, 2), round(brightness, 2), None
