import os
import random
from typing import Optional, List
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# Pillow 10+ moved resampling filters — provide backward-compatible alias.
_BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR


class RandomRotationWithFill:
    """
    Randomly rotates an image while filling corners with the image's mean color
    or reflect-padding, preventing black corner artifacts that never appear at inference.
    """
    def __init__(self, degrees: float = 30, mode: str = "mean"):
        self.degrees = degrees
        self.mode = mode

    def __call__(self, img: Image.Image) -> Image.Image:
        angle = random.uniform(-self.degrees, self.degrees)
        if abs(angle) < 0.1:
            return img

        if self.mode == "reflect":
            arr = np.array(img)
            h, w = arr.shape[:2]
            diag = int(np.ceil(np.sqrt(h**2 + w**2)))
            pad_y = (diag - h) // 2 + 2
            pad_x = (diag - w) // 2 + 2
            padded = np.pad(arr, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode="reflect")
            pad_img = Image.fromarray(padded)
            rotated = pad_img.rotate(angle, resample=_BILINEAR)
            rw, rh = rotated.size
            cx, cy = rw // 2, rh // 2
            return rotated.crop((cx - w // 2, cy - h // 2, cx - w // 2 + w, cy - h // 2 + h))
        else:
            # Mean color fill
            arr = np.array(img)
            mean_color = tuple(int(round(c)) for c in np.mean(arr, axis=(0, 1))[:3])
            return img.rotate(angle, resample=_BILINEAR, fillcolor=mean_color)


class FieldBackgroundVariation:
    """
    Simulates variable field background conditions to break lab background shortcut learning.
    Segments the leaf using excess-green + HSV foliar thresholding (reusing predict.py logic style),
    and with probability p replaces non-leaf pixels with random crops from a background folder
    or falls back to procedural soil, grass, and sky textures.
    
    Only applies when the leaf segmentation mask covers between 10% and 90% of the image,
    ensuring the leaf cannot be erased or corrupted.
    """
    def __init__(self, p: float = 0.4, backgrounds_dir: Optional[str] = None):
        self.p = p
        self.backgrounds_dir = backgrounds_dir
        self.background_files: List[str] = []

        if self.backgrounds_dir and os.path.isdir(self.backgrounds_dir):
            valid_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
            self.background_files = [
                os.path.join(self.backgrounds_dir, f)
                for f in os.listdir(self.backgrounds_dir)
                if f.lower().endswith(valid_exts)
            ]

    def _generate_procedural_background(self, h: int, w: int) -> np.ndarray:
        """Generates synthetic outdoor soil, grass, or sky background textures."""
        bg_type = random.choice(["soil", "grass", "sky"])

        if bg_type == "soil":
            base_r = random.uniform(90, 140)
            base_g = random.uniform(60, 95)
            base_b = random.uniform(30, 65)
            base = np.zeros((h, w, 3), dtype=np.float32)
            base[:, :, 0] = base_r
            base[:, :, 1] = base_g
            base[:, :, 2] = base_b

            # Multi-scale clumping and granular soil noise
            coarse_h = max(2, h // 8)
            coarse_w = max(2, w // 8)
            coarse_noise = cv2.resize(
                np.random.normal(0, 20, (coarse_h, coarse_w)), (w, h)
            )
            fine_noise = np.random.normal(0, 8, (h, w))
            tot_noise = (coarse_noise + fine_noise)[:, :, np.newaxis]
            return np.clip(base + tot_noise, 0, 255)

        elif bg_type == "grass":
            base_r = random.uniform(40, 80)
            base_g = random.uniform(105, 155)
            base_b = random.uniform(30, 65)
            base = np.zeros((h, w, 3), dtype=np.float32)
            base[:, :, 0] = base_r
            base[:, :, 1] = base_g
            base[:, :, 2] = base_b

            # Grassy streaks and mottled foliage clumps
            coarse_h = max(2, h // 12)
            coarse_w = max(2, w // 12)
            streak_w = max(2, w // 4)
            coarse = cv2.resize(np.random.normal(0, 25, (coarse_h, coarse_w)), (w, h))
            streaks = cv2.resize(np.random.normal(0, 15, (h, streak_w)), (w, h))
            return np.clip(base + (coarse + streaks)[:, :, np.newaxis], 0, 255)

        else:  # sky
            top_color = np.array(
                [random.uniform(70, 110), random.uniform(130, 175), random.uniform(215, 245)],
                dtype=np.float32
            )
            bot_color = np.array(
                [random.uniform(180, 220), random.uniform(210, 235), random.uniform(240, 255)],
                dtype=np.float32
            )
            grad = np.linspace(0, 1, h, dtype=np.float32).reshape(h, 1, 1)
            base = top_color * (1.0 - grad) + bot_color * grad
            cloud_noise = cv2.GaussianBlur(np.random.normal(0, 10, (h, w)), (15, 15), 0)[:, :, np.newaxis]
            return np.clip(base + cloud_noise, 0, 255)

    def _get_background_patch(self, h: int, w: int) -> np.ndarray:
        """Retrieves a background patch from images or procedural generation."""
        if self.background_files:
            bg_path = random.choice(self.background_files)
            try:
                bg_img = Image.open(bg_path).convert("RGB")
                bw, bh = bg_img.size
                if bw < w or bh < h:
                    scale = max(w / bw, h / bh) * 1.15
                    bg_img = bg_img.resize((int(round(bw * scale)), int(round(bh * scale))), Image.BILINEAR)
                    bw, bh = bg_img.size
                x0 = random.randint(0, bw - w)
                y0 = random.randint(0, bh - h)
                crop_img = bg_img.crop((x0, y0, x0 + w, y0 + h))
                return np.array(crop_img, dtype=np.float32)
            except Exception:
                pass  # Fall back to procedural background on read failure

        return self._generate_procedural_background(h, w)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        arr = np.array(img)
        if len(arr.shape) != 3 or arr.shape[2] != 3:
            return img

        h, w = arr.shape[:2]
        total_pixels = h * w
        if total_pixels == 0:
            return img

        # Segment leaf using excess-green + HSV foliar logic
        r = arr[:, :, 0].astype(np.float32)
        g = arr[:, :, 1].astype(np.float32)
        b = arr[:, :, 2].astype(np.float32)
        exg = 2.0 * g - r - b
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        green_foliage = (exg > 5.0) & (g >= b - 10) & (g >= r - 10) & (sat >= 30) & (val >= 30)
        veg_foliage = (hue >= 25) & (hue <= 95) & (sat >= 30) & (val >= 30)
        chlorophyll = green_foliage | veg_foliage

        # Dilate chlorophyll to encompass lesion / necrotic regions within or immediately bordering leaf tissue
        k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated_green = cv2.dilate(chlorophyll.astype(np.uint8), k_dil) > 0

        rust_spots = (hue >= 10) & (hue <= 26) & (sat >= 70) & (val >= 70) & dilated_green
        necrotic_spots = (hue >= 10) & (hue <= 35) & (sat >= 25) & (val >= 25) & (val <= 170) & dilated_green

        leaf_mask = chlorophyll | rust_spots | necrotic_spots
        leaf_u8 = (leaf_mask.astype(np.uint8)) * 255

        # Clean mask: fill minor interior pinholes while preserving external boundary
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        leaf_clean = cv2.morphologyEx(leaf_u8, cv2.MORPH_CLOSE, k_close)

        coverage = float(np.count_nonzero(leaf_clean)) / float(total_pixels)

        # Only apply when mask covers between 10% and 90% of the image
        if coverage < 0.10 or coverage > 0.90:
            return img

        # Fetch replacement background patch
        bg_arr = self._get_background_patch(h, w)

        # Alpha-blend with subtle feathering along leaf edges to avoid jagged boundary artifacts
        mask_f = cv2.GaussianBlur(leaf_clean.astype(np.float32) / 255.0, (5, 5), 0)[:, :, np.newaxis]
        blended = arr.astype(np.float32) * mask_f + bg_arr * (1.0 - mask_f)
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        return Image.fromarray(blended)


def get_train_transforms(img_size: int = 224, backgrounds_dir: Optional[str] = None) -> T.Compose:
    """
    Full training augmentation pipeline:
    - Random resized crop
    - Flips & rotations with mean-color / reflection corner fill
    - Real background replacement (leaf segmentation + natural/procedural backgrounds)
    - Color jitter & slight blur
    - Tensor normalization
    - Random erasing (occlusions)
    """
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        RandomRotationWithFill(degrees=30, mode="mean"),
        FieldBackgroundVariation(p=0.4, backgrounds_dir=backgrounds_dir),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.25),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random")
    ])


def get_eval_transforms(img_size: int = 224) -> T.Compose:
    """
    Deterministic evaluation / test pipeline:
    Resize -> CenterCrop -> Normalize
    """
    return T.Compose([
        T.Resize(int(img_size * 1.14)),  # e.g., 256 for 224
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
