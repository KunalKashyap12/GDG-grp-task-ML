import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter
import random
import numpy as np

class FieldBackgroundVariation:
    """
    Simulates variable field background conditions to break lab background shortcut learning.
    Randomly injects background noise, earth tones (soil, sky, foliage),
    or edge perturbations to emulate in-field leaf captures.
    """
    def __init__(self, p: float = 0.4):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        # Convert image to numpy array
        arr = np.array(img).astype(np.float32)
        h, w, c = arr.shape

        # Choose perturbation type
        mode = random.choice(["field_tint", "edge_noise", "shadow_gradient"])

        if mode == "field_tint":
            # Simulate variable natural outdoor ambient tint (sunlight / canopy reflection)
            r_scale = random.uniform(0.85, 1.15)
            g_scale = random.uniform(0.90, 1.10)
            b_scale = random.uniform(0.80, 1.05)
            tint = np.array([r_scale, g_scale, b_scale], dtype=np.float32)
            arr = np.clip(arr * tint, 0, 255)

        elif mode == "edge_noise":
            # Add synthetic background texture to corner/border regions
            # where lab backgrounds are typically solid/uniform
            mask = np.ones((h, w, 1), dtype=np.float32)
            # Create a center-focused vignette mask
            y, x = np.ogrid[:h, :w]
            center_dist = np.sqrt((x - w/2)**2 + (y - h/2)**2)
            max_dist = np.sqrt((w/2)**2 + (h/2)**2)
            normalized_dist = (center_dist / max_dist)[:, :, np.newaxis]
            
            # Soil / outdoor color palette
            soil_color = np.array([random.uniform(50, 120), random.uniform(40, 90), random.uniform(20, 60)], dtype=np.float32)
            noise = np.random.normal(0, 15, arr.shape)
            blended = arr * (1 - normalized_dist * 0.4) + (soil_color + noise) * (normalized_dist * 0.4)
            arr = np.clip(blended, 0, 255)

        elif mode == "shadow_gradient":
            # Emulate natural canopy shadow gradient across the leaf
            gradient = np.linspace(random.uniform(0.65, 0.9), 1.0, w).reshape(1, w, 1)
            if random.random() > 0.5:
                gradient = np.flip(gradient, axis=1)
            arr = np.clip(arr * gradient, 0, 255)

        return Image.fromarray(arr.astype(np.uint8))


def get_train_transforms(img_size: int = 224) -> T.Compose:
    """
    Full training augmentation pipeline:
    - Random resized crop
    - Flips & rotations
    - Field background variation
    - Color jitter & slight blur
    - Tensor normalization
    - Random erasing (occlusions)
    """
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(degrees=30),
        FieldBackgroundVariation(p=0.4),
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
