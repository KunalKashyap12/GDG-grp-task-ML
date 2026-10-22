"""
Data Augmentation & Preprocessing Pipelines.
Provides production-grade transforms for training, validation, and inference.
"""

from typing import Tuple
from torchvision.transforms import InterpolationMode
import torchvision.transforms as T


IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def build_train_transforms(img_size: int = 224) -> T.Compose:
    """
    Constructs robust stochastic augmentation pipeline for training:
    - Random Resized Crop (scale: 0.7 to 1.0)
    - Horizontal & Vertical Flips (plants have rotational invariance)
    - Color Jitter (brightness, contrast, saturation, hue)
    - AutoAugment / Random Affine
    - ImageNet standard normalization
    """
    interpolation = InterpolationMode.BILINEAR

    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0), interpolation=interpolation),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(degrees=25, interpolation=interpolation),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_val_transforms(img_size: int = 224) -> T.Compose:
    """
    Constructs deterministic preprocessing pipeline for validation & test:
    - Resize to 256
    - Center Crop to target resolution
    - ImageNet standard normalization
    """
    interpolation = InterpolationMode.BILINEAR

    return T.Compose([
        T.Resize(round(img_size * 1.14), interpolation=interpolation),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
