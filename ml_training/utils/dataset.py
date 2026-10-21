import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import List, Dict, Optional, Tuple, Callable
import numpy as np

class CropDiseaseDataset(Dataset):
    """
    Dataset loader for multi-crop disease classification.
    Reads images based on manifest DataFrame or folder paths, tracking
    image path, label index, and origin source dataset.
    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        label_to_idx: Dict[str, int],
        transform: Optional[Callable] = None,
        max_failure_rate: float = 0.005,
        max_logged_failures: int = 5
    ):
        self.df = dataframe.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.transform = transform
        self.max_failure_rate = float(max_failure_rate)
        self.max_logged_failures = int(max_logged_failures)
        self.failure_count = 0
        self.failed_paths: List[str] = []

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str]:
        row = self.df.iloc[idx]
        img_path = row["path"]
        label_str = row["label"]
        source = row.get("source", "unknown")

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            self.failure_count += 1
            if len(self.failed_paths) < self.max_logged_failures:
                self.failed_paths.append(img_path)
                print(f"[!] Warning: Failed to open image at '{img_path}': {e}")

            failure_rate = self.failure_count / max(len(self.df), 1)
            if failure_rate > self.max_failure_rate:
                raise RuntimeError(
                    f"Dataset loading failure rate exceeded limit ({failure_rate:.2%} > {self.max_failure_rate:.2%})! "
                    f"Total failed images: {self.failure_count}/{len(self.df)}. "
                    f"Sample broken paths: {self.failed_paths[:self.max_logged_failures]}"
                )

            # Fallback for occasional corrupt image under threshold
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        if self.transform:
            image = self.transform(image)

        label_idx = self.label_to_idx[label_str]
        return image, label_idx, source, img_path


def compute_class_weights(
    labels: List[str],
    label_to_idx: Dict[str, int],
    smoothing: float = 0.15
) -> torch.Tensor:
    """
    Computes inverse frequency balanced class weights:
    W_c = (N / (C * count_c)) ** (1 - smoothing)
    Includes smoothing to avoid overly aggressive weights for rare classes.
    """
    counts = np.zeros(len(label_to_idx), dtype=np.float32)
    for label in labels:
        if label in label_to_idx:
            counts[label_to_idx[label]] += 1

    total_samples = len(labels)
    num_classes = len(label_to_idx)

    # Avoid zero division
    counts = np.maximum(counts, 1.0)
    weights = total_samples / (num_classes * counts)

    # Apply exponential smoothing factor
    weights = np.power(weights, 1.0 - smoothing)

    # Normalize weights so mean is 1.0
    weights = weights / np.mean(weights)

    return torch.tensor(weights, dtype=torch.float32)


def create_dataloader(
    df: pd.DataFrame,
    label_to_idx: Dict[str, int],
    transform: Callable,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 2,
    max_failure_rate: float = 0.005
) -> DataLoader:
    """Helper to build standard PyTorch DataLoader."""
    dataset = CropDiseaseDataset(
        df,
        label_to_idx,
        transform=transform,
        max_failure_rate=max_failure_rate
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
