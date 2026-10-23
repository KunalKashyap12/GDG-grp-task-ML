"""
DataLoader Construction Factory.
Optimizes memory pinning, multi-process prefetching, and batch delivery.
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0 if os.name == "nt" else 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Constructs an optimized DataLoader for high-throughput model training and evaluation.
    Defaults num_workers to 0 on Windows to prevent multiprocessing spawn deadlocks.
    """
    is_cuda = torch.cuda.is_available()
    effective_workers = 0 if os.name == "nt" else num_workers

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=effective_workers,
        pin_memory=pin_memory and is_cuda,
        persistent_workers=(effective_workers > 0),
        drop_last=drop_last,
    )
