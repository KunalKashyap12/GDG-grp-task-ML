"""
DataLoader Construction Factory.
Optimizes memory pinning, multi-process prefetching, and batch delivery.
"""

import torch
from torch.utils.data import DataLoader, Dataset


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Constructs an optimized DataLoader for high-throughput model training and evaluation.
    """
    is_cuda = torch.cuda.is_available()

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and is_cuda,
        persistent_workers=(num_workers > 0),
        drop_last=drop_last,
    )
