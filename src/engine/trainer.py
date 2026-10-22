"""
Model Training & Validation Engine.
Decouples optimization, gradient scaling, and metric tracking from model definitions.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from src.utils.logger import get_logger


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float


class ModelTrainer:
    """
    Encapsulated Training and Validation Manager with native
    Automatic Mixed Precision (AMP), gradient clipping, and scheduler stepping.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        scheduler: Optional[_LRScheduler] = None,
        use_amp: bool = True,
        grad_clip_norm: Optional[float] = 1.0,
        logger=None,
    ) -> None:
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp and (device.type == "cuda")
        self.grad_clip_norm = grad_clip_norm
        self.logger = logger or get_logger("Trainer")

        # Hardware-accelerated gradient scaler for float16 operations
        self.scaler = GradScaler(device=self.device.type, enabled=self.use_amp)

    def train_one_epoch(self, dataloader: DataLoader) -> EpochMetrics:
        """
        Executes one full training epoch over the provided DataLoader.
        """
        self.model.train()
        total_loss: float = 0.0
        correct_predictions: int = 0
        total_samples: int = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Tensor Shape: inputs -> [B, 3, 224, 224], targets -> [B]
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            batch_size = inputs.size(0)

            self.optimizer.zero_grad(set_to_none=True)

            # Mixed precision context
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                # Shape: [B, Num_Classes]
                logits = self.model(inputs)
                loss = self.criterion(logits, targets)

            # Backward pass with scaled gradients
            self.scaler.scale(loss).backward()

            # Unscale before clipping gradients
            if self.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Compute running statistics
            total_loss += float(loss.item()) * batch_size
            preds = torch.argmax(logits, dim=-1)
            correct_predictions += int((preds == targets).sum().item())
            total_samples += batch_size

        if self.scheduler is not None:
            self.scheduler.step()

        return EpochMetrics(
            loss=total_loss / total_samples,
            accuracy=correct_predictions / total_samples,
        )

    @torch.inference_mode()
    def validate(self, dataloader: DataLoader) -> EpochMetrics:
        """
        Executes validation pass without building an autograd graph.
        """
        self.model.eval()
        total_loss: float = 0.0
        correct_predictions: int = 0
        total_samples: int = 0

        for inputs, targets in dataloader:
            # Tensor Shape: inputs -> [B, 3, 224, 224], targets -> [B]
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            batch_size = inputs.size(0)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                # Shape: [B, Num_Classes]
                logits = self.model(inputs)
                loss = self.criterion(logits, targets)

            total_loss += float(loss.item()) * batch_size
            preds = torch.argmax(logits, dim=-1)
            correct_predictions += int((preds == targets).sum().item())
            total_samples += batch_size

        return EpochMetrics(
            loss=total_loss / total_samples,
            accuracy=correct_predictions / total_samples,
        )
