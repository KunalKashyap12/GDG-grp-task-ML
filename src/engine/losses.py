"""
Loss Functions for Robust Classification.
Includes Label-Smoothed CrossEntropy and Multi-Class Focal Loss.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss down-weighting well-classified examples:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cross entropy loss with optional label smoothing
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def build_criterion(label_smoothing: float = 0.1, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """
    Constructs default robust classification loss.
    """
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
