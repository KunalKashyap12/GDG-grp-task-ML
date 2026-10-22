"""
Crop Disease Neural Network Architectures.
Decouples backbone representation learning from task-specific classification heads.
"""

from typing import Optional
import torch
import torch.nn as nn
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    mobilenet_v3_large,
    mobilenet_v3_small,
)
from src.models.base import BaseModel


class ClassificationHead(nn.Module):
    """
    Multi-layer classification head with BatchNorm, Hardswish activation,
    and Dropout for regularized feature mapping.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            # Shape: [B, in_features] -> [B, hidden_dim]
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout_rate),
            # Shape: [B, hidden_dim] -> [B, num_classes]
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input:  [B, in_features]
        # Output: [B, num_classes]
        return self.head(x)


class ModularCropClassifier(BaseModel):
    """
    Production-grade image classifier with progressive fine-tuning hooks.
    """

    def __init__(
        self,
        num_classes: int = 39,
        architecture: str = "mobilenet_v3_large",
        pretrained: bool = True,
        hidden_dim: int = 512,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.architecture = architecture

        # 1. Feature Extractor Backbone
        if architecture == "mobilenet_v3_large":
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            base_model = mobilenet_v3_large(weights=weights)
            in_features = 960
        elif architecture == "mobilenet_v3_small":
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            base_model = mobilenet_v3_small(weights=weights)
            in_features = 576
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        self.features: nn.Module = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 2. Modular Classification Head
        self.classifier = ClassificationHead(
            in_features=in_features,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
        )

    def set_backbone_frozen(self, freeze: bool = True) -> None:
        """Freezes or unfreezes backbone parameters for progressive warm-up."""
        for param in self.features.parameters():
            param.requires_grad = not freeze

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation pass with documented tensor dimensions.

        Args:
            x: Input image batch tensor.
               Shape: [Batch_Size, Channels (3), Height, Width]

        Returns:
            logits: Unnormalized class scores.
               Shape: [Batch_Size, Num_Classes]
        """
        # Feature extraction -> Shape: [B, in_channels, H', W']
        feats = self.features(x)

        # Global average pooling -> Shape: [B, in_channels, 1, 1]
        pooled = self.pool(feats)

        # Flatten -> Shape: [B, in_channels]
        flattened = torch.flatten(pooled, start_dim=1)

        # Classification logits -> Shape: [B, Num_Classes]
        logits = self.classifier(flattened)
        return logits
