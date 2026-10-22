"""
Base Model Abstract Specification.
Provides serialization, backbone freezing, and parameter profiling hooks.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Union
import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
    Abstract Base Class for all Deep Learning Models in the framework.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass contract."""
        pass

    @abstractmethod
    def set_backbone_frozen(self, freeze: bool = True) -> None:
        """Hooks to freeze or unfreeze backbone layers."""
        pass

    def count_parameters(self) -> Dict[str, int]:
        """Returns total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def save_weights(self, path: Union[str, Path]) -> None:
        """Saves state_dict safely to disk."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), target_path)

    def load_weights(self, path: Union[str, Path], map_location: Union[str, torch.device] = "cpu") -> None:
        """Loads state_dict safely into current model."""
        state_dict = torch.load(path, map_location=map_location, weights_only=False)
        self.load_state_dict(state_dict)
