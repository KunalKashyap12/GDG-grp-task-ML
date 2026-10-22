"""
Model Factory Pattern.
Instantiates registered architectures based on configuration schemas.
"""

from typing import Union
from src.models.base import BaseModel
from src.models.classifier import ModularCropClassifier
from src.utils.config import ModelConfig


def build_model(config: Union[ModelConfig, dict]) -> BaseModel:
    """
    Constructs and returns a neural network instance from configuration.
    """
    if isinstance(config, dict):
        arch = config.get("architecture", "mobilenet_v3_large")
        num_classes = config.get("num_classes", 39)
        pretrained = config.get("pretrained", True)
        dropout_rate = config.get("dropout_rate", 0.3)
        hidden_dim = config.get("hidden_dim", 512)
    else:
        arch = config.architecture
        num_classes = config.num_classes
        pretrained = config.pretrained
        dropout_rate = config.dropout_rate
        hidden_dim = config.hidden_dim

    return ModularCropClassifier(
        num_classes=num_classes,
        architecture=arch,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        hidden_dim=hidden_dim,
    )
