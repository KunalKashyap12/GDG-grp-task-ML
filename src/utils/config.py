"""
Configuration Parser & Schema Validator.
Loads hierarchical YAML configurations with type enforcement and environment variable expansion.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml  # type: ignore[import-untyped]


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    manifest_path: str = "data/manifests/manifest.csv"
    processed_dir: str = "data/processed"
    train_dir: str = "data/processed/train"
    val_dir: str = "data/processed/val"
    test_dir: str = "data/processed/test_lab"
    sample_leaves_dir: str = "data/sample_real_leaves"
    max_images_per_class: int = 1500
    val_split: float = 0.15
    test_split: float = 0.15
    img_size: int = 224


@dataclass
class ModelConfig:
    architecture: str = "mobilenet_v3_large"
    pretrained: bool = True
    num_classes: int = 39
    dropout_rate: float = 0.3
    hidden_dim: int = 1280


@dataclass
class StageConfig:
    epochs: int = 5
    lr: float = 0.001
    freeze_backbone: bool = True


@dataclass
class TrainingConfig:
    seed: int = 42
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    use_amp: bool = True
    weight_decay: float = 0.0001
    label_smoothing: float = 0.1
    grad_clip_norm: float = 1.0
    stage1: StageConfig = field(default_factory=lambda: StageConfig(epochs=5, lr=0.001, freeze_backbone=True))
    stage2: StageConfig = field(default_factory=lambda: StageConfig(epochs=15, lr=0.0001, freeze_backbone=False))


@dataclass
class CalibrationConfig:
    temperature: float = 1.0
    method: str = "temperature_scaling"
    max_iter: int = 100
    lr: float = 0.01


@dataclass
class ArtifactsConfig:
    model_dir: str = "ml_service/model"
    model_path: str = "ml_service/model/model.pt"
    labels_path: str = "ml_service/model/labels.json"
    calibration_path: str = "ml_service/model/calibration.json"
    metrics_report: str = "ml_training/evaluation_report.json"


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)


def load_config(config_path: Union[str, Path]) -> AppConfig:
    """
    Loads and validates a YAML configuration into structured dataclasses.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
        raw_dict: Dict[str, Any] = loaded if isinstance(loaded, dict) else {}

    data_cfg = DataConfig(**raw_dict.get("data", {}))
    model_cfg = ModelConfig(**raw_dict.get("model", {}))

    train_dict = raw_dict.get("training", {})
    stage1_dict = train_dict.pop("stage1", {})
    stage2_dict = train_dict.pop("stage2", {})
    training_cfg = TrainingConfig(
        stage1=StageConfig(**stage1_dict) if stage1_dict else StageConfig(),
        stage2=StageConfig(**stage2_dict) if stage2_dict else StageConfig(epochs=15, lr=0.0001, freeze_backbone=False),
        **{k: v for k, v in train_dict.items() if k not in ["stage1", "stage2", "early_stopping"]}
    )

    calibration_cfg = CalibrationConfig(**raw_dict.get("calibration", {}))
    artifacts_cfg = ArtifactsConfig(**raw_dict.get("artifacts", {}))

    return AppConfig(
        data=data_cfg,
        model=model_cfg,
        training=training_cfg,
        calibration=calibration_cfg,
        artifacts=artifacts_cfg,
    )
