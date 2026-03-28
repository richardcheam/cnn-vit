from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
RGB_DEFAULT_MEAN = (0.485, 0.456, 0.406)
RGB_DEFAULT_STD = (0.229, 0.224, 0.225)


@dataclass
class DataConfig:
    data_dir: Path = Path("data")
    image_size: int = 32
    num_classes: int = 10
    batch_size: int = 128
    num_workers: int = 2
    val_fraction: float = 0.1
    mean: tuple[float, float, float] = CIFAR10_MEAN
    std: tuple[float, float, float] = CIFAR10_STD


@dataclass
class EuroSATDataConfig:
    data_dir: Path = Path("data")
    image_size: int = 64
    num_classes: int = 10
    batch_size: int = 64
    num_workers: int = 2
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    train_fraction: float = 1.0
    mean: tuple[float, float, float] = RGB_DEFAULT_MEAN
    std: tuple[float, float, float] = RGB_DEFAULT_STD


@dataclass
class AugmentationConfig:
    occlusion_mask_size: int = 8
    occlusion_fill_value: float = 0.0
    texture_patch_size: int = 4
    texture_shuffle_fraction: float = 0.75
    texture_noise_std: float = 0.05


@dataclass
class CNNConfig:
    channels: tuple[int, int, int] = (64, 128, 256)
    dropout: float = 0.2


@dataclass
class ViTConfig:
    patch_size: int = 4
    embed_dim: int = 192
    depth: int = 4
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attention_dropout: float = 0.1


@dataclass
class TrainingConfig:
    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "auto"
    log_interval: int = 50


@dataclass
class TransferConfig:
    output_dir: Path = Path("outputs/eurosat_transfer")
    checkpoint_dir: Path = Path("outputs/checkpoints")
    cnn_checkpoint: Path | None = None
    vit_checkpoint: Path | None = None
    run_mode: str = "both"
    epochs: int = 10
    scratch_learning_rate: float = 1e-3
    backbone_learning_rate: float = 1e-4
    head_learning_rate: float = 1e-3
    weight_decay: float = 1e-4


@dataclass
class ExperimentSettings:
    output_dir: Path = Path("outputs")
    data_fractions: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0)
    full_run: bool = False
    interpretability_samples: int = 4


@dataclass
class ProjectConfig:
    title: str = "Understanding CNN vs Vision Transformer"
    data: DataConfig = field(default_factory=DataConfig)
    eurosat: EuroSATDataConfig = field(default_factory=EuroSATDataConfig)
    augmentations: AugmentationConfig = field(default_factory=AugmentationConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    vit: ViTConfig = field(default_factory=ViTConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    transfer: TransferConfig = field(default_factory=TransferConfig)
    experiment: ExperimentSettings = field(default_factory=ExperimentSettings)

    def to_dict(self) -> dict:
        return _stringify_paths(asdict(self))


def build_config(full_run: bool = False) -> ProjectConfig:
    config = ProjectConfig()
    config.training.epochs = 20 if full_run else 5
    config.transfer.epochs = 25 if full_run else 10
    config.experiment.full_run = full_run
    return config


def _stringify_paths(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _stringify_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stringify_paths(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_stringify_paths(item) for item in value)
    return value
