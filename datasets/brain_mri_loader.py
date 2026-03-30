from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from configs.config import ProjectConfig
from datasets.cifar_loader import DataBundle


@dataclass
class BrainMRIProtocol:
    dataset_name: str
    dataset_root: str
    train_source_size: int
    test_source_size: int
    train_size: int
    val_size: int
    test_size: int
    num_classes: int
    class_names: tuple[str, ...]
    image_size: int
    channels: int


class ConvertToRGB:
    """Top-level callable so DataLoader workers can pickle the transform on macOS."""

    def __call__(self, image):
        return image.convert("RGB")


def build_brain_mri_train_transform(config: ProjectConfig) -> transforms.Compose:
    """Use light geometry-only augmentation to avoid unrealistic MRI distortions."""
    return transforms.Compose(
        [
            ConvertToRGB(),
            transforms.Resize((config.brain_mri.image_size, config.brain_mri.image_size)),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(config.brain_mri.mean, config.brain_mri.std),
        ]
    )


def build_brain_mri_eval_transform(config: ProjectConfig) -> transforms.Compose:
    return transforms.Compose(
        [
            ConvertToRGB(),
            transforms.Resize((config.brain_mri.image_size, config.brain_mri.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(config.brain_mri.mean, config.brain_mri.std),
        ]
    )


def describe_brain_mri_protocol(config: ProjectConfig) -> BrainMRIProtocol:
    train_dir, test_dir = _resolve_brain_mri_layout(config.brain_mri.data_dir)
    train_dataset = datasets.ImageFolder(root=train_dir)
    test_dataset = datasets.ImageFolder(root=test_dir)
    train_indices, val_indices = _build_stratified_train_val_split(
        labels=train_dataset.targets,
        val_fraction=config.brain_mri.val_fraction,
        seed=config.training.seed,
    )
    train_indices = _fraction_indices(
        indices=train_indices,
        fraction=config.brain_mri.train_fraction,
        seed=config.training.seed,
    )
    return BrainMRIProtocol(
        dataset_name=config.brain_mri.name,
        dataset_root=str(Path(config.brain_mri.data_dir)),
        train_source_size=len(train_dataset),
        test_source_size=len(test_dataset),
        train_size=len(train_indices),
        val_size=len(val_indices),
        test_size=len(test_dataset),
        num_classes=len(train_dataset.classes),
        class_names=tuple(train_dataset.classes),
        image_size=config.brain_mri.image_size,
        channels=config.brain_mri.channels,
    )


def build_brain_mri_datasets(
    config: ProjectConfig,
) -> tuple[Dataset, Dataset, Dataset, tuple[str, ...]]:
    train_dir, test_dir = _resolve_brain_mri_layout(config.brain_mri.data_dir)
    train_dataset_augmented = datasets.ImageFolder(
        root=train_dir,
        transform=build_brain_mri_train_transform(config),
    )
    train_dataset_eval = datasets.ImageFolder(
        root=train_dir,
        transform=build_brain_mri_eval_transform(config),
    )
    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=build_brain_mri_eval_transform(config),
    )

    train_indices, val_indices = _build_stratified_train_val_split(
        labels=train_dataset_eval.targets,
        val_fraction=config.brain_mri.val_fraction,
        seed=config.training.seed,
    )
    train_indices = _fraction_indices(
        indices=train_indices,
        fraction=config.brain_mri.train_fraction,
        seed=config.training.seed,
    )

    train_subset = Subset(train_dataset_augmented, train_indices)
    val_subset = Subset(train_dataset_eval, val_indices)
    return train_subset, val_subset, test_dataset, tuple(train_dataset_eval.classes)


def build_brain_mri_dataloaders(config: ProjectConfig) -> DataBundle:
    train_dataset, val_dataset, test_dataset, classes = build_brain_mri_datasets(config)
    generator = torch.Generator().manual_seed(config.training.seed)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.brain_mri.batch_size,
        shuffle=True,
        num_workers=config.brain_mri.num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.brain_mri.batch_size,
        shuffle=False,
        num_workers=config.brain_mri.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.brain_mri.batch_size,
        shuffle=False,
        num_workers=config.brain_mri.num_workers,
        pin_memory=pin_memory,
    )

    return DataBundle(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        classes=classes,
    )


def _resolve_brain_mri_layout(root: str | Path) -> tuple[Path, Path]:
    root = Path(root)
    direct_train = root / "Training"
    direct_test = root / "Testing"
    if direct_train.exists() and direct_test.exists():
        return direct_train, direct_test

    for child in sorted(root.iterdir()) if root.exists() else []:
        if not child.is_dir():
            continue
        nested_train = child / "Training"
        nested_test = child / "Testing"
        if nested_train.exists() and nested_test.exists():
            return nested_train, nested_test

    raise FileNotFoundError(
        "Unable to find the Brain Tumor MRI dataset. Expected a directory containing "
        "`Training/` and `Testing/` folders, for example "
        "`data/brain_tumor_mri_dataset/Training` and `data/brain_tumor_mri_dataset/Testing`."
    )


def _build_stratified_train_val_split(
    labels: list[int],
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if val_fraction <= 0.0 or val_fraction >= 1.0:
        raise ValueError("Brain MRI val_fraction must be between 0 and 1.")

    indices = np.arange(len(labels))
    labels_array = np.asarray(labels)
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=seed,
        stratify=labels_array,
    )
    return train_indices.tolist(), val_indices.tolist()


def _fraction_indices(indices: list[int], fraction: float, seed: int) -> list[int]:
    if fraction >= 1.0:
        return indices
    if fraction <= 0.0:
        raise ValueError("Brain MRI train_fraction must be positive.")
    subset_size = max(1, int(len(indices) * fraction))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(indices, size=subset_size, replace=False)
    return chosen.tolist()
