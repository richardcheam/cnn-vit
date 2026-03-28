from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from configs.config import ProjectConfig
from datasets.cifar_loader import DataBundle


@dataclass
class EuroSATProtocol:
    dataset_size: int
    train_size: int
    val_size: int
    test_size: int
    num_classes: int
    class_names: tuple[str, ...]


def build_eurosat_train_transform(config: ProjectConfig) -> transforms.Compose:
    """Satellite images benefit from light spatial augmentation and orientation flips."""
    return transforms.Compose(
        [
            transforms.Resize((config.eurosat.image_size, config.eurosat.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(config.eurosat.mean, config.eurosat.std),
        ]
    )


def build_eurosat_eval_transform(config: ProjectConfig) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((config.eurosat.image_size, config.eurosat.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(config.eurosat.mean, config.eurosat.std),
        ]
    )


def describe_eurosat_protocol(config: ProjectConfig) -> EuroSATProtocol:
    eval_dataset = _load_eurosat_dataset(config=config, train_transform=False)
    train_indices, val_indices, test_indices = _build_stratified_splits(
        labels=eval_dataset.targets,
        val_fraction=config.eurosat.val_fraction,
        test_fraction=config.eurosat.test_fraction,
        seed=config.training.seed,
    )
    train_indices = _fraction_indices(
        indices=train_indices,
        fraction=config.eurosat.train_fraction,
        seed=config.training.seed,
    )
    return EuroSATProtocol(
        dataset_size=len(eval_dataset),
        train_size=len(train_indices),
        val_size=len(val_indices),
        test_size=len(test_indices),
        num_classes=len(eval_dataset.classes),
        class_names=tuple(eval_dataset.classes),
    )


def build_eurosat_datasets(
    config: ProjectConfig,
) -> tuple[Dataset, Dataset, Dataset, tuple[str, ...]]:
    train_dataset = _load_eurosat_dataset(config=config, train_transform=True)
    eval_dataset = _load_eurosat_dataset(config=config, train_transform=False)

    train_indices, val_indices, test_indices = _build_stratified_splits(
        labels=eval_dataset.targets,
        val_fraction=config.eurosat.val_fraction,
        test_fraction=config.eurosat.test_fraction,
        seed=config.training.seed,
    )
    train_indices = _fraction_indices(
        indices=train_indices,
        fraction=config.eurosat.train_fraction,
        seed=config.training.seed,
    )

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(eval_dataset, val_indices)
    test_subset = Subset(eval_dataset, test_indices)
    return train_subset, val_subset, test_subset, tuple(eval_dataset.classes)


def build_eurosat_dataloaders(config: ProjectConfig) -> DataBundle:
    train_dataset, val_dataset, test_dataset, classes = build_eurosat_datasets(config)
    generator = torch.Generator().manual_seed(config.training.seed)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.eurosat.batch_size,
        shuffle=True,
        num_workers=config.eurosat.num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eurosat.batch_size,
        shuffle=False,
        num_workers=config.eurosat.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eurosat.batch_size,
        shuffle=False,
        num_workers=config.eurosat.num_workers,
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


def _load_eurosat_dataset(config: ProjectConfig, train_transform: bool) -> datasets.EuroSAT:
    transform = build_eurosat_train_transform(config) if train_transform else build_eurosat_eval_transform(config)
    try:
        return datasets.EuroSAT(
            root=config.eurosat.data_dir,
            transform=transform,
            download=True,
        )
    except Exception as error:
        message = (
            "Unable to access EuroSAT. If the dataset is not cached locally, "
            "please ensure the machine can download it once and rerun the project."
        )
        raise RuntimeError(message) from error


def _build_stratified_splits(
    labels: list[int],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    if val_fraction <= 0.0 or test_fraction <= 0.0 or val_fraction + test_fraction >= 1.0:
        raise ValueError("EuroSAT val_fraction and test_fraction must be positive and sum to less than 1.")

    indices = np.arange(len(labels))
    labels_array = np.asarray(labels)

    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=test_fraction,
        random_state=seed,
        stratify=labels_array,
    )
    train_val_labels = labels_array[train_val_indices]
    adjusted_val_fraction = val_fraction / (1.0 - test_fraction)
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=adjusted_val_fraction,
        random_state=seed,
        stratify=train_val_labels,
    )
    return train_indices.tolist(), val_indices.tolist(), test_indices.tolist()


def _fraction_indices(indices: list[int], fraction: float, seed: int) -> list[int]:
    if fraction >= 1.0:
        return indices
    if fraction <= 0.0:
        raise ValueError("EuroSAT train_fraction must be positive.")
    subset_size = max(1, int(len(indices) * fraction))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(indices, size=subset_size, replace=False)
    return chosen.tolist()
