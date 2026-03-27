from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from configs.config import ProjectConfig
from datasets.occlusion import OcclusionWrapperDataset
from datasets.texture_modification import TextureModifiedDataset


@dataclass
class DataBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    classes: tuple[str, ...]


def build_train_transform(config: ProjectConfig) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomCrop(config.data.image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(config.data.mean, config.data.std),
        ]
    )


def build_eval_transform(config: ProjectConfig) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(config.data.mean, config.data.std),
        ]
    )


def _split_indices(dataset_size: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    indices = np.arange(dataset_size)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_size = int(dataset_size * val_fraction)
    val_indices = indices[:val_size].tolist()
    train_indices = indices[val_size:].tolist()
    return train_indices, val_indices


def _fraction_indices(indices: list[int], fraction: float, seed: int) -> list[int]:
    if fraction >= 1.0:
        return indices
    subset_size = max(1, int(len(indices) * fraction))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(indices, size=subset_size, replace=False)
    return chosen.tolist()


def _apply_test_variant(
    dataset: Dataset,
    variant: str,
    config: ProjectConfig,
) -> Dataset:
    if variant == "clean":
        return dataset
    if variant == "occluded":
        return OcclusionWrapperDataset(
            dataset,
            mask_size=config.augmentations.occlusion_mask_size,
            fill_value=config.augmentations.occlusion_fill_value,
            seed=config.training.seed,
        )
    if variant == "texture":
        return TextureModifiedDataset(
            dataset,
            patch_size=config.augmentations.texture_patch_size,
            shuffle_fraction=config.augmentations.texture_shuffle_fraction,
            noise_std=config.augmentations.texture_noise_std,
            seed=config.training.seed,
        )
    raise ValueError(f"Unknown test variant: {variant}")


def build_cifar_datasets(
    config: ProjectConfig,
    train_fraction: float = 1.0,
    test_variant: str = "clean",
) -> tuple[Dataset, Dataset, Dataset, tuple[str, ...]]:
    train_transform = build_train_transform(config)
    eval_transform = build_eval_transform(config)

    try:
        train_dataset_augmented = datasets.CIFAR10(
            root=config.data.data_dir,
            train=True,
            transform=train_transform,
            download=True,
        )
        train_dataset_eval = datasets.CIFAR10(
            root=config.data.data_dir,
            train=True,
            transform=eval_transform,
            download=True,
        )
        test_dataset = datasets.CIFAR10(
            root=config.data.data_dir,
            train=False,
            transform=eval_transform,
            download=True,
        )
    except Exception as error:
        message = (
            "Unable to access CIFAR-10. If the dataset is not cached locally, "
            "please ensure the machine can download it once and rerun the project."
        )
        raise RuntimeError(message) from error

    train_indices, val_indices = _split_indices(
        dataset_size=len(train_dataset_augmented),
        val_fraction=config.data.val_fraction,
        seed=config.training.seed,
    )
    train_indices = _fraction_indices(train_indices, fraction=train_fraction, seed=config.training.seed)

    train_subset = Subset(train_dataset_augmented, train_indices)
    val_subset = Subset(train_dataset_eval, val_indices)
    test_subset = _apply_test_variant(test_dataset, variant=test_variant, config=config)

    return train_subset, val_subset, test_subset, tuple(train_dataset_eval.classes)


def build_dataloaders(
    config: ProjectConfig,
    train_fraction: float = 1.0,
    test_variant: str = "clean",
) -> DataBundle:
    train_dataset, val_dataset, test_dataset, classes = build_cifar_datasets(
        config=config,
        train_fraction=train_fraction,
        test_variant=test_variant,
    )
    generator = torch.Generator().manual_seed(config.training.seed)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
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
