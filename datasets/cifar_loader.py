from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from configs.config import ProjectConfig
from datasets.occlusion import OcclusionWrapperDataset
from datasets.texture_modification import TextureModifiedDataset


CIFAR10_TRAINSET_SIZE = 50_000
CIFAR10_TESTSET_SIZE = 10_000


@dataclass
class DataBundle:
    """Small container so downstream code can pass around all splits together."""
    train: DataLoader
    val: DataLoader
    test: DataLoader
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    classes: tuple[str, ...]


def describe_cifar_protocol(
    config: ProjectConfig,
    fractions: tuple[float, ...] | None = None,
) -> dict:
    """Describe the planned source-dataset split sizes without touching the dataset."""
    fractions = fractions or (1.0,)
    val_size = int(CIFAR10_TRAINSET_SIZE * config.data.val_fraction)
    train_pool_size = CIFAR10_TRAINSET_SIZE - val_size
    runs = []

    for fraction in fractions:
        train_size = train_pool_size if fraction >= 1.0 else max(1, int(train_pool_size * fraction))
        runs.append(
            {
                "train_fraction": fraction,
                "train_size": train_size,
                "val_size": val_size,
                "clean_test_size": CIFAR10_TESTSET_SIZE,
                "occluded_test_size": CIFAR10_TESTSET_SIZE,
                "texture_test_size": CIFAR10_TESTSET_SIZE,
            }
        )

    return {
        "dataset_name": config.data.name,
        "image_size": config.data.image_size,
        "channels": config.data.channels,
        "source_train_size": CIFAR10_TRAINSET_SIZE,
        "source_test_size": CIFAR10_TESTSET_SIZE,
        "val_fraction": config.data.val_fraction,
        "val_size": val_size,
        "train_pool_size": train_pool_size,
        "runs": runs,
    }


def build_train_transform(config: ProjectConfig) -> transforms.Compose:
    """Training uses light augmentation to improve generalization on the source dataset."""
    return transforms.Compose(
        [
            transforms.RandomCrop(config.data.image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(config.data.mean, config.data.std),
        ]
    )


def build_eval_transform(config: ProjectConfig) -> transforms.Compose:
    """Validation and test stay deterministic so comparisons remain controlled."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(config.data.mean, config.data.std),
        ]
    )


def _split_indices(dataset_size: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    # We split indices once with a fixed seed so CNN and ViT see the same
    # train/validation partition.
    indices = np.arange(dataset_size)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_size = int(dataset_size * val_fraction)
    val_indices = indices[:val_size].tolist()
    train_indices = indices[val_size:].tolist()
    return train_indices, val_indices


def _fraction_indices(indices: list[int], fraction: float, seed: int) -> list[int]:
    # Data-efficiency experiments reuse the same sampling procedure at each
    # fraction, which keeps the comparison reproducible.
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
    """Wrap the clean test set with the requested corruption at evaluation time."""
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
    """Build clean training data and optionally shifted test data.

    The project always trains on the clean distribution. Robustness is measured
    by replacing only the test set with an occluded or texture-modified view.
    """
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
            f"Unable to access {config.data.name}. If the dataset is not cached locally, "
            "please ensure the machine can download it once and rerun the project."
        )
        raise RuntimeError(message) from error

    train_indices, val_indices = _split_indices(
        dataset_size=len(train_dataset_augmented),
        val_fraction=config.data.val_fraction,
        seed=config.training.seed,
    )
    # The training subset is applied after the train/val split so the validation
    # set remains a consistent reference across data-efficiency runs.
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
    """Create the three loaders used throughout training and evaluation."""
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
