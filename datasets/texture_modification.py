from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class TextureShuffleTransform:
    patch_size: int
    shuffle_fraction: float = 0.75
    noise_std: float = 0.05

    def __call__(
        self,
        image: torch.Tensor,
        rng: np.random.Generator | None = None,
    ) -> torch.Tensor:
        rng = rng or np.random.default_rng()
        modified = image.clone()
        channels, height, width = modified.shape

        for top in range(0, height, self.patch_size):
            for left in range(0, width, self.patch_size):
                bottom = min(top + self.patch_size, height)
                right = min(left + self.patch_size, width)
                patch = modified[:, top:bottom, left:right]
                if patch.numel() == 0:
                    continue

                if rng.random() <= self.shuffle_fraction:
                    flattened = patch.reshape(channels, -1)
                    permutation = torch.as_tensor(
                        rng.permutation(flattened.shape[-1]),
                        dtype=torch.long,
                    )
                    patch = flattened[:, permutation].reshape_as(patch)

                if self.noise_std > 0.0:
                    noise = torch.as_tensor(
                        rng.normal(0.0, self.noise_std, size=patch.shape),
                        dtype=patch.dtype,
                    )
                    patch = patch + noise

                modified[:, top:bottom, left:right] = patch

        return modified


class TextureModifiedDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        patch_size: int,
        shuffle_fraction: float = 0.75,
        noise_std: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.transform = TextureShuffleTransform(
            patch_size=patch_size,
            shuffle_fraction=shuffle_fraction,
            noise_std=noise_std,
        )
        self.seed = seed

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        rng = np.random.default_rng(self.seed + index)
        return self.transform(image, rng=rng), label
