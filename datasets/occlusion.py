from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SquareOcclusionTransform:
    mask_size: int
    fill_value: float = 0.0

    def __call__(
        self,
        image: torch.Tensor,
        rng: np.random.Generator | None = None,
    ) -> torch.Tensor:
        rng = rng or np.random.default_rng()
        occluded = image.clone()
        _, height, width = occluded.shape
        patch_size = min(self.mask_size, height, width)
        top = int(rng.integers(0, height - patch_size + 1))
        left = int(rng.integers(0, width - patch_size + 1))
        occluded[:, top : top + patch_size, left : left + patch_size] = self.fill_value
        return occluded


class OcclusionWrapperDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        mask_size: int,
        fill_value: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.transform = SquareOcclusionTransform(mask_size=mask_size, fill_value=fill_value)
        self.seed = seed

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        rng = np.random.default_rng(self.seed + index)
        return self.transform(image, rng=rng), label
