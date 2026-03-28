from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from models.cnn import CNN
from models.vit import VisionTransformer


def replace_classification_head(model: nn.Module, num_classes: int) -> None:
    """Replace the dataset-specific classifier while keeping the backbone."""
    if isinstance(model, CNN):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return

    if isinstance(model, VisionTransformer):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
        nn.init.xavier_uniform_(model.head.weight)
        nn.init.zeros_(model.head.bias)
        return

    raise TypeError(f"Unsupported model type for classifier replacement: {type(model)!r}")


def build_finetune_parameter_groups(
    model: nn.Module,
    backbone_learning_rate: float,
    head_learning_rate: float,
) -> list[dict]:
    """Use a smaller learning rate for pretrained features than for the new head."""
    head_parameters = _classifier_parameters(model)
    head_parameter_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in head_parameter_ids
    ]

    return [
        {"params": backbone_parameters, "lr": backbone_learning_rate},
        {"params": head_parameters, "lr": head_learning_rate},
    ]


def resolve_checkpoint_path(
    model_name: str,
    checkpoint_dir: str | Path,
    explicit_path: str | Path | None = None,
) -> Path:
    if explicit_path is not None:
        return Path(explicit_path)
    return Path(checkpoint_dir) / f"{model_name}_100pct_best.pt"


def load_pretrained_backbone(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> dict[str, list[str] | str]:
    """Load only transferable weights and skip the source-task classifier."""
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    filtered_state_dict = {}

    for key, value in state_dict.items():
        if _is_classifier_key(model, key):
            continue

        if isinstance(model, VisionTransformer) and key == "positional_embedding":
            value = _resize_vit_positional_embedding(
                saved_embedding=value,
                target_tokens=model.positional_embedding.shape[1],
            )

        filtered_state_dict[key] = value

    incompatible = model.load_state_dict(filtered_state_dict, strict=False)
    return {
        "checkpoint_path": str(checkpoint_path),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def _classifier_parameters(model: nn.Module) -> list[nn.Parameter]:
    if isinstance(model, CNN):
        return list(model.classifier.parameters())
    if isinstance(model, VisionTransformer):
        return list(model.head.parameters())
    raise TypeError(f"Unsupported model type for classifier lookup: {type(model)!r}")


def _is_classifier_key(model: nn.Module, key: str) -> bool:
    if isinstance(model, CNN):
        return key.startswith("classifier.")
    if isinstance(model, VisionTransformer):
        return key.startswith("head.")
    return False


def _resize_vit_positional_embedding(
    saved_embedding: torch.Tensor,
    target_tokens: int,
) -> torch.Tensor:
    """Interpolate patch-token embeddings when the downstream image grid changes."""
    if saved_embedding.shape[1] == target_tokens:
        return saved_embedding

    cls_token = saved_embedding[:, :1]
    patch_tokens = saved_embedding[:, 1:]
    embedding_dim = patch_tokens.shape[-1]
    source_grid = int(math.sqrt(patch_tokens.shape[1]))
    target_grid = int(math.sqrt(target_tokens - 1))

    if source_grid * source_grid != patch_tokens.shape[1]:
        raise ValueError("Saved ViT positional embedding does not form a square patch grid.")
    if target_grid * target_grid != target_tokens - 1:
        raise ValueError("Target ViT positional embedding does not form a square patch grid.")

    patch_tokens = patch_tokens.reshape(1, source_grid, source_grid, embedding_dim).permute(0, 3, 1, 2)
    patch_tokens = F.interpolate(
        patch_tokens,
        size=(target_grid, target_grid),
        mode="bicubic",
        align_corners=False,
    )
    patch_tokens = patch_tokens.permute(0, 2, 3, 1).reshape(1, target_grid * target_grid, embedding_dim)
    return torch.cat([cls_token, patch_tokens], dim=1)
