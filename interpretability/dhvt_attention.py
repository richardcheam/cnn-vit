from __future__ import annotations

import torch
import torch.nn.functional as F

from interpretability.vit_attention import attention_rollout


def head_token_influence(full_attentions: list[torch.Tensor], image_size: int) -> torch.Tensor:
    """Aggregate DHVT head-token attention over patch tokens across layers."""
    if not full_attentions:
        raise ValueError("DHVT head-token influence requires at least one attention tensor.")

    layer_maps: list[torch.Tensor] = []
    for attention in full_attentions:
        num_attention_heads = attention.size(1)
        num_total_tokens = attention.size(-1)
        num_original_tokens = num_total_tokens - num_attention_heads

        # Queries from the generated head tokens, keys over the original patch tokens.
        patch_attention = attention[:, :, num_original_tokens:, 1:num_original_tokens]
        layer_map = patch_attention.mean(dim=(1, 2))
        layer_maps.append(layer_map)

    influence = torch.stack(layer_maps, dim=0).mean(dim=0)
    batch_size = influence.size(0)
    grid_size = int(influence.size(-1) ** 0.5)
    influence = influence.reshape(batch_size, 1, grid_size, grid_size)
    influence = F.interpolate(
        influence,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    influence = influence - influence.amin(dim=(1, 2), keepdim=True)
    influence = influence / influence.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return influence


@torch.no_grad()
def generate_dhvt_attention_maps(model, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return logits, rollout maps, and DHVT-specific head-token influence maps."""
    logits, attentions, _, full_attentions = model(images, return_attention_details=True)
    rollout_maps = attention_rollout(attentions, image_size=images.shape[-1])
    head_maps = head_token_influence(full_attentions, image_size=images.shape[-1])
    return logits, rollout_maps, head_maps
