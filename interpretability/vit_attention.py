from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from utils.helpers import to_numpy_image


def attention_rollout(attentions: list[torch.Tensor], image_size: int) -> torch.Tensor:
    if not attentions:
        raise ValueError("Attention rollout requires at least one attention tensor.")

    batch_size = attentions[0].size(0)
    num_tokens = attentions[0].size(-1)
    result = torch.eye(num_tokens, device=attentions[0].device).unsqueeze(0).repeat(batch_size, 1, 1)

    for attention in attentions:
        fused_attention = attention.mean(dim=1)
        identity = torch.eye(num_tokens, device=attention.device).unsqueeze(0)
        fused_attention = fused_attention + identity
        fused_attention = fused_attention / fused_attention.sum(dim=-1, keepdim=True)
        result = fused_attention @ result

    cls_attention = result[:, 0, 1:]
    grid_size = int(cls_attention.size(-1) ** 0.5)
    cls_attention = cls_attention.reshape(batch_size, 1, grid_size, grid_size)
    cls_attention = F.interpolate(
        cls_attention,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    cls_attention = cls_attention - cls_attention.amin(dim=(1, 2), keepdim=True)
    cls_attention = cls_attention / cls_attention.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return cls_attention


@torch.no_grad()
def generate_attention_maps(model, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits, attentions, _ = model(images, return_attentions=True)
    return logits, attention_rollout(attentions, image_size=images.shape[-1])


def overlay_attention_map(
    image: torch.Tensor,
    attention_map: torch.Tensor,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    alpha: float = 0.4,
) -> np.ndarray:
    base_image = to_numpy_image(image, mean=mean, std=std)
    heatmap_uint8 = np.uint8(255 * attention_map.detach().cpu().numpy())
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_VIRIDIS)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB) / 255.0
    return np.clip((1.0 - alpha) * base_image + alpha * colored_heatmap, 0.0, 1.0)
