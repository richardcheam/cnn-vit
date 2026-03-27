from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from utils.helpers import to_numpy_image


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._forward_handle = target_layer.register_forward_hook(self._capture_activations)
        self._backward_handle = target_layer.register_full_backward_hook(self._capture_gradients)

    def _capture_activations(self, module, inputs, output) -> None:
        self.activations = output.detach()

    def _capture_gradients(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def generate(
        self,
        images: torch.Tensor,
        target_classes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(images)

        if target_classes is None:
            target_classes = logits.argmax(dim=1)

        selected_scores = logits.gather(1, target_classes.view(-1, 1)).sum()
        selected_scores.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmaps = (weights * self.activations).sum(dim=1)
        heatmaps = F.relu(heatmaps)
        heatmaps = F.interpolate(
            heatmaps.unsqueeze(1),
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        heatmaps = heatmaps - heatmaps.amin(dim=(1, 2), keepdim=True)
        heatmaps = heatmaps / heatmaps.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        return heatmaps


def overlay_heatmap(
    image: torch.Tensor,
    heatmap: torch.Tensor,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    alpha: float = 0.4,
) -> np.ndarray:
    base_image = to_numpy_image(image, mean=mean, std=std)
    heatmap_uint8 = np.uint8(255 * heatmap.detach().cpu().numpy())
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB) / 255.0
    return np.clip((1.0 - alpha) * base_image + alpha * colored_heatmap, 0.0, 1.0)
