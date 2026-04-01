from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from interpretability.dhvt_attention import generate_dhvt_attention_maps
from interpretability.gradcam import GradCAM, overlay_heatmap
from interpretability.vit_attention import generate_attention_maps, overlay_attention_map
from utils.helpers import ensure_dir, to_numpy_image


def sample_batch(dataset, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return next(iter(loader))


def save_single_model_interpretability(
    *,
    model_name: str,
    model: torch.nn.Module,
    dataset,
    class_names: tuple[str, ...] | list[str],
    device: torch.device,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    batch_size: int,
    output_dir: Path,
    output_stem: str,
    dataset_label: str,
) -> dict[str, str]:
    ensure_dir(output_dir)
    images, labels = sample_batch(dataset, batch_size=batch_size)
    images = images.to(device)
    labels = labels.to(device)

    if model_name == "cnn":
        model.eval()
        images = images.detach().clone().requires_grad_(True)
        gradcam = GradCAM(model, model.conv3)
        heatmaps = gradcam.generate(images)
        gradcam.close()
        with torch.no_grad():
            predictions = model(images).argmax(dim=1)

        figure, axes = plt.subplots(len(images), 3, figsize=(9, 3 * len(images)))
        if len(images) == 1:
            axes = [axes]

        for index in range(len(images)):
            base_image = to_numpy_image(images[index].detach().cpu(), mean=mean, std=std)
            overlay = overlay_heatmap(
                images[index].detach().cpu(),
                heatmaps[index].detach().cpu(),
                mean=mean,
                std=std,
            )
            axes[index][0].imshow(base_image)
            axes[index][0].set_title(f"true={class_names[int(labels[index].detach().cpu().item())]}")
            axes[index][0].axis("off")
            axes[index][1].imshow(overlay)
            axes[index][1].set_title("overlay")
            axes[index][1].axis("off")
            axes[index][2].imshow(heatmaps[index].detach().cpu(), cmap="inferno")
            axes[index][2].set_title(f"pred={class_names[int(predictions[index].detach().cpu().item())]}")
            axes[index][2].axis("off")

        figure.suptitle(f"{dataset_label} CNN Grad-CAM", fontsize=13, fontweight="semibold")
        figure.tight_layout()
        path = output_dir / f"{output_stem}_cnn_gradcam.png"
        figure.savefig(path, dpi=200)
        plt.close(figure)
        return {"cnn_gradcam": str(path)}

    if model_name == "vit":
        model.eval()
        with torch.no_grad():
            logits, attention_maps = generate_attention_maps(model, images)
            predictions = logits.argmax(dim=1)

        figure, axes = plt.subplots(len(images), 3, figsize=(9, 3 * len(images)))
        if len(images) == 1:
            axes = [axes]

        for index in range(len(images)):
            base_image = to_numpy_image(images[index].detach().cpu(), mean=mean, std=std)
            overlay = overlay_attention_map(
                images[index].detach().cpu(),
                attention_maps[index].detach().cpu(),
                mean=mean,
                std=std,
            )
            axes[index][0].imshow(base_image)
            axes[index][0].set_title(f"true={class_names[int(labels[index].detach().cpu().item())]}")
            axes[index][0].axis("off")
            axes[index][1].imshow(overlay)
            axes[index][1].set_title("overlay")
            axes[index][1].axis("off")
            axes[index][2].imshow(attention_maps[index].detach().cpu(), cmap="viridis")
            axes[index][2].set_title(f"pred={class_names[int(predictions[index].detach().cpu().item())]}")
            axes[index][2].axis("off")

        model_label = "ViT" if model_name == "vit" else "DHVT"
        figure.suptitle(f"{dataset_label} {model_label} Attention Rollout", fontsize=13, fontweight="semibold")
        figure.tight_layout()
        path = output_dir / f"{output_stem}_{model_name}_attention.png"
        figure.savefig(path, dpi=200)
        plt.close(figure)
        return {f"{model_name}_attention": str(path)}

    if model_name == "dhvt":
        model.eval()
        with torch.no_grad():
            logits, rollout_maps, head_maps = generate_dhvt_attention_maps(model, images)
            predictions = logits.argmax(dim=1)

        figure, axes = plt.subplots(len(images), 3, figsize=(9, 3 * len(images)))
        if len(images) == 1:
            axes = [axes]

        for index in range(len(images)):
            base_image = to_numpy_image(images[index].detach().cpu(), mean=mean, std=std)
            rollout_overlay = overlay_attention_map(
                images[index].detach().cpu(),
                rollout_maps[index].detach().cpu(),
                mean=mean,
                std=std,
            )
            head_overlay = overlay_attention_map(
                images[index].detach().cpu(),
                head_maps[index].detach().cpu(),
                mean=mean,
                std=std,
            )
            axes[index][0].imshow(base_image)
            axes[index][0].set_title(f"true={class_names[int(labels[index].detach().cpu().item())]}")
            axes[index][0].axis("off")
            axes[index][1].imshow(rollout_overlay)
            axes[index][1].set_title(f"rollout pred={class_names[int(predictions[index].detach().cpu().item())]}")
            axes[index][1].axis("off")
            axes[index][2].imshow(head_overlay)
            axes[index][2].set_title("head-token influence")
            axes[index][2].axis("off")

        figure.suptitle(f"{dataset_label} DHVT Attention", fontsize=13, fontweight="semibold")
        figure.tight_layout()
        path = output_dir / f"{output_stem}_dhvt_attention.png"
        figure.savefig(path, dpi=200)
        plt.close(figure)
        return {"dhvt_attention": str(path)}

    return {}
