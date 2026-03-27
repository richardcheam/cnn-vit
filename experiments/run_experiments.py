from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from configs.config import ProjectConfig
from datasets.cifar_loader import DataBundle, build_dataloaders
from evaluation.metrics import model_summary
from evaluation.robustness import summarize_shift
from interpretability.gradcam import GradCAM, overlay_heatmap
from interpretability.vit_attention import generate_attention_maps, overlay_attention_map
from models.cnn import SimpleCNN
from models.vit import VisionTransformer
from training.trainer import Trainer
from utils.helpers import ensure_dir, format_seconds, save_csv, save_json


def build_model(model_name: str, config: ProjectConfig) -> torch.nn.Module:
    if model_name == "cnn":
        return SimpleCNN(
            num_classes=config.data.num_classes,
            channels=config.cnn.channels,
            dropout=config.cnn.dropout,
        )
    if model_name == "vit":
        return VisionTransformer(
            image_size=config.data.image_size,
            patch_size=config.vit.patch_size,
            num_classes=config.data.num_classes,
            embed_dim=config.vit.embed_dim,
            depth=config.vit.depth,
            num_heads=config.vit.num_heads,
            mlp_ratio=config.vit.mlp_ratio,
            dropout=config.vit.dropout,
            attention_dropout=config.vit.attention_dropout,
        )
    raise ValueError(f"Unsupported model name: {model_name}")


def run_training(
    model_name: str,
    config: ProjectConfig,
    device: torch.device,
    train_fraction: float,
) -> tuple[torch.nn.Module, Trainer, DataBundle, dict]:
    data_bundle = build_dataloaders(config=config, train_fraction=train_fraction, test_variant="clean")
    model = build_model(model_name=model_name, config=config)
    trainer = Trainer(
        model=model,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        device=device,
    )
    history = trainer.fit(
        train_loader=data_bundle.train,
        val_loader=data_bundle.val,
        epochs=config.training.epochs,
    )
    test_metrics = trainer.evaluate(data_bundle.test)
    summary = {
        "model": model_name,
        "train_fraction": train_fraction,
        "test_accuracy": round(test_metrics["accuracy"], 4),
        "test_loss": round(test_metrics["loss"], 4),
        "best_val_accuracy": round(history["best_val_accuracy"], 4),
        "training_time_seconds": round(history["training_time_seconds"], 2),
        "training_time_readable": format_seconds(history["training_time_seconds"]),
        **model_summary(model),
        "history": history,
    }
    return model, trainer, data_bundle, summary


def _save_training_curves(full_run_results: dict[str, dict], output_dir: Path) -> None:
    ensure_dir(output_dir)
    for model_name, result in full_run_results.items():
        history = result["history"]
        epochs = range(1, len(history["train_loss"]) + 1)

        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(epochs, history["train_loss"], label="train")
        axes[0].plot(epochs, history["val_loss"], label="val")
        axes[0].set_title(f"{model_name.upper()} loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Cross-entropy")
        axes[0].legend()

        axes[1].plot(epochs, history["train_accuracy"], label="train")
        axes[1].plot(epochs, history["val_accuracy"], label="val")
        axes[1].set_title(f"{model_name.upper()} accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()

        figure.tight_layout()
        figure.savefig(output_dir / f"{model_name}_training_curves.png", dpi=200)
        plt.close(figure)


def _save_data_efficiency_plot(rows: list[dict], output_dir: Path) -> None:
    ensure_dir(output_dir)
    figure, axis = plt.subplots(figsize=(7, 4))
    for model_name in ("cnn", "vit"):
        model_rows = [row for row in rows if row["model"] == model_name]
        fractions = [row["train_fraction"] for row in model_rows]
        accuracies = [row["test_accuracy"] for row in model_rows]
        axis.plot(fractions, accuracies, marker="o", label=model_name.upper())

    axis.set_title("Data efficiency on CIFAR-10")
    axis.set_xlabel("Training fraction")
    axis.set_ylabel("Test accuracy")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "data_efficiency.png", dpi=200)
    plt.close(figure)


def _save_robustness_plot(rows: list[dict], output_dir: Path) -> None:
    ensure_dir(output_dir)
    figure, axis = plt.subplots(figsize=(8, 4))
    positions = [0, 1]
    bar_width = 0.35
    occlusion_drops = [row["robustness_drop"] for row in rows if row["shift"] == "occluded"]
    texture_drops = [row["robustness_drop"] for row in rows if row["shift"] == "texture"]

    axis.bar([position - bar_width / 2 for position in positions], occlusion_drops, bar_width, label="Occlusion")
    axis.bar([position + bar_width / 2 for position in positions], texture_drops, bar_width, label="Texture")
    axis.set_xticks(positions)
    axis.set_xticklabels([row["model"].upper() for row in rows if row["shift"] == "occluded"])
    axis.set_ylabel("Accuracy drop")
    axis.set_title("Robustness drop relative to clean CIFAR-10")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "robustness_drop.png", dpi=200)
    plt.close(figure)


def _sample_batch(dataset, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return next(iter(loader))


def _save_interpretability_examples(
    config: ProjectConfig,
    cnn_model: torch.nn.Module,
    vit_model: torch.nn.Module,
    test_dataset,
    class_names: tuple[str, ...],
    device: torch.device,
    output_dir: Path,
) -> dict[str, str]:
    ensure_dir(output_dir)
    images, labels = _sample_batch(test_dataset, batch_size=config.experiment.interpretability_samples)
    images = images.to(device)
    labels = labels.to(device)

    cnn_model.eval()
    vit_model.eval()

    gradcam = GradCAM(cnn_model, cnn_model.conv3)
    gradcam_maps = gradcam.generate(images)
    gradcam.close()

    with torch.no_grad():
        cnn_predictions = cnn_model(images).argmax(dim=1)
        vit_logits, vit_maps = generate_attention_maps(vit_model, images)
        vit_predictions = vit_logits.argmax(dim=1)

    gradcam_figure, gradcam_axes = plt.subplots(len(images), 2, figsize=(6, 3 * len(images)))
    vit_figure, vit_axes = plt.subplots(len(images), 2, figsize=(6, 3 * len(images)))

    if len(images) == 1:
        gradcam_axes = [gradcam_axes]
        vit_axes = [vit_axes]

    for index in range(len(images)):
        base_image = images[index].detach().cpu()
        gradcam_overlay = overlay_heatmap(
            base_image,
            gradcam_maps[index].detach().cpu(),
            mean=config.data.mean,
            std=config.data.std,
        )
        vit_overlay = overlay_attention_map(
            base_image,
            vit_maps[index].detach().cpu(),
            mean=config.data.mean,
            std=config.data.std,
        )

        title = f"true={class_names[labels[index].item()]}"
        gradcam_axes[index][0].imshow(gradcam_overlay)
        gradcam_axes[index][0].set_title(title)
        gradcam_axes[index][0].axis("off")
        gradcam_axes[index][1].imshow(gradcam_maps[index].detach().cpu(), cmap="inferno")
        gradcam_axes[index][1].set_title(f"pred={class_names[cnn_predictions[index].item()]}")
        gradcam_axes[index][1].axis("off")

        vit_axes[index][0].imshow(vit_overlay)
        vit_axes[index][0].set_title(title)
        vit_axes[index][0].axis("off")
        vit_axes[index][1].imshow(vit_maps[index].detach().cpu(), cmap="viridis")
        vit_axes[index][1].set_title(f"pred={class_names[vit_predictions[index].item()]}")
        vit_axes[index][1].axis("off")

    gradcam_figure.tight_layout()
    vit_figure.tight_layout()
    gradcam_path = output_dir / "cnn_gradcam.png"
    vit_path = output_dir / "vit_attention.png"
    gradcam_figure.savefig(gradcam_path, dpi=200)
    vit_figure.savefig(vit_path, dpi=200)
    plt.close(gradcam_figure)
    plt.close(vit_figure)

    return {
        "cnn_gradcam": str(gradcam_path),
        "vit_attention": str(vit_path),
    }


def run_experiments(config: ProjectConfig, device: torch.device) -> dict:
    root_output_dir = ensure_dir(config.experiment.output_dir)
    plots_dir = ensure_dir(root_output_dir / "plots")
    interpretability_dir = ensure_dir(root_output_dir / "interpretability")

    data_efficiency_rows: list[dict] = []
    robustness_rows: list[dict] = []
    full_run_results: dict[str, dict] = {}
    trainers: dict[str, Trainer] = {}
    clean_bundles: dict[str, DataBundle] = {}
    trained_models: dict[str, torch.nn.Module] = {}

    for model_name in ("cnn", "vit"):
        for fraction in config.experiment.data_fractions:
            model, trainer, bundle, summary = run_training(
                model_name=model_name,
                config=config,
                device=device,
                train_fraction=fraction,
            )
            data_efficiency_rows.append(
                {
                    key: value
                    for key, value in summary.items()
                    if key != "history"
                }
            )

            if fraction == 1.0:
                full_run_results[model_name] = summary
                trainers[model_name] = trainer
                clean_bundles[model_name] = bundle
                trained_models[model_name] = model

    baseline_rows = [
        {
            key: value
            for key, value in result.items()
            if key != "history"
        }
        for result in full_run_results.values()
    ]

    for model_name, trainer in trainers.items():
        clean_accuracy = full_run_results[model_name]["test_accuracy"]
        occluded_bundle = build_dataloaders(config=config, train_fraction=1.0, test_variant="occluded")
        texture_bundle = build_dataloaders(config=config, train_fraction=1.0, test_variant="texture")
        occluded_metrics = trainer.evaluate(occluded_bundle.test)
        texture_metrics = trainer.evaluate(texture_bundle.test)

        robustness_rows.append(
            summarize_shift(
                model_name=model_name,
                shift_name="occluded",
                clean_accuracy=clean_accuracy,
                shifted_accuracy=round(occluded_metrics["accuracy"], 4),
            )
        )
        robustness_rows.append(
            summarize_shift(
                model_name=model_name,
                shift_name="texture",
                clean_accuracy=clean_accuracy,
                shifted_accuracy=round(texture_metrics["accuracy"], 4),
            )
        )

    interpretability_paths = _save_interpretability_examples(
        config=config,
        cnn_model=trained_models["cnn"],
        vit_model=trained_models["vit"],
        test_dataset=clean_bundles["cnn"].test_dataset,
        class_names=clean_bundles["cnn"].classes,
        device=device,
        output_dir=interpretability_dir,
    )

    _save_training_curves(full_run_results=full_run_results, output_dir=plots_dir)
    _save_data_efficiency_plot(rows=data_efficiency_rows, output_dir=plots_dir)
    _save_robustness_plot(rows=robustness_rows, output_dir=plots_dir)

    summary = {
        "config": config.to_dict(),
        "baseline": baseline_rows,
        "data_efficiency": data_efficiency_rows,
        "robustness": robustness_rows,
        "interpretability": interpretability_paths,
    }

    save_json(summary, root_output_dir / "summary.json")
    save_csv(data_efficiency_rows, root_output_dir / "data_efficiency.csv")
    save_csv(robustness_rows, root_output_dir / "robustness.csv")
    return summary
