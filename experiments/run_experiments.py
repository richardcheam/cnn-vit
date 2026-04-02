from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import PercentFormatter
from torch.utils.data import DataLoader

from configs.config import ProjectConfig
from datasets.cifar_loader import DataBundle, build_dataloaders, describe_cifar_protocol
from evaluation.metrics import model_summary
from evaluation.robustness import summarize_shift
from interpretability.dhvt_attention import generate_dhvt_attention_maps
from interpretability.gradcam import GradCAM, overlay_heatmap
from interpretability.vit_attention import generate_attention_maps, overlay_attention_map
from models.cnn import CNN
from models.dhvt import DHVisionTransformer
from models.vit import VisionTransformer
from training.trainer import Trainer
from utils.helpers import ensure_dir, format_seconds, save_csv, save_json, save_torch_checkpoint, to_numpy_image


def _log(message: str) -> None:
    print(message, flush=True)


def _fraction_tag(fraction: float) -> str:
    return f"{int(round(fraction * 100))}pct"


PLOT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "#fbfcfe",
    "axes.edgecolor": "#667085",
    "axes.labelcolor": "#1f2937",
    "axes.titlecolor": "#111827",
    "axes.titlesize": 13,
    "axes.titleweight": "semibold",
    "axes.labelsize": 11,
    "xtick.color": "#344054",
    "ytick.color": "#344054",
    "grid.color": "#d0d7e2",
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "legend.frameon": True,
    "legend.facecolor": "white",
    "legend.edgecolor": "#d0d7e2",
    "legend.fancybox": False,
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
}

ARCHITECTURE_COLORS = {
    "cnn": "#1f4e79",
    "vit": "#b85c38",
    "dhvt": "#2f855a",
}

ARCHITECTURE_LINESTYLES = {
    "cnn": "-",
    "vit": "--",
    "dhvt": "-.",
}

SPLIT_STYLES = {
    "train": {"linestyle": "-"},
    "val": {"linestyle": "--"},
}

FRACTION_COLORS = {
    0.1: "#9fb3c8",
    0.25: "#6b8ba4",
    0.5: "#355c7d",
    1.0: "#b85c38",
}


def _style_axis(axis, title: str, xlabel: str, ylabel: str, *, percent_y: bool = False) -> None:
    axis.set_title(title, pad=10)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, axis="y", alpha=0.85)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.9)
    axis.spines["bottom"].set_linewidth(0.9)
    axis.tick_params(axis="both", labelsize=10)
    if percent_y:
        axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))


def _save_figure(figure, path: Path) -> None:
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(path, dpi=240)
    plt.close(figure)


def _add_bar_labels(axis, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#344054",
        )


def _log_dataset_protocol(config: ProjectConfig) -> None:
    protocol = describe_cifar_protocol(config, fractions=config.experiment.data_fractions)
    _log("Dataset protocol overview:")
    _log(
        f"  Source dataset: {protocol['dataset_name']} | "
        f"train_images={protocol['source_train_size']:,} | "
        f"test_images={protocol['source_test_size']:,} | "
        f"image_size={protocol['image_size']}x{protocol['image_size']}x{protocol['channels']}"
    )
    _log(
        f"  Validation split: {protocol['val_fraction']:.0%} | "
        f"validation_images={protocol['val_size']:,} | "
        f"remaining_train_pool={protocol['train_pool_size']:,}"
    )
    _log("  Planned data-efficiency runs:")
    for run in protocol["runs"]:
        _log(
            f"    - fraction={run['train_fraction']:.0%} | "
            f"train={run['train_size']:,} | "
            f"val={run['val_size']:,} | "
            f"clean_test={run['clean_test_size']:,}"
        )
    _log("  Robustness evaluation after full-data training:")
    _log(f"    - occluded_test={protocol['source_test_size']:,}")
    _log(f"    - texture_modified_test={protocol['source_test_size']:,}")


def build_model(model_name: str, config: ProjectConfig) -> torch.nn.Module:
    """Factory so every experiment uses the same model construction path."""
    if model_name == "cnn":
        return CNN(
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
    if model_name == "dhvt":
        return DHVisionTransformer(
            image_size=config.data.image_size,
            patch_size=config.dhvt.patch_size,
            num_classes=config.data.num_classes,
            embed_dim=config.dhvt.embed_dim,
            depth=config.dhvt.depth,
            num_heads=config.dhvt.num_heads,
            mlp_ratio=config.dhvt.mlp_ratio,
            dropout=config.dhvt.dropout,
            attention_dropout=config.dhvt.attention_dropout,
            drop_path_rate=config.dhvt.drop_path_rate,
        )
    raise ValueError(f"Unsupported model name: {model_name}")


def run_training(
    model_name: str,
    config: ProjectConfig,
    device: torch.device,
    train_fraction: float,
    checkpoint_dir: Path,
) -> tuple[torch.nn.Module, Trainer, DataBundle, dict]:
    """Train one model under one data-budget setting and return its summary."""
    dataset_name = config.data.name
    run_label = f"{model_name.upper()} | {dataset_name} | data={train_fraction:.0%}"
    _log(f"\n[{run_label}] Preparing {dataset_name} dataloaders.")
    data_bundle = build_dataloaders(config=config, train_fraction=train_fraction, test_variant="clean")
    _log(
        f"[{run_label}] Loader config | batch_size={config.data.batch_size}, "
        f"num_workers={config.data.num_workers}, pin_memory={data_bundle.train.pin_memory}"
    )
    _log(
        f"[{run_label}] Dataset sizes | "
        f"train={len(data_bundle.train_dataset)}, val={len(data_bundle.val_dataset)}, "
        f"test={len(data_bundle.test_dataset)}"
    )
    _log(f"[{run_label}] Building model.")
    model = build_model(model_name=model_name, config=config)
    parameter_count = model_summary(model)["parameter_count"]
    _log(f"[{run_label}] Model ready | trainable_parameters={parameter_count:,}")
    trainer = Trainer(
        model=model,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        device=device,
    )
    _log(
        f"[{run_label}] Starting training on {device} | "
        f"epochs={config.training.epochs}, batch_size={config.data.batch_size}"
    )
    history = trainer.fit(
        train_loader=data_bundle.train,
        val_loader=data_bundle.val,
        epochs=config.training.epochs,
        run_name=run_label,
    )
    _log(f"[{run_label}] Running clean test evaluation.")
    test_metrics = trainer.evaluate(data_bundle.test, label=f"{run_label} test")
    summary = {
        "model": model_name,
        "train_fraction": train_fraction,
        "train_size": len(data_bundle.train_dataset),
        "val_size": len(data_bundle.val_dataset),
        "test_size": len(data_bundle.test_dataset),
        "test_accuracy": round(test_metrics["accuracy"], 4),
        "test_loss": round(test_metrics["loss"], 4),
        "best_val_accuracy": round(history["best_val_accuracy"], 4),
        "training_time_seconds": round(history["training_time_seconds"], 2),
        "training_time_readable": format_seconds(history["training_time_seconds"]),
        "parameter_count": parameter_count,
        "history": history,
    }
    checkpoint_path = checkpoint_dir / f"{model_name}_{_fraction_tag(train_fraction)}_best.pt"
    checkpoint = {
        "model_name": model_name,
        "dataset": dataset_name,
        "dataset_slug": config.data.slug,
        "checkpoint_type": "best_validation_model",
        "train_fraction": train_fraction,
        "train_size": len(data_bundle.train_dataset),
        "val_size": len(data_bundle.val_dataset),
        "test_size": len(data_bundle.test_dataset),
        "best_val_accuracy": history["best_val_accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "num_classes": config.data.num_classes,
        "class_names": list(data_bundle.classes),
        "config": config.to_dict(),
        "history": history,
        "model_state_dict": model.state_dict(),
    }
    save_torch_checkpoint(checkpoint, checkpoint_path)
    summary["checkpoint_path"] = str(checkpoint_path)
    _log(f"[{run_label}] Saved best checkpoint to {checkpoint_path}")
    _log(
        f"[{run_label}] Finished | test_acc={summary['test_accuracy']:.4f}, "
        f"best_val_acc={summary['best_val_accuracy']:.4f}, "
        f"time={summary['training_time_readable']}"
    )
    return model, trainer, data_bundle, summary


def _save_training_curves(full_run_results: dict[str, dict], output_dir: Path) -> None:
    """Persist loss/accuracy trajectories for the full-data runs."""
    ensure_dir(output_dir)
    with plt.rc_context(PLOT_STYLE):
        for model_name, result in full_run_results.items():
            history = result["history"]
            epochs = range(1, len(history["train_loss"]) + 1)
            color = ARCHITECTURE_COLORS[model_name]

            figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
            axes[0].plot(
                epochs,
                history["train_loss"],
                color=color,
                linewidth=2.2,
                label="Train",
                **SPLIT_STYLES["train"],
            )
            axes[0].plot(
                epochs,
                history["val_loss"],
                color=color,
                linewidth=2.2,
                alpha=0.9,
                label="Validation",
                **SPLIT_STYLES["val"],
            )
            _style_axis(
                axes[0],
                title=f"{model_name.upper()} Loss",
                xlabel="Epoch",
                ylabel="Cross-Entropy Loss",
            )
            axes[0].legend(loc="upper right")

            axes[1].plot(
                epochs,
                history["train_accuracy"],
                color=color,
                linewidth=2.2,
                label="Train",
                **SPLIT_STYLES["train"],
            )
            axes[1].plot(
                epochs,
                history["val_accuracy"],
                color=color,
                linewidth=2.2,
                alpha=0.9,
                label="Validation",
                **SPLIT_STYLES["val"],
            )
            _style_axis(
                axes[1],
                title=f"{model_name.upper()} Accuracy",
                xlabel="Epoch",
                ylabel="Accuracy",
                percent_y=True,
            )
            axes[1].set_ylim(0.0, 1.0)
            axes[1].legend(loc="lower right")

            figure.suptitle(
                f"{model_name.upper()} Learning Curves",
                fontsize=14,
                fontweight="semibold",
                y=1.03,
            )
            _save_figure(figure, output_dir / f"{model_name}_training_curves.png")


def _save_combined_training_curves(full_run_results: dict[str, dict], output_dir: Path) -> None:
    """Save a shared full-data comparison figure for all available architectures."""
    if len(full_run_results) < 2:
        return
    ensure_dir(output_dir)
    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

        for model_name in full_run_results:
            history = full_run_results[model_name]["history"]
            epochs = range(1, len(history["train_loss"]) + 1)
            color = ARCHITECTURE_COLORS[model_name]
            label_prefix = model_name.upper()

            axes[0].plot(
                epochs,
                history["train_loss"],
                color=color,
                linewidth=2.3,
                label=f"{label_prefix} train",
                **SPLIT_STYLES["train"],
            )
            axes[0].plot(
                epochs,
                history["val_loss"],
                color=color,
                linewidth=2.3,
                alpha=0.9,
                label=f"{label_prefix} val",
                **SPLIT_STYLES["val"],
            )

            axes[1].plot(
                epochs,
                history["train_accuracy"],
                color=color,
                linewidth=2.3,
                label=f"{label_prefix} train",
                **SPLIT_STYLES["train"],
            )
            axes[1].plot(
                epochs,
                history["val_accuracy"],
                color=color,
                linewidth=2.3,
                alpha=0.9,
                label=f"{label_prefix} val",
                **SPLIT_STYLES["val"],
            )

        _style_axis(
            axes[0],
            title="Loss Comparison",
            xlabel="Epoch",
            ylabel="Cross-Entropy Loss",
        )
        _style_axis(
            axes[1],
            title="Accuracy Comparison",
            xlabel="Epoch",
            ylabel="Accuracy",
            percent_y=True,
        )
        axes[1].set_ylim(0.0, 1.0)
        axes[0].legend(loc="upper right", ncol=2)
        axes[1].legend(loc="lower right", ncol=2)

        model_labels = ", ".join(model_name.upper() for model_name in full_run_results)

        figure.suptitle(
            f"Full-Data Learning Curves: {model_labels}",
            fontsize=14,
            fontweight="semibold",
            y=1.03,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.98))
        figure.savefig(output_dir / "architecture_training_comparison.png", dpi=240)
        # Keep the older filename as a compatibility alias for existing notebook cells.
        figure.savefig(output_dir / "cnn_vit_training_comparison.png", dpi=240)
        plt.close(figure)


def _save_fraction_learning_curves(data_efficiency_runs: list[dict], output_dir: Path) -> None:
    """Show how each model's learning dynamics change with the available data budget."""
    ensure_dir(output_dir)
    model_names = list(dict.fromkeys(row["model"] for row in data_efficiency_runs))

    with plt.rc_context(PLOT_STYLE):
        for model_name in model_names:
            model_rows = sorted(
                [row for row in data_efficiency_runs if row["model"] == model_name],
                key=lambda row: row["train_fraction"],
            )
            if not model_rows:
                continue

            figure, axes = plt.subplots(1, 2, figsize=(13, 5))

            for row in model_rows:
                history = row.get("history")
                if not history:
                    continue

                fraction = row["train_fraction"]
                epochs = range(1, len(history["val_loss"]) + 1)
                color = FRACTION_COLORS.get(fraction, ARCHITECTURE_COLORS[model_name])
                label = f"{fraction:.0%} of train pool"

                axes[0].plot(
                    epochs,
                    history["val_loss"],
                    color=color,
                    linewidth=2.2,
                    label=label,
                )
                axes[1].plot(
                    epochs,
                    history["val_accuracy"],
                    color=color,
                    linewidth=2.2,
                    label=label,
                )

            _style_axis(
                axes[0],
                title=f"{model_name.upper()} Validation Loss by Data Fraction",
                xlabel="Epoch",
                ylabel="Cross-Entropy Loss",
            )
            _style_axis(
                axes[1],
                title=f"{model_name.upper()} Validation Accuracy by Data Fraction",
                xlabel="Epoch",
                ylabel="Accuracy",
                percent_y=True,
            )
            axes[1].set_ylim(0.0, 1.0)
            axes[0].legend(loc="upper right", title="Training data")
            axes[1].legend(loc="lower right", title="Training data")
            figure.suptitle(
                f"{model_name.upper()} Learning Curves Across Data Budgets",
                fontsize=14,
                fontweight="semibold",
                y=1.03,
            )
            _save_figure(figure, output_dir / f"{model_name}_data_fraction_curves.png")


def _save_data_efficiency_plot(config: ProjectConfig, rows: list[dict], output_dir: Path) -> None:
    """Plot how accuracy changes as the training set gets smaller."""
    ensure_dir(output_dir)
    with plt.rc_context(PLOT_STYLE):
        figure, axis = plt.subplots(figsize=(8.2, 4.8))
        model_names = list(dict.fromkeys(row["model"] for row in rows))
        for model_name in model_names:
            model_rows = sorted(
                [row for row in rows if row["model"] == model_name],
                key=lambda row: row["train_fraction"],
            )
            fractions = [row["train_fraction"] for row in model_rows]
            accuracies = [row["test_accuracy"] for row in model_rows]
            axis.plot(
                fractions,
                accuracies,
                color=ARCHITECTURE_COLORS[model_name],
                linestyle=ARCHITECTURE_LINESTYLES[model_name],
                linewidth=2.4,
                label=model_name.upper(),
            )

        _style_axis(
            axis,
            title=f"Data Efficiency on {config.data.name}",
            xlabel="Training Fraction",
            ylabel="Test Accuracy",
            percent_y=True,
        )
        axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        axis.set_ylim(0.0, 1.0)
        axis.legend(loc="lower right")
        _save_figure(figure, output_dir / "data_efficiency.png")


def _save_robustness_plot(config: ProjectConfig, rows: list[dict], output_dir: Path) -> None:
    """Visualize the clean-to-shift accuracy drop for each architecture."""
    if not rows:
        return
    ensure_dir(output_dir)
    with plt.rc_context(PLOT_STYLE):
        figure, axis = plt.subplots(figsize=(8.6, 4.8))
        shifts = ("occluded", "texture")
        shift_labels = {
            "occluded": "Occlusion",
            "texture": "Texture Shift",
        }
        positions = [0, 1]
        bar_width = 0.34

        model_names = list(dict.fromkeys(row["model"] for row in rows))
        center_offset = (len(model_names) - 1) / 2
        for index, model_name in enumerate(model_names):
            offsets = [position + (index - center_offset) * bar_width for position in positions]
            values = []
            for shift in shifts:
                row = next(item for item in rows if item["model"] == model_name and item["shift"] == shift)
                values.append(row["robustness_drop"])

            bars = axis.bar(
                offsets,
                values,
                width=bar_width,
                color=ARCHITECTURE_COLORS[model_name],
                alpha=0.92,
                label=model_name.upper(),
            )
            _add_bar_labels(axis, bars)

        _style_axis(
            axis,
            title=f"Robustness Drop Relative to Clean {config.data.name}",
            xlabel="Distribution Shift",
            ylabel="Accuracy Drop",
            percent_y=True,
        )
        axis.set_xticks(positions)
        axis.set_xticklabels([shift_labels[shift] for shift in shifts])
        max_drop = max(row["robustness_drop"] for row in rows)
        axis.set_ylim(0.0, max(0.08, max_drop * 1.25))
        axis.legend(loc="upper left")
        _save_figure(figure, output_dir / "robustness_drop.png")


def _sample_batch(dataset, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Grab a deterministic batch for qualitative visualization."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return next(iter(loader))


def _save_interpretability_examples(
    config: ProjectConfig,
    models: dict[str, torch.nn.Module],
    test_dataset,
    class_names: tuple[str, ...],
    device: torch.device,
    output_dir: Path,
) -> dict[str, str]:
    """Generate one qualitative figure for each interpretability method."""
    ensure_dir(output_dir)
    if not models:
        return {}

    _log("[Interpretability] Sampling examples and generating visual explanations.")
    images, labels = _sample_batch(test_dataset, batch_size=config.experiment.interpretability_samples)
    images = images.to(device)
    labels = labels.to(device)
    _log(
        f"[Interpretability] Batch moved to {device} | "
        f"images_shape={tuple(images.shape)} | labels_shape={tuple(labels.shape)}"
    )

    interpretability_paths: dict[str, str] = {}
    label_indices = labels.detach().cpu().tolist()

    if "cnn" in models:
        cnn_model = models["cnn"]
        cnn_model.eval()
        _log("[Interpretability] CNN Grad-CAM running.")
        gradcam = GradCAM(cnn_model, cnn_model.conv3)
        gradcam_maps = gradcam.generate(images)
        gradcam.close()

        with torch.no_grad():
            cnn_predictions = cnn_model(images).argmax(dim=1)
        cnn_prediction_indices = cnn_predictions.detach().cpu().tolist()

        gradcam_figure, gradcam_axes = plt.subplots(len(images), 2, figsize=(6, 3 * len(images)))
        if len(images) == 1:
            gradcam_axes = [gradcam_axes]

        for index in range(len(images)):
            base_image = images[index].detach().cpu()
            gradcam_overlay = overlay_heatmap(
                base_image,
                gradcam_maps[index].detach().cpu(),
                mean=config.data.mean,
                std=config.data.std,
            )
            title = f"true={class_names[label_indices[index]]}"
            gradcam_axes[index][0].imshow(gradcam_overlay)
            gradcam_axes[index][0].set_title(title)
            gradcam_axes[index][0].axis("off")
            gradcam_axes[index][1].imshow(gradcam_maps[index].detach().cpu(), cmap="inferno")
            gradcam_axes[index][1].set_title(f"pred={class_names[cnn_prediction_indices[index]]}")
            gradcam_axes[index][1].axis("off")

        gradcam_figure.tight_layout()
        gradcam_path = output_dir / "cnn_gradcam.png"
        gradcam_figure.savefig(gradcam_path, dpi=200)
        plt.close(gradcam_figure)
        interpretability_paths["cnn_gradcam"] = str(gradcam_path)

    if "vit" in models:
        vit_model = models["vit"]
        vit_model.eval()
        with torch.no_grad():
            _log("[Interpretability] VIT attention rollout running.")
            vit_logits, vit_maps = generate_attention_maps(vit_model, images)
            vit_predictions = vit_logits.argmax(dim=1)
        vit_prediction_indices = vit_predictions.detach().cpu().tolist()

        vit_figure, vit_axes = plt.subplots(len(images), 2, figsize=(6, 3 * len(images)))
        if len(images) == 1:
            vit_axes = [vit_axes]

        for index in range(len(images)):
            base_image = images[index].detach().cpu()
            vit_overlay = overlay_attention_map(
                base_image,
                vit_maps[index].detach().cpu(),
                mean=config.data.mean,
                std=config.data.std,
            )
            title = f"true={class_names[label_indices[index]]}"
            vit_axes[index][0].imshow(vit_overlay)
            vit_axes[index][0].set_title(title)
            vit_axes[index][0].axis("off")
            vit_axes[index][1].imshow(vit_maps[index].detach().cpu(), cmap="viridis")
            vit_axes[index][1].set_title(f"pred={class_names[vit_prediction_indices[index]]}")
            vit_axes[index][1].axis("off")

        vit_figure.tight_layout()
        vit_path = output_dir / "vit_attention.png"
        vit_figure.savefig(vit_path, dpi=200)
        plt.close(vit_figure)
        interpretability_paths["vit_attention"] = str(vit_path)

    if "dhvt" in models:
        dhvt_model = models["dhvt"]
        dhvt_model.eval()
        with torch.no_grad():
            _log("[Interpretability] DHVT rollout + head-token influence running.")
            dhvt_logits, rollout_maps, head_maps = generate_dhvt_attention_maps(dhvt_model, images)
            dhvt_predictions = dhvt_logits.argmax(dim=1)
        dhvt_prediction_indices = dhvt_predictions.detach().cpu().tolist()

        dhvt_figure, dhvt_axes = plt.subplots(len(images), 3, figsize=(9, 3 * len(images)))
        if len(images) == 1:
            dhvt_axes = [dhvt_axes]

        for index in range(len(images)):
            base_image = images[index].detach().cpu()
            rollout_overlay = overlay_attention_map(
                base_image,
                rollout_maps[index].detach().cpu(),
                mean=config.data.mean,
                std=config.data.std,
            )
            head_overlay = overlay_attention_map(
                base_image,
                head_maps[index].detach().cpu(),
                mean=config.data.mean,
                std=config.data.std,
            )
            dhvt_axes[index][0].imshow(to_numpy_image(base_image, mean=config.data.mean, std=config.data.std))
            dhvt_axes[index][0].set_title(f"true={class_names[label_indices[index]]}")
            dhvt_axes[index][0].axis("off")
            dhvt_axes[index][1].imshow(rollout_overlay)
            dhvt_axes[index][1].set_title(f"rollout pred={class_names[dhvt_prediction_indices[index]]}")
            dhvt_axes[index][1].axis("off")
            dhvt_axes[index][2].imshow(head_overlay)
            dhvt_axes[index][2].set_title("head-token influence")
            dhvt_axes[index][2].axis("off")

        dhvt_figure.tight_layout()
        dhvt_path = output_dir / "dhvt_attention.png"
        dhvt_figure.savefig(dhvt_path, dpi=200)
        plt.close(dhvt_figure)
        interpretability_paths["dhvt_attention"] = str(dhvt_path)

    return interpretability_paths


def run_experiments(config: ProjectConfig, device: torch.device) -> dict:
    """Run the complete comparison protocol and save all resulting artifacts."""
    root_output_dir = ensure_dir(config.experiment.output_dir)
    plots_dir = ensure_dir(root_output_dir / "plots")
    interpretability_dir = ensure_dir(root_output_dir / "interpretability")
    checkpoints_dir = ensure_dir(root_output_dir / "checkpoints")

    _log("=" * 80)
    _log(config.title)
    _log(f"Output directory: {root_output_dir}")
    _log(f"Checkpoint directory: {checkpoints_dir}")
    _log(f"Device: {device}")
    _log(f"Source dataset: {config.data.name}")
    _log(
        "Training fractions: "
        + ", ".join(f"{fraction:.0%}" for fraction in config.experiment.data_fractions)
    )
    _log(
        f"Epochs per run: {config.training.epochs} | "
        f"Batch size: {config.data.batch_size} | "
        f"Learning rate: {config.training.learning_rate}"
    )
    _log("Model families: " + ", ".join(model_name.upper() for model_name in config.experiment.model_names))
    _log("Planned stages: data-efficiency training, robustness evaluation, interpretability, artifact export")
    _log_dataset_protocol(config)
    _log("=" * 80)

    data_efficiency_rows: list[dict] = []
    data_efficiency_runs: list[dict] = []
    robustness_rows: list[dict] = []
    full_run_results: dict[str, dict] = {}
    trainers: dict[str, Trainer] = {}
    clean_bundles: dict[str, DataBundle] = {}
    trained_models: dict[str, torch.nn.Module] = {}
    selected_models = config.experiment.model_names

    # First, train both architectures at each data fraction. The full-data runs
    # are kept for later robustness and interpretability analyses.
    for model_name in selected_models:
        _log(f"\n===== Training family: {model_name.upper()} =====")
        for fraction in config.experiment.data_fractions:
            model, trainer, bundle, summary = run_training(
                model_name=model_name,
                config=config,
                device=device,
                train_fraction=fraction,
                checkpoint_dir=checkpoints_dir,
            )
            data_efficiency_rows.append(
                {
                    key: value
                    for key, value in summary.items()
                    if key != "history"
                }
            )
            data_efficiency_runs.append(summary)

            if fraction == 1.0:
                full_run_results[model_name] = summary
                trainers[model_name] = trainer
                clean_bundles[model_name] = bundle
                trained_models[model_name] = model
                _log(f"[{model_name.upper()}] Stored full-data model for later robustness and interpretability analysis.")

    baseline_rows = [
        {
            key: value
            for key, value in result.items()
            if key != "history"
        }
        for result in full_run_results.values()
    ]

    # Robustness is measured by testing the already-trained clean models on
    # shifted test sets, not by retraining on corrupted data.
    _log("\n===== Robustness evaluation =====")
    for model_name, trainer in trainers.items():
        clean_accuracy = full_run_results[model_name]["test_accuracy"]
        _log(f"[{model_name.upper()}] Building occluded and texture-shifted test sets.")
        occluded_bundle = build_dataloaders(config=config, train_fraction=1.0, test_variant="occluded")
        texture_bundle = build_dataloaders(config=config, train_fraction=1.0, test_variant="texture")
        occluded_metrics = trainer.evaluate(
            occluded_bundle.test,
            label=f"{model_name.upper()} occluded test",
        )
        texture_metrics = trainer.evaluate(
            texture_bundle.test,
            label=f"{model_name.upper()} texture test",
        )

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
        _log(
            f"[{model_name.upper()}] Robustness summary | "
            f"clean={clean_accuracy:.4f}, occluded={occluded_metrics['accuracy']:.4f}, "
            f"texture={texture_metrics['accuracy']:.4f}"
        )

    # Qualitative explanations are saved only for the full-data models because
    # they are the clearest reference point for comparison.
    _log("\n===== Interpretability =====")
    interpretability_paths = _save_interpretability_examples(
        config=config,
        models=trained_models,
        test_dataset=clean_bundles[selected_models[0]].test_dataset,
        class_names=clean_bundles[selected_models[0]].classes,
        device=device,
        output_dir=interpretability_dir,
    )
    if interpretability_paths:
        _log(
            "[Interpretability] Saved visualizations | "
            + " | ".join(f"{name}={path}" for name, path in interpretability_paths.items())
        )
    else:
        _log("[Interpretability] No interpretability artifacts were generated.")

    _log("\n===== Saving plots and tables =====")
    _save_training_curves(full_run_results=full_run_results, output_dir=plots_dir)
    if len(full_run_results) >= 2:
        _save_combined_training_curves(full_run_results=full_run_results, output_dir=plots_dir)
    else:
        _log("[Artifacts] Skipping combined training-curve plot because only one model family was selected.")
    _save_fraction_learning_curves(data_efficiency_runs=data_efficiency_runs, output_dir=plots_dir)
    _save_data_efficiency_plot(config=config, rows=data_efficiency_rows, output_dir=plots_dir)
    _save_robustness_plot(config=config, rows=robustness_rows, output_dir=plots_dir)
    _log(f"[Artifacts] Saved plots to {plots_dir}")

    summary = {
        "config": config.to_dict(),
        "dataset": config.data.name,
        "baseline": baseline_rows,
        "data_efficiency": data_efficiency_rows,
        "data_efficiency_runs": data_efficiency_runs,
        "robustness": robustness_rows,
        "checkpoints": {
            model_name: result["checkpoint_path"]
            for model_name, result in full_run_results.items()
        },
        "full_run_histories": {
            model_name: result["history"]
            for model_name, result in full_run_results.items()
        },
        "interpretability": interpretability_paths,
    }

    save_json(summary, root_output_dir / "summary.json")
    save_json(data_efficiency_runs, root_output_dir / "data_efficiency_runs.json")
    save_csv(data_efficiency_rows, root_output_dir / "data_efficiency.csv")
    save_csv(robustness_rows, root_output_dir / "robustness.csv")
    _log(
        f"[Artifacts] Saved summary files to {root_output_dir} | "
        "summary.json, data_efficiency.csv, robustness.csv"
    )
    _log("\nExperiment suite complete.")
    return summary
