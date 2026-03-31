from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import f1_score

from configs.config import ProjectConfig
from datasets.brain_mri_loader import build_brain_mri_dataloaders, describe_brain_mri_protocol
from evaluation.metrics import model_summary
from interpretability.downstream import save_single_model_interpretability
from models.cnn import CNN
from models.vit import VisionTransformer
from training.trainer import Trainer
from utils.artifacts import load_brain_mri_runs_with_histories
from utils.helpers import ensure_dir, format_seconds, save_csv, save_json, save_torch_checkpoint
from utils.transfer import build_finetune_parameter_groups, load_pretrained_backbone, resolve_checkpoint_path


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
}

TRANSFER_LINESTYLES = {
    "scratch": "-",
    "pretrained": "--",
}


def _log(message: str) -> None:
    print(message, flush=True)


def _brain_mri_run_key(row: dict) -> tuple:
    return (
        row.get("dataset_slug", row.get("dataset")),
        row.get("model"),
        row.get("initialization"),
        row.get("train_size"),
        row.get("val_size"),
        row.get("test_size"),
    )


def _merge_brain_mri_runs(existing_runs: list[dict], new_runs: list[dict]) -> list[dict]:
    merged: dict[tuple, dict] = {}
    for row in existing_runs:
        merged[_brain_mri_run_key(row)] = row
    for row in new_runs:
        merged[_brain_mri_run_key(row)] = row
    return sorted(
        merged.values(),
        key=lambda row: (
            row.get("model", ""),
            row.get("initialization", ""),
            row.get("train_size", 0),
            row.get("val_size", 0),
            row.get("test_size", 0),
        ),
    )


def _summary_run_id(row: dict) -> str:
    train_size = row.get("train_size", "na")
    initialization = row.get("initialization", "run")
    model = row.get("model", "model")
    return f"{model}_{initialization}_{train_size}"


def _style_axis(axis, title: str, xlabel: str, ylabel: str, *, percent_y: bool = False) -> None:
    axis.set_title(title, pad=10)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, axis="y", alpha=0.85)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if percent_y:
        axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))


def _save_figure(figure, path: Path) -> None:
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(path, dpi=240)
    plt.close(figure)


@torch.no_grad()
def _collect_predictions(model: torch.nn.Module, loader, device: torch.device) -> tuple[list[int], list[int]]:
    predictions: list[int] = []
    targets: list[int] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=device.type != "cpu")
        labels = labels.to(device, non_blocking=device.type != "cpu")
        logits = model(images)
        batch_predictions = logits.argmax(dim=1)
        predictions.extend(batch_predictions.detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())
    return predictions, targets


def _build_brain_mri_model(model_name: str, config: ProjectConfig) -> torch.nn.Module:
    if model_name == "cnn":
        return CNN(
            num_classes=config.brain_mri.num_classes,
            channels=config.cnn.channels,
            dropout=config.cnn.dropout,
        )
    if model_name == "vit":
        return VisionTransformer(
            image_size=config.brain_mri.image_size,
            patch_size=config.vit.patch_size,
            num_classes=config.brain_mri.num_classes,
            embed_dim=config.vit.embed_dim,
            depth=config.vit.depth,
            num_heads=config.vit.num_heads,
            mlp_ratio=config.vit.mlp_ratio,
            dropout=config.vit.dropout,
            attention_dropout=config.vit.attention_dropout,
        )
    raise ValueError(f"Unsupported model name: {model_name}")


def _save_transfer_accuracy_plot(rows: list[dict], output_dir: Path) -> None:
    ensure_dir(output_dir)
    ordered_rows = sorted(rows, key=lambda row: (row["model"], row["initialization"]))
    unique_train_sizes = {row.get("train_size") for row in ordered_rows}
    show_train_size = len(unique_train_sizes) > 1
    labels = []
    for row in ordered_rows:
        label = f"{row['model'].upper()}\n{row['initialization']}"
        if show_train_size and row.get("train_size") is not None:
            label += f"\n{row['train_size']:,} train"
        labels.append(label)
    accuracies = [row["test_accuracy"] for row in ordered_rows]
    colors = [ARCHITECTURE_COLORS[row["model"]] for row in ordered_rows]
    alphas = [0.65 if row["initialization"] == "scratch" else 1.0 for row in ordered_rows]

    with plt.rc_context(PLOT_STYLE):
        figure, axis = plt.subplots(figsize=(9, 5.2))
        bars = axis.bar(labels, accuracies, color=colors, edgecolor="#344054", linewidth=0.8)
        for bar, alpha in zip(bars, alphas):
            bar.set_alpha(alpha)
        for bar, accuracy in zip(bars, accuracies):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                accuracy + 0.01,
                f"{accuracy:.1%}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#344054",
            )
        _style_axis(
            axis,
            title=f"{rows[0]['dataset']} Transfer Accuracy" if rows else "Transfer Accuracy",
            xlabel="Model and initialization",
            ylabel="Test accuracy",
            percent_y=True,
        )
        axis.set_ylim(0.0, 1.0)
        _save_figure(figure, output_dir / "brain_mri_transfer_accuracy.png")


def _save_transfer_validation_curves(results: list[dict], output_dir: Path) -> None:
    ensure_dir(output_dir)
    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(1, 2, figsize=(13, 5))
        dataset_name = results[0]["dataset"] if results else "Downstream"
        source_dataset = results[0].get("source_dataset", "source-stage") if results else "source-stage"
        unique_train_sizes = {row.get("train_size") for row in results}
        show_train_size = len(unique_train_sizes) > 1

        for result in results:
            history = result["history"]
            epochs = range(1, len(history["val_loss"]) + 1)
            color = ARCHITECTURE_COLORS[result["model"]]
            linestyle = TRANSFER_LINESTYLES[result["initialization"]]
            label = f"{result['model'].upper()} {result['initialization']}"
            if show_train_size and result.get("train_size") is not None:
                label += f" ({result['train_size']:,})"

            axes[0].plot(
                epochs,
                history["val_loss"],
                color=color,
                linestyle=linestyle,
                linewidth=2.1,
                label=label,
            )
            axes[1].plot(
                epochs,
                history["val_accuracy"],
                color=color,
                linestyle=linestyle,
                linewidth=2.1,
                label=label,
            )

        _style_axis(axes[0], f"{dataset_name} Validation Loss", "Epoch", "Cross-Entropy Loss")
        _style_axis(axes[1], f"{dataset_name} Validation Accuracy", "Epoch", "Accuracy", percent_y=True)
        axes[1].set_ylim(0.0, 1.0)
        axes[0].legend(loc="upper right")
        axes[1].legend(loc="lower right")
        figure.suptitle(
            f"Scratch vs {source_dataset}-Pretrained Fine-Tuning on {dataset_name}",
            fontsize=14,
            fontweight="semibold",
        )
        _save_figure(figure, output_dir / "brain_mri_transfer_validation_curves.png")


def _save_downstream_checkpoint(
    model: torch.nn.Module,
    config: ProjectConfig,
    output_dir: Path,
    model_name: str,
    initialization: str,
    summary: dict,
    classes: tuple[str, ...],
) -> Path:
    checkpoint_path = output_dir / f"{model_name}_{initialization}_{config.brain_mri.slug}_best.pt"
    checkpoint = {
        "model_name": model_name,
        "dataset": config.brain_mri.name,
        "dataset_slug": config.brain_mri.slug,
        "source_dataset": config.data.name,
        "source_dataset_slug": config.data.slug,
        "initialization": initialization,
        "config": config.to_dict(),
        "class_names": list(classes),
        "history": summary["history"],
        "model_state_dict": model.state_dict(),
        "summary": {
            key: value
            for key, value in summary.items()
            if key != "history"
        },
    }
    save_torch_checkpoint(checkpoint, checkpoint_path)
    return checkpoint_path


def run_brain_mri_transfer(config: ProjectConfig, device: torch.device) -> dict:
    root_output_dir = ensure_dir(config.brain_transfer.output_dir)
    plots_dir = ensure_dir(root_output_dir / "plots")
    checkpoints_dir = ensure_dir(root_output_dir / "checkpoints")

    _log("=" * 80)
    _log(f"{config.brain_mri.name} transfer-learning study")
    _log(f"Output directory: {root_output_dir}")
    _log(f"Checkpoint directory: {checkpoints_dir}")
    _log(f"Device: {device}")
    _log(f"Source dataset: {config.data.name}")
    _log(f"Downstream dataset: {config.brain_mri.name}")
    _log(
        f"Downstream epochs: {config.brain_transfer.epochs} | "
        f"batch_size={config.brain_mri.batch_size} | "
        f"train_fraction={config.brain_mri.train_fraction:.0%}"
    )
    _log("Model families: " + ", ".join(model_name.upper() for model_name in config.brain_transfer.model_names))
    protocol = describe_brain_mri_protocol(config)
    _log(
        f"{protocol.dataset_name} overview | train_source={protocol.train_source_size:,} | "
        f"test_source={protocol.test_source_size:,} | classes={protocol.num_classes} | "
        f"image_size={protocol.image_size}x{protocol.image_size}x{protocol.channels}"
    )
    _log(
        f"Split sizes | train={protocol.train_size:,}, val={protocol.val_size:,}, test={protocol.test_size:,}"
    )
    _log(f"Dataset root: {protocol.dataset_root}")
    _log("Class names: " + ", ".join(protocol.class_names))

    run_modes = ("scratch", "pretrained")
    if config.brain_transfer.run_mode == "scratch":
        run_modes = ("scratch",)
    if config.brain_transfer.run_mode == "pretrained":
        run_modes = ("pretrained",)
    selected_models = config.brain_transfer.model_names

    checkpoint_paths = {}
    if "pretrained" in run_modes:
        explicit_checkpoint_paths = {
            "cnn": config.brain_transfer.cnn_checkpoint,
            "vit": config.brain_transfer.vit_checkpoint,
        }
        for model_name in selected_models:
            checkpoint_paths[model_name] = resolve_checkpoint_path(
                model_name=model_name,
                checkpoint_dir=config.brain_transfer.checkpoint_dir,
                explicit_path=explicit_checkpoint_paths.get(model_name),
                dataset_slug=config.data.slug,
            )
        for model_name, checkpoint_path in checkpoint_paths.items():
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Missing pretrained checkpoint for {model_name.upper()}: {checkpoint_path}. "
                    f"Run the {config.data.name} experiment first or pass an explicit checkpoint path."
                )

    detailed_results: list[dict] = []

    for model_name in selected_models:
        _log(f"\n===== {config.brain_mri.name} model family: {model_name.upper()} =====")
        for initialization in run_modes:
            run_label = f"{model_name.upper()} | {config.brain_mri.name} | {initialization}"
            _log(f"[{run_label}] Preparing {config.brain_mri.name} dataloaders.")
            data_bundle = build_brain_mri_dataloaders(config)
            _log(
                f"[{run_label}] Dataset sizes | "
                f"train={len(data_bundle.train_dataset)}, val={len(data_bundle.val_dataset)}, "
                f"test={len(data_bundle.test_dataset)}"
            )
            model = _build_brain_mri_model(model_name, config)
            parameter_groups = None
            learning_rate = config.brain_transfer.scratch_learning_rate
            preload_info = None

            if initialization == "pretrained":
                checkpoint_path = checkpoint_paths[model_name]
                preload_info = load_pretrained_backbone(model=model, checkpoint_path=checkpoint_path)
                parameter_groups = build_finetune_parameter_groups(
                    model=model,
                    backbone_learning_rate=config.brain_transfer.backbone_learning_rate,
                    head_learning_rate=config.brain_transfer.head_learning_rate,
                )
                learning_rate = config.brain_transfer.head_learning_rate
                _log(
                    f"[{run_label}] Loaded {config.data.name} checkpoint: {checkpoint_path} | "
                    f"missing={len(preload_info['missing_keys'])}, unexpected={len(preload_info['unexpected_keys'])}"
                )
            else:
                _log(f"[{run_label}] Using random initialization.")

            parameter_count = model_summary(model)["parameter_count"]
            trainer = Trainer(
                model=model,
                learning_rate=learning_rate,
                weight_decay=config.brain_transfer.weight_decay,
                device=device,
                parameter_groups=parameter_groups,
            )
            _log(
                f"[{run_label}] Starting downstream training | "
                f"epochs={config.brain_transfer.epochs}, scratch_lr={config.brain_transfer.scratch_learning_rate}, "
                f"backbone_lr={config.brain_transfer.backbone_learning_rate}, "
                f"head_lr={config.brain_transfer.head_learning_rate}"
            )
            history = trainer.fit(
                train_loader=data_bundle.train,
                val_loader=data_bundle.val,
                epochs=config.brain_transfer.epochs,
                run_name=run_label,
            )
            test_metrics = trainer.evaluate(data_bundle.test, label=f"{run_label} test")
            test_predictions, test_targets = _collect_predictions(model, data_bundle.test, device=device)
            macro_f1 = f1_score(test_targets, test_predictions, average="macro")
            weighted_f1 = f1_score(test_targets, test_predictions, average="weighted")
            interpretability_paths = save_single_model_interpretability(
                model_name=model_name,
                model=model,
                dataset=data_bundle.test_dataset,
                class_names=data_bundle.classes,
                device=device,
                mean=config.brain_mri.mean,
                std=config.brain_mri.std,
                batch_size=config.experiment.interpretability_samples,
                output_dir=ensure_dir(root_output_dir / "interpretability"),
                output_stem=f"{model_name}_{initialization}",
                dataset_label=config.brain_mri.name,
            )

            summary = {
                "dataset": config.brain_mri.name,
                "dataset_slug": config.brain_mri.slug,
                "source_dataset": config.data.name,
                "source_dataset_slug": config.data.slug,
                "model": model_name,
                "initialization": initialization,
                "train_size": len(data_bundle.train_dataset),
                "val_size": len(data_bundle.val_dataset),
                "test_size": len(data_bundle.test_dataset),
                "image_size": config.brain_mri.image_size,
                "test_accuracy": round(test_metrics["accuracy"], 4),
                "test_loss": round(test_metrics["loss"], 4),
                "macro_f1": round(macro_f1, 4),
                "weighted_f1": round(weighted_f1, 4),
                "best_val_accuracy": round(history["best_val_accuracy"], 4),
                "parameter_count": parameter_count,
                "training_time_seconds": round(history["training_time_seconds"], 2),
                "training_time_readable": format_seconds(history["training_time_seconds"]),
                "source_checkpoint": str(checkpoint_paths[model_name]) if initialization == "pretrained" else None,
                "interpretability": interpretability_paths,
                "history": history,
            }
            checkpoint_path = _save_downstream_checkpoint(
                model=model,
                config=config,
                output_dir=checkpoints_dir,
                model_name=model_name,
                initialization=initialization,
                summary=summary,
                classes=data_bundle.classes,
            )
            summary["checkpoint_path"] = str(checkpoint_path)
            if preload_info is not None:
                summary["preload_info"] = preload_info

            detailed_results.append(summary)
            _log(
                f"[{run_label}] Finished | "
                f"test_acc={summary['test_accuracy']:.4f}, macro_f1={summary['macro_f1']:.4f}, "
                f"best_val_acc={summary['best_val_accuracy']:.4f}, "
                f"checkpoint={checkpoint_path}"
            )

    existing_detailed_results = load_brain_mri_runs_with_histories(root_output_dir)
    merged_detailed_results = _merge_brain_mri_runs(existing_detailed_results, detailed_results)
    merged_rows = [
        {
            key: value
            for key, value in result.items()
            if key not in {"history", "preload_info"}
        }
        for result in merged_detailed_results
    ]

    summary = {
        "config": config.to_dict(),
        "dataset": config.brain_mri.name,
        "source_dataset": config.data.name,
        "protocol": {
            "dataset_name": protocol.dataset_name,
            "dataset_root": protocol.dataset_root,
            "train_source_size": protocol.train_source_size,
            "test_source_size": protocol.test_source_size,
            "train_size": protocol.train_size,
            "val_size": protocol.val_size,
            "test_size": protocol.test_size,
            "class_names": list(protocol.class_names),
        },
        "runs": merged_rows,
        "detailed_runs": merged_detailed_results,
        "histories": {
            _summary_run_id(result): result["history"]
            for result in merged_detailed_results
        },
        "checkpoints": {
            _summary_run_id(result): result["checkpoint_path"]
            for result in merged_detailed_results
        },
    }
    save_json(summary, root_output_dir / "summary.json")
    save_json(merged_detailed_results, root_output_dir / "transfer_runs.json")
    save_csv(merged_rows, root_output_dir / "brain_mri_transfer_results.csv")

    _save_transfer_accuracy_plot(rows=merged_rows, output_dir=plots_dir)
    _save_transfer_validation_curves(results=merged_detailed_results, output_dir=plots_dir)
    _log(f"[Artifacts] Saved {config.brain_mri.name} results to {root_output_dir}")
    _log("=" * 80)
    return summary
