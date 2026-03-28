from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import PercentFormatter

from configs.config import ProjectConfig
from datasets.cifar_loader import DataBundle
from datasets.eurosat_loader import build_eurosat_dataloaders, describe_eurosat_protocol
from evaluation.metrics import model_summary
from models.cnn import CNN
from models.vit import VisionTransformer
from training.trainer import Trainer
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


def _build_eurosat_model(model_name: str, config: ProjectConfig) -> torch.nn.Module:
    if model_name == "cnn":
        return CNN(
            num_classes=config.eurosat.num_classes,
            channels=config.cnn.channels,
            dropout=config.cnn.dropout,
        )
    if model_name == "vit":
        return VisionTransformer(
            image_size=config.eurosat.image_size,
            patch_size=config.vit.patch_size,
            num_classes=config.eurosat.num_classes,
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
    labels = [f"{row['model'].upper()}\n{row['initialization']}" for row in ordered_rows]
    accuracies = [row["test_accuracy"] for row in ordered_rows]
    colors = [ARCHITECTURE_COLORS[row["model"]] for row in ordered_rows]
    alphas = [0.65 if row["initialization"] == "scratch" else 1.0 for row in ordered_rows]

    with plt.rc_context(PLOT_STYLE):
        figure, axis = plt.subplots(figsize=(9, 5.2))
        bars = axis.bar(labels, accuracies, color=colors, alpha=alphas, edgecolor="#344054", linewidth=0.8)
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
            title="EuroSAT Transfer Accuracy",
            xlabel="Model and initialization",
            ylabel="Test accuracy",
            percent_y=True,
        )
        axis.set_ylim(0.0, 1.0)
        _save_figure(figure, output_dir / "eurosat_transfer_accuracy.png")


def _save_transfer_validation_curves(results: list[dict], output_dir: Path) -> None:
    ensure_dir(output_dir)
    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(1, 2, figsize=(13, 5))

        for result in results:
            history = result["history"]
            epochs = range(1, len(history["val_loss"]) + 1)
            color = ARCHITECTURE_COLORS[result["model"]]
            linestyle = TRANSFER_LINESTYLES[result["initialization"]]
            label = f"{result['model'].upper()} {result['initialization']}"

            axes[0].plot(
                epochs,
                history["val_loss"],
                color=color,
                linestyle=linestyle,
                marker="o",
                linewidth=2.1,
                markersize=4.2,
                label=label,
            )
            axes[1].plot(
                epochs,
                history["val_accuracy"],
                color=color,
                linestyle=linestyle,
                marker="o",
                linewidth=2.1,
                markersize=4.2,
                label=label,
            )

        _style_axis(axes[0], "EuroSAT Validation Loss", "Epoch", "Cross-Entropy Loss")
        _style_axis(axes[1], "EuroSAT Validation Accuracy", "Epoch", "Accuracy", percent_y=True)
        axes[1].set_ylim(0.0, 1.0)
        axes[0].legend(loc="upper right")
        axes[1].legend(loc="lower right")
        figure.suptitle("Scratch vs CIFAR-Pretrained Fine-Tuning on EuroSAT", fontsize=14, fontweight="semibold")
        _save_figure(figure, output_dir / "eurosat_transfer_validation_curves.png")


def _save_downstream_checkpoint(
    model: torch.nn.Module,
    config: ProjectConfig,
    output_dir: Path,
    model_name: str,
    initialization: str,
    summary: dict,
    classes: tuple[str, ...],
) -> Path:
    checkpoint_path = output_dir / f"{model_name}_{initialization}_eurosat_best.pt"
    checkpoint = {
        "model_name": model_name,
        "dataset": "EuroSAT",
        "initialization": initialization,
        "config": config.to_dict(),
        "class_names": list(classes),
        "model_state_dict": model.state_dict(),
        "summary": {
            key: value
            for key, value in summary.items()
            if key != "history"
        },
    }
    save_torch_checkpoint(checkpoint, checkpoint_path)
    return checkpoint_path


def run_eurosat_transfer(config: ProjectConfig, device: torch.device) -> dict:
    root_output_dir = ensure_dir(config.transfer.output_dir)
    plots_dir = ensure_dir(root_output_dir / "plots")
    checkpoints_dir = ensure_dir(root_output_dir / "checkpoints")

    _log("=" * 80)
    _log("EuroSAT transfer-learning study")
    _log(f"Output directory: {root_output_dir}")
    _log(f"Checkpoint directory: {checkpoints_dir}")
    _log(f"Device: {device}")
    _log(
        f"Downstream epochs: {config.transfer.epochs} | "
        f"batch_size={config.eurosat.batch_size} | "
        f"train_fraction={config.eurosat.train_fraction:.0%}"
    )
    protocol = describe_eurosat_protocol(config)
    _log(
        f"EuroSAT overview | images={protocol.dataset_size:,} | classes={protocol.num_classes} | "
        f"image_size={config.eurosat.image_size}x{config.eurosat.image_size}x3"
    )
    _log(
        f"Split sizes | train={protocol.train_size:,}, val={protocol.val_size:,}, test={protocol.test_size:,}"
    )
    _log("Class names: " + ", ".join(protocol.class_names))

    run_modes = ("scratch", "pretrained")
    if config.transfer.run_mode == "scratch":
        run_modes = ("scratch",)
    if config.transfer.run_mode == "pretrained":
        run_modes = ("pretrained",)

    checkpoint_paths = {}
    if "pretrained" in run_modes:
        checkpoint_paths["cnn"] = resolve_checkpoint_path(
            model_name="cnn",
            checkpoint_dir=config.transfer.checkpoint_dir,
            explicit_path=config.transfer.cnn_checkpoint,
        )
        checkpoint_paths["vit"] = resolve_checkpoint_path(
            model_name="vit",
            checkpoint_dir=config.transfer.checkpoint_dir,
            explicit_path=config.transfer.vit_checkpoint,
        )
        for model_name, checkpoint_path in checkpoint_paths.items():
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Missing pretrained checkpoint for {model_name.upper()}: {checkpoint_path}. "
                    "Run the CIFAR-10 experiment first or pass an explicit checkpoint path."
                )

    rows: list[dict] = []
    detailed_results: list[dict] = []

    for model_name in ("cnn", "vit"):
        _log(f"\n===== EuroSAT model family: {model_name.upper()} =====")
        for initialization in run_modes:
            run_label = f"{model_name.upper()} | EuroSAT | {initialization}"
            _log(f"[{run_label}] Preparing EuroSAT dataloaders.")
            data_bundle = build_eurosat_dataloaders(config)
            _log(
                f"[{run_label}] Dataset sizes | "
                f"train={len(data_bundle.train_dataset)}, val={len(data_bundle.val_dataset)}, "
                f"test={len(data_bundle.test_dataset)}"
            )
            model = _build_eurosat_model(model_name, config)
            parameter_groups = None
            learning_rate = config.transfer.scratch_learning_rate
            preload_info = None

            if initialization == "pretrained":
                checkpoint_path = checkpoint_paths[model_name]
                preload_info = load_pretrained_backbone(model=model, checkpoint_path=checkpoint_path)
                parameter_groups = build_finetune_parameter_groups(
                    model=model,
                    backbone_learning_rate=config.transfer.backbone_learning_rate,
                    head_learning_rate=config.transfer.head_learning_rate,
                )
                learning_rate = config.transfer.head_learning_rate
                _log(
                    f"[{run_label}] Loaded CIFAR checkpoint: {checkpoint_path} | "
                    f"missing={len(preload_info['missing_keys'])}, unexpected={len(preload_info['unexpected_keys'])}"
                )
            else:
                _log(f"[{run_label}] Using random initialization.")

            parameter_count = model_summary(model)["parameter_count"]
            trainer = Trainer(
                model=model,
                learning_rate=learning_rate,
                weight_decay=config.transfer.weight_decay,
                device=device,
                parameter_groups=parameter_groups,
            )
            _log(
                f"[{run_label}] Starting downstream training | "
                f"epochs={config.transfer.epochs}, scratch_lr={config.transfer.scratch_learning_rate}, "
                f"backbone_lr={config.transfer.backbone_learning_rate}, head_lr={config.transfer.head_learning_rate}"
            )
            history = trainer.fit(
                train_loader=data_bundle.train,
                val_loader=data_bundle.val,
                epochs=config.transfer.epochs,
                run_name=run_label,
            )
            test_metrics = trainer.evaluate(data_bundle.test, label=f"{run_label} test")

            summary = {
                "dataset": "EuroSAT",
                "model": model_name,
                "initialization": initialization,
                "train_size": len(data_bundle.train_dataset),
                "val_size": len(data_bundle.val_dataset),
                "test_size": len(data_bundle.test_dataset),
                "image_size": config.eurosat.image_size,
                "test_accuracy": round(test_metrics["accuracy"], 4),
                "test_loss": round(test_metrics["loss"], 4),
                "best_val_accuracy": round(history["best_val_accuracy"], 4),
                "parameter_count": parameter_count,
                "training_time_seconds": round(history["training_time_seconds"], 2),
                "training_time_readable": format_seconds(history["training_time_seconds"]),
                "source_checkpoint": str(checkpoint_paths[model_name]) if initialization == "pretrained" else None,
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

            rows.append(
                {
                    key: value
                    for key, value in summary.items()
                    if key not in {"history", "preload_info"}
                }
            )
            detailed_results.append(summary)
            _log(
                f"[{run_label}] Finished | "
                f"test_acc={summary['test_accuracy']:.4f}, best_val_acc={summary['best_val_accuracy']:.4f}, "
                f"checkpoint={checkpoint_path}"
            )

    _save_transfer_accuracy_plot(rows=rows, output_dir=plots_dir)
    _save_transfer_validation_curves(results=detailed_results, output_dir=plots_dir)

    summary = {
        "config": config.to_dict(),
        "protocol": {
            "dataset_size": protocol.dataset_size,
            "train_size": protocol.train_size,
            "val_size": protocol.val_size,
            "test_size": protocol.test_size,
            "class_names": list(protocol.class_names),
        },
        "runs": rows,
        "histories": {
            f"{result['model']}_{result['initialization']}": result["history"]
            for result in detailed_results
        },
        "checkpoints": {
            f"{result['model']}_{result['initialization']}": result["checkpoint_path"]
            for result in detailed_results
        },
    }
    save_json(summary, root_output_dir / "summary.json")
    save_csv(rows, root_output_dir / "eurosat_transfer_results.csv")
    _log(f"[Artifacts] Saved EuroSAT results to {root_output_dir}")
    _log("=" * 80)
    return summary
