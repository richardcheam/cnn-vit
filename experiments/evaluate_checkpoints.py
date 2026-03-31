from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import build_config
from datasets.brain_mri_loader import build_brain_mri_dataloaders
from datasets.cifar_loader import DataBundle, build_dataloaders
from datasets.eurosat_loader import build_eurosat_dataloaders
from evaluation.metrics import model_summary
from evaluation.robustness import summarize_shift
from experiments.run_brain_mri_transfer import PLOT_STYLE as BRAIN_TRANSFER_PLOT_STYLE
from experiments.run_brain_mri_transfer import _build_brain_mri_model
from experiments.run_eurosat_transfer import PLOT_STYLE as TRANSFER_PLOT_STYLE
from experiments.run_eurosat_transfer import _build_eurosat_model
from experiments.run_experiments import PLOT_STYLE as SOURCE_PLOT_STYLE
from experiments.run_experiments import build_model
from interpretability.gradcam import GradCAM, overlay_heatmap
from interpretability.vit_attention import generate_attention_maps, overlay_attention_map
from utils.artifacts import load_torch_checkpoint
from utils.helpers import ensure_dir, get_device, runtime_diagnostics, save_json, set_seed, to_numpy_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved checkpoints and generate interpretability artifacts without retraining.",
    )
    parser.add_argument(
        "--checkpoint-paths",
        nargs="+",
        type=Path,
        default=None,
        help="One or more explicit checkpoint paths to evaluate.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Optional directory containing checkpoints to evaluate.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pt",
        help="Glob pattern used when --checkpoint-dir is provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/checkpoint_evaluation"),
        help="Directory where evaluation summaries and figures will be saved.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Execution device such as cpu, cuda, or mps.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional evaluation batch size override.")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional DataLoader worker override.")
    parser.add_argument(
        "--interpretability-samples",
        type=int,
        default=4,
        help="Number of test images to visualize for Grad-CAM or attention rollout.",
    )
    return parser.parse_args()


def _log(message: str) -> None:
    print(message, flush=True)


def _discover_checkpoints(args: argparse.Namespace) -> list[Path]:
    paths = [path.resolve() for path in (args.checkpoint_paths or [])]
    if args.checkpoint_dir is not None:
        paths.extend(sorted(path.resolve() for path in args.checkpoint_dir.glob(args.pattern)))

    unique_paths = []
    seen = set()
    for path in paths:
        if path in seen:
            continue
        unique_paths.append(path)
        seen.add(path)

    if not unique_paths:
        raise ValueError("No checkpoints were provided. Pass --checkpoint-paths or --checkpoint-dir.")
    return unique_paths


def _tuple_or_default(value: Any, default: tuple) -> tuple:
    if value is None:
        return default
    return tuple(value)


def _apply_checkpoint_runtime_settings(config, checkpoint: dict[str, Any]) -> None:
    stored_config = checkpoint.get("config", {})
    source_data = stored_config.get("data", {})
    downstream_data = stored_config.get("eurosat", {})
    brain_data = stored_config.get("brain_mri", {})
    cnn_config = stored_config.get("cnn", {})
    vit_config = stored_config.get("vit", {})
    dhvt_config = stored_config.get("dhvt", {})
    augmentations = stored_config.get("augmentations", {})
    training = stored_config.get("training", {})

    if source_data:
        config.data.name = source_data.get("name", config.data.name)
        config.data.slug = source_data.get("slug", config.data.slug)
        config.data.image_size = source_data.get("image_size", config.data.image_size)
        config.data.channels = source_data.get("channels", config.data.channels)
        config.data.num_classes = source_data.get("num_classes", config.data.num_classes)
        config.data.mean = _tuple_or_default(source_data.get("mean"), config.data.mean)
        config.data.std = _tuple_or_default(source_data.get("std"), config.data.std)
        config.data.val_fraction = source_data.get("val_fraction", config.data.val_fraction)

    if downstream_data:
        config.eurosat.name = downstream_data.get("name", config.eurosat.name)
        config.eurosat.slug = downstream_data.get("slug", config.eurosat.slug)
        config.eurosat.image_size = downstream_data.get("image_size", config.eurosat.image_size)
        config.eurosat.channels = downstream_data.get("channels", config.eurosat.channels)
        config.eurosat.num_classes = downstream_data.get("num_classes", config.eurosat.num_classes)
        config.eurosat.mean = _tuple_or_default(downstream_data.get("mean"), config.eurosat.mean)
        config.eurosat.std = _tuple_or_default(downstream_data.get("std"), config.eurosat.std)
        config.eurosat.val_fraction = downstream_data.get("val_fraction", config.eurosat.val_fraction)
        config.eurosat.test_fraction = downstream_data.get("test_fraction", config.eurosat.test_fraction)
        config.eurosat.train_fraction = downstream_data.get("train_fraction", config.eurosat.train_fraction)

    if brain_data:
        config.brain_mri.name = brain_data.get("name", config.brain_mri.name)
        config.brain_mri.slug = brain_data.get("slug", config.brain_mri.slug)
        config.brain_mri.data_dir = Path(brain_data.get("data_dir", config.brain_mri.data_dir))
        config.brain_mri.image_size = brain_data.get("image_size", config.brain_mri.image_size)
        config.brain_mri.channels = brain_data.get("channels", config.brain_mri.channels)
        config.brain_mri.num_classes = brain_data.get("num_classes", config.brain_mri.num_classes)
        config.brain_mri.mean = _tuple_or_default(brain_data.get("mean"), config.brain_mri.mean)
        config.brain_mri.std = _tuple_or_default(brain_data.get("std"), config.brain_mri.std)
        config.brain_mri.val_fraction = brain_data.get("val_fraction", config.brain_mri.val_fraction)
        config.brain_mri.train_fraction = brain_data.get("train_fraction", config.brain_mri.train_fraction)

    if cnn_config:
        config.cnn.channels = _tuple_or_default(cnn_config.get("channels"), config.cnn.channels)
        config.cnn.dropout = cnn_config.get("dropout", config.cnn.dropout)

    if vit_config:
        config.vit.patch_size = vit_config.get("patch_size", config.vit.patch_size)
        config.vit.embed_dim = vit_config.get("embed_dim", config.vit.embed_dim)
        config.vit.depth = vit_config.get("depth", config.vit.depth)
        config.vit.num_heads = vit_config.get("num_heads", config.vit.num_heads)
        config.vit.mlp_ratio = vit_config.get("mlp_ratio", config.vit.mlp_ratio)
        config.vit.dropout = vit_config.get("dropout", config.vit.dropout)
        config.vit.attention_dropout = vit_config.get("attention_dropout", config.vit.attention_dropout)

    if dhvt_config:
        config.dhvt.patch_size = dhvt_config.get("patch_size", config.dhvt.patch_size)
        config.dhvt.embed_dim = dhvt_config.get("embed_dim", config.dhvt.embed_dim)
        config.dhvt.depth = dhvt_config.get("depth", config.dhvt.depth)
        config.dhvt.num_heads = dhvt_config.get("num_heads", config.dhvt.num_heads)
        config.dhvt.mlp_ratio = dhvt_config.get("mlp_ratio", config.dhvt.mlp_ratio)
        config.dhvt.dropout = dhvt_config.get("dropout", config.dhvt.dropout)
        config.dhvt.attention_dropout = dhvt_config.get("attention_dropout", config.dhvt.attention_dropout)
        config.dhvt.drop_path_rate = dhvt_config.get("drop_path_rate", config.dhvt.drop_path_rate)

    if augmentations:
        config.augmentations.occlusion_mask_size = augmentations.get(
            "occlusion_mask_size",
            config.augmentations.occlusion_mask_size,
        )
        config.augmentations.occlusion_fill_value = augmentations.get(
            "occlusion_fill_value",
            config.augmentations.occlusion_fill_value,
        )
        config.augmentations.texture_patch_size = augmentations.get(
            "texture_patch_size",
            config.augmentations.texture_patch_size,
        )
        config.augmentations.texture_shuffle_fraction = augmentations.get(
            "texture_shuffle_fraction",
            config.augmentations.texture_shuffle_fraction,
        )
        config.augmentations.texture_noise_std = augmentations.get(
            "texture_noise_std",
            config.augmentations.texture_noise_std,
        )

    if training:
        config.training.seed = training.get("seed", config.training.seed)


def _checkpoint_stage(checkpoint: dict[str, Any]) -> str:
    if checkpoint.get("train_fraction") is not None:
        return "source"
    if checkpoint.get("initialization") is not None:
        return "downstream"
    if checkpoint.get("summary", {}).get("initialization") is not None:
        return "downstream"
    if checkpoint.get("source_dataset") is not None:
        return "downstream"
    return "source"


def _resolve_dataset_name(checkpoint: dict[str, Any], config, stage: str) -> str:
    summary = checkpoint.get("summary", {})
    stored_config = checkpoint.get("config", {})

    if checkpoint.get("dataset"):
        return checkpoint["dataset"]
    if isinstance(summary, dict) and summary.get("dataset"):
        return summary["dataset"]

    if stage == "source":
        return stored_config.get("data", {}).get("name", config.data.name)
    return stored_config.get("eurosat", {}).get("name", config.eurosat.name)


def _resolve_source_dataset_name(checkpoint: dict[str, Any], config) -> str:
    summary = checkpoint.get("summary", {})
    stored_config = checkpoint.get("config", {})

    if checkpoint.get("source_dataset"):
        return checkpoint["source_dataset"]
    if isinstance(summary, dict) and summary.get("source_dataset"):
        return summary["source_dataset"]
    return stored_config.get("data", {}).get("name", config.data.name)


def _resolve_class_names(
    checkpoint: dict[str, Any],
    fallback_class_names: list[str] | tuple[str, ...],
    expected_count: int,
) -> list[str]:
    candidate = checkpoint.get("class_names")
    if isinstance(candidate, (list, tuple)):
        normalized = [str(item) for item in candidate]
        if len(normalized) == expected_count:
            return normalized

    summary = checkpoint.get("summary", {})
    summary_candidate = summary.get("class_names") if isinstance(summary, dict) else None
    if isinstance(summary_candidate, (list, tuple)):
        normalized = [str(item) for item in summary_candidate]
        if len(normalized) == expected_count:
            return normalized

    fallback = [str(item) for item in fallback_class_names]
    if len(fallback) == expected_count:
        return fallback

    return [f"class_{index}" for index in range(expected_count)]


def _class_name(index: int, class_names: list[str]) -> str:
    if 0 <= index < len(class_names):
        return class_names[index]
    return f"class_{index}"


def _build_model_from_checkpoint(checkpoint: dict[str, Any], config, device: torch.device) -> tuple[torch.nn.Module, dict]:
    stage = _checkpoint_stage(checkpoint)
    model_name = checkpoint["model_name"]
    if stage == "source":
        model = build_model(model_name=model_name, config=config)
    else:
        dataset_slug = checkpoint.get("dataset_slug", checkpoint.get("summary", {}).get("dataset_slug"))
        if dataset_slug in {"brain_tumor_mri", "brain_mri"}:
            model = _build_brain_mri_model(model_name=model_name, config=config)
        else:
            model = _build_eurosat_model(model_name=model_name, config=config)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model, {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


@torch.no_grad()
def _evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    label: str,
) -> dict[str, Any]:
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    predictions: list[int] = []
    targets: list[int] = []
    confidences: list[float] = []
    logged_first_batch = False

    iterator = tqdm(loader, desc=label, leave=False)
    for images, labels in iterator:
        images = images.to(device, non_blocking=device.type != "cpu")
        labels = labels.to(device, non_blocking=device.type != "cpu")

        if not logged_first_batch:
            tqdm.write(
                f"[{label}] First batch on {images.device} | "
                f"images_shape={tuple(images.shape)} | labels_shape={tuple(labels.shape)}"
            )
            logged_first_batch = True

        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        loss = criterion(logits, labels)
        batch_predictions = logits.argmax(dim=1)
        batch_confidences = probabilities.max(dim=1).values

        batch_size = labels.size(0)
        total_examples += batch_size
        total_loss += loss.item() * batch_size
        total_correct += (batch_predictions == labels).sum().item()
        predictions.extend(batch_predictions.detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())
        confidences.extend(batch_confidences.detach().cpu().tolist())

        iterator.set_postfix(
            loss=f"{total_loss / total_examples:.4f}",
            acc=f"{total_correct / total_examples:.4f}",
        )

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
        "predictions": predictions,
        "targets": targets,
        "confidences": confidences,
    }


def _save_confusion_matrix(
    targets: list[int],
    predictions: list[int],
    class_names: list[str],
    title: str,
    output_path: Path,
    plot_style: dict,
    normalize: str | None = None,
) -> None:
    with plt.rc_context(plot_style):
        figure, axis = plt.subplots(figsize=(8.5, 7.2))
        matrix = confusion_matrix(
            targets,
            predictions,
            labels=list(range(len(class_names))),
            normalize=normalize,
        )
        display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
        display.plot(
            ax=axis,
            cmap="Blues",
            colorbar=False,
            xticks_rotation=45,
            values_format=".2f" if normalize else "d",
        )
        axis.set_title(title, pad=12)
        figure.tight_layout()
        figure.savefig(output_path, dpi=220)
        plt.close(figure)


def _build_prediction_analysis(
    targets: list[int],
    predictions: list[int],
    confidences: list[float],
    class_names: list[str],
    report: dict[str, Any],
) -> dict[str, Any]:
    matrix = confusion_matrix(targets, predictions, labels=list(range(len(class_names))))
    per_class_metrics = []
    predicted_counts = [0 for _ in class_names]
    for prediction in predictions:
        predicted_counts[prediction] += 1

    for index, class_name in enumerate(class_names):
        support = int(matrix[index].sum())
        class_report = report.get(class_name, {})
        per_class_metrics.append(
            {
                "class_index": index,
                "class_name": class_name,
                "support": support,
                "correct": int(matrix[index, index]),
                "accuracy": round(matrix[index, index] / support, 4) if support else 0.0,
                "precision": round(float(class_report.get("precision", 0.0)), 4),
                "recall": round(float(class_report.get("recall", 0.0)), 4),
                "f1_score": round(float(class_report.get("f1-score", 0.0)), 4),
                "predicted_count": predicted_counts[index],
            }
        )

    confusion_pairs = []
    for true_index, true_name in enumerate(class_names):
        for predicted_index, predicted_name in enumerate(class_names):
            if true_index == predicted_index:
                continue
            count = int(matrix[true_index, predicted_index])
            if count <= 0:
                continue
            confusion_pairs.append(
                {
                    "true_class": true_name,
                    "predicted_class": predicted_name,
                    "count": count,
                }
            )
    confusion_pairs.sort(key=lambda row: row["count"], reverse=True)

    correct_confidences = [
        confidence
        for target, prediction, confidence in zip(targets, predictions, confidences)
        if target == prediction
    ]
    error_confidences = [
        confidence
        for target, prediction, confidence in zip(targets, predictions, confidences)
        if target != prediction
    ]

    return {
        "macro_avg": {
            "precision": round(float(report.get("macro avg", {}).get("precision", 0.0)), 4),
            "recall": round(float(report.get("macro avg", {}).get("recall", 0.0)), 4),
            "f1_score": round(float(report.get("macro avg", {}).get("f1-score", 0.0)), 4),
        },
        "weighted_avg": {
            "precision": round(float(report.get("weighted avg", {}).get("precision", 0.0)), 4),
            "recall": round(float(report.get("weighted avg", {}).get("recall", 0.0)), 4),
            "f1_score": round(float(report.get("weighted avg", {}).get("f1-score", 0.0)), 4),
        },
        "confidence": {
            "mean": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
            "mean_correct": round(sum(correct_confidences) / len(correct_confidences), 4) if correct_confidences else 0.0,
            "mean_incorrect": round(sum(error_confidences) / len(error_confidences), 4) if error_confidences else 0.0,
        },
        "per_class": per_class_metrics,
        "top_confusions": confusion_pairs[:10],
    }


def _sample_batch(dataset, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return next(iter(loader))


@torch.no_grad()
def _save_example_grid(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    class_names: list[str],
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    output_path: Path,
    *,
    selection: str,
    title: str,
    plot_style: dict,
    max_examples: int = 12,
) -> str | None:
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    examples = []

    for images, labels in loader:
        images = images.to(device, non_blocking=device.type != "cpu")
        labels = labels.to(device, non_blocking=device.type != "cpu")
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)
        confidences = probabilities.max(dim=1).values
        images_cpu = images.detach().cpu()
        label_indices = labels.detach().cpu().tolist()
        prediction_indices = predictions.detach().cpu().tolist()
        confidence_values = confidences.detach().cpu().tolist()

        for image, label_index, prediction_index, confidence in zip(
            images_cpu,
            label_indices,
            prediction_indices,
            confidence_values,
        ):
            is_correct = label_index == prediction_index
            if selection == "correct" and not is_correct:
                continue
            if selection == "misclassified" and is_correct:
                continue

            examples.append(
                {
                    "image": to_numpy_image(image, mean=mean, std=std),
                    "true_label": _class_name(label_index, class_names),
                    "predicted_label": _class_name(prediction_index, class_names),
                    "confidence": float(confidence),
                }
            )
            if len(examples) >= max_examples:
                break
        if len(examples) >= max_examples:
            break

    if not examples:
        return None

    columns = min(4, len(examples))
    rows = (len(examples) + columns - 1) // columns
    with plt.rc_context(plot_style):
        figure, axes = plt.subplots(rows, columns, figsize=(3.6 * columns, 3.2 * rows))
        axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

        for axis, example in zip(axes, examples):
            axis.imshow(example["image"], interpolation="nearest")
            axis.set_title(
                f"true={example['true_label']}\npred={example['predicted_label']} ({example['confidence']:.2f})",
                fontsize=9,
            )
            axis.axis("off")

        for axis in axes[len(examples) :]:
            axis.axis("off")

        figure.suptitle(title, fontsize=13, fontweight="semibold")
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        figure.savefig(output_path, dpi=220)
        plt.close(figure)
    return str(output_path)


def _save_interpretability(
    model_name: str,
    model: torch.nn.Module,
    dataset,
    class_names: list[str],
    device: torch.device,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    batch_size: int,
    output_dir: Path,
) -> list[str]:
    output_paths = []
    images, labels = _sample_batch(dataset, batch_size=batch_size)
    images = images.to(device)
    labels = labels.to(device)

    _log(
        f"[Interpretability] {model_name.upper()} batch on {device} | "
        f"images_shape={tuple(images.shape)} | labels_shape={tuple(labels.shape)}"
    )

    if model_name == "cnn":
        gradcam = GradCAM(model, model.conv3)
        heatmaps = gradcam.generate(images)
        gradcam.close()
        with torch.no_grad():
            predictions = model(images).argmax(dim=1)
        label_indices = labels.detach().cpu().tolist()
        prediction_indices = predictions.detach().cpu().tolist()

        figure, axes = plt.subplots(len(images), 3, figsize=(9, 3 * len(images)))
        if len(images) == 1:
            axes = [axes]

        for index in range(len(images)):
            base_image = to_numpy_image(images[index].detach().cpu(), mean=mean, std=std)
            overlay = overlay_heatmap(images[index].detach().cpu(), heatmaps[index].detach().cpu(), mean=mean, std=std)
            axes[index][0].imshow(base_image)
            axes[index][0].set_title(f"true={_class_name(label_indices[index], class_names)}")
            axes[index][0].axis("off")
            axes[index][1].imshow(overlay)
            axes[index][1].set_title("overlay")
            axes[index][1].axis("off")
            axes[index][2].imshow(heatmaps[index].detach().cpu(), cmap="inferno")
            axes[index][2].set_title(f"pred={_class_name(prediction_indices[index], class_names)}")
            axes[index][2].axis("off")

        figure.tight_layout()
        path = output_dir / "cnn_gradcam.png"
        figure.savefig(path, dpi=220)
        plt.close(figure)
        output_paths.append(str(path))
    else:
        with torch.no_grad():
            logits, attention_maps = generate_attention_maps(model, images)
            predictions = logits.argmax(dim=1)
        label_indices = labels.detach().cpu().tolist()
        prediction_indices = predictions.detach().cpu().tolist()

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
            axes[index][0].set_title(f"true={_class_name(label_indices[index], class_names)}")
            axes[index][0].axis("off")
            axes[index][1].imshow(overlay)
            axes[index][1].set_title("overlay")
            axes[index][1].axis("off")
            axes[index][2].imshow(attention_maps[index].detach().cpu(), cmap="viridis")
            axes[index][2].set_title(f"pred={_class_name(prediction_indices[index], class_names)}")
            axes[index][2].axis("off")

        figure.tight_layout()
        path = output_dir / f"{model_name}_attention.png"
        figure.savefig(path, dpi=220)
        plt.close(figure)
        output_paths.append(str(path))

    return output_paths


def _evaluate_source_checkpoint(
    checkpoint: dict[str, Any],
    config,
    model: torch.nn.Module,
    device: torch.device,
    output_dir: Path,
    interpretability_samples: int,
) -> dict[str, Any]:
    clean_bundle = build_dataloaders(config=config, train_fraction=1.0, test_variant="clean")
    occluded_bundle = build_dataloaders(config=config, train_fraction=1.0, test_variant="occluded")
    texture_bundle = build_dataloaders(config=config, train_fraction=1.0, test_variant="texture")

    class_names = _resolve_class_names(
        checkpoint=checkpoint,
        fallback_class_names=clean_bundle.classes,
        expected_count=config.data.num_classes,
    )
    clean_metrics = _evaluate_loader(model, clean_bundle.test, device, label="clean test")
    occluded_metrics = _evaluate_loader(model, occluded_bundle.test, device, label="occluded test")
    texture_metrics = _evaluate_loader(model, texture_bundle.test, device, label="texture test")

    report = classification_report(
        clean_metrics["targets"],
        clean_metrics["predictions"],
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    analysis = _build_prediction_analysis(
        targets=clean_metrics["targets"],
        predictions=clean_metrics["predictions"],
        confidences=clean_metrics["confidences"],
        class_names=class_names,
        report=report,
    )
    save_json(report, output_dir / "classification_report.json")
    save_json(analysis, output_dir / "prediction_analysis.json")
    _save_confusion_matrix(
        clean_metrics["targets"],
        clean_metrics["predictions"],
        class_names=class_names,
        title=f"{config.data.name} Confusion Matrix",
        output_path=output_dir / "confusion_matrix.png",
        plot_style=SOURCE_PLOT_STYLE,
    )
    _save_confusion_matrix(
        clean_metrics["targets"],
        clean_metrics["predictions"],
        class_names=class_names,
        title=f"{config.data.name} Normalized Confusion Matrix",
        output_path=output_dir / "confusion_matrix_normalized.png",
        plot_style=SOURCE_PLOT_STYLE,
        normalize="true",
    )
    interpretability_paths = _save_interpretability(
        model_name=checkpoint["model_name"],
        model=model,
        dataset=clean_bundle.test_dataset,
        class_names=class_names,
        device=device,
        mean=config.data.mean,
        std=config.data.std,
        batch_size=interpretability_samples,
        output_dir=ensure_dir(output_dir / "interpretability"),
    )
    example_dir = ensure_dir(output_dir / "examples")
    misclassified_examples_path = _save_example_grid(
        model=model,
        dataset=clean_bundle.test_dataset,
        device=device,
        class_names=class_names,
        mean=config.data.mean,
        std=config.data.std,
        output_path=example_dir / "misclassified_examples.png",
        selection="misclassified",
        title=f"{config.data.name} Misclassified Examples",
        plot_style=SOURCE_PLOT_STYLE,
    )
    correct_examples_path = _save_example_grid(
        model=model,
        dataset=clean_bundle.test_dataset,
        device=device,
        class_names=class_names,
        mean=config.data.mean,
        std=config.data.std,
        output_path=example_dir / "correct_examples.png",
        selection="correct",
        title=f"{config.data.name} Correct Predictions",
        plot_style=SOURCE_PLOT_STYLE,
    )

    summary = {
        "dataset": config.data.name,
        "dataset_slug": config.data.slug,
        "model": checkpoint["model_name"],
        "train_fraction": checkpoint.get("train_fraction"),
        "parameter_count": model_summary(model)["parameter_count"],
        "classification": analysis,
        "clean": {
            "loss": round(clean_metrics["loss"], 4),
            "accuracy": round(clean_metrics["accuracy"], 4),
        },
        "occluded": {
            "loss": round(occluded_metrics["loss"], 4),
            "accuracy": round(occluded_metrics["accuracy"], 4),
        },
        "texture": {
            "loss": round(texture_metrics["loss"], 4),
            "accuracy": round(texture_metrics["accuracy"], 4),
        },
        "robustness": [
            summarize_shift(
                model_name=checkpoint["model_name"],
                shift_name="occluded",
                clean_accuracy=round(clean_metrics["accuracy"], 4),
                shifted_accuracy=round(occluded_metrics["accuracy"], 4),
            ),
            summarize_shift(
                model_name=checkpoint["model_name"],
                shift_name="texture",
                clean_accuracy=round(clean_metrics["accuracy"], 4),
                shifted_accuracy=round(texture_metrics["accuracy"], 4),
            ),
        ],
        "artifacts": {
            "classification_report": str(output_dir / "classification_report.json"),
            "prediction_analysis": str(output_dir / "prediction_analysis.json"),
            "confusion_matrix": str(output_dir / "confusion_matrix.png"),
            "confusion_matrix_normalized": str(output_dir / "confusion_matrix_normalized.png"),
            "interpretability": interpretability_paths,
            "misclassified_examples": misclassified_examples_path,
            "correct_examples": correct_examples_path,
        },
    }
    return summary


def _evaluate_downstream_checkpoint(
    checkpoint: dict[str, Any],
    config,
    model: torch.nn.Module,
    device: torch.device,
    output_dir: Path,
    interpretability_samples: int,
) -> dict[str, Any]:
    dataset_slug = checkpoint.get("dataset_slug", checkpoint.get("summary", {}).get("dataset_slug"))
    if dataset_slug in {"brain_tumor_mri", "brain_mri"}:
        data_bundle = build_brain_mri_dataloaders(config)
        class_count = config.brain_mri.num_classes
        dataset_name = config.brain_mri.name
        dataset_slug = config.brain_mri.slug
        mean = config.brain_mri.mean
        std = config.brain_mri.std
        plot_style = BRAIN_TRANSFER_PLOT_STYLE
    else:
        data_bundle = build_eurosat_dataloaders(config)
        class_count = config.eurosat.num_classes
        dataset_name = config.eurosat.name
        dataset_slug = config.eurosat.slug
        mean = config.eurosat.mean
        std = config.eurosat.std
        plot_style = TRANSFER_PLOT_STYLE

    class_names = _resolve_class_names(
        checkpoint=checkpoint,
        fallback_class_names=data_bundle.classes,
        expected_count=class_count,
    )
    metrics = _evaluate_loader(model, data_bundle.test, device, label="downstream test")
    report = classification_report(
        metrics["targets"],
        metrics["predictions"],
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    analysis = _build_prediction_analysis(
        targets=metrics["targets"],
        predictions=metrics["predictions"],
        confidences=metrics["confidences"],
        class_names=class_names,
        report=report,
    )
    save_json(report, output_dir / "classification_report.json")
    save_json(analysis, output_dir / "prediction_analysis.json")
    _save_confusion_matrix(
        metrics["targets"],
        metrics["predictions"],
        class_names=class_names,
        title=f"{dataset_name} Confusion Matrix",
        output_path=output_dir / "confusion_matrix.png",
        plot_style=plot_style,
    )
    _save_confusion_matrix(
        metrics["targets"],
        metrics["predictions"],
        class_names=class_names,
        title=f"{dataset_name} Normalized Confusion Matrix",
        output_path=output_dir / "confusion_matrix_normalized.png",
        plot_style=plot_style,
        normalize="true",
    )
    interpretability_paths = _save_interpretability(
        model_name=checkpoint["model_name"],
        model=model,
        dataset=data_bundle.test_dataset,
        class_names=class_names,
        device=device,
        mean=mean,
        std=std,
        batch_size=interpretability_samples,
        output_dir=ensure_dir(output_dir / "interpretability"),
    )
    example_dir = ensure_dir(output_dir / "examples")
    misclassified_examples_path = _save_example_grid(
        model=model,
        dataset=data_bundle.test_dataset,
        device=device,
        class_names=class_names,
        mean=mean,
        std=std,
        output_path=example_dir / "misclassified_examples.png",
        selection="misclassified",
        title=f"{dataset_name} Misclassified Examples",
        plot_style=plot_style,
    )
    correct_examples_path = _save_example_grid(
        model=model,
        dataset=data_bundle.test_dataset,
        device=device,
        class_names=class_names,
        mean=mean,
        std=std,
        output_path=example_dir / "correct_examples.png",
        selection="correct",
        title=f"{dataset_name} Correct Predictions",
        plot_style=plot_style,
    )

    summary = {
        "dataset": dataset_name,
        "dataset_slug": dataset_slug,
        "source_dataset": checkpoint.get("source_dataset", config.data.name),
        "model": checkpoint["model_name"],
        "initialization": checkpoint.get("initialization", checkpoint.get("summary", {}).get("initialization")),
        "parameter_count": model_summary(model)["parameter_count"],
        "classification": analysis,
        "test": {
            "loss": round(metrics["loss"], 4),
            "accuracy": round(metrics["accuracy"], 4),
        },
        "artifacts": {
            "classification_report": str(output_dir / "classification_report.json"),
            "prediction_analysis": str(output_dir / "prediction_analysis.json"),
            "confusion_matrix": str(output_dir / "confusion_matrix.png"),
            "confusion_matrix_normalized": str(output_dir / "confusion_matrix_normalized.png"),
            "interpretability": interpretability_paths,
            "misclassified_examples": misclassified_examples_path,
            "correct_examples": correct_examples_path,
        },
    }
    return summary


def evaluate_checkpoint(
    checkpoint_path: Path,
    base_output_dir: Path,
    device: torch.device,
    batch_size: int | None,
    num_workers: int | None,
    interpretability_samples: int,
) -> dict[str, Any]:
    checkpoint = load_torch_checkpoint(checkpoint_path)
    config = build_config()
    _apply_checkpoint_runtime_settings(config, checkpoint)
    stage = _checkpoint_stage(checkpoint)
    set_seed(config.training.seed)
    dataset_name = _resolve_dataset_name(checkpoint, config, stage)
    source_dataset_name = _resolve_source_dataset_name(checkpoint, config)

    if batch_size is not None:
        config.data.batch_size = batch_size
        config.eurosat.batch_size = batch_size
        config.brain_mri.batch_size = batch_size
    if num_workers is not None:
        config.data.num_workers = num_workers
        config.eurosat.num_workers = num_workers
        config.brain_mri.num_workers = num_workers

    checkpoint_output_dir = ensure_dir(base_output_dir / checkpoint_path.stem)
    _log("=" * 80)
    _log(f"Evaluating checkpoint: {checkpoint_path}")
    _log(f"Detected stage: {stage}")
    _log(f"Output directory: {checkpoint_output_dir}")
    _log(f"Model: {checkpoint['model_name'].upper()} | dataset={dataset_name}")
    if stage == "downstream":
        _log(f"Source dataset: {source_dataset_name}")

    model, load_info = _build_model_from_checkpoint(checkpoint, config, device)
    if load_info["missing_keys"] or load_info["unexpected_keys"]:
        _log(
            f"Checkpoint load info | missing={len(load_info['missing_keys'])}, "
            f"unexpected={len(load_info['unexpected_keys'])}"
        )

    if stage == "source":
        summary = _evaluate_source_checkpoint(
            checkpoint=checkpoint,
            config=config,
            model=model,
            device=device,
            output_dir=checkpoint_output_dir,
            interpretability_samples=interpretability_samples,
        )
    else:
        summary = _evaluate_downstream_checkpoint(
            checkpoint=checkpoint,
            config=config,
            model=model,
            device=device,
            output_dir=checkpoint_output_dir,
            interpretability_samples=interpretability_samples,
        )

    summary["checkpoint_path"] = str(checkpoint_path)
    summary["load_info"] = load_info
    save_json(summary, checkpoint_output_dir / "summary.json")
    _log(f"Saved evaluation summary to {checkpoint_output_dir / 'summary.json'}")
    return summary


def main() -> None:
    args = parse_args()
    checkpoint_paths = _discover_checkpoints(args)
    device = get_device(args.device)

    print("Launching checkpoint evaluation...", flush=True)
    print(f"Checkpoints to evaluate: {len(checkpoint_paths)}", flush=True)
    print("\nRuntime diagnostics:", flush=True)
    for line in runtime_diagnostics(device):
        print(f"  - {line}", flush=True)

    output_dir = ensure_dir(args.output_dir)
    all_summaries = []
    for checkpoint_path in checkpoint_paths:
        all_summaries.append(
            evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                base_output_dir=output_dir,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                interpretability_samples=args.interpretability_samples,
            )
        )

    save_json(all_summaries, output_dir / "evaluation_index.json")
    print(f"\nSaved checkpoint evaluation index to {output_dir / 'evaluation_index.json'}", flush=True)


if __name__ == "__main__":
    main()
