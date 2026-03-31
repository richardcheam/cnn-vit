from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_torch_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu")


def load_cifar_runs_with_histories(output_dir: str | Path) -> list[dict]:
    output_dir = Path(output_dir)
    summary_path = output_dir / "summary.json"
    runs_path = output_dir / "data_efficiency_runs.json"

    if runs_path.exists():
        return load_json(runs_path)

    if summary_path.exists():
        summary = load_json(summary_path)
        if "data_efficiency_runs" in summary:
            return summary["data_efficiency_runs"]

    return _load_cifar_runs_from_checkpoints(output_dir / "checkpoints")


def load_eurosat_runs_with_histories(output_dir: str | Path) -> list[dict]:
    output_dir = Path(output_dir)
    summary_path = output_dir / "summary.json"
    runs_path = output_dir / "transfer_runs.json"
    checkpoint_runs = _load_eurosat_runs_from_checkpoints(output_dir / "checkpoints")

    if runs_path.exists():
        return _merge_transfer_runs(load_json(runs_path), checkpoint_runs)

    if summary_path.exists():
        summary = load_json(summary_path)
        if "detailed_runs" in summary:
            return _merge_transfer_runs(summary["detailed_runs"], checkpoint_runs)
        if "runs" in summary and "histories" in summary:
            detailed_runs = []
            for row in summary["runs"]:
                key = (
                    f"{row['model']}_{row['initialization']}_{row.get('adaptation', 'na')}_{row.get('train_size', 'na')}"
                )
                detailed = dict(row)
                detailed["history"] = summary["histories"].get(key)
                detailed_runs.append(detailed)
            return _merge_transfer_runs(detailed_runs, checkpoint_runs)

    return checkpoint_runs


def load_brain_mri_runs_with_histories(output_dir: str | Path) -> list[dict]:
    output_dir = Path(output_dir)
    summary_path = output_dir / "summary.json"
    runs_path = output_dir / "transfer_runs.json"
    checkpoint_runs = _load_brain_mri_runs_from_checkpoints(output_dir / "checkpoints")

    if runs_path.exists():
        return _merge_transfer_runs(load_json(runs_path), checkpoint_runs)

    if summary_path.exists():
        summary = load_json(summary_path)
        if "detailed_runs" in summary:
            return _merge_transfer_runs(summary["detailed_runs"], checkpoint_runs)
        if "runs" in summary and "histories" in summary:
            detailed_runs = []
            for row in summary["runs"]:
                key = (
                    f"{row['model']}_{row['initialization']}_{row.get('adaptation', 'na')}_{row.get('train_size', 'na')}"
                )
                detailed = dict(row)
                detailed["history"] = summary["histories"].get(key)
                detailed_runs.append(detailed)
            return _merge_transfer_runs(detailed_runs, checkpoint_runs)

    return checkpoint_runs


def _load_cifar_runs_from_checkpoints(checkpoint_dir: Path) -> list[dict]:
    if not checkpoint_dir.exists():
        return []

    runs = []
    for checkpoint_path in sorted(checkpoint_dir.glob("*_best.pt")):
        checkpoint = load_torch_checkpoint(checkpoint_path)
        if "train_fraction" not in checkpoint or "train_size" not in checkpoint:
            continue
        history = checkpoint.get("history")
        if history is None:
            continue

        runs.append(
            {
                "model": checkpoint["model_name"],
                "train_fraction": checkpoint["train_fraction"],
                "train_size": checkpoint["train_size"],
                "val_size": checkpoint["val_size"],
                "test_size": checkpoint["test_size"],
                "test_accuracy": checkpoint.get("test_accuracy"),
                "best_val_accuracy": checkpoint.get("best_val_accuracy"),
                "checkpoint_path": str(checkpoint_path),
                "history": history,
            }
        )

    return sorted(runs, key=lambda row: (row["model"], row["train_fraction"]))


def _load_eurosat_runs_from_checkpoints(checkpoint_dir: Path) -> list[dict]:
    if not checkpoint_dir.exists():
        return []

    runs = []
    for checkpoint_path in sorted(checkpoint_dir.glob("*_best.pt")):
        checkpoint = load_torch_checkpoint(checkpoint_path)
        history = checkpoint.get("history")
        summary = checkpoint.get("summary", {})
        if history is None or summary.get("initialization") is None:
            continue

        detailed = dict(summary)
        detailed["history"] = history
        detailed["checkpoint_path"] = str(checkpoint_path)
        runs.append(detailed)

    return sorted(runs, key=lambda row: (row["model"], row["initialization"]))


def _load_brain_mri_runs_from_checkpoints(checkpoint_dir: Path) -> list[dict]:
    if not checkpoint_dir.exists():
        return []

    runs = []
    for checkpoint_path in sorted(checkpoint_dir.glob("*_best.pt")):
        checkpoint = load_torch_checkpoint(checkpoint_path)
        history = checkpoint.get("history")
        summary = checkpoint.get("summary", {})
        if history is None or summary.get("initialization") is None:
            continue
        if summary.get("dataset_slug") not in {"brain_tumor_mri", "brain_mri"}:
            continue

        detailed = dict(summary)
        detailed["history"] = history
        detailed["checkpoint_path"] = str(checkpoint_path)
        runs.append(detailed)

    return sorted(runs, key=lambda row: (row["model"], row["initialization"]))


def _merge_transfer_runs(existing_runs: list[dict], checkpoint_runs: list[dict]) -> list[dict]:
    merged: dict[tuple, dict] = {}
    for row in existing_runs:
        merged[_transfer_run_key(row)] = row
    for row in checkpoint_runs:
        merged[_transfer_run_key(row)] = row
    return sorted(
        merged.values(),
        key=lambda row: (
            row.get("model", ""),
            row.get("initialization", ""),
            row.get("adaptation", ""),
            row.get("train_size", 0),
            row.get("val_size", 0),
            row.get("test_size", 0),
        ),
    )


def _transfer_run_key(row: dict) -> tuple:
    return (
        row.get("dataset_slug", row.get("dataset")),
        row.get("model"),
        row.get("initialization"),
        row.get("adaptation"),
        row.get("train_size"),
        row.get("val_size"),
        row.get("test_size"),
    )
