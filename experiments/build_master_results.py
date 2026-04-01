from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import ensure_dir, save_csv, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a canonical master results table from JSON-backed artifacts.",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Root outputs directory containing stage summaries and checkpoint evaluations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/master_results"),
        help="Directory where the aggregated master tables will be saved.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_adaptation(initialization: str | None, adaptation: str | None) -> str:
    if initialization == "scratch":
        return "full_finetune"
    if adaptation in {None, "", "na"}:
        return "full_finetune" if initialization == "pretrained" else "na"
    return adaptation


def _source_fraction_label(train_fraction: float | None) -> str | None:
    if train_fraction is None:
        return None
    return f"{int(round(train_fraction * 100))}%"


def _load_checkpoint_evaluations(outputs_dir: Path) -> dict[str, tuple[dict[str, Any], Path]]:
    evaluations: dict[str, tuple[dict[str, Any], Path]] = {}
    canonical_root = outputs_dir / "checkpoint_evaluation"
    if canonical_root.is_dir():
        evaluation_roots = [canonical_root]
    else:
        evaluation_roots = sorted(eval_root for eval_root in outputs_dir.glob("checkpoint_evaluation*") if eval_root.is_dir())

    for eval_root in evaluation_roots:
        if not eval_root.is_dir():
            continue
        for summary_path in sorted(eval_root.glob("*/summary.json")):
            data = _read_json(summary_path)
            checkpoint_path = data.get("checkpoint_path")
            if checkpoint_path:
                evaluations[str(Path(checkpoint_path))] = (data, summary_path)
    return evaluations


def _source_rows(outputs_dir: Path, evaluations: dict[str, tuple[dict[str, Any], Path]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for checkpoint_path, (summary, summary_path) in sorted(evaluations.items()):
        if summary.get("dataset_slug") != "cifar10":
            continue
        classification = summary.get("classification", {})
        macro_avg = classification.get("macro_avg", {})
        robustness_rows = summary.get("robustness", [])
        rows.append(
            {
                "stage": "source",
                "dataset": summary.get("dataset"),
                "dataset_slug": summary.get("dataset_slug"),
                "source_dataset": None,
                "source_dataset_slug": None,
                "model": summary.get("model"),
                "initialization": "pretrained_source",
                "adaptation": "na",
                "train_fraction": summary.get("train_fraction"),
                "train_fraction_label": _source_fraction_label(summary.get("train_fraction")),
                "train_size": None,
                "val_size": None,
                "test_size": None,
                "accuracy": summary.get("clean", {}).get("accuracy"),
                "macro_f1": macro_avg.get("f1_score"),
                "macro_precision": macro_avg.get("precision"),
                "macro_recall": macro_avg.get("recall"),
                "best_val_accuracy": None,
                "clean_accuracy": summary.get("clean", {}).get("accuracy"),
                "occluded_accuracy": summary.get("occluded", {}).get("accuracy"),
                "texture_accuracy": summary.get("texture", {}).get("accuracy"),
                "occluded_drop": _shift_value(robustness_rows, "occluded"),
                "texture_drop": _shift_value(robustness_rows, "texture"),
                "parameter_count": summary.get("parameter_count"),
                "training_time_seconds": None,
                "training_time_readable": None,
                "checkpoint_path": checkpoint_path,
                "evaluation_summary_path": str(summary_path),
            }
        )
    return rows


def _downstream_rows(
    outputs_dir: Path,
    evaluations: dict[str, tuple[dict[str, Any], Path]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage_dir in [outputs_dir / "eurosat_transfer", outputs_dir / "brain_mri_transfer"]:
        runs_path = stage_dir / "transfer_runs.json"
        summary_path = stage_dir / "summary.json"
        if runs_path.exists():
            run_rows = _read_json(runs_path)
        elif summary_path.exists():
            summary = _read_json(summary_path)
            run_rows = summary.get("runs") or summary.get("detailed_runs") or []
        else:
            continue

        for row in run_rows:
            checkpoint_path = row.get("checkpoint_path")
            evaluation, evaluation_summary_path = evaluations.get(str(Path(checkpoint_path)), ({}, None))
            classification = evaluation.get("classification", {})
            macro_avg = classification.get("macro_avg", {})
            initialization = row.get("initialization")
            rows.append(
                {
                    "stage": "downstream",
                    "dataset": row.get("dataset"),
                    "dataset_slug": row.get("dataset_slug"),
                    "source_dataset": row.get("source_dataset"),
                    "source_dataset_slug": row.get("source_dataset_slug"),
                    "model": row.get("model"),
                    "initialization": initialization,
                    "adaptation": _normalize_adaptation(initialization, row.get("adaptation")),
                    "train_fraction": None,
                    "train_fraction_label": None,
                    "train_size": row.get("train_size"),
                    "val_size": row.get("val_size"),
                    "test_size": row.get("test_size"),
                    "accuracy": row.get("test_accuracy"),
                    "macro_f1": row.get("macro_f1"),
                    "macro_precision": macro_avg.get("precision"),
                    "macro_recall": macro_avg.get("recall"),
                    "best_val_accuracy": row.get("best_val_accuracy"),
                    "clean_accuracy": None,
                    "occluded_accuracy": None,
                    "texture_accuracy": None,
                    "occluded_drop": None,
                    "texture_drop": None,
                    "parameter_count": row.get("parameter_count"),
                    "training_time_seconds": row.get("training_time_seconds"),
                    "training_time_readable": row.get("training_time_readable"),
                    "checkpoint_path": checkpoint_path,
                    "evaluation_summary_path": str(evaluation_summary_path) if evaluation_summary_path else None,
                }
            )
    return rows


def _shift_value(rows: list[dict[str, Any]], shift_name: str) -> Any:
    for row in rows:
        if row.get("shift") == shift_name:
            return row.get("robustness_drop")
    return None


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    evaluations = _load_checkpoint_evaluations(args.outputs_dir)
    rows = _source_rows(args.outputs_dir, evaluations) + _downstream_rows(args.outputs_dir, evaluations)
    rows = sorted(
        rows,
        key=lambda row: (
            row.get("stage", ""),
            row.get("dataset", ""),
            row.get("model", ""),
            row.get("initialization", ""),
            row.get("adaptation", ""),
            row.get("train_fraction") or 0.0,
        ),
    )

    save_json(rows, output_dir / "master_results.json")
    save_csv(rows, output_dir / "master_results.csv")
    print(f"Saved {len(rows)} rows to {output_dir / 'master_results.json'}", flush=True)
    print(f"Saved {len(rows)} rows to {output_dir / 'master_results.csv'}", flush=True)


if __name__ == "__main__":
    main()
