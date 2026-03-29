from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import build_config
from experiments.run_eurosat_transfer import (
    _save_transfer_accuracy_plot,
    _save_transfer_validation_curves,
)
from experiments.run_experiments import (
    _save_combined_training_curves,
    _save_data_efficiency_plot,
    _save_fraction_learning_curves,
    _save_robustness_plot,
    _save_training_curves,
)
from utils.artifacts import load_cifar_runs_with_histories, load_eurosat_runs_with_histories, load_json
from utils.helpers import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate saved experiment plots from persisted histories without retraining.",
    )
    parser.add_argument(
        "--experiment",
        choices=("cifar", "eurosat"),
        required=True,
        help="Choose which saved experiment outputs to regenerate plots for.",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Directory containing the saved experiment artifacts, such as `outputs` or `outputs/eurosat_transfer`.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory to write the regenerated plots. Defaults to `<source-dir>/plots`.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = ensure_dir(Path(args.output_dir) if args.output_dir is not None else source_dir / "plots")

    if args.experiment == "cifar":
        regenerate_cifar_plots(source_dir=source_dir, output_dir=output_dir)
    else:
        regenerate_eurosat_plots(source_dir=source_dir, output_dir=output_dir)


def regenerate_cifar_plots(source_dir: Path, output_dir: Path) -> None:
    summary_path = source_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing CIFAR summary file: {summary_path}")

    summary = load_json(summary_path)
    config = build_config()
    config.data.name = summary.get("dataset", summary.get("config", {}).get("data", {}).get("name", config.data.name))
    detailed_runs = load_cifar_runs_with_histories(source_dir)
    if not detailed_runs:
        raise RuntimeError(
            "No CIFAR runs with saved histories were found. "
            "Use outputs generated with the updated code, or rerun once so histories are persisted."
        )

    full_run_results = {
        row["model"]: row
        for row in detailed_runs
        if float(row["train_fraction"]) >= 1.0
    }
    data_efficiency_rows = [
        {
            key: value
            for key, value in row.items()
            if key != "history"
        }
        for row in detailed_runs
    ]

    _save_training_curves(full_run_results=full_run_results, output_dir=output_dir)
    if len(full_run_results) >= 2:
        _save_combined_training_curves(full_run_results=full_run_results, output_dir=output_dir)
    _save_fraction_learning_curves(data_efficiency_runs=detailed_runs, output_dir=output_dir)
    _save_data_efficiency_plot(config=config, rows=data_efficiency_rows, output_dir=output_dir)
    _save_robustness_plot(config=config, rows=summary.get("robustness", []), output_dir=output_dir)
    print(f"Regenerated CIFAR plots in {output_dir}")


def regenerate_eurosat_plots(source_dir: Path, output_dir: Path) -> None:
    detailed_runs = load_eurosat_runs_with_histories(source_dir)
    if not detailed_runs:
        raise RuntimeError(
            "No EuroSAT runs with saved histories were found. "
            "Use outputs generated with the updated code, or rerun once so histories are persisted."
        )

    rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"history", "preload_info"}
        }
        for row in detailed_runs
    ]
    _save_transfer_accuracy_plot(rows=rows, output_dir=output_dir)
    _save_transfer_validation_curves(results=detailed_runs, output_dir=output_dir)
    print(f"Regenerated EuroSAT plots in {output_dir}")


if __name__ == "__main__":
    main()
