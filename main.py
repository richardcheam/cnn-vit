from __future__ import annotations

import argparse

from configs.config import build_config
from experiments.run_experiments import run_experiments
from utils.helpers import get_device, runtime_diagnostics, set_seed


def parse_args() -> argparse.Namespace:
    """Expose the main knobs so the same code supports quick and longer runs."""
    parser = argparse.ArgumentParser(
        description="Compare CNNs and Vision Transformers on CIFAR-10 with controlled robustness and interpretability experiments.",
    )
    parser.add_argument("--full", action="store_true", help="Run the longer 20-epoch protocol.")
    parser.add_argument("--epochs", type=int, default=None, help="Override the number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override the mini-batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader worker count.")
    parser.add_argument("--device", type=str, default=None, help="Choose a device such as cpu, cuda, or mps.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory where results are saved.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_config(full_run=args.full)

    # Command-line overrides make it easy to rerun the same pipeline under a
    # different budget without editing the source code.
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
    if args.output_dir is not None:
        config.experiment.output_dir = args.output_dir
    if args.device is not None:
        config.training.device = args.device

    # Reproducibility matters for comparisons, so we seed before building data
    # loaders or models.
    set_seed(config.training.seed)
    device = get_device(config.training.device)
    print("Launching experiment runner...", flush=True)
    print("\nRuntime diagnostics:", flush=True)
    for line in runtime_diagnostics(device):
        print(f"  - {line}", flush=True)
    summary = run_experiments(config=config, device=device)

    print("\nFinal baseline summary:", flush=True)
    print(f"Device: {device}", flush=True)
    print("Baseline results:")
    for row in summary["baseline"]:
        print(
            f"  {row['model'].upper()}: acc={row['test_accuracy']:.4f}, "
            f"params={row['parameter_count']:,}, time={row['training_time_readable']}"
        )
    print(f"Artifacts saved to: {config.experiment.output_dir}", flush=True)


if __name__ == "__main__":
    main()
