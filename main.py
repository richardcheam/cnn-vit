from __future__ import annotations

import argparse

from configs.config import build_config
from experiments.run_experiments import run_experiments
from utils.helpers import get_device, set_seed


def parse_args() -> argparse.Namespace:
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

    set_seed(config.training.seed)
    device = get_device(config.training.device)
    summary = run_experiments(config=config, device=device)

    print(f"Device: {device}")
    print("Baseline results:")
    for row in summary["baseline"]:
        print(
            f"  {row['model'].upper()}: acc={row['test_accuracy']:.4f}, "
            f"params={row['parameter_count']:,}, time={row['training_time_readable']}"
        )
    print(f"Artifacts saved to: {config.experiment.output_dir}")


if __name__ == "__main__":
    main()
