from __future__ import annotations

import argparse

from configs.config import build_config
from experiments.run_eurosat_transfer import run_eurosat_transfer
from experiments.run_experiments import run_experiments
from utils.helpers import get_device, runtime_diagnostics, set_seed


def parse_args() -> argparse.Namespace:
    """Expose the main knobs so the same code supports quick and longer runs."""
    parser = argparse.ArgumentParser(
        description="Compare CNNs and Vision Transformers on CIFAR-10 and transfer the learned weights to EuroSAT.",
    )
    parser.add_argument(
        "--experiment",
        choices=("cifar", "eurosat"),
        default="cifar",
        help="Choose between the CIFAR-10 source study and the EuroSAT transfer stage.",
    )
    parser.add_argument("--full", action="store_true", help="Run the longer 20-epoch protocol.")
    parser.add_argument("--epochs", type=int, default=None, help="Override the number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override the mini-batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader worker count.")
    parser.add_argument("--device", type=str, default=None, help="Choose a device such as cpu, cuda, or mps.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory where results are saved.")
    parser.add_argument(
        "--transfer-mode",
        choices=("both", "scratch", "pretrained"),
        default=None,
        help="When running EuroSAT, compare scratch, pretrained, or both.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory containing the CIFAR checkpoints used for EuroSAT fine-tuning.",
    )
    parser.add_argument(
        "--cnn-checkpoint",
        type=str,
        default=None,
        help="Explicit CNN checkpoint path for EuroSAT transfer. Overrides --checkpoint-dir.",
    )
    parser.add_argument(
        "--vit-checkpoint",
        type=str,
        default=None,
        help="Explicit ViT checkpoint path for EuroSAT transfer. Overrides --checkpoint-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_config(full_run=args.full)

    # Command-line overrides make it easy to rerun the same pipeline under a
    # different budget without editing the source code.
    if args.epochs is not None:
        if args.experiment == "cifar":
            config.training.epochs = args.epochs
        else:
            config.transfer.epochs = args.epochs
    if args.batch_size is not None:
        if args.experiment == "cifar":
            config.data.batch_size = args.batch_size
        else:
            config.eurosat.batch_size = args.batch_size
    if args.num_workers is not None:
        if args.experiment == "cifar":
            config.data.num_workers = args.num_workers
        else:
            config.eurosat.num_workers = args.num_workers
    if args.output_dir is not None:
        if args.experiment == "cifar":
            config.experiment.output_dir = args.output_dir
        else:
            config.transfer.output_dir = args.output_dir
    if args.device is not None:
        config.training.device = args.device
    if args.transfer_mode is not None:
        config.transfer.run_mode = args.transfer_mode
    if args.checkpoint_dir is not None:
        config.transfer.checkpoint_dir = args.checkpoint_dir
    if args.cnn_checkpoint is not None:
        config.transfer.cnn_checkpoint = args.cnn_checkpoint
    if args.vit_checkpoint is not None:
        config.transfer.vit_checkpoint = args.vit_checkpoint

    # Reproducibility matters for comparisons, so we seed before building data
    # loaders or models.
    set_seed(config.training.seed)
    device = get_device(config.training.device)
    print("Launching experiment runner...", flush=True)
    print(f"Selected experiment: {args.experiment}", flush=True)
    print("\nRuntime diagnostics:", flush=True)
    for line in runtime_diagnostics(device):
        print(f"  - {line}", flush=True)
    if args.experiment == "cifar":
        summary = run_experiments(config=config, device=device)
    else:
        summary = run_eurosat_transfer(config=config, device=device)

    print("\nFinal summary:", flush=True)
    print(f"Device: {device}", flush=True)
    if args.experiment == "cifar":
        print("Baseline results:")
        for row in summary["baseline"]:
            print(
                f"  {row['model'].upper()}: acc={row['test_accuracy']:.4f}, "
                f"params={row['parameter_count']:,}, time={row['training_time_readable']}"
            )
        print(f"Artifacts saved to: {config.experiment.output_dir}", flush=True)
    else:
        print("EuroSAT transfer results:")
        for row in summary["runs"]:
            print(
                f"  {row['model'].upper()} ({row['initialization']}): "
                f"acc={row['test_accuracy']:.4f}, checkpoint={row['checkpoint_path']}"
            )
        print(f"Artifacts saved to: {config.transfer.output_dir}", flush=True)


if __name__ == "__main__":
    main()
