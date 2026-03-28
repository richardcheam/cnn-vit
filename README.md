# Understanding CNN vs Vision Transformer

Research-oriented PyTorch project for comparing Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) on CIFAR-10 with a focus on:

- texture vs. shape bias
- robustness to occlusion
- data efficiency
- interpretability through Grad-CAM and attention visualization

The implementation is intentionally lightweight so it can run on a single GPU or CPU, while still following a modular academic-project layout.

## Project Structure

```text
project/
├── configs/
│   └── config.py
├── datasets/
│   ├── cifar_loader.py
│   ├── occlusion.py
│   └── texture_modification.py
├── evaluation/
│   ├── metrics.py
│   └── robustness.py
├── experiments/
│   └── run_experiments.py
├── interpretability/
│   ├── gradcam.py
│   └── vit_attention.py
├── models/
│   ├── cnn.py
│   └── vit.py
├── training/
│   └── trainer.py
├── utils/
│   └── helpers.py
├── main.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## Features

- CIFAR-10 with train / validation / test split
- Standard preprocessing: normalization, random crop, horizontal flip
- Controlled evaluation shifts:
  - square occlusion masking
  - texture distortion via local patch shuffling and mild noise
- Two lightweight models:
  - batch-normalized CNN
  - custom ViT with patch embeddings, class token, and transformer encoder blocks
- AdamW training pipeline with loss / accuracy tracking
- Experiments for:
  - baseline accuracy
  - occlusion robustness
  - texture bias
  - data efficiency at 10%, 25%, 50%, 100%
- Interpretability outputs:
  - Grad-CAM for the CNN
  - attention rollout for the ViT

## Dependency Management

This project uses `uv` instead of `pip` or `requirements.txt`.

```bash
uv sync
uv run python main.py
```

For a longer research-style run:

```bash
uv run python main.py --full
```

The default run is a lightweight protocol with 5 epochs per experiment for quick iteration. The `--full` flag switches to 20 epochs.

## Outputs

Running the project writes artifacts to `outputs/`:

- `summary.json`: experiment summary
- `data_efficiency.csv`: accuracy across data fractions
- `robustness.csv`: robustness drops for occlusion and texture shifts
- `plots/`: per-model training curves, a combined CNN-vs-ViT learning-curve figure, data-efficiency plots, and robustness plots
- `interpretability/`: Grad-CAM and ViT attention visualizations

## Notes

- CIFAR-10 is downloaded automatically the first time you run the project.
- CPU runs are supported, but GPU is recommended for the full experiment schedule.
- `timm` is included as a dependency so the project can be extended with pretrained reference models later without changing the environment setup.
