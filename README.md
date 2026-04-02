# Interpretable Source and Transfer Learning with CNNs, ViTs, and DHVT

Research-oriented PyTorch project for comparing Convolutional Neural Networks (CNNs), Vision Transformers (ViTs), and a Dynamic Hybrid Vision Transformer (DHVT) across a controlled CIFAR-10 source stage and two downstream transfer benchmarks with a focus on:

- texture vs. shape bias
- robustness to occlusion
- data efficiency
- interpretability through Grad-CAM and attention visualization

The implementation is intentionally lightweight so it can run on a single GPU or CPU, while still following a modular setup.

## Research Goal

The project has two linked goals:

1. Measure how CNN, ViT, and DHVT behave on a controlled source task using CIFAR-10.
2. Test how those learned representations transfer to EuroSAT and Brain Tumor MRI under scratch training, linear probing, and full fine-tuning.

The comparison is not limited to top-line accuracy. It asks:

- Which architecture is most data-efficient on the source task?
- Which one is most robust to occlusion and texture disruption?
- How much does frozen-backbone transfer lose relative to full fine-tuning?
- What image regions or token interactions drive the prediction?

In practice, the repository studies architectural behavior at the source stage, then follows the same models into downstream transfer to see which biases remain useful under domain shift.

## Architecture Attribution

The models in this repository are lightweight research baselines, not exact reproductions of one published architecture.

- `CNN`: custom small CNN inspired by LeNet-style convolution/pooling classifiers, VGG-style stacked `3x3` convolutions, and Batch Normalization.
- `ViT`: compact Vision Transformer inspired by the original ViT design and the standard Transformer encoder.
- `DHVT`: compact reimplementation inspired by Lu et al., 2022, using convolutional patch embedding, head-token interaction attention, and a dynamic feed-forward block.

Useful architecture references:

- LeCun et al., 1998, `Gradient-Based Learning Applied to Document Recognition`  
  https://ieeexplore.ieee.org/document/726791
- Simonyan and Zisserman, 2014, `Very Deep Convolutional Networks for Large-Scale Image Recognition`  
  https://arxiv.org/abs/1409.1556
- Ioffe and Szegedy, 2015, `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`  
  https://arxiv.org/abs/1502.03167
- Vaswani et al., 2017, `Attention Is All You Need`  
  https://arxiv.org/abs/1706.03762
- Dosovitskiy et al., 2021, `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  
  https://arxiv.org/abs/2010.11929
- Lu et al., 2022, `Bridging the Gap Between Vision Transformers and Convolutional Neural Networks on Small Datasets`  
  https://openreview.net/forum?id=bfz-jhJ8wn

The repository now supports three connected stages:

1. Source-stage pretraining and controlled analysis on CIFAR-10.
2. Downstream transfer to EuroSAT using either random initialization or the saved CIFAR checkpoints.
3. Downstream transfer to the Brain Tumor MRI dataset using the same scratch-vs-pretrained comparison.

## Current Results

These are the currently saved results in `outputs/`.
All reported source and downstream results below were trained for `100` epochs per run.

### Metrics used

- `Accuracy`: fraction of correct predictions over the evaluation split.
- `Precision`: for one class, among samples predicted as that class, how many are correct.
- `Recall`: for one class, among true samples of that class, how many are recovered.
- `F1-score`: harmonic mean of precision and recall for one class.
- `Macro-F1`: average of the class-wise F1 scores, giving every class equal weight.
- `Weighted-F1`: average of the class-wise F1 scores weighted by class size.
- `Support`: number of true samples belonging to a class in the evaluation split.

In single-label multiclass classification, `macro-F1` is especially useful when class balance matters, because it does not let large classes dominate the summary.
In the current downstream runs, class supports are close to balanced, so `weighted-F1` is almost identical to `macro-F1`. For that reason, the main tables report `macro-F1` only.

### CIFAR-10 source study

| Model | Accuracy | Macro precision | Macro recall | Macro-F1 | Time |
| --- | ---: | ---: | ---: | ---: | ---: |
| DHVT | `88.88%` | `0.8906` | `0.8888` | `0.8891` | `59m 56s` |
| CNN | `85.70%` | `0.8611` | `0.8570` | `0.8580` | `13m 56s` |
| ViT | `73.36%` | `0.7330` | `0.7336` | `0.7330` | `15m 46s` |

Robustness on the full-data models:

| Model | Clean accuracy | Occluded accuracy | Texture accuracy |
| --- | ---: | ---: | ---: |
| DHVT | `88.88%` | `83.78%` | `41.96%` |
| CNN | `85.70%` | `80.32%` | `32.98%` |
| ViT | `73.36%` | `68.81%` | `50.26%` |

- DHVT is the strongest source-stage model on clean CIFAR-10 accuracy.
- DHVT also improves the low-data behavior of vanilla ViT, but it remains more texture-sensitive than the vanilla ViT.
- ViT is still the most texture-robust of the three source-stage models.
- DHVT rollout maps and head-token influence maps largely overlap on CIFAR-10, which suggests that the extra head-token mechanism reinforces the same discriminative regions rather than redirecting attention to different parts of the image.
- Checkpoint-based evaluation shows the hardest source classes are `cat`, `dog`, and `bird`, with strong `cat`/`dog` confusion for both models.

### EuroSAT transfer

| Model | Setup | Accuracy | Macro-F1 | Time |
| --- | --- | ---: | ---: | ---: |
| CNN | scratch | `96.52%` | `0.9642` | `11m 19s` |
| CNN | pretrained + linear probe | `85.78%` | `0.8496` | `11m 21s` |
| CNN | pretrained + full fine-tune | `96.74%` | `0.9660` | `11m 23s` |
| DHVT | scratch | `97.52%` | `0.9745` | `1h 53m 56s` |
| DHVT | pretrained + linear probe | `87.52%` | `0.8682` | `1h 10m 27s` |
| DHVT | pretrained + full fine-tune | `96.96%` | `0.9685` | `1h 54m 19s` |
| ViT | scratch | `93.59%` | `0.9331` | `31m 29s` |
| ViT | pretrained + linear probe | `79.00%` | `0.7784` | `11m 43s` |
| ViT | pretrained + full fine-tune | `94.15%` | `0.9385` | `31m 28s` |

- DHVT scratch is the strongest EuroSAT result in the repository at `97.52%`.
- CNN changes very little on EuroSAT: scratch `96.52%`, linear probe `85.78%`, full fine-tune `96.74%`, and all three runs take about `11` minutes.
- ViT shows a stronger adaptation gap on EuroSAT: scratch `93.59%`, linear probe `79.00%`, full fine-tune `94.15%`.
- CIFAR pretraining does not help DHVT on EuroSAT: `96.96%` with full fine-tuning is slightly below the DHVT scratch run.
- Linear probing is clearly weaker than full fine-tuning for all transformer-style models.
- For DHVT, linear probing saves about `44` minutes relative to full fine-tuning, but loses about `9.4` accuracy points.
- The ViT is especially weak under a frozen backbone on EuroSAT: `79.00%` with linear probing versus `94.15%` with full fine-tuning.
- Evaluation results show the most difficult EuroSAT classes are `PermanentCrop`, `Highway`, and `River`. The most common confusions are `PermanentCrop -> HerbaceousVegetation` and `Highway <-> River`.

### Brain Tumor MRI transfer

| Model | Setup | Accuracy | Macro-F1 | Time |
| --- | --- | ---: | ---: | ---: |
| CNN | scratch | `82.56%` | `0.8230` | `13m 18s` |
| CNN | pretrained + linear probe | `67.31%` | `0.6715` | `42m 10s` |
| CNN | pretrained + full fine-tune | `89.38%` | `0.8917` | `13m 16s` |
| DHVT | scratch | `87.88%` | `0.8776` | `2h 57m 00s` |
| DHVT | pretrained + linear probe | `76.88%` | `0.7648` | `59m 24s` |
| DHVT | pretrained + full fine-tune | `94.00%` | `0.9386` | `2h 31m 18s` |
| ViT | scratch | `85.88%` | `0.8554` | `47m 47s` |
| ViT | pretrained + linear probe | `76.00%` | `0.7524` | `15m 28s` |
| ViT | pretrained + full fine-tune | `93.00%` | `0.9279` | `47m 56s` |

- Transfer helps both models substantially on Brain Tumor MRI.
- CNN gains strongly from full fine-tuning on Brain Tumor MRI: scratch `82.56%`, linear probe `67.31%`, full fine-tune `89.38%`.
- ViT shows the same pattern: scratch `85.88%`, linear probe `76.00%`, full fine-tune `93.00%`.
- DHVT full fine-tuning is now the strongest Brain Tumor MRI result in the repository at `94.00%`.
- Linear probing is clearly insufficient on Brain Tumor MRI. The DHVT drops from `94.00%` with full fine-tuning to `76.88%` with a frozen backbone, but the linear-probe run is also much cheaper: `59m` instead of `2h 31m`.
- For CNN, linear probing is not even cheaper than full fine-tuning here (`42m` versus `13m`) and performs much worse, so it is not a useful tradeoff.
- For ViT, linear probing is cheaper (`15m` versus `48m`) but still loses `17` accuracy points relative to full fine-tuning.
- Full fine-tuning gives a clear gain over scratch for both ViT-style models on this medical task.
- Brain MRI is the clearest case where CIFAR pretraining matters for DHVT.
- Full class-wise precision, recall, F1, and support remain available in the saved `classification_report.json` artifacts.

## Visual Results

These previews use the saved artifacts under `outputs/`. The section uses plain Markdown image embeds so it renders more reliably across viewers.

### Source-stage distribution shifts

![CIFAR-10 clean, occluded, and texture-modified example](outputs/plots/cifar_shift_examples.png)

### Learning dynamics

![Full-data source-stage learning curves](outputs/plots/architecture_training_comparison.png)

![EuroSAT validation dynamics by model](outputs/plots/eurosat_transfer_validation_curves.png)

![Brain Tumor MRI validation dynamics by model](outputs/plots/brain_mri_transfer_validation_curves.png)

### Robustness and data efficiency

![CIFAR-10 data efficiency](outputs/plots/data_efficiency.png)

![CIFAR-10 robustness drop](outputs/plots/robustness_drop.png)

### Source-stage interpretability examples

![CNN Grad-CAM](outputs/interpretability/cnn_gradcam.png)

![ViT attention rollout](outputs/interpretability/vit_attention.png)

![DHVT rollout and head-token influence](outputs/interpretability/dhvt_attention.png)

### Error analysis examples

![CNN hardest CIFAR-10 classes](outputs/checkpoint_evaluation/cnn_100pct_best/examples/class_diagnostics/hardest_classes.png)

![DHVT misclassified interpretability](outputs/checkpoint_evaluation/dhvt_100pct_best/examples/misclassified_interpretability.png)

## Pipeline Overview

```mermaid
flowchart TD
    A[CIFAR-10 source stage<br/>50,000 train / 10,000 test]
    A --> B[Split source data<br/>45,000 train pool / 5,000 val]
    B --> C[Run source study<br/>10% / 25% / 50% / 100%]

    C --> D1[Train CNN]
    C --> D2[Train ViT]
    C --> D3[Train DHVT]

    D1 --> E1[Clean / occluded / texture test]
    D2 --> E2[Clean / occluded / texture test]
    D3 --> E3[Clean / occluded / texture test]

    D1 --> F1[Grad-CAM]
    D2 --> F2[Attention rollout]
    D3 --> F3[Rollout + head-token influence]

    E1 --> G[Save source checkpoints<br/>plots / summaries / interpretability]
    E2 --> G
    E3 --> G
    F1 --> G
    F2 --> G
    F3 --> G

    G --> H[CIFAR checkpoints become transfer initialization]

    H --> I1[EuroSAT downstream stage<br/>64 x 64 x 3]
    H --> I2[Brain Tumor MRI downstream stage<br/>128 x 128 x 3]

    I1 --> J1[Scratch]
    I1 --> J2[Pretrained + linear probe]
    I1 --> J3[Pretrained + full fine-tune]

    I2 --> K1[Scratch]
    I2 --> K2[Pretrained + linear probe]
    I2 --> K3[Pretrained + full fine-tune]

    J1 --> L[Save downstream checkpoints<br/>metrics / plots / evaluation artifacts]
    J2 --> L
    J3 --> L
    K1 --> L
    K2 --> L
    K3 --> L

    L --> M[Checkpoint evaluation<br/>confusion matrices / class diagnostics / error interpretability]

    classDef source fill:#e8f1ff,stroke:#3563a9,stroke-width:1.5px,color:#12243d;
    classDef model fill:#eef7ec,stroke:#2f855a,stroke-width:1.5px,color:#17351f;
    classDef eval fill:#fff4de,stroke:#b7791f,stroke-width:1.5px,color:#4a2a02;
    classDef transfer fill:#f7ebff,stroke:#7b3fa0,stroke-width:1.5px,color:#34144d;
    classDef artifact fill:#f4f4f5,stroke:#52525b,stroke-width:1.5px,color:#18181b;

    class A,B,C source;
    class D1,D2,D3,J1,J2,J3,K1,K2,K3 model;
    class E1,E2,E3,F1,F2,F3,M eval;
    class I1,I2 transfer;
    class G,H,L artifact;
```

## Experimental Setup

### 1. Input data

| Item | Value |
| --- | --- |
| Dataset | CIFAR-10 |
| Number of classes | `10` |
| Native image size | `32 x 32 x 3` |
| Color format | RGB |
| Resize step | None |

The project keeps CIFAR-10 at its native resolution. Images are not downscaled below `32 x 32`, and they are not resized to a larger training resolution either.

### 2. Preprocessing and transforms

The training and evaluation pipelines are intentionally different. Here, `ToTensor()` and `Normalize(...)` are preprocessing transforms. The random crop, flip, rotation, and affine steps are augmentation transforms.

| Split | Transform sequence | Purpose |
| --- | --- | --- |
| Train | `RandomCrop(32, padding=4)` -> `RandomHorizontalFlip()` -> `ToTensor()` -> `Normalize(mean, std)` | Add light augmentation while keeping the task realistic |
| Validation | `ToTensor()` -> `Normalize(mean, std)` | Deterministic validation |
| Test | `ToTensor()` -> `Normalize(mean, std)` | Deterministic testing |

Normalization uses the CIFAR-10 channel statistics:

- Mean: `(0.4914, 0.4822, 0.4465)`
- Std: `(0.2470, 0.2435, 0.2616)`

Important design choice:

- Augmentation is applied only to the training split.
- Validation and test images are kept deterministic so CNN, ViT, and DHVT are compared on the same reference distribution.

### 3. Dataset splits

| Split | Size | Notes |
| --- | ---: | --- |
| Original training set | `50,000` | Standard CIFAR-10 training split |
| Validation set | `5,000` | `10%` of the original training set |
| Remaining training pool | `45,000` | Used for data-efficiency experiments |
| Clean test set | `10,000` | Standard CIFAR-10 test split |
| Occluded test set | `10,000` | Clean test images with square masking |
| Texture-modified test set | `10,000` | Clean test images with patch shuffling and noise |

The validation split is created once with a fixed random seed and reused across all experiments. This keeps the comparison controlled.

Augmentation is applied online during loading. It changes the appearance of a sample seen in an epoch, but it does not increase the stored dataset size.

### 4. Data-efficiency design

The data-efficiency experiment changes only the size of the training subset. The validation set and test set do not change.

| Train fraction | Training images used | Meaning |
| --- | ---: | --- |
| `10%` | `4,500` | Very low-data regime |
| `25%` | `11,250` | Low-data regime |
| `50%` | `22,500` | Medium-data regime |
| `100%` | `45,000` | Full-data reference |

This is not continuous training. Each fraction is a separate run trained from scratch.

For each architecture, the project performs:

- 1 run at `10%`
- 1 run at `25%`
- 1 run at `50%`
- 1 run at `100%`

That means:

- `4` independent CNN runs
- `4` independent ViT runs
- `4` independent DHVT runs
- `12` independent training runs in total

The purpose is to measure how quickly each architecture benefits from additional labeled data.

### 5. Robustness design

Both models are always trained on clean CIFAR-10. The robustness experiment modifies only the test distribution.

| Test condition | Description | Goal |
| --- | --- | --- |
| Clean | Standard CIFAR-10 test set | Baseline accuracy |
| Occluded | Random square region is masked | Test robustness to missing local evidence |
| Texture-modified | Local patches are shuffled and mild noise is added | Test sensitivity to texture disruption |

Only the full-data models from the `100%` training runs are reused for the robustness evaluation.

### 6. Interpretability design

| Model | Method | What it shows |
| --- | --- | --- |
| CNN | Grad-CAM | Which spatial image regions most influenced the prediction |
| ViT | Attention rollout | How information flows from image patches toward the class token |
| DHVT | Attention rollout approximation | Patch-token attention after trimming the extra DHVT head tokens for compatibility with the current visualization pipeline |

Interpretability is also generated from the full-data models so the visualizations reflect the strongest trained version of each architecture.

### 7. Model and optimization settings

| Component | Setting |
| --- | --- |
| CNN | 3 convolutional stages with batch normalization, ReLU, pooling, dropout, and a classifier head |
| ViT | Patch embedding, learnable class token, positional embeddings, transformer encoder blocks, classifier head |
| DHVT | Convolutional patch embedding, head-token interaction attention, dynamic feed-forward block, classifier head |
| Optimizer | AdamW |
| Learning rate | `1e-3` |
| Weight decay | `1e-4` |
| Batch size | `128` by default |
| Epochs | Reported results here use `100`; quick CLI defaults remain shorter for lightweight reruns |
| Random seed | `42` |

## What The Pipeline Measures

| Question | Measurement |
| --- | --- |
| Which model is more accurate on standard classification? | Clean test accuracy |
| Which model is more robust to partial information loss? | Accuracy drop on occluded test images |
| Which model is more sensitive to texture corruption? | Accuracy drop on texture-modified test images |
| Which model learns better from small datasets? | Accuracy as training fraction grows from `10%` to `100%` |
| What drives the prediction? | Grad-CAM for CNN, attention rollout for ViT and DHVT |

## EuroSAT Transfer Setup

The first downstream transfer stage uses EuroSAT as an out-of-distribution image-classification benchmark.

### 1. Downstream dataset design

| Item | Value |
| --- | --- |
| Dataset | EuroSAT RGB |
| Number of classes | `10` |
| Working image size | `64 x 64 x 3` |
| Split strategy | Stratified random split with fixed seed |
| Default split | `80%` train, `10%` validation, `10%` test |

The EuroSAT pipeline uses a stratified split so every class stays represented across train, validation, and test.

### 2. EuroSAT transforms

| Split | Transform sequence |
| --- | --- |
| Train | `Resize(64, 64)` -> `RandomHorizontalFlip()` -> `RandomVerticalFlip()` -> `ToTensor()` -> `Normalize(mean, std)` |
| Validation | `Resize(64, 64)` -> `ToTensor()` -> `Normalize(mean, std)` |
| Test | `Resize(64, 64)` -> `ToTensor()` -> `Normalize(mean, std)` |

In this stage, `Resize`, `ToTensor`, and `Normalize` are preprocessing. The horizontal and vertical flips are augmentation.

The EuroSAT normalization defaults are standard RGB values:

- Mean: `(0.485, 0.456, 0.406)`
- Std: `(0.229, 0.224, 0.225)`

### 3. EuroSAT comparison matrix

For each architecture, the transfer runner compares:

- `scratch`: train directly on EuroSAT from random initialization
- `pretrained + linear_probe`: load the CIFAR-10 backbone checkpoint, freeze the backbone, and train only the new EuroSAT classifier head
- `pretrained + full_finetune`: load the CIFAR-10 backbone checkpoint, keep a new EuroSAT classifier head, and fine-tune end to end

So the EuroSAT transfer stage runs:

- CNN scratch
- CNN pretrained on CIFAR-10 then linear-probed on EuroSAT
- CNN pretrained on CIFAR-10 then fine-tuned on EuroSAT
- ViT scratch
- ViT pretrained on CIFAR-10 then linear-probed on EuroSAT
- ViT pretrained on CIFAR-10 then fine-tuned on EuroSAT

### 4. Adaptation modes

```mermaid
flowchart LR
    A[CIFAR-10 checkpoint]
    A --> B[Replace downstream classifier head]
    B --> C1[Linear probe<br/>freeze backbone<br/>train head only]
    B --> C2[Full fine-tune<br/>train backbone + head]
```

`linear_probe` tests whether the pretrained representation is already useful without adapting the backbone.

`full_finetune` tests whether downstream adaptation of the backbone gives additional gains.

Interpretability note:

- For linear probing, the backbone remains frozen during training.
- Grad-CAM and attention visualization may still use gradients during the visualization pass.
- This does not update the frozen backbone weights. It is only an inference-time explanation step.

### 5. Optimization details

| Setting | Value |
| --- | --- |
| Optimizer | AdamW |
| Scratch learning rate | `1e-3` |
| Pretrained backbone learning rate | `1e-4` |
| Pretrained classifier-head learning rate | `1e-3` |
| Weight decay | `1e-4` |
| Epochs | Reported results here use `100`; quick CLI defaults remain shorter for lightweight reruns |

For the ViT transfer runs, the positional embeddings are automatically interpolated when moving from CIFAR-10 resolution to EuroSAT resolution.

## Brain MRI Transfer Setup

The second downstream transfer stage uses the Kaggle Brain Tumor MRI dataset as a medical-image classification benchmark.

### 1. Expected local dataset layout

This dataset is not auto-downloaded by the project. Download it manually from Kaggle and place it under a directory that contains:

```text
data/brain_tumor_mri_dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

If your extracted folder has one extra wrapper directory, the loader will detect that automatically.

### 2. Brain MRI setup

| Item | Value |
| --- | --- |
| Dataset | Brain Tumor MRI |
| Number of classes | `4` |
| Working image size | `128 x 128 x 3` |
| Split strategy | Provided Kaggle `Training/` and `Testing/` folders, plus a validation split carved from `Training/` |
| Default validation split | `15%` of the training folder |

### 3. Brain MRI transforms

| Split | Transform sequence |
| --- | --- |
| Train | `RGB convert` -> `Resize(128, 128)` -> `RandomRotation(10)` -> `RandomAffine(translate/scale)` -> `ToTensor()` -> `Normalize(mean, std)` |
| Validation | `RGB convert` -> `Resize(128, 128)` -> `ToTensor()` -> `Normalize(mean, std)` |
| Test | `RGB convert` -> `Resize(128, 128)` -> `ToTensor()` -> `Normalize(mean, std)` |

In this stage, `RGB convert`, `Resize`, `ToTensor`, and `Normalize` are preprocessing. `RandomRotation` and `RandomAffine` are augmentation. The medical pipeline uses light geometry-only augmentation and avoids the stronger orientation changes used in EuroSAT.

### 4. Brain MRI comparison matrix

For each architecture, the runner compares:

- `scratch`: train directly on Brain MRI from random initialization
- `pretrained + linear_probe`: load the CIFAR-10 backbone checkpoint, freeze the backbone, and train only the new classifier head
- `pretrained + full_finetune`: load the CIFAR-10 backbone checkpoint, replace the classifier head, and fine-tune end to end

### 5. Run commands

```bash
uv run python main.py --experiment brain_mri --brain-mri-data-dir data/brain_tumor_mri_dataset
uv run python main.py --experiment brain_mri --transfer-mode pretrained --brain-mri-data-dir data/brain_tumor_mri_dataset
uv run python main.py --experiment brain_mri --models cnn --brain-mri-data-dir data/brain_tumor_mri_dataset
uv run python main.py --experiment brain_mri --brain-mri-train-fraction 0.25 --brain-mri-data-dir data/brain_tumor_mri_dataset
```

Brain MRI artifacts are saved to `outputs/brain_mri_transfer/`.

## Project Structure

```text
project/
├── configs/
│   └── config.py
├── datasets/
│   ├── brain_mri_loader.py
│   ├── cifar_loader.py
│   ├── eurosat_loader.py
│   ├── occlusion.py
│   └── texture_modification.py
├── evaluation/
│   ├── metrics.py
│   └── robustness.py
├── experiments/
│   ├── run_brain_mri_transfer.py
│   ├── run_eurosat_transfer.py
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
│   ├── helpers.py
│   └── transfer.py
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
- EuroSAT transfer-learning experiments:
  - scratch training
  - CIFAR-pretrained fine-tuning
- Interpretability outputs:
  - Grad-CAM for the CNN
  - attention rollout for the ViT

## Dependency Management

This project uses `uv` instead of `pip` or `requirements.txt`.

```bash
uv sync
uv run python main.py --experiment cifar
```

PyTorch is pinned to a CUDA 12.8-compatible stack for Linux and Windows:

- `torch==2.9.1`
- `torchvision==0.24.1`
- Linux/Windows GPU installs use the official PyTorch `cu128` index
- macOS falls back to the default CPU wheels

For a longer research-style run:

```bash
uv run python main.py --experiment cifar --full
```

To run the downstream EuroSAT stage after CIFAR checkpoints have been created:

```bash
uv run python main.py --experiment eurosat
```

Useful EuroSAT options:

```bash
uv run python main.py --experiment eurosat --transfer-mode pretrained
uv run python main.py --experiment eurosat --checkpoint-dir outputs/checkpoints
uv run python main.py --experiment eurosat --cnn-checkpoint /path/to/cnn_100pct_best.pt --vit-checkpoint /path/to/vit_100pct_best.pt
uv run python main.py --experiment eurosat --models cnn
uv run python main.py --experiment eurosat --eurosat-train-fraction 0.25
```

The current saved results in this repository were produced with 100-epoch runs. The CLI defaults remain shorter for quick iteration, and you can override them with `--epochs`.

If you want to run only one architecture, use `--models`:

```bash
uv run python main.py --experiment cifar --models cnn
uv run python main.py --experiment cifar --models vit
uv run python main.py --experiment eurosat --models cnn
uv run python main.py --experiment eurosat --models vit
```

To regenerate plots later from saved artifacts without retraining:

```bash
uv run python experiments/regenerate_plots.py --experiment cifar --source-dir outputs
uv run python experiments/regenerate_plots.py --experiment eurosat --source-dir outputs/eurosat_transfer
```

This is useful when you want to change labels, update styles, or rebuild figures on another machine after copying back the saved outputs and checkpoints.

To evaluate pulled checkpoints directly and generate confusion matrices plus interpretability figures without retraining:

```bash
uv run python experiments/evaluate_checkpoints.py --checkpoint-dir outputs/checkpoints
uv run python experiments/evaluate_checkpoints.py --checkpoint-paths outputs/checkpoints/cnn_100pct_best.pt outputs/checkpoints/vit_100pct_best.pt
```

This checkpoint-only workflow saves per-checkpoint summaries, classification reports, confusion matrices, and Grad-CAM or ViT attention figures into `outputs/checkpoint_evaluation/`.
It also saves normalized confusion matrices, class-wise prediction analysis, grids of correct and misclassified examples, class-diagnostic panels for the easiest and hardest classes, and misclassification-interpretability panels that show where the model was looking when it made a wrong prediction.

## CLI Execution Flow

When you run `uv run python main.py`, the project executes the following stages:

1. Load the project configuration and apply any command-line overrides.
2. Set the random seed for reproducibility.
3. Detect whether the run will use CPU, CUDA, or MPS.
4. Print the runtime diagnostics and planned dataset protocol.
5. Train the CNN on each training fraction: `10%`, `25%`, `50%`, `100%`.
6. Train the ViT on the same training fractions.
7. Keep the full-data runs for downstream robustness and interpretability analysis.
8. Evaluate the full-data CNN and ViT on:
   - the clean test set
   - the occluded test set
   - the texture-modified test set
9. Generate Grad-CAM for the CNN and attention maps for the ViT.
10. Save summaries, CSV files, plots, and interpretability figures to `outputs/`.

When you run `uv run python main.py --experiment eurosat`, the project executes the following stages:

1. Load EuroSAT with a stratified train / validation / test split.
2. Build one CNN and one ViT for EuroSAT resolution.
3. Run the requested initialization modes:
   - scratch
   - pretrained from CIFAR checkpoints
4. For pretrained runs, load only the transferable backbone weights and keep a fresh EuroSAT classifier head.
5. Fine-tune on the EuroSAT training split and select the best validation checkpoint in memory.
6. Save EuroSAT checkpoints, plots, CSV summaries, and `summary.json` into `outputs/eurosat_transfer/`.

The EuroSAT summary rows also include `macro_f1` and `weighted_f1`, so the downstream stage is not limited to accuracy alone.

## Why The Train Split Grows From 10% To 100%

The increasing training split is the core of the data-efficiency experiment.

- `10%` asks: can the architecture learn well from very little data?
- `25%` and `50%` show how performance improves as more supervision becomes available.
- `100%` provides the full-data reference point.

If a model performs strongly even at `10%`, it is more data-efficient. If it needs much more data before it becomes competitive, it is less data-efficient. The changing split is therefore intentional and central to the research question.

## Outputs

Running the CIFAR source-stage experiment writes artifacts to `outputs/`:

- `summary.json`: experiment summary
- `data_efficiency_runs.json`: detailed per-run metadata with saved training histories
- `data_efficiency.csv`: accuracy across data fractions
- `robustness.csv`: robustness drops for occlusion and texture shifts
- `checkpoints/`: saved best-model checkpoints for each architecture and training fraction
- `plots/`: per-model training curves, a combined CNN-vs-ViT learning-curve figure, per-model data-fraction learning-curve figures, data-efficiency plots, and robustness plots
- `interpretability/`: Grad-CAM and ViT attention visualizations

## Canonical Results Export

Because top-level stage summaries can be overwritten by later runs, the safer source of truth is the checkpoint-backed artifacts. The repository includes a small aggregator that rebuilds a canonical table from checkpoints and checkpoint evaluations:

```bash
uv run python experiments/build_master_results.py
```

This writes:

- `outputs/master_results/master_results.json`
- `outputs/master_results/master_results.csv`

Those files are the recommended base for any final reporting table that should survive later reruns.

Checkpoint filenames follow the pattern:

- `outputs/checkpoints/cnn_10pct_best.pt`
- `outputs/checkpoints/cnn_100pct_best.pt`
- `outputs/checkpoints/vit_50pct_best.pt`

Each checkpoint stores the model `state_dict` together with basic metadata such as the model name, training fraction, class names, configuration, and validation/test accuracy.
Newer checkpoints also store the per-epoch training history so plots can be regenerated later without retraining.

Running the EuroSAT transfer stage writes artifacts to `outputs/eurosat_transfer/`:

- `summary.json`: transfer summary and split metadata
- `transfer_runs.json`: detailed scratch / pretrained runs with saved histories
- `eurosat_transfer_results.csv`: scratch vs pretrained transfer results
- `checkpoints/`: saved best EuroSAT models for each run
- `plots/eurosat_transfer_accuracy.png`: final test-accuracy comparison
- `plots/eurosat_transfer_validation_curves.png`: validation learning curves across transfer runs

## Notes

- CIFAR-10 is downloaded automatically the first time you run the project.
- EuroSAT is also downloaded automatically the first time you run the transfer stage.
- CPU runs are supported, but GPU is recommended for the full experiment schedule.
- Checkpoints are saved in `outputs/checkpoints/` in a portable CPU-friendly format so they can be loaded later on another machine with `torch.load(..., map_location="cpu")`.
- EuroSAT fine-tuning also saves downstream checkpoints in `outputs/eurosat_transfer/checkpoints/`.
- The repo keeps both checkpoint folders trackable: `outputs/checkpoints/` and `outputs/eurosat_transfer/checkpoints/`. Other generated output folders remain ignored. If you plan to version many checkpoints, Git LFS is recommended.
- `timm` is included as a dependency so the project can be extended with pretrained reference models later without changing the environment setup.
