from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageColor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "assets" / "readme"
BACKGROUND = ImageColor.getrgb("#ffffff")
PADDING = 24
GAP = 18


def _open_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _resize_to_width(image: Image.Image, width: int) -> Image.Image:
    scale = width / image.width
    return image.resize((width, int(round(image.height * scale))), Image.Resampling.LANCZOS)


def _resize_to_height(image: Image.Image, height: int) -> Image.Image:
    scale = height / image.height
    return image.resize((int(round(image.width * scale)), height), Image.Resampling.LANCZOS)


def _hstack(images: list[Image.Image], *, gap: int = GAP, background: tuple[int, int, int] = BACKGROUND) -> Image.Image:
    height = max(image.height for image in images)
    normalized = [_resize_to_height(image, height) for image in images]
    width = sum(image.width for image in normalized) + gap * (len(normalized) - 1)
    canvas = Image.new("RGB", (width, height), background)
    x = 0
    for image in normalized:
        canvas.paste(image, (x, 0))
        x += image.width + gap
    return canvas


def _vstack(images: list[Image.Image], *, gap: int = GAP, background: tuple[int, int, int] = BACKGROUND) -> Image.Image:
    width = max(image.width for image in images)
    normalized = [_resize_to_width(image, width) for image in images]
    height = sum(image.height for image in normalized) + gap * (len(normalized) - 1)
    canvas = Image.new("RGB", (width, height), background)
    y = 0
    for image in normalized:
        canvas.paste(image, (0, y))
        y += image.height + gap
    return canvas


def _add_margin(image: Image.Image, padding: int = PADDING) -> Image.Image:
    canvas = Image.new(
        "RGB",
        (image.width + 2 * padding, image.height + 2 * padding),
        BACKGROUND,
    )
    canvas.paste(image, (padding, padding))
    return canvas


def build_panels() -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_shift = _open_image(PROJECT_ROOT / "outputs/plots/cifar_shift_examples.png")
    source_curves = _open_image(PROJECT_ROOT / "outputs/plots/architecture_training_comparison.png")
    data_efficiency = _open_image(PROJECT_ROOT / "outputs/plots/data_efficiency.png")
    robustness = _open_image(PROJECT_ROOT / "outputs/plots/robustness_drop.png")
    eurosat_curves = _open_image(PROJECT_ROOT / "outputs/plots/eurosat_transfer_validation_curves.png")
    brain_curves = _open_image(PROJECT_ROOT / "outputs/plots/brain_mri_transfer_validation_curves.png")
    cnn_interp = _open_image(PROJECT_ROOT / "outputs/interpretability/cnn_gradcam.png")
    vit_interp = _open_image(PROJECT_ROOT / "outputs/interpretability/vit_attention.png")
    dhvt_interp = _open_image(PROJECT_ROOT / "outputs/interpretability/dhvt_attention.png")
    eurosat_cnn_interp = _open_image(
        PROJECT_ROOT / "outputs/checkpoint_evaluation/cnn_pretrained_eurosat_best/interpretability/cnn_gradcam.png"
    )
    eurosat_vit_interp = _open_image(
        PROJECT_ROOT / "outputs/checkpoint_evaluation/vit_pretrained_eurosat_best/interpretability/vit_attention.png"
    )
    eurosat_dhvt_interp = _open_image(
        PROJECT_ROOT / "outputs/checkpoint_evaluation/dhvt_pretrained_full_finetune_eurosat_best/interpretability/dhvt_attention.png"
    )
    brain_cnn_interp = _open_image(
        PROJECT_ROOT / "outputs/checkpoint_evaluation/cnn_pretrained_brain_tumor_mri_best/interpretability/cnn_gradcam.png"
    )
    brain_vit_interp = _open_image(
        PROJECT_ROOT / "outputs/checkpoint_evaluation/vit_pretrained_brain_tumor_mri_best/interpretability/vit_attention.png"
    )
    brain_dhvt_interp = _open_image(
        PROJECT_ROOT / "outputs/checkpoint_evaluation/dhvt_pretrained_full_finetune_brain_tumor_mri_best/interpretability/dhvt_attention.png"
    )
    hardest_classes = _open_image(
        PROJECT_ROOT / "outputs/checkpoint_evaluation/cnn_100pct_best/examples/class_diagnostics/hardest_classes.png"
    )
    misclassified_interp = _open_image(
        PROJECT_ROOT / "outputs/checkpoint_evaluation/cnn_100pct_best/examples/misclassified_interpretability.png"
    )

    source_panel = _vstack(
        [
            source_shift,
            source_curves,
            _hstack([data_efficiency, robustness]),
        ]
    )
    downstream_panel = _vstack([eurosat_curves, brain_curves])
    interpretability_panel = _hstack([cnn_interp, vit_interp, dhvt_interp])
    downstream_interpretability_panel = _vstack(
        [
            _hstack([eurosat_cnn_interp, eurosat_vit_interp, eurosat_dhvt_interp]),
            _hstack([brain_cnn_interp, brain_vit_interp, brain_dhvt_interp]),
        ]
    )
    error_panel = _hstack([hardest_classes, misclassified_interp])

    panels = {
        "source_overview.png": _add_margin(source_panel),
        "downstream_dynamics.png": _add_margin(downstream_panel),
        "interpretability_overview.png": _add_margin(interpretability_panel),
        "downstream_interpretability_overview.png": _add_margin(downstream_interpretability_panel),
        "error_analysis_overview.png": _add_margin(error_panel),
    }

    written_paths: list[Path] = []
    for filename, image in panels.items():
        path = OUTPUT_DIR / filename
        image.save(path, format="PNG")
        written_paths.append(path)
    return written_paths


if __name__ == "__main__":
    for path in build_panels():
        print(path)
