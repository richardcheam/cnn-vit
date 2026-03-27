from __future__ import annotations

from evaluation.metrics import robustness_drop


def summarize_shift(
    model_name: str,
    shift_name: str,
    clean_accuracy: float,
    shifted_accuracy: float,
) -> dict[str, float | str]:
    return {
        "model": model_name,
        "shift": shift_name,
        "clean_accuracy": clean_accuracy,
        "shifted_accuracy": shifted_accuracy,
        "robustness_drop": robustness_drop(clean_accuracy, shifted_accuracy),
    }
