from __future__ import annotations

from utils.helpers import count_parameters


def robustness_drop(clean_accuracy: float, shifted_accuracy: float) -> float:
    return clean_accuracy - shifted_accuracy


def model_summary(model) -> dict[str, int]:
    return {"parameter_count": count_parameters(model)}
