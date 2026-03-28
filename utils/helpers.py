from __future__ import annotations

import json
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def runtime_diagnostics(selected_device: torch.device) -> list[str]:
    lines = [
        f"Python version: {platform.python_version()}",
        f"PyTorch version: {torch.__version__}",
        f"Selected execution device: {selected_device}",
        f"CUDA available: {torch.cuda.is_available()}",
        f"MPS available: {_mps_available()}",
        f"MPS built: {_mps_built()}",
    ]

    if torch.cuda.is_available():
        lines.append(f"CUDA device count: {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            total_memory_gb = properties.total_memory / (1024**3)
            lines.append(
                f"CUDA device {index}: {properties.name} | "
                f"compute_capability={properties.major}.{properties.minor} | "
                f"memory={total_memory_gb:.2f} GB"
            )

    if selected_device.type == "cuda" and torch.cuda.is_available():
        current_index = torch.cuda.current_device()
        lines.append(f"Current CUDA device index: {current_index}")
        lines.append(f"Current CUDA device name: {torch.cuda.get_device_name(current_index)}")

    if selected_device.type == "mps":
        lines.append("Using Apple Metal Performance Shaders backend.")
    if selected_device.type == "cpu":
        lines.append("Using CPU execution path.")

    return lines


def save_json(data: Any, path: str | Path) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def save_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    if not rows:
        return
    target = Path(path)
    ensure_dir(target.parent)
    headers = list(rows[0].keys())
    with target.open("w", encoding="utf-8") as handle:
        handle.write(",".join(headers) + "\n")
        for row in rows:
            handle.write(",".join(str(row[header]) for header in headers) + "\n")


def save_torch_checkpoint(data: Any, path: str | Path) -> None:
    """Save a torch checkpoint after moving tensors to CPU for portability."""
    target = Path(path)
    ensure_dir(target.parent)
    torch.save(_move_to_cpu(data), target)


def format_seconds(seconds: float) -> str:
    minutes, remaining = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {remaining:02d}s"
    if minutes:
        return f"{minutes:d}m {remaining:02d}s"
    return f"{remaining:d}s"


def denormalize_image(
    image: torch.Tensor,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> torch.Tensor:
    mean_tensor = torch.tensor(mean, device=image.device).view(-1, 1, 1)
    std_tensor = torch.tensor(std, device=image.device).view(-1, 1, 1)
    return image * std_tensor + mean_tensor


def to_numpy_image(
    image: torch.Tensor,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> np.ndarray:
    image = denormalize_image(image.detach().cpu(), mean, std)
    image = image.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return image


def _mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def _mps_built() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_built()


def _move_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _move_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_cpu(item) for item in value)
    return value
