from __future__ import annotations

import copy
import time

import torch
from torch import nn
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float,
        weight_decay: float,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def _run_epoch(self, loader, training: bool) -> dict[str, float]:
        self.model.train(training)
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        phase = "train" if training else "eval"

        iterator = tqdm(loader, desc=phase, leave=False)
        for images, labels in iterator:
            images = images.to(self.device, non_blocking=self.device.type != "cpu")
            labels = labels.to(self.device, non_blocking=self.device.type != "cpu")

            with torch.set_grad_enabled(training):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()

            predictions = logits.argmax(dim=1)
            batch_size = labels.size(0)
            total_examples += batch_size
            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()

            iterator.set_postfix(
                loss=f"{total_loss / total_examples:.4f}",
                acc=f"{total_correct / total_examples:.4f}",
            )

        return {
            "loss": total_loss / max(total_examples, 1),
            "accuracy": total_correct / max(total_examples, 1),
        }

    def fit(self, train_loader, val_loader, epochs: int) -> dict:
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        best_state = copy.deepcopy(self.model.state_dict())
        best_val_accuracy = float("-inf")
        start_time = time.perf_counter()

        for _ in range(epochs):
            train_metrics = self._run_epoch(train_loader, training=True)
            val_metrics = self._run_epoch(val_loader, training=False)

            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])

            if val_metrics["accuracy"] > best_val_accuracy:
                best_val_accuracy = val_metrics["accuracy"]
                best_state = copy.deepcopy(self.model.state_dict())

        history["best_val_accuracy"] = best_val_accuracy
        history["training_time_seconds"] = time.perf_counter() - start_time
        self.model.load_state_dict(best_state)
        return history

    @torch.no_grad()
    def evaluate(self, loader) -> dict[str, float]:
        return self._run_epoch(loader, training=False)
