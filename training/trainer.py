from __future__ import annotations

import copy
import time

import torch
from torch import nn
from tqdm import tqdm


class Trainer:
    """Minimal training helper that keeps the research loop easy to read."""
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float,
        weight_decay: float,
        device: torch.device,
        parameter_groups: list[dict] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        optimizer_parameters = parameter_groups or self.model.parameters()
        self.optimizer = torch.optim.AdamW(
            optimizer_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def _run_epoch(self, loader, training: bool, desc: str | None = None) -> dict[str, float]:
        """Run one full pass over a loader and aggregate loss/accuracy."""
        self.model.train(training)
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        logged_first_batch = False
        phase = "train" if training else "eval"

        iterator = tqdm(loader, desc=desc or phase, leave=False)
        for images, labels in iterator:
            images = images.to(self.device, non_blocking=self.device.type != "cpu")
            labels = labels.to(self.device, non_blocking=self.device.type != "cpu")

            if not logged_first_batch:
                tqdm.write(
                    f"[{desc or phase}] First batch on {images.device} | "
                    f"images_shape={tuple(images.shape)} | labels_shape={tuple(labels.shape)}"
                )
                logged_first_batch = True

            with torch.set_grad_enabled(training):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                if training:
                    # The optimization step happens only in training mode; the
                    # evaluation path reuses the same code without touching gradients.
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

    def fit(self, train_loader, val_loader, epochs: int, run_name: str = "run") -> dict:
        """Train for several epochs and keep the best validation checkpoint in memory."""
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        best_state = copy.deepcopy(self.model.state_dict())
        best_val_accuracy = float("-inf")
        start_time = time.perf_counter()
        model_device = next(self.model.parameters()).device

        tqdm.write(f"[{run_name}] Training started.")
        tqdm.write(
            f"[{run_name}] Model parameters are on {model_device} | "
            f"train_batches={len(train_loader)} | val_batches={len(val_loader)}"
        )
        for epoch_index in range(epochs):
            epoch_number = epoch_index + 1
            tqdm.write(f"[{run_name}] Epoch {epoch_number}/{epochs} started.")
            train_metrics = self._run_epoch(
                train_loader,
                training=True,
                desc=f"{run_name} train {epoch_number}/{epochs}",
            )
            val_metrics = self._run_epoch(
                val_loader,
                training=False,
                desc=f"{run_name} val {epoch_number}/{epochs}",
            )

            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])

            # We restore the best validation state at the end so later test-set
            # comparisons use the same model-selection rule for CNN and ViT.
            improved = val_metrics["accuracy"] > best_val_accuracy
            if val_metrics["accuracy"] > best_val_accuracy:
                best_val_accuracy = val_metrics["accuracy"]
                best_state = copy.deepcopy(self.model.state_dict())

            best_note = " | new best validation checkpoint" if improved else ""
            tqdm.write(
                f"[{run_name}] Epoch {epoch_number}/{epochs} complete | "
                f"train_loss={train_metrics['loss']:.4f}, train_acc={train_metrics['accuracy']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['accuracy']:.4f}"
                f"{best_note}"
            )

        history["best_val_accuracy"] = best_val_accuracy
        history["training_time_seconds"] = time.perf_counter() - start_time
        self.model.load_state_dict(best_state)
        tqdm.write(
            f"[{run_name}] Training finished in {history['training_time_seconds']:.2f}s | "
            f"best_val_acc={best_val_accuracy:.4f}"
        )
        return history

    @torch.no_grad()
    def evaluate(self, loader, label: str = "eval") -> dict[str, float]:
        """Run the evaluation branch of the epoch loop under no-grad."""
        model_device = next(self.model.parameters()).device
        tqdm.write(f"[{label}] Evaluation started.")
        tqdm.write(f"[{label}] Model parameters are on {model_device} | eval_batches={len(loader)}")
        metrics = self._run_epoch(loader, training=False, desc=label)
        tqdm.write(
            f"[{label}] Evaluation complete | "
            f"loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.4f}"
        )
        return metrics
