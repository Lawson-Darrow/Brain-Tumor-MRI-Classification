from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import save_json


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, list[int], list[int]]:
    """Run one train or eval epoch and return loss with predictions."""
    is_train = optimizer is not None
    model.train(is_train)

    losses: list[float] = []
    all_preds: list[int] = []
    all_targets: list[int] = []

    for images, targets in tqdm(loader, leave=False, disable=True):
        images = images.to(device)
        targets = targets.to(device)

        if is_train:
            # Clear stale gradients before each optimizer step.
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, targets)
            if is_train:
                # Backprop + parameter update only in train mode.
                loss.backward()
                optimizer.step()

        losses.append(float(loss.item()))
        all_preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
        all_targets.extend(targets.detach().cpu().tolist())

    avg_loss = float(sum(losses) / max(len(losses), 1))
    return avg_loss, all_targets, all_preds


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_names: list[str],
    output_dir: str | Path,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    scheduler_patience: int = 2,
    early_stopping_patience: int = 6,
    min_delta: float = 1e-4,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train model with early stopping and validation-macro-F1 tracking."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        # Restrict optimization to parameters currently marked trainable.
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=scheduler_patience
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_macro_f1": [],
        "val_macro_f1": [],
        "learning_rate": [],
    }
    best_val_f1 = -1.0
    epochs_without_improvement = 0
    best_checkpoint = output_dir / "best_model.pt"

    for _epoch in range(1, epochs + 1):
        # Compute both train and validation summaries every epoch for reporting.
        train_loss, train_true, train_pred = _run_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_true, val_pred = _run_epoch(model, val_loader, criterion, None, device)

        train_metrics = classification_report(
            train_true, train_pred, target_names=class_names, output_dict=True, zero_division=0
        )
        val_metrics = classification_report(
            val_true, val_pred, target_names=class_names, output_dict=True, zero_division=0
        )

        train_f1 = float(train_metrics["macro avg"]["f1-score"])
        val_f1 = float(val_metrics["macro avg"]["f1-score"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_macro_f1"].append(train_f1)
        history["val_macro_f1"].append(val_f1)
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
        # Reduce LR when validation quality plateaus.
        scheduler.step(val_f1)

        if val_f1 > (best_val_f1 + min_delta):
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_checkpoint)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            break

    if best_checkpoint.exists():
        # Restore best observed validation checkpoint before final testing.
        model.load_state_dict(torch.load(best_checkpoint, map_location=device))

    save_json({"history": history, "best_val_macro_f1": best_val_f1}, output_dir / "history.json")
    return model, {"history": history, "best_val_macro_f1": best_val_f1}


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader) -> tuple[list[int], list[int]]:
    """Run inference and return ground-truth/predicted labels."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    for images, targets in loader:
        logits = model(images.to(device))
        preds = logits.argmax(dim=1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(targets.tolist())
    return y_true, y_pred


@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader):
    """Run inference and return (y_true, softmax_probs) for ROC-AUC and calibration.

    The base predict() returns labels only, which is insufficient for AUC/ECE; this
    returns per-class probabilities so threshold-free and calibration metrics work.
    """
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    y_true: list[int] = []
    probs: list[list[float]] = []
    for images, targets in loader:
        logits = model(images.to(device))
        p = torch.softmax(logits, dim=1).cpu().numpy()
        probs.extend(p.tolist())
        y_true.extend(targets.tolist())
    return np.array(y_true), np.array(probs)

