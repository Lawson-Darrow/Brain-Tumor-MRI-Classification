from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from .utils import save_json


def build_metrics_dict(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
) -> dict[str, Any]:
    """Build a sklearn classification report dict with consistent types."""
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report["accuracy"] = float(report["accuracy"])
    return report


def save_confusion_matrix_plot(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    out_path: str | Path,
    title: str,
) -> None:
    """Save labeled confusion matrix heatmap for qualitative error analysis."""
    cm = confusion_matrix(y_true, y_pred)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_metrics_bundle(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    metrics_path: str | Path,
    matrix_plot_path: str | Path,
    matrix_title: str,
) -> dict[str, Any]:
    # Keep JSON metrics and plot generation coupled so every run has both artifacts.
    metrics = build_metrics_dict(y_true, y_pred, class_names)
    save_json(metrics, metrics_path)
    save_confusion_matrix_plot(y_true, y_pred, class_names, matrix_plot_path, matrix_title)
    return metrics

