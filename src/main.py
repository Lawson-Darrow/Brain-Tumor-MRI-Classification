from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .baselines import run_baseline_experiments
from .cnn import CustomCNN
from .data import create_dataloaders, load_dataset_frames
from .evaluate import save_metrics_bundle
from .train import predict, train_model
from .transfer import build_transfer_model, unfreeze_last_n_feature_blocks
from .utils import ensure_dir, save_json, set_global_seed


def _class_distribution(frame: pd.DataFrame, class_names: list[str]) -> dict[str, int]:
    # Return class counts in fixed class-name order for stable reports.
    counts = frame["class_name"].value_counts().to_dict()
    return {name: int(counts.get(name, 0)) for name in class_names}


def _summarize_for_comparison(metrics: dict[str, Any]) -> dict[str, float]:
    # Normalize metrics payload shape so all model families can share one table.
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_precision": float(metrics["macro avg"]["precision"]),
        "macro_recall": float(metrics["macro avg"]["recall"]),
        "macro_f1": float(metrics["macro avg"]["f1-score"]),
    }


def _train_custom_cnn(
    split_frames,
    image_size: int,
    batch_size: int,
    epochs: int,
    output_root: Path,
) -> dict[str, Any]:
    """Train custom CNN and persist test metrics + training artifacts."""
    run_dir = ensure_dir(output_root / "cnn")
    train_loader, val_loader, test_loader = create_dataloaders(
        split_frames,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=0,
    )

    model = CustomCNN(num_classes=len(split_frames.class_names))
    model, train_info = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_names=split_frames.class_names,
        output_dir=run_dir,
        epochs=epochs,
        learning_rate=1e-3,
    )

    y_true, y_pred = predict(model, test_loader)
    metrics = save_metrics_bundle(
        y_true=y_true,
        y_pred=y_pred,
        class_names=split_frames.class_names,
        metrics_path=run_dir / "test_metrics.json",
        matrix_plot_path=run_dir / "test_confusion_matrix.png",
        matrix_title="Custom CNN Test Confusion Matrix",
    )
    metrics["training"] = train_info
    save_json(metrics, run_dir / "cnn_bundle.json")
    return metrics


def _train_transfer_model(
    split_frames,
    image_size: int,
    batch_size: int,
    epochs: int,
    transfer_model: str,
    output_root: Path,
) -> dict[str, Any]:
    """Train transfer model in two stages (head warmup, then partial fine-tuning)."""
    run_dir = ensure_dir(output_root / "transfer")
    train_loader, val_loader, test_loader = create_dataloaders(
        split_frames,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=0,
    )

    model = build_transfer_model(
        model_name=transfer_model,
        num_classes=len(split_frames.class_names),
        freeze_backbone=True,
    )
    head_stage_epochs = max(2, epochs // 2)
    finetune_stage_epochs = max(1, epochs - head_stage_epochs)

    model, head_stage_info = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_names=split_frames.class_names,
        output_dir=run_dir / "stage_head",
        epochs=head_stage_epochs,
        learning_rate=2e-4,
    )

    # Fine-tune part of the frozen backbone after warming up the classifier head.
    unfreeze_last_n_feature_blocks(model, model_name=transfer_model, n_blocks=1)
    model, finetune_stage_info = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_names=split_frames.class_names,
        output_dir=run_dir / "stage_finetune",
        epochs=finetune_stage_epochs,
        learning_rate=5e-5,
    )

    y_true, y_pred = predict(model, test_loader)
    metrics = save_metrics_bundle(
        y_true=y_true,
        y_pred=y_pred,
        class_names=split_frames.class_names,
        metrics_path=run_dir / "test_metrics.json",
        matrix_plot_path=run_dir / "test_confusion_matrix.png",
        matrix_title=f"{transfer_model} Test Confusion Matrix",
    )
    metrics["training"] = {
        "head_stage": head_stage_info,
        "finetune_stage": finetune_stage_info,
        "head_stage_epochs": head_stage_epochs,
        "finetune_stage_epochs": finetune_stage_epochs,
    }
    save_json(metrics, run_dir / "transfer_bundle.json")
    return metrics


def _build_comparison_file(output_root: Path, rows: list[dict[str, Any]]) -> None:
    # Save both CSV and JSON so results are easy to use in code and slides.
    comp_dir = ensure_dir(output_root / "comparison")
    frame = pd.DataFrame(rows)
    frame.to_csv(comp_dir / "model_comparison.csv", index=False)
    save_json({"models": rows}, comp_dir / "model_comparison.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Brain Tumor MRI Classification pipeline")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument(
        "--task",
        choices=["baselines", "cnn", "transfer", "all"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument(
        "--transfer-model",
        type=str,
        default="resnet50",
        choices=["resnet50", "efficientnet_b0", "vgg16", "vit_b_16"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Seed first so every downstream random operation is reproducible.
    set_global_seed(args.seed)
    output_root = ensure_dir(args.output_dir)

    split_frames = load_dataset_frames(
        dataset_root=args.dataset_root,
        val_size=args.val_size,
        random_seed=args.seed,
    )

    split_report = {
        "class_names": split_frames.class_names,
        "train_size": int(len(split_frames.train)),
        "validation_size": int(len(split_frames.val)),
        "test_size": int(len(split_frames.test)),
        "train_class_distribution": _class_distribution(
            split_frames.train, split_frames.class_names
        ),
        "validation_class_distribution": _class_distribution(
            split_frames.val, split_frames.class_names
        ),
        "test_class_distribution": _class_distribution(
            split_frames.test, split_frames.class_names
        ),
    }
    save_json(split_report, output_root / "split_report.json")

    comparison_rows: list[dict[str, Any]] = []

    if args.task in {"baselines", "all"}:
        baseline_metrics = run_baseline_experiments(
            train_df=split_frames.train,
            val_df=split_frames.val,
            test_df=split_frames.test,
            class_names=split_frames.class_names,
            output_dir=output_root / "baselines",
        )
        best_name = baseline_metrics["best_model_by_validation_macro_f1"]
        best_test = baseline_metrics["best_model_test_metrics"]
        comparison_rows.append({"model": f"baseline::{best_name}", **_summarize_for_comparison(best_test)})

    if args.task in {"cnn", "all"}:
        cnn_metrics = _train_custom_cnn(
            split_frames=split_frames,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            output_root=output_root,
        )
        comparison_rows.append({"model": "custom_cnn", **_summarize_for_comparison(cnn_metrics)})

    if args.task in {"transfer", "all"}:
        transfer_metrics = _train_transfer_model(
            split_frames=split_frames,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            transfer_model=args.transfer_model,
            output_root=output_root,
        )
        comparison_rows.append(
            {"model": f"transfer::{args.transfer_model}", **_summarize_for_comparison(transfer_metrics)}
        )

    if comparison_rows:
        _build_comparison_file(output_root=output_root, rows=comparison_rows)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Finished task={args.task} using device={device}. Artifacts saved in '{output_root}'.")


if __name__ == "__main__":
    main()

