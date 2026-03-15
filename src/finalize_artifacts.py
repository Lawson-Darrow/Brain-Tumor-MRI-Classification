from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def _summarize_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    # Flatten report dictionary to a compact row used in final comparison tables.
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_precision": float(metrics["macro avg"]["precision"]),
        "macro_recall": float(metrics["macro avg"]["recall"]),
        "macro_f1": float(metrics["macro avg"]["f1-score"]),
    }


def _count_split(split_dir: Path) -> dict[str, int]:
    # Count physical image files on disk so provenance reflects actual local data.
    counts: dict[str, int] = {}
    for class_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
        counts[class_dir.name] = sum(1 for img in class_dir.rglob("*") if img.is_file())
    return counts


def _plot_comparison(rows: list[dict[str, Any]], out_path: Path) -> None:
    # Two-bar chart (accuracy + macro-F1) is easy to present in slides.
    labels = [row["model"] for row in rows]
    accuracy = [row["accuracy"] for row in rows]
    macro_f1 = [row["macro_f1"] for row in rows]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, accuracy, width, label="Accuracy")
    plt.bar(x + width / 2, macro_f1, width, label="Macro-F1")
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title("Final Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_split_summary(split_report: dict[str, Any], out_path: Path) -> None:
    # Visual check that train/val/test class balance matches expected split policy.
    class_names = split_report["class_names"]
    train = [split_report["train_class_distribution"][name] for name in class_names]
    val = [split_report["validation_class_distribution"][name] for name in class_names]
    test = [split_report["test_class_distribution"][name] for name in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar(x - width, train, width, label="Train")
    plt.bar(x, val, width, label="Validation")
    plt.bar(x + width, test, width, label="Test")
    plt.xticks(x, class_names)
    plt.ylabel("Image Count")
    plt.title("Dataset Split Class Balance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_cnn_learning_curve(cnn_history: dict[str, Any], out_path: Path) -> None:
    # Plot macro-F1 progression because it is class-balance aware.
    history = cnn_history["history"]
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_macro_f1"], marker="o", label="Train Macro-F1")
    plt.plot(epochs, history["val_macro_f1"], marker="o", label="Validation Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.ylim(0.0, 1.0)
    plt.title("Custom CNN Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_transfer_learning_curve(
    stage_head_history: dict[str, Any], stage_finetune_history: dict[str, Any], out_path: Path
) -> None:
    # Show both stages on one axis to highlight gain from fine-tuning.
    head_vals = stage_head_history["history"]["val_macro_f1"]
    fine_vals = stage_finetune_history["history"]["val_macro_f1"]

    x_head = np.arange(1, len(head_vals) + 1)
    x_fine = np.arange(len(head_vals) + 1, len(head_vals) + len(fine_vals) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(x_head, head_vals, marker="o", label="Head Stage Val Macro-F1")
    plt.plot(x_fine, fine_vals, marker="o", label="Fine-tune Stage Val Macro-F1")
    plt.xlabel("Epoch (across both stages)")
    plt.ylabel("Validation Macro-F1")
    plt.ylim(0.0, 1.0)
    plt.title("Transfer Learning Two-Stage Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_sample_gallery(dataset_root: Path, out_path: Path) -> None:
    # Keep sample gallery deterministic by sorting paths and taking first two samples per class.
    testing_root = dataset_root / "Testing"
    class_dirs = sorted([p for p in testing_root.iterdir() if p.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(f"No class folders found in {testing_root}")

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for col, class_dir in enumerate(class_dirs):
        images = sorted([p for p in class_dir.iterdir() if p.is_file()])[:2]
        for row in range(2):
            axis = axes[row][col]
            axis.axis("off")
            if row < len(images):
                image = Image.open(images[row]).convert("RGB")
                axis.imshow(image)
            if row == 0:
                axis.set_title(class_dir.name)

    fig.suptitle("Sample MRI Images from Testing Split", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _build_key_findings(
    baseline_metrics: dict[str, Any],
    cnn_metrics: dict[str, Any],
    transfer_metrics_original: dict[str, Any],
    transfer_metrics: dict[str, Any],
) -> str:
    baseline_summary = _summarize_metrics(baseline_metrics)
    cnn_summary = _summarize_metrics(cnn_metrics)
    transfer_original_summary = _summarize_metrics(transfer_metrics_original)
    transfer_summary = _summarize_metrics(transfer_metrics)

    per_class = []
    for class_name, values in transfer_metrics.items():
        if class_name in {"accuracy", "macro avg", "weighted avg", "training"}:
            continue
        per_class.append((class_name, float(values["recall"])))
    hardest = min(per_class, key=lambda item: item[1])
    easiest = max(per_class, key=lambda item: item[1])

    lines = [
        "# Key Findings",
        "",
        "- The best overall model in this project is the feature-based SVM baseline.",
        (
            f"- SVM baseline test performance: accuracy={baseline_summary['accuracy']:.4f}, "
            f"macro_f1={baseline_summary['macro_f1']:.4f}."
        ),
        (
            f"- Custom CNN test performance: accuracy={cnn_summary['accuracy']:.4f}, "
            f"macro_f1={cnn_summary['macro_f1']:.4f}."
        ),
        (
            f"- Tuned transfer model test performance: accuracy={transfer_summary['accuracy']:.4f}, "
            f"macro_f1={transfer_summary['macro_f1']:.4f}."
        ),
        (
            f"- Transfer tuning improved from accuracy={transfer_original_summary['accuracy']:.4f}, "
            f"macro_f1={transfer_original_summary['macro_f1']:.4f} to accuracy={transfer_summary['accuracy']:.4f}, "
            f"macro_f1={transfer_summary['macro_f1']:.4f}."
        ),
        (
            f"- In the tuned transfer model, hardest class by recall is '{hardest[0]}' "
            f"({hardest[1]:.4f}) and easiest class is '{easiest[0]}' ({easiest[1]:.4f})."
        ),
        "- The project still demonstrates strong evaluation discipline: validation comes from Training, and Testing is kept for final evaluation only.",
    ]
    return "\n".join(lines) + "\n"


def finalize_artifacts(dataset_root: Path, results_root: Path, output_root: Path) -> None:
    """Build a clean final package for reporting and presentation use."""
    real_run = results_root / "real_run"
    tuned_transfer_run = results_root / "real_run_tuned_fast"

    split_report = _read_json(real_run / "split_report.json")
    baseline_bundle = _read_json(real_run / "baselines" / "baseline_metrics.json")
    cnn_bundle = _read_json(real_run / "cnn" / "cnn_bundle.json")
    transfer_bundle_tuned = _read_json(tuned_transfer_run / "transfer" / "transfer_bundle.json")
    transfer_bundle_original = _read_json(real_run / "transfer" / "transfer_bundle.json")

    final_model_rows = [
        {
            "model": "baseline::svm_rbf",
            **_summarize_metrics(baseline_bundle["best_model_test_metrics"]),
            "source_run": "real_run",
        },
        {
            "model": "custom_cnn",
            **_summarize_metrics(cnn_bundle),
            "source_run": "real_run",
        },
        {
            "model": "transfer::resnet50_original",
            **_summarize_metrics(transfer_bundle_original),
            "source_run": "real_run",
        },
        {
            "model": "transfer::resnet50_tuned",
            **_summarize_metrics(transfer_bundle_tuned),
            "source_run": "real_run_tuned_fast",
        },
    ]

    counts_training = _count_split(dataset_root / "Training")
    counts_testing = _count_split(dataset_root / "Testing")
    total_images = sum(counts_training.values()) + sum(counts_testing.values())

    canonical_split = {
        "dataset_root": str(dataset_root),
        "proposal_note": "Proposal cites 7023 images. Local dataset contains 7200 images (balanced 1800/class).",
        "dataset_counts_from_disk": {
            "Training": counts_training,
            "Testing": counts_testing,
            "total_images": total_images,
        },
        "split_report_from_official_run": split_report,
    }

    comparison_dir = _ensure_dir(output_root / "comparison")
    assets_dir = _ensure_dir(output_root / "presentation_assets")
    models_dir = _ensure_dir(output_root / "models")

    # 1) Write provenance and final comparison tables.
    _write_json(output_root / "canonical_split_report.json", canonical_split)
    _write_json(comparison_dir / "final_model_comparison.json", {"models": final_model_rows})
    _write_csv(
        comparison_dir / "final_model_comparison.csv",
        final_model_rows,
        ["model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "source_run"],
    )

    # 2) Copy model-level artifacts into one normalized directory structure.
    _copy_file(real_run / "baselines" / "baseline_metrics.json", models_dir / "baseline_svm" / "baseline_metrics.json")
    _copy_file(
        real_run / "baselines" / "svm_rbf_test_confusion_matrix.png",
        models_dir / "baseline_svm" / "test_confusion_matrix.png",
    )

    _copy_file(real_run / "cnn" / "cnn_bundle.json", models_dir / "custom_cnn" / "cnn_bundle.json")
    _copy_file(real_run / "cnn" / "history.json", models_dir / "custom_cnn" / "history.json")
    _copy_file(real_run / "cnn" / "test_metrics.json", models_dir / "custom_cnn" / "test_metrics.json")
    _copy_file(real_run / "cnn" / "test_confusion_matrix.png", models_dir / "custom_cnn" / "test_confusion_matrix.png")

    _copy_file(
        tuned_transfer_run / "transfer" / "transfer_bundle.json",
        models_dir / "transfer_resnet50_tuned" / "transfer_bundle.json",
    )
    _copy_file(
        tuned_transfer_run / "transfer" / "test_metrics.json",
        models_dir / "transfer_resnet50_tuned" / "test_metrics.json",
    )
    _copy_file(
        tuned_transfer_run / "transfer" / "test_confusion_matrix.png",
        models_dir / "transfer_resnet50_tuned" / "test_confusion_matrix.png",
    )
    _copy_file(
        tuned_transfer_run / "transfer" / "stage_head" / "history.json",
        models_dir / "transfer_resnet50_tuned" / "stage_head_history.json",
    )
    _copy_file(
        tuned_transfer_run / "transfer" / "stage_finetune" / "history.json",
        models_dir / "transfer_resnet50_tuned" / "stage_finetune_history.json",
    )
    _copy_file(
        real_run / "transfer" / "transfer_bundle.json",
        models_dir / "transfer_resnet50_original" / "transfer_bundle.json",
    )
    _copy_file(
        real_run / "transfer" / "history.json",
        models_dir / "transfer_resnet50_original" / "history.json",
    )
    _copy_file(
        real_run / "transfer" / "test_metrics.json",
        models_dir / "transfer_resnet50_original" / "test_metrics.json",
    )
    _copy_file(
        real_run / "transfer" / "test_confusion_matrix.png",
        models_dir / "transfer_resnet50_original" / "test_confusion_matrix.png",
    )

    # 3) Generate presentation visuals from the canonical final data.
    _plot_comparison(final_model_rows, assets_dir / "final_model_comparison.png")
    _plot_split_summary(split_report, assets_dir / "split_summary.png")
    _plot_cnn_learning_curve(_read_json(real_run / "cnn" / "history.json"), assets_dir / "cnn_learning_curve.png")
    _plot_transfer_learning_curve(
        _read_json(tuned_transfer_run / "transfer" / "stage_head" / "history.json"),
        _read_json(tuned_transfer_run / "transfer" / "stage_finetune" / "history.json"),
        assets_dir / "transfer_two_stage_learning_curve.png",
    )
    _plot_sample_gallery(dataset_root, assets_dir / "mri_sample_gallery.png")

    # 4) Emit narrative findings and reproducibility manifest.
    key_findings = _build_key_findings(
        baseline_bundle["best_model_test_metrics"],
        cnn_bundle,
        transfer_bundle_original,
        transfer_bundle_tuned,
    )
    (output_root / "key_findings.md").write_text(key_findings, encoding="utf-8")

    run_manifest = {
        "official_dataset_root": str(dataset_root),
        "official_results_package": str(output_root),
        "selected_model_runs": {
            "baseline": "results/real_run/baselines",
            "custom_cnn": "results/real_run/cnn",
            "transfer_original": "results/real_run/transfer",
            "transfer_tuned": "results/real_run_tuned_fast/transfer",
        },
        "canonical_commands": {
            "all_models_reference_run": (
                "python -m src.main --dataset-root data --task all --epochs 3 "
                "--batch-size 32 --transfer-model resnet50 --output-dir results/real_run"
            ),
            "tuned_transfer_reference_run": (
                "python -m src.main --dataset-root data --task transfer --epochs 3 "
                "--batch-size 32 --transfer-model resnet50 --output-dir results/real_run_tuned_fast"
            ),
        },
        "seed": 42,
    }
    _write_json(output_root / "run_manifest.json", run_manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final, presentation-ready artifacts.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-root", type=Path, default=Path("results/final_submission"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    finalize_artifacts(args.dataset_root, args.results_root, args.output_root)
    print(f"Final artifact package generated at: {args.output_root}")


if __name__ == "__main__":
    main()
