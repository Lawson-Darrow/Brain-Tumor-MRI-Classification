"""Research-grade brain-tumor classification experiments.

Trains the deep models (custom CNN, ResNet50, EfficientNet-B0, ViT-B/16) to
convergence across multiple seeds on a leakage-aware split, and evaluates each on
BOTH the official test set and the deduplicated ("clean") test set. Reports
accuracy, macro precision/recall/F1, macro one-vs-rest ROC-AUC, and expected
calibration error, with mean +/- std across seeds and bootstrap confidence
intervals. Classical HOG baselines (LogReg/SVM/RF) are run once for continuity.

VGG16 is intentionally excluded (heavy, low marginal value). Saves a master
results JSON + comparison CSV, and the seed-0 weights of each deep model for the
Grad-CAM step.

Usage:
    python scripts/run_brain_experiments.py
    python scripts/run_brain_experiments.py --epochs 25 --seeds 0 1 2 --models custom_cnn resnet50 efficientnet_b0 vit_b_16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cnn import CustomCNN
from src.data import BrainTumorImageDataset, SplitDataFrames, build_transforms, create_dataloaders
from src.dedup_split import build_dedup_frames
from src.train import predict_proba, train_model
from src.transfer import build_transfer_model, unfreeze_last_n_feature_blocks
from src.utils import ensure_dir, save_json, set_global_seed

DEEP_MODELS = ["custom_cnn", "resnet50", "efficientnet_b0", "vit_b_16"]


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() > 0:
            ece += abs(correct[m].mean() - conf[m].mean()) * m.mean()
    return float(ece)


def metrics_from_probs(y_true: np.ndarray, probs: np.ndarray, class_names: list[str]) -> dict:
    y_pred = probs.argmax(axis=1)
    rep = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    try:
        auc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except ValueError:
        auc = None
    return {
        "accuracy": float(rep["accuracy"]),
        "macro_precision": float(rep["macro avg"]["precision"]),
        "macro_recall": float(rep["macro avg"]["recall"]),
        "macro_f1": float(rep["macro avg"]["f1-score"]),
        "roc_auc_ovr_macro": auc,
        "ece": expected_calibration_error(y_true, probs),
        "per_class_f1": {c: float(rep[c]["f1-score"]) for c in class_names},
    }


def bootstrap_acc_ci(y_true: np.ndarray, probs: np.ndarray, n: int = 1000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    n_s = len(y_true)
    vals = [correct[rng.integers(0, n_s, n_s)].mean() for _ in range(n)]
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def build_model(name: str, num_classes: int):
    if name == "custom_cnn":
        return CustomCNN(num_classes=num_classes), "scratch"
    return build_transfer_model(name, num_classes=num_classes, freeze_backbone=True), "transfer"


def train_and_eval(name: str, splits, args, seed: int, out_dir: Path) -> dict:
    set_global_seed(seed)
    n_classes = len(splits.class_names)
    sf = SplitDataFrames(splits.train, splits.val, splits.test_official, splits.class_names)
    train_loader, val_loader, test_loader = create_dataloaders(
        sf, image_size=args.image_size, batch_size=args.batch, num_workers=0
    )
    _, eval_tf = build_transforms(image_size=args.image_size)
    clean_loader = DataLoader(
        BrainTumorImageDataset(splits.test_clean, transform=eval_tf), batch_size=args.batch, shuffle=False
    )

    model, kind = build_model(name, n_classes)
    if kind == "scratch":
        model, _ = train_model(model, train_loader, val_loader, splits.class_names, out_dir, epochs=args.epochs, learning_rate=1e-3)
    else:
        head_ep = max(2, args.epochs // 3)
        model, _ = train_model(model, train_loader, val_loader, splits.class_names, out_dir / "head", epochs=head_ep, learning_rate=2e-4)
        unfreeze_last_n_feature_blocks(model, name, n_blocks=1)
        model, _ = train_model(model, train_loader, val_loader, splits.class_names, out_dir / "ft", epochs=max(1, args.epochs - head_ep), learning_rate=5e-5)

    yt_o, pr_o = predict_proba(model, test_loader)
    yt_c, pr_c = predict_proba(model, clean_loader)
    if seed == args.seeds[0]:
        torch.save(model.state_dict(), out_dir / "model_seed0.pt")  # for Grad-CAM

    official = metrics_from_probs(yt_o, pr_o, splits.class_names)
    official["acc_ci95"] = bootstrap_acc_ci(yt_o, pr_o)
    clean = metrics_from_probs(yt_c, pr_c, splits.class_names)
    clean["acc_ci95"] = bootstrap_acc_ci(yt_c, pr_c)
    return {"seed": seed, "official": official, "clean": clean}


def aggregate(runs: list[dict]) -> dict:
    def ms(key_path):
        vals = []
        for r in runs:
            v = r
            for k in key_path:
                v = v[k]
            if v is not None:
                vals.append(v)
        return {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)} if vals else None

    out = {}
    for split in ("official", "clean"):
        out[split] = {m: ms([split, m]) for m in ("accuracy", "macro_f1", "roc_auc_ovr_macro", "ece")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", default=r"C:\Users\lawso\.cache\kagglehub\datasets\masoudnickparvar\brain-tumor-mri-dataset\versions\2")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--models", nargs="+", default=DEEP_MODELS)
    ap.add_argument("--hamming", type=int, default=5)
    ap.add_argument("--out", default="results/research_grade")
    ap.add_argument("--skip-baselines", action="store_true")
    args = ap.parse_args()

    out_root = ensure_dir(Path(args.out))
    print("Building leakage-aware splits...")
    splits = build_dedup_frames(args.dataset_root, random_seed=args.seeds[0], hamming_threshold=args.hamming)
    print("Split stats:", splits.stats)
    save_json(splits.stats, out_root / "split_stats.json")

    results = {"split_stats": splits.stats, "models": {}}

    # Classical baselines (once, official split) for continuity with the original study.
    if not args.skip_baselines:
        from src.baselines import run_baseline_experiments

        print("Running classical baselines (HOG features)...")
        base = run_baseline_experiments(splits.train, splits.val, splits.test_official, splits.class_names, out_root / "baselines")
        results["baselines_best"] = {
            "model": base["best_model_by_validation_macro_f1"],
            "official_macro_f1": float(base["best_model_test_metrics"]["macro avg"]["f1-score"]),
            "official_accuracy": float(base["best_model_test_metrics"]["accuracy"]),
        }

    for name in args.models:
        print(f"\n=== {name} ===")
        runs = []
        for seed in args.seeds:
            print(f"  seed {seed}...", flush=True)
            run = train_and_eval(name, splits, args, seed, ensure_dir(out_root / name / f"seed{seed}"))
            runs.append(run)
            print(f"    official acc={run['official']['accuracy']:.4f} auc={run['official']['roc_auc_ovr_macro']} "
                  f"| clean acc={run['clean']['accuracy']:.4f}", flush=True)
        results["models"][name] = {"runs": runs, "aggregate": aggregate(runs)}
        save_json(results, out_root / "results.json")  # checkpoint after each model

    save_json(results, out_root / "results.json")
    print("\n=== SUMMARY (mean across seeds) ===")
    for name in args.models:
        agg = results["models"][name]["aggregate"]
        o, c = agg["official"], agg["clean"]
        print(f"{name:18s} official acc={o['accuracy']['mean']:.3f} f1={o['macro_f1']['mean']:.3f} "
              f"auc={o['roc_auc_ovr_macro']['mean']:.3f} | clean acc={c['accuracy']['mean']:.3f} f1={c['macro_f1']['mean']:.3f}")
    print(f"\nLeakage: dropped {splits.stats['test_dropped_as_leaked']} leaked test imgs "
          f"({splits.stats['test_leaked_pct']}%); clean test n={splits.stats['n_test_clean']}")
    print(f"Results -> {out_root / 'results.json'}")


if __name__ == "__main__":
    main()
