"""Near-duplicate / data-leakage audit for the brain-tumor MRI dataset.

The masoudnickparvar set is assembled from multiple public sources and contains
near-duplicate MRI slices. If duplicates straddle the Training/Testing boundary,
the official test metrics are optimistic; if they straddle the train/val split we
carve from Training, validation-based model selection is optimistic too.

This computes a 64-bit perceptual hash (dHash) per image and reports, per class
and overall: exact duplicates and near-duplicates (Hamming <= threshold) that
cross Train<->Test, plus near-duplicate density inside Training (the train/val
risk). No deep model, no GPU. Findings feed the README limitations section.

Usage:
    python scripts/audit_leakage.py
    python scripts/audit_leakage.py --root <dataset_root> --threshold 5
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_ROOT = r"C:\Users\lawso\.cache\kagglehub\datasets\masoudnickparvar\brain-tumor-mri-dataset\versions\2"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def dhash_bits(path: Path, size: int = 8) -> np.ndarray:
    """64-bit dHash as a (64,) uint8 bit array (row-wise horizontal gradient)."""
    img = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]  # (size, size)
    return diff.flatten().astype(np.uint8)


def collect(split_dir: Path) -> tuple[list[Path], dict]:
    paths, by_class = [], defaultdict(list)
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for p in sorted(class_dir.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                idx = len(paths)
                paths.append(p)
                by_class[class_dir.name].append(idx)
    return paths, by_class


def hash_matrix(paths: list[Path]) -> np.ndarray:
    return np.stack([dhash_bits(p) for p in paths]).astype(np.uint8)  # (N,64)


def cross_min_distances(test_bits: np.ndarray, train_bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each test row, the min Hamming distance to any train row, and that train index."""
    min_d = np.empty(len(test_bits), dtype=np.int32)
    arg = np.empty(len(test_bits), dtype=np.int64)
    for i, row in enumerate(test_bits):
        d = np.count_nonzero(train_bits ^ row, axis=1)  # Hamming to all train
        arg[i] = int(np.argmin(d))
        min_d[i] = int(d[arg[i]])
    return min_d, arg


def within_near_dup_count(bits: np.ndarray, threshold: int) -> int:
    """Number of images that have at least one near-duplicate elsewhere in the same set."""
    n = len(bits)
    flagged = np.zeros(n, dtype=bool)
    for i in range(n):
        d = np.count_nonzero(bits[i + 1 :] ^ bits[i], axis=1)
        hit = np.where(d <= threshold)[0]
        if len(hit):
            flagged[i] = True
            flagged[i + 1 + hit] = True
    return int(flagged.sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--threshold", type=int, default=5, help="Hamming <= threshold counts as near-duplicate")
    ap.add_argument("--out", default="results/leakage_audit.json")
    args = ap.parse_args()

    root = Path(args.root)
    train_paths, train_by_class = collect(root / "Training")
    test_paths, test_by_class = collect(root / "Testing")
    print(f"Hashing {len(train_paths)} train + {len(test_paths)} test images...")
    train_bits = hash_matrix(train_paths)
    test_bits = hash_matrix(test_paths)

    min_d, arg = cross_min_distances(test_bits, train_bits)
    exact = int(np.count_nonzero(min_d == 0))
    near = int(np.count_nonzero(min_d <= args.threshold))

    # Worst offenders for the writeup.
    order = np.argsort(min_d)[:15]
    examples = [
        {
            "test": str(test_paths[i].relative_to(root)),
            "nearest_train": str(train_paths[arg[i]].relative_to(root)),
            "hamming": int(min_d[i]),
        }
        for i in order
    ]

    within_train = within_near_dup_count(train_bits, args.threshold)

    report = {
        "threshold": args.threshold,
        "n_train": len(train_paths),
        "n_test": len(test_paths),
        "test_exact_dup_in_train": exact,
        "test_near_dup_in_train": near,
        "test_near_dup_pct": round(100 * near / len(test_paths), 2),
        "train_images_with_near_dup_in_train": within_train,
        "train_near_dup_pct": round(100 * within_train / len(train_paths), 2),
        "worst_cross_split_examples": examples,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("\n=== Leakage audit (dHash, Hamming <= {}) ===".format(args.threshold))
    print(f"Test images with EXACT duplicate in Training : {exact} ({100*exact/len(test_paths):.2f}%)")
    print(f"Test images with NEAR duplicate in Training  : {near} ({report['test_near_dup_pct']}%)")
    print(f"Training images with a near-dup inside Training (train/val risk): "
          f"{within_train} ({report['train_near_dup_pct']}%)")
    print("\nWorst cross-split matches:")
    for e in examples[:8]:
        print(f"  H={e['hamming']:2d}  {e['test']}  ~=  {e['nearest_train']}")
    print(f"\nReport -> {out}")


if __name__ == "__main__":
    main()
