"""Leakage-aware data splits for the brain-tumor MRI dataset.

The audit (scripts/audit_leakage.py) showed the official split leaks: ~63% of test
images have a near-duplicate in Training, and ~65% of Training has an internal
near-duplicate. This module builds splits that account for that:

  * group-aware train/val split  -- near-duplicate images share a group id and are
    kept on the same side of the train/val boundary (honest model selection);
  * a deduplicated ("clean") test set -- official Test images whose perceptual-hash
    group has NO member in Training, i.e. images with no near-twin the model trained on.

We report BOTH the official test split (comparable to published numbers, leakage
disclosed) and the clean split (honest generalization estimate).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

from .data import _collect_split_records

IMG_HASH_SIZE = 8


@dataclass
class DedupSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test_official: pd.DataFrame
    test_clean: pd.DataFrame
    class_names: list[str]
    stats: dict


def _dhash_bits(path: str, size: int = IMG_HASH_SIZE) -> np.ndarray:
    img = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.int16)
    return (arr[:, 1:] > arr[:, :-1]).flatten().astype(np.uint8)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _group_by_near_dup(bits: np.ndarray, threshold: int) -> np.ndarray:
    """Connected-components grouping: edge between i,j if Hamming(bits) <= threshold."""
    n = len(bits)
    uf = _UnionFind(n)
    for i in range(n):
        d = np.count_nonzero(bits[i + 1 :] ^ bits[i], axis=1)
        for j in np.where(d <= threshold)[0]:
            uf.union(i, int(i + 1 + j))
    roots = np.array([uf.find(i) for i in range(n)])
    # Re-label roots to contiguous group ids.
    _, group_ids = np.unique(roots, return_inverse=True)
    return group_ids


def build_dedup_frames(
    dataset_root: str | Path,
    val_size: float = 0.15,
    random_seed: int = 42,
    hamming_threshold: int = 5,
) -> DedupSplits:
    dataset_root = Path(dataset_root)
    train_records = _collect_split_records(dataset_root / "Training")
    test_records = _collect_split_records(dataset_root / "Testing")

    train_df = pd.DataFrame(train_records)
    test_df = pd.DataFrame(test_records)
    class_names = sorted(train_df["class_name"].unique().tolist())
    class_to_label = {n: i for i, n in enumerate(class_names)}
    train_df["label"] = train_df["class_name"].map(class_to_label)
    test_df["label"] = test_df["class_name"].map(class_to_label)

    n_train = len(train_df)
    all_paths = train_df["path"].tolist() + test_df["path"].tolist()
    bits = np.stack([_dhash_bits(p) for p in all_paths]).astype(np.uint8)
    groups = _group_by_near_dup(bits, hamming_threshold)
    train_df = train_df.assign(group=groups[:n_train])
    test_df = test_df.assign(group=groups[n_train:])

    # Group-aware train/val split: near-duplicates stay on the same side.
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_seed)
    tr_idx, va_idx = next(gss.split(train_df, train_df["label"], groups=train_df["group"]))
    train_split = train_df.iloc[tr_idx].reset_index(drop=True)
    val_split = train_df.iloc[va_idx].reset_index(drop=True)

    # Clean test = official Test images whose group never appears in Training.
    train_groups = set(train_df["group"].tolist())
    clean_mask = ~test_df["group"].isin(train_groups)
    test_clean = test_df[clean_mask].reset_index(drop=True)

    stats = {
        "hamming_threshold": hamming_threshold,
        "n_train": int(len(train_split)),
        "n_val": int(len(val_split)),
        "n_test_official": int(len(test_df)),
        "n_test_clean": int(len(test_clean)),
        "test_dropped_as_leaked": int(len(test_df) - len(test_clean)),
        "test_leaked_pct": round(100 * (len(test_df) - len(test_clean)) / len(test_df), 2),
    }
    return DedupSplits(
        train=train_split,
        val=val_split,
        test_official=test_df,
        test_clean=test_clean,
        class_names=class_names,
        stats=stats,
    )
