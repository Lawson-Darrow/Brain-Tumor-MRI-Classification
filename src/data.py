from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SplitDataFrames:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    class_names: list[str]


def _collect_split_records(split_dir: Path) -> list[dict[str, str | int]]:
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    # Class folders are read in sorted order so label assignment stays stable
    # across runs and machines.
    class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    records: list[dict[str, str | int]] = []

    for class_idx, class_dir in enumerate(class_dirs):
        # rglob allows nested folder layouts while still collecting only image files.
        for image_path in class_dir.rglob("*"):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append(
                    {
                        "path": str(image_path),
                        "class_name": class_dir.name,
                        "label": class_idx,
                    }
                )
    return records


def load_dataset_frames(
    dataset_root: str | Path,
    val_size: float = 0.15,
    random_seed: int = 42,
) -> SplitDataFrames:
    """Load official Training/Testing directories and create a stratified validation split.

    The test set is never split or shuffled into validation to avoid leakage.
    """
    dataset_root = Path(dataset_root)
    train_root = dataset_root / "Training"
    test_root = dataset_root / "Testing"

    train_records = _collect_split_records(train_root)
    test_records = _collect_split_records(test_root)
    if not train_records or not test_records:
        raise ValueError("Training and Testing folders must both contain labeled images.")

    full_train_df = pd.DataFrame(train_records)
    test_df = pd.DataFrame(test_records)

    class_names = sorted(full_train_df["class_name"].unique().tolist())
    class_to_label = {name: i for i, name in enumerate(class_names)}

    # Force label mapping from class names so train/test share the same numeric IDs.
    full_train_df["label"] = full_train_df["class_name"].map(class_to_label)
    test_df["label"] = test_df["class_name"].map(class_to_label)

    # Stratification keeps class ratios consistent between train and validation.
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_seed)
    train_idx, val_idx = next(splitter.split(full_train_df["path"], full_train_df["label"]))

    train_df = full_train_df.iloc[train_idx].reset_index(drop=True)
    val_df = full_train_df.iloc[val_idx].reset_index(drop=True)

    return SplitDataFrames(train=train_df, val=val_df, test=test_df, class_names=class_names)


class BrainTumorImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: Callable | None = None) -> None:
        # Reset index so DataLoader indexing is always contiguous.
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"])


def build_transforms(image_size: int = 224) -> tuple[Callable, Callable]:
    # Training transform includes augmentation to improve generalization.
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.08, contrast=0.08),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Validation/test transform mirrors model input formatting without augmentation.
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, eval_transform


def create_dataloaders(
    split_frames: SplitDataFrames,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/validation/test dataloaders with task-appropriate transforms."""
    train_tf, eval_tf = build_transforms(image_size=image_size)

    train_dataset = BrainTumorImageDataset(split_frames.train, transform=train_tf)
    val_dataset = BrainTumorImageDataset(split_frames.val, transform=eval_tf)
    test_dataset = BrainTumorImageDataset(split_frames.test, transform=eval_tf)

    train_loader = DataLoader(
        # Shuffle only training batches so optimization does not learn order patterns.
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader

