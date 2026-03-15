from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from skimage.feature import hog


@dataclass
class FeatureConfig:
    # Smaller feature extraction size keeps classical model runtime manageable.
    image_size: int = 128
    # Toggle between HOG features and raw flatten features.
    use_hog: bool = True
    pixels_per_cell: int = 16
    cells_per_block: int = 2
    # Set to None to disable PCA.
    pca_components: int | None = 200


def _load_image_array(path: str, image_size: int = 128) -> np.ndarray:
    # Baseline branch uses grayscale to reduce feature dimensionality.
    image = Image.open(path).convert("L").resize((image_size, image_size))
    return np.asarray(image, dtype=np.float32) / 255.0


def _extract_single_feature(image: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    if cfg.use_hog:
        # HOG captures edge/shape structure and usually outperforms raw pixels in small-data settings.
        return hog(
            image,
            orientations=9,
            pixels_per_cell=(cfg.pixels_per_cell, cfg.pixels_per_cell),
            cells_per_block=(cfg.cells_per_block, cfg.cells_per_block),
            block_norm="L2-Hys",
            feature_vector=True,
        ).astype(np.float32)
    return image.flatten().astype(np.float32)


def extract_feature_matrix(frame: pd.DataFrame, cfg: FeatureConfig) -> tuple[np.ndarray, np.ndarray]:
    """Convert image paths into a 2D feature matrix used by sklearn models."""
    features = []
    labels = frame["label"].astype(int).to_numpy()
    for path in frame["path"].tolist():
        image = _load_image_array(path, image_size=cfg.image_size)
        features.append(_extract_single_feature(image, cfg))
    x = np.vstack(features)
    return x, labels


def maybe_apply_pca(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    pca_components: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, PCA | None]:
    """Optionally reduce feature dimensionality using PCA fit on train only."""
    if pca_components is None:
        return x_train, x_val, x_test, None

    # Cap components to valid bounds so PCA never requests impossible dimensions.
    n_components = min(pca_components, x_train.shape[0], x_train.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    x_train_pca = pca.fit_transform(x_train)
    x_val_pca = pca.transform(x_val)
    x_test_pca = pca.transform(x_test)
    return x_train_pca, x_val_pca, x_test_pca, pca

