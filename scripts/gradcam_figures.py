"""Grad-CAM interpretability figures for the brain-tumor classifiers.

Produces class-activation overlays for a trained model on sample MRI slices (one
column per true class). Supports the CNN/ResNet/EfficientNet conv models and the
ViT (via a reshape transform on the final encoder block). Saliency is qualitative
context, NOT evidence of clinical reasoning -- see the README caveat.

Usage:
    python scripts/gradcam_figures.py --model resnet50 --weights results/research_grade/resnet50/seed0/model_seed0.pt
    python scripts/gradcam_figures.py --model vit_b_16 --weights results/research_grade/vit_b_16/seed0/model_seed0.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import build_transforms
from src.dedup_split import build_dedup_frames
from src.cnn import CustomCNN
from src.transfer import build_transfer_model


def get_target_layers(model, name: str):
    if name == "custom_cnn":
        return [model.features[-2]]  # last ReLU before adaptive pool
    if name == "resnet50":
        return [model.layer4[-1]]
    if name == "efficientnet_b0":
        return [model.features[-1]]
    if name == "vit_b_16":
        return [model.encoder.layers[-1].ln_1]
    raise ValueError(name)


def vit_reshape_transform(tensor, height: int = 14, width: int = 14):
    # Drop CLS token, fold patch tokens back into a (C,H,W) feature map.
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=["custom_cnn", "resnet50", "efficientnet_b0", "vit_b_16"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset-root", default=r"C:\Users\lawso\.cache\kagglehub\datasets\masoudnickparvar\brain-tumor-mri-dataset\versions\2")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-per-class", type=int, default=3)
    ap.add_argument("--image-size", type=int, default=224)
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    splits = build_dedup_frames(args.dataset_root)
    class_names = splits.class_names
    n_classes = len(class_names)

    if args.model == "custom_cnn":
        model = CustomCNN(num_classes=n_classes)
    else:
        model = build_transfer_model(args.model, num_classes=n_classes, freeze_backbone=False)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    _, eval_tf = build_transforms(image_size=args.image_size)
    reshape = vit_reshape_transform if args.model == "vit_b_16" else None
    cam = GradCAM(model=model, target_layers=get_target_layers(model, args.model), reshape_transform=reshape)

    out = Path(args.out) if args.out else Path(args.weights).resolve().parent / "gradcam"
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(n_classes, args.n_per_class * 2, figsize=(args.n_per_class * 4, n_classes * 2.2))
    test = splits.test_clean
    for ci, cname in enumerate(class_names):
        rows = test[test["class_name"] == cname].head(args.n_per_class)
        for j, (_, row) in enumerate(rows.iterrows()):
            img = Image.open(row["path"]).convert("RGB").resize((args.image_size, args.image_size))
            rgb = np.asarray(img, dtype=np.float32) / 255.0
            x = eval_tf(img).unsqueeze(0)
            grayscale = cam(input_tensor=x, targets=None)[0]
            overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)
            ax_img = axes[ci, j * 2] if n_classes > 1 else axes[j * 2]
            ax_cam = axes[ci, j * 2 + 1] if n_classes > 1 else axes[j * 2 + 1]
            ax_img.imshow(rgb); ax_img.set_title(cname, fontsize=8); ax_img.axis("off")
            ax_cam.imshow(overlay); ax_cam.set_title("Grad-CAM", fontsize=8); ax_cam.axis("off")

    fig.suptitle(f"Grad-CAM: {args.model}", fontsize=12)
    fig.tight_layout()
    path = out / f"gradcam_{args.model}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
