from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_transfer_model(
    model_name: str,
    num_classes: int,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Load a pretrained backbone and swap its classifier for project classes."""
    model_name = model_name.lower()

    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(
            f"Unsupported transfer model '{model_name}'. "
            "Choose from: resnet50, efficientnet_b0, vgg16."
        )

    if freeze_backbone:
        # Head-only warmup is more stable on small datasets than full-network training from epoch 1.
        for param in model.parameters():
            param.requires_grad = False

        if model_name == "resnet50":
            for param in model.fc.parameters():
                param.requires_grad = True
        elif model_name == "efficientnet_b0":
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif model_name == "vgg16":
            for param in model.classifier.parameters():
                param.requires_grad = True

    return model


def unfreeze_last_n_feature_blocks(
    model: nn.Module, model_name: str, n_blocks: int = 1
) -> None:
    """Unfreeze a small tail of feature blocks for controlled fine-tuning."""
    if n_blocks <= 0:
        return

    model_name = model_name.lower()
    blocks: list[nn.Module]

    if model_name == "resnet50":
        blocks = [model.layer4, model.layer3, model.layer2, model.layer1]
    elif model_name == "efficientnet_b0":
        blocks = list(model.features)[::-1]
    elif model_name == "vgg16":
        blocks = [model.features]
    else:
        return

    # Unfreeze only the requested number of deepest blocks.
    for module in blocks[:n_blocks]:
        for param in module.parameters():
            param.requires_grad = True

