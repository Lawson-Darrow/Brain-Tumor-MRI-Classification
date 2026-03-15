from __future__ import annotations

import torch.nn as nn


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int, base_filters: int = 32, dropout: float = 0.35) -> None:
        super().__init__()
        # Progressive conv blocks capture local texture and larger spatial patterns.
        self.features = nn.Sequential(
            nn.Conv2d(3, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_filters * 4, base_filters * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Compact classifier head maps pooled features to class logits.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_filters * 8, base_filters * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(base_filters * 4, num_classes),
        )

    def forward(self, x):
        # Keep forward path explicit so tensor flow is easy to trace.
        x = self.features(x)
        return self.classifier(x)

