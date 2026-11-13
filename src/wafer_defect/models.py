from __future__ import annotations

import torch.nn as nn
import torchvision

MODEL_NAMES = ("custom_cnn", "resnet18")


class CustomCNN(nn.Module):
    """4 blocks of (conv-BN-ReLU) x2 + max-pool, then global average pooling and an MLP head."""

    def __init__(self, n_classes: int, channels=(32, 64, 128, 256), dropout: float = 0.4):
        super().__init__()
        layers, cin = [], 3
        for c in channels:
            layers += [
                nn.Conv2d(cin, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(),
                nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(),
                nn.MaxPool2d(2),
            ]
            cin = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(cin, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def make_resnet18(n_classes: int, pretrained: bool = False) -> nn.Module:
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.resnet18(weights=weights)
    if not pretrained:
        # small-image stem: 3x3 stride-1 conv, no max-pool (inputs are only 64x64)
        m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


def _disable_inplace(model: nn.Module) -> None:
    """In-place ReLUs break backward hooks (needed for Grad-CAM)."""
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False


def build_model(name: str, n_classes: int, pretrained: bool = False) -> nn.Module:
    if name == "custom_cnn":
        model = CustomCNN(n_classes)
    elif name == "resnet18":
        model = make_resnet18(n_classes, pretrained)
    else:
        raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}.")
    _disable_inplace(model)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
