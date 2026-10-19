from __future__ import annotations

import torch.nn as nn
import torchvision.models as tvm

_WEIGHTS = {
    "resnet18": "ResNet18_Weights",
    "resnet50": "ResNet50_Weights",
    "resnet101": "ResNet101_Weights",
}


def build_model(name: str = "resnet50", num_classes: int = 8, pretrained: bool = True) -> nn.Module:
    if name not in _WEIGHTS:
        raise ValueError(f"model {name} not supported; choose from {list(_WEIGHTS)}")
    weights = getattr(tvm, _WEIGHTS[name]).IMAGENET1K_V1 if pretrained else None
    model = getattr(tvm, name)(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
