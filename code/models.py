"""Model factory — load models for experiments."""

import torch
import torch.nn as nn
import timm
import torchvision.models as tv_models


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CIFARResNet(nn.Module):
    """
    ResNet adapted for CIFAR (32x32 input).
    Standard approach: stride=1 in first conv, remove maxpool.
    """

    def __init__(self, base_model: str, num_classes: int = 10, pretrained: bool = False):
        super().__init__()
        if base_model == "resnet50":
            self.backbone = tv_models.resnet50(weights="DEFAULT" if pretrained else None)
        elif base_model == "resnet101":
            self.backbone = tv_models.resnet101(weights="DEFAULT" if pretrained else None)
        else:
            raise ValueError(f"Unknown ResNet variant: {base_model}")

        # Adapt first conv for 32x32: 7x7 → 3x3, stride 2 → 1, no padding
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Remove maxpool
        self.backbone.maxpool = nn.Identity()

        # Replace classifier
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def build_model(model_name: str, num_classes: int, pretrained: bool = True):
    """
    Build model for experiment.

    Args:
        model_name: key into MODELS dict ('vit_s', 'vit_b', 'deit_s', 'resnet50', etc.)
        num_classes: number of output classes
        pretrained: use pretrained weights (ImageNet)
    """
    if model_name == "vit_s":
        model = timm.create_model(
            "vit_small_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )
    elif model_name == "vit_b":
        model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )
    elif model_name == "deit_s":
        model = timm.create_model(
            "deit_small_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )
    elif model_name == "deit_b":
        model = timm.create_model(
            "deit_base_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )
    elif model_name in ("resnet50", "resnet101"):
        # Use CIFAR-adapted ResNet
        model = CIFARResNet(
            base_model=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model
