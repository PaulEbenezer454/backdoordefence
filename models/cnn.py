"""Small CNN for MNIST with named, inspectable trainable layers."""

from __future__ import annotations

import torch
from torch import nn


class MNISTCNN(nn.Module):
    """Two-convolution MNIST classifier with explicit layer names."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.output = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        return self.output(x)


class CIFAR10CNN(nn.Module):
    """Small three-convolution CNN for 32x32 color images."""
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.output = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        return self.output(self.relu(self.fc1(x)))


def create_model(name: str, num_classes: int = 10) -> nn.Module:
    """Build a supported small CNN by configured name."""
    factories = {"mnist_cnn": MNISTCNN, "cifar10_cnn": CIFAR10CNN}
    try:
        return factories[name.lower()](num_classes=num_classes)
    except KeyError as exc:
        raise ValueError(f"Unsupported model architecture: {name}") from exc


def layer_parameters(model: nn.Module) -> dict[str, dict[str, nn.Parameter]]:
    """Return trainable parameters grouped by owning module name."""
    result: dict[str, dict[str, nn.Parameter]] = {}
    for module_name, module in model.named_modules():
        params = {name: param for name, param in module.named_parameters(recurse=False)
                  if param.requires_grad}
        if params:
            result[module_name or "root"] = params
    return result


def extract_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copy all trainable named parameters to CPU tensors."""
    return {name: value.detach().cpu().clone()
            for name, value in model.named_parameters() if value.requires_grad}


def parameter_differences(
    newer: dict[str, torch.Tensor], older: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Compute named parameter deltas after validating matching structures."""
    if newer.keys() != older.keys():
        raise ValueError("Parameter mappings have different keys")
    return {name: newer[name] - older[name] for name in newer}


def restore_parameters(model: nn.Module, values: dict[str, torch.Tensor]) -> None:
    """Restore named parameters, rejecting missing or extra entries."""
    params = dict(model.named_parameters())
    if params.keys() != values.keys():
        raise ValueError("Parameter mapping does not match model")
    with torch.no_grad():
        for name, parameter in params.items():
            parameter.copy_(values[name].to(device=parameter.device, dtype=parameter.dtype))
