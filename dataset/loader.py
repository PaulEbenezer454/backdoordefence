"""Dataset loading helpers with explicit, opt-in downloads."""
from __future__ import annotations
from pathlib import Path
from torchvision import datasets, transforms


def dataset_transform(name: str):
    """Return deterministic normalization for the supported image dataset."""
    if name.lower() == "mnist":
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    if name.lower() == "cifar10":
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    raise ValueError(f"Unsupported dataset: {name}")


def load_torchvision_dataset(name: str, data_dir: Path, train: bool,
                             download: bool = False):
    """Load MNIST or CIFAR-10; no download occurs unless explicitly requested."""
    transform = dataset_transform(name)
    if name.lower() == "mnist":
        return datasets.MNIST(str(data_dir), train=train, download=download, transform=transform)
    if name.lower() == "cifar10":
        return datasets.CIFAR10(str(data_dir), train=train, download=download, transform=transform)
    raise ValueError(f"Unsupported dataset: {name}")
