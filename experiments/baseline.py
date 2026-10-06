"""Reproducible centralized image-classification baseline training."""

from __future__ import annotations

import json
import logging
import platform
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
import torch
import torchvision
import yaml
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from dataset.loader import load_torchvision_dataset
from models.cnn import create_model

LOGGER = logging.getLogger("tclad.centralized")


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Set Python, NumPy, and PyTorch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Return mean cross-entropy and accuracy for a data loader."""
    model.eval()
    loss_sum = 0.0
    correct = total = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += float(criterion(logits, labels).item())
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += len(labels)
    if total == 0:
        raise ValueError("Evaluation dataset is empty")
    return loss_sum / total, correct / total


def run(config_path: Path, experiment_id: str | None = None) -> Path:
    """Train centralized MNIST, saving config, metrics, checkpoint, and plot."""
    started = time.perf_counter()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    dataset_name = config["dataset"]["name"].lower()

    seed = int(config["project"]["seed"])
    seed_everything(seed, config["training"].get("deterministic", True))
    requested_device = config["training"].get("device", "auto")
    device = torch.device("cuda" if requested_device == "auto" and torch.cuda.is_available()
                          else "cpu" if requested_device == "auto" else requested_device)
    LOGGER.info("Training on %s", device)

    data_dir = Path(config["dataset"]["data_dir"])
    if not data_dir.is_absolute():
        data_dir = config_path.resolve().parent.parent / data_dir
    download = bool(config["dataset"].get("download", False))
    train_data = load_torchvision_dataset(dataset_name, data_dir, train=True, download=download)
    test_data = load_torchvision_dataset(dataset_name, data_dir, train=False, download=download)

    subset_size = config["dataset"].get("subset_size")
    indices = np.arange(len(train_data))
    if subset_size is not None:
        indices = indices[:min(int(subset_size), len(indices))]
    train_idx, val_idx = train_test_split(indices, test_size=float(config["dataset"]["validation_fraction"]),
                                           random_state=seed, stratify=np.asarray(train_data.targets)[indices])
    batch_size = int(config["training"]["batch_size"])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(Subset(train_data, train_idx.tolist()), batch_size=batch_size,
                              shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(Subset(train_data, val_idx.tolist()), batch_size=batch_size,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    model = create_model(config["model"]["name"], int(config["model"]["num_classes"])).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=float(config["training"]["learning_rate"]))
    criterion = nn.CrossEntropyLoss()
    epochs = int(config["training"].get("centralized_epochs", 3))
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(labels)
            seen += len(labels)
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": running_loss / seen,
                        "validation_loss": val_loss, "validation_accuracy": val_accuracy})
        LOGGER.info("epoch=%d train_loss=%.4f val_accuracy=%.4f", epoch,
                    running_loss / seen, val_accuracy)

    test_loss, test_accuracy = evaluate(model, test_loader, device)
    root = Path(config["project"].get("output_root", "results"))
    if not root.is_absolute():
        root = config_path.resolve().parent.parent / root
    identifier = experiment_id or datetime.now(timezone.utc).strftime("centralized-%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    output = root / "raw" / "centralized" / identifier
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    metrics = {"experiment_id": identifier, "seed": seed, "device": str(device),
               "runtime_seconds": time.perf_counter() - started,
               "environment": {"python": platform.python_version(), "platform": platform.platform(),
                               "torch": torch.__version__, "torchvision": torchvision.__version__,
                               "numpy": np.__version__, "scikit_learn": sklearn.__version__},
               "train_examples": len(train_idx), "validation_examples": len(val_idx),
               "test_examples": len(test_data), "history": history,
               "test_loss": test_loss, "test_accuracy": test_accuracy}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"model_state_dict": model.state_dict(), "metrics": metrics}, output / "checkpoint.pt")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([row["epoch"] for row in history], [row["train_loss"] for row in history], label="Train loss")
    ax.plot([row["epoch"] for row in history], [row["validation_loss"] for row in history], label="Validation loss")
    ax.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Centralized MNIST baseline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "loss.png", dpi=160)
    plt.close(fig)
    LOGGER.info("Saved experiment to %s", output)
    return output

