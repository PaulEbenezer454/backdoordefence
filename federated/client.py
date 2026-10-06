"""In-process simulated Flower client using local MNIST subsets."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from models.cnn import create_model
from attacks.base_attack import TriggerConfig, poison_batch


class FederatedClient:
    """A deterministic local learner; raw examples remain in its local subset."""
    def __init__(self, client_id: int, dataset: Dataset, indices: np.ndarray,
                 device: torch.device, batch_size: int, learning_rate: float,
                 local_epochs: int, seed: int, model_name: str = "mnist_cnn",
                 num_classes: int = 10) -> None:
        self.client_id = client_id
        self.dataset = dataset
        self.indices = indices
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.seed = seed
        self.model_name = model_name
        self.num_classes = num_classes

    def fit(self, parameters: list[np.ndarray], attack: dict | None = None) -> tuple[list[np.ndarray], int, dict[str, float]]:
        """Train from received global parameters and return updated weights."""
        model = create_model(self.model_name, self.num_classes).to(self.device)
        named = list(model.named_parameters())
        with torch.no_grad():
            for (name, parameter), value in zip(named, parameters, strict=True):
                parameter.copy_(torch.from_numpy(value).to(self.device, dtype=parameter.dtype))
        generator = torch.Generator().manual_seed(self.seed + self.client_id)
        loader = DataLoader(torch.utils.data.Subset(self.dataset, self.indices.tolist()),
                            batch_size=self.batch_size, shuffle=True, generator=generator)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        poison_generator = torch.Generator().manual_seed(self.seed + 100_003 * self.client_id)
        losses: list[float] = []
        model.train()
        for _ in range(self.local_epochs):
            total_loss = 0.0
            total = 0
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                if attack is not None:
                    trigger = TriggerConfig(size=int(attack.get("trigger_size", 3)),
                                            value=float(attack.get("trigger_value", 1.0)))
                    images, labels = poison_batch(images, labels, int(attack["target_class"]),
                        float(attack.get("poison_fraction", 0.3)), trigger, poison_generator)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * len(labels)
                total += len(labels)
            losses.append(total_loss / total)
        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                correct += int((model(images).argmax(dim=1) == labels).sum().item())
                total += len(labels)
        updated = [parameter.detach().cpu().numpy().copy() for _, parameter in model.named_parameters()]
        return updated, len(self.indices), {"train_loss": float(np.mean(losses)),
                                             "client_accuracy": correct / total}
