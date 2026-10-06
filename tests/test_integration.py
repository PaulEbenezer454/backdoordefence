"""Tiny local Flower-training path using only generated synthetic data."""
from __future__ import annotations
import numpy as np
import torch
import yaml
from torch.utils.data import TensorDataset
import federated.server as server


class SyntheticImageDataset(TensorDataset):
    def __init__(self, seed: int = 3):
        generator = torch.Generator().manual_seed(seed)
        images = torch.rand((20, 1, 28, 28), generator=generator)
        labels = torch.arange(20) % 10
        super().__init__(images, labels)
        self.targets = labels.numpy()


def test_one_round_federated_run_saves_metrics_without_dataset_download(tmp_path, monkeypatch):
    data = SyntheticImageDataset()
    monkeypatch.setattr(server, "load_torchvision_dataset", lambda *args, **kwargs: data)
    output_root = tmp_path / "results"
    config = {
        "project": {"seed": 17, "output_root": str(output_root)},
        "dataset": {"name": "mnist", "data_dir": "unused", "download": False,
                    "subset_size": 20, "validation_fraction": 0.1,
                    "partition": {"type": "iid", "clients": 2, "dirichlet_alpha": 0.5}},
        "model": {"name": "mnist_cnn", "num_classes": 10},
        "training": {"device": "cpu", "rounds": 1, "local_epochs": 1,
                     "batch_size": 8, "learning_rate": 0.01,
                     "client_fraction": 1.0, "deterministic": True},
        "attack": {"enabled": False},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = server.run(config_path, "synthetic-integration")
    import json
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["experiment_id"] == "synthetic-integration"
    assert len(metrics["history"]) == 1
    assert np.isfinite(metrics["history"][0]["global_accuracy"])
    assert (output / "fingerprints.csv").is_file()
    assert (output / "checkpoint.pt").is_file()
