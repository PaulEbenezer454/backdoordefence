"""Flower aggregator and TCLAD mitigation tests."""
import numpy as np
import pytest
from federated.strategy import aggregate_round, fedavg_strategy


def test_flower_fedavg_uses_example_and_mitigation_weights():
    initial = [np.array([0.0], dtype=np.float32)]
    strategy = fedavg_strategy(initial, 2)
    updates = [([np.array([2.0], dtype=np.float32)], 10, {"train_loss": 1.0, "client_accuracy": 0.5}),
               ([np.array([10.0], dtype=np.float32)], 10, {"train_loss": 1.0, "client_accuracy": 0.5})]
    aggregated, _ = aggregate_round(strategy, 1, updates, [1.0, 0.0])
    assert aggregated[0][0] == pytest.approx(2.0)
    with pytest.raises(ValueError):
        aggregate_round(strategy, 1, updates, [0.0, 0.0])
