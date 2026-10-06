"""Robust aggregation baseline test using five synthetic client updates."""
import numpy as np
import pytest
from federated.strategy import aggregate_round, fedavg_strategy


def _updates(values):
    return [([np.array([float(value)], dtype=np.float32)], 1, {"train_loss": 1.0, "client_accuracy": 1.0})
            for value in values]


def test_coordinate_median_and_trimmed_mean_resist_one_extreme_value():
    strategy = fedavg_strategy([np.array([0.0], dtype=np.float32)], 5)
    median, _ = aggregate_round(strategy, 1, _updates([0, 1, 2, 3, 100]), method="coordinate_median")
    trimmed, _ = aggregate_round(strategy, 1, _updates([0, 1, 2, 3, 100]), method="trimmed_mean", trim_count=1)
    assert median[0][0] == pytest.approx(2.0)
    assert trimmed[0][0] == pytest.approx(2.0)


def test_multi_krum_selects_neighbor_consistent_updates():
    strategy = fedavg_strategy([np.array([0.0], dtype=np.float32)], 5)
    result, _ = aggregate_round(strategy, 1, _updates([0, 0.1, -0.1, 0.2, 100]),
                                method="multi_krum", byzantine_count=1)
    assert abs(float(result[0][0])) < 1.0
