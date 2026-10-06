"""Layer fingerprint tests with hand-checkable vectors."""
import numpy as np
import torch
from defense.fingerprints import compute_layer_statistics, fingerprint_rows


def test_known_statistics_and_finite_degenerate_moments():
    result = compute_layer_statistics(torch.tensor([0.0, 1.0, -1.0, 2.0]))
    assert result["mean"] == 0.5
    assert result["median"] == 0.5
    assert result["min"] == -1.0 and result["max"] == 2.0
    assert result["l1"] == 4.0
    assert result["sparsity"] == 0.25
    constant = compute_layer_statistics(np.ones(4))
    assert constant["skewness"] == 0.0 and constant["kurtosis"] == 0.0


def test_fingerprint_label_is_metadata():
    frame = fingerprint_rows({"conv1": np.array([1.0, -1.0])}, "e", 1, 2, 1)
    assert set(frame.columns) == {"experiment_id", "round", "client_id", "layer_name", "feature_name", "feature_value", "ground_truth"}
    assert frame.ground_truth.unique().tolist() == [1]
    assert "ground_truth" not in frame.feature_name.tolist()
