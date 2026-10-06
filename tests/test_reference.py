"""Benign reference fitting must respect its explicit rounds and labels."""
import pandas as pd
import pytest

from defense.reference import build_benign_reference


def test_reference_excludes_attack_labels_and_unselected_rounds():
    fingerprints = pd.DataFrame([
        {"round": 1, "layer_name": "conv", "feature_name": "l2", "feature_value": 1.0, "ground_truth": 0},
        {"round": 1, "layer_name": "conv", "feature_name": "l2", "feature_value": 3.0, "ground_truth": 0},
        {"round": 1, "layer_name": "conv", "feature_name": "l2", "feature_value": 1000.0, "ground_truth": 1},
        {"round": 2, "layer_name": "conv", "feature_name": "l2", "feature_value": 500.0, "ground_truth": 0},
    ])
    reference = build_benign_reference(fingerprints, {1})
    assert reference.center.iloc[0] == 2.0
    assert reference.columns.tolist() == ["layer_name", "feature_name", "center", "scale"]
    assert "ground_truth" not in reference.columns


def test_reference_rejects_empty_benign_fit():
    fingerprints = pd.DataFrame([
        {"round": 1, "layer_name": "conv", "feature_name": "l2", "feature_value": 1.0, "ground_truth": 1}
    ])
    with pytest.raises(ValueError, match="No benign rows"):
        build_benign_reference(fingerprints, {1})
