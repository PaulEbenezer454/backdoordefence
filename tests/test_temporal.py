"""Benign reference and temporal feature tests."""
import pandas as pd
import pytest
from defense.reference import build_benign_reference
from defense.temporal import temporal_features


def test_reference_uses_only_benign_fit_rounds():
    frame = pd.DataFrame([
        {"round": 1, "layer_name": "a", "feature_name": "l2", "feature_value": 2.0, "ground_truth": 0},
        {"round": 2, "layer_name": "a", "feature_name": "l2", "feature_value": 4.0, "ground_truth": 0},
        {"round": 1, "layer_name": "a", "feature_name": "l2", "feature_value": 100.0, "ground_truth": 1},
    ])
    result = build_benign_reference(frame, {1, 2})
    assert result.center.iloc[0] == 3.0
    assert "ground_truth" not in result.columns


def test_temporal_features_keep_actual_missing_round_gap():
    frame = pd.DataFrame([
        {"experiment_id": "e", "client_id": 1, "layer_name": "a", "feature_name": "l2", "round": 1, "feature_value": 2.0},
        {"experiment_id": "e", "client_id": 1, "layer_name": "a", "feature_name": "l2", "round": 3, "feature_value": 6.0},
    ])
    result = temporal_features(frame)
    assert result.iloc[1].first_difference == 4.0
    assert result.iloc[1].elapsed_rounds == 2.0
    assert result.iloc[1].trend_slope == pytest.approx(2.0)
