"""Small synthetic component test for the combined detector score."""
import numpy as np
import pandas as pd
from defense.anomaly_detector import fit_robust_reference, robust_location_scale, score_tclad
from defense.cross_layer import cross_layer_features
from defense.reference import build_benign_reference
from defense.temporal import temporal_features
from defense.fingerprints import fingerprint_rows


def test_tclad_scores_have_components_without_label_feature_leak():
    rows = []
    for client in range(3):
        for rnd in range(1, 4):
            magnitude = 1.0 if client < 2 else 8.0
            fp = fingerprint_rows({"conv1": np.array([magnitude, -magnitude]),
                                  "output": np.array([magnitude * 2, -magnitude * 2])},
                                 "synthetic", rnd, client, ground_truth=0 if client < 2 else 1)
            rows.append(fp)
    fingerprints = pd.concat(rows, ignore_index=True)
    layer_ref = build_benign_reference(fingerprints, {1, 2})
    temporal = temporal_features(fingerprints, layer_ref)
    benign_ids = temporal.client_id < 2
    temporal_values = ["absolute_difference", "moving_std", "temporal_variance", "deviation_persistence",
                       "deviation_duration", "trend_slope", "ewma"]
    temporal_ref = fit_robust_reference(temporal[benign_ids], ["layer_name", "feature_name"], temporal_values)
    cross = cross_layer_features(fingerprints)
    cross_ref = fit_robust_reference(cross[cross.client_id < 2], ["layer_a", "layer_b"],
                                     ["summary_cosine", "normalized_summary_distance", "relative_l2_magnitude"])
    score = score_tclad(fingerprints[fingerprints["round"] == 3], layer_ref,
                        temporal[temporal["round"] == 3], temporal_ref,
                        cross[cross["round"] == 3], cross_ref,
                        {"layer": 1, "temporal": 1, "cross_layer": 1}, threshold=1.0)
    assert len(score) == 3
    assert {"layer_score", "temporal_score", "cross_layer_score", "overall_anomaly_score",
            "predicted_label", "layer_scores", "suspicious_layers"}.issubset(score.columns)
    assert "ground_truth" not in score.columns
    assert np.isfinite(score.overall_anomaly_score).all()
    assert score.overall_anomaly_score.between(0.0, 1.0, inclusive="left").all()
    assert score[["layer_score", "temporal_score", "cross_layer_score"]].max().max() < 1.0


def test_robust_scale_falls_back_when_mad_is_zero():
    center, scale = robust_location_scale([0.0, 0.0, 0.0, 0.0, 4.0])
    assert center == 0.0
    assert scale > 0.0
