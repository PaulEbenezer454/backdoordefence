"""Held-out detector and ablation evaluation with a separate benign reference run."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from defense.anomaly_detector import (bounded_robust_deviation, fit_robust_reference,
                                      robust_location_scale, score_tclad, select_threshold)
from defense.cross_layer import cross_layer_features
from defense.reference import build_benign_reference
from defense.temporal import temporal_features
from evaluation.metrics import detection_metrics

TEMPORAL_COLUMNS = ["absolute_difference", "moving_std", "temporal_variance", "deviation_persistence",
                    "deviation_duration", "trend_slope", "ewma"]
CROSS_COLUMNS = ["summary_cosine", "normalized_summary_distance", "relative_l2_magnitude"]
EVENT = ["experiment_id", "round", "client_id"]
ABLATIONS = {
    "whole_update": None,
    "A_layer_only": {"layer": 1},
    "B_layer_temporal": {"layer": 1, "temporal": 1},
    "C_layer_cross_layer": {"layer": 1, "cross_layer": 1},
    "D_temporal_cross_layer": {"temporal": 1, "cross_layer": 1},
    "temporal_only": {"temporal": 1},
    "cross_layer_only": {"cross_layer": 1},
    "E_full_tclad": {"layer": 1, "temporal": 1, "cross_layer": 1},
}


def _fit_components(reference_dir: Path):
    fp = pd.read_csv(reference_dir / "fingerprints.csv")
    if fp.ground_truth.fillna(0).any():
        raise ValueError("Calibration input must be an attack-free benign run")
    rounds = sorted(fp["round"].unique())
    fit_end = max(1, int(len(rounds) * 0.5))
    validation_end = min(len(rounds), fit_end + max(1, int(len(rounds) * 0.25)))
    fit_rounds = set(rounds[:fit_end])
    validation_rounds = set(rounds[fit_end:validation_end])
    if not validation_rounds:
        raise ValueError("Benign calibration run needs more rounds")
    clean_fp = fp.drop(columns=["ground_truth"], errors="ignore")
    layer_ref = build_benign_reference(fp, fit_rounds)
    temporal = temporal_features(clean_fp, layer_ref)
    temporal_ref = fit_robust_reference(temporal[temporal["round"].isin(fit_rounds)],
        ["layer_name", "feature_name"], TEMPORAL_COLUMNS)
    cross = cross_layer_features(clean_fp)
    cross_ref = fit_robust_reference(cross[cross["round"].isin(fit_rounds)],
        ["layer_a", "layer_b"], CROSS_COLUMNS)

    geometry = pd.read_csv(reference_dir / "whole_update_features.csv")
    if "experiment_id" not in geometry:
        geometry.insert(0, "experiment_id", json.loads((reference_dir / "metrics.json").read_text()).get("experiment_id", reference_dir.name))
    geometry_features = [name for name in geometry.select_dtypes(include="number").columns
                         if name not in {"round", "client_id", "ground_truth"}]
    fit_geometry = geometry[geometry["round"].isin(fit_rounds)]
    geometry_reference = {}
    for feature in geometry_features:
        values = fit_geometry[feature].to_numpy(dtype=float)
        geometry_reference[feature] = robust_location_scale(values)
    return fp, clean_fp, layer_ref, temporal, temporal_ref, cross, cross_ref, fit_rounds, validation_rounds, geometry_features, geometry_reference


def _whole_update_scores(frame: pd.DataFrame, features: list[str], reference: dict) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        deviations = [bounded_robust_deviation(float(row[name]), reference[name][0], reference[name][1])
                      for name in features]
        rows.append({**{key: row[key] for key in EVENT}, "overall_anomaly_score": float(np.mean(deviations))})
    return pd.DataFrame(rows)


def run(reference_dir: Path, evaluation_dir: Path, output_dir: Path,
        threshold_percentile: float = 95.0) -> Path:
    """Score full detector, ablations, and whole-update baseline out of sample."""
    (ref_fp, clean_ref, layer_ref, ref_temporal, temporal_ref, ref_cross, cross_ref,
     fit_rounds, validation_rounds, geometry_features, geometry_ref) = _fit_components(reference_dir)
    ref_geometry = pd.read_csv(reference_dir / "whole_update_features.csv")
    if "experiment_id" not in ref_geometry:
        ref_geometry.insert(0, "experiment_id", json.loads((reference_dir / "metrics.json").read_text()).get("experiment_id", reference_dir.name))
    val_fp = clean_ref[clean_ref["round"].isin(validation_rounds)]
    val_temporal = ref_temporal[ref_temporal["round"].isin(validation_rounds)]
    val_cross = ref_cross[ref_cross["round"].isin(validation_rounds)]
    eval_fp = pd.read_csv(evaluation_dir / "fingerprints.csv")
    truth = eval_fp.groupby(EVENT).ground_truth.first().reset_index()
    eval_clean = eval_fp.drop(columns=["ground_truth"], errors="ignore")
    eval_temporal = temporal_features(eval_clean, layer_ref)
    eval_cross = cross_layer_features(eval_clean)
    eval_geometry = pd.read_csv(evaluation_dir / "whole_update_features.csv")
    if "experiment_id" not in eval_geometry:
        eval_geometry.insert(0, "experiment_id", json.loads((evaluation_dir / "metrics.json").read_text()).get("experiment_id", evaluation_dir.name))

    output_dir.mkdir(parents=True, exist_ok=False)
    summary = []
    for method, weights in ABLATIONS.items():
        if weights is None:
            val_scores = _whole_update_scores(ref_geometry[ref_geometry["round"].isin(validation_rounds)],
                                              geometry_features, geometry_ref)
            threshold = select_threshold(val_scores.overall_anomaly_score, "percentile", percentile=threshold_percentile)
            test_scores = _whole_update_scores(eval_geometry, geometry_features, geometry_ref)
            scores = test_scores.merge(truth, on=EVENT, how="left")
            scores["predicted_label"] = (scores.overall_anomaly_score >= threshold).astype(int)
        else:
            val_scores = score_tclad(val_fp, layer_ref, val_temporal, temporal_ref, val_cross,
                                     cross_ref, weights)
            threshold = select_threshold(val_scores.overall_anomaly_score, "percentile", percentile=threshold_percentile)
            test_scores = score_tclad(eval_clean, layer_ref, eval_temporal, temporal_ref,
                                      eval_cross, cross_ref, weights, threshold=threshold)
            scores = test_scores.merge(truth, on=EVENT, how="left")
        metric = detection_metrics(scores.ground_truth.astype(int), scores.predicted_label.astype(int),
                                  scores.overall_anomaly_score)
        metric.update({"method": method, "threshold": threshold,
                       "detection_threshold_source": f"benign validation rounds; {threshold_percentile}th percentile"})
        summary.append(metric)
        scores.to_csv(output_dir / f"predictions_{method}.csv", index=False)
    pd.DataFrame(summary).to_csv(output_dir / "comparison.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps({
        "reference_run": str(reference_dir), "evaluation_run": str(evaluation_dir),
        "fit_rounds": [int(value) for value in sorted(fit_rounds)],
        "threshold_validation_rounds": [int(value) for value in sorted(validation_rounds)],
        "threshold_percentile": threshold_percentile,
        "comparison": summary}, indent=2), encoding="utf-8")
    return output_dir
"""Prepare frozen TCLAD references and validation threshold from a benign run."""
from pathlib import Path
from experiments.defense_experiment import _fit_components
from defense.anomaly_detector import score_tclad, select_threshold


def fit_detector_bundle(reference_dir: Path, weights: dict[str, float] | None = None,
                        percentile: float = 95.0) -> dict:
    """Fit only on the designated attack-free run and benign validation rounds."""
    (fp, clean_fp, layer_ref, temporal, temporal_ref, cross, cross_ref,
     _, validation_rounds, _, _) = _fit_components(reference_dir)
    validation_scores = score_tclad(
        clean_fp[clean_fp["round"].isin(validation_rounds)], layer_ref,
        temporal[temporal["round"].isin(validation_rounds)], temporal_ref,
        cross[cross["round"].isin(validation_rounds)], cross_ref,
        weights or {"layer": 1, "temporal": 1, "cross_layer": 1})
    threshold = select_threshold(validation_scores.overall_anomaly_score,
                                 "percentile", percentile=percentile)
    return {"layer_reference": layer_ref, "temporal_reference": temporal_ref,
            "cross_reference": cross_ref, "threshold": threshold,
            "weights": weights or {"layer": 1, "temporal": 1, "cross_layer": 1},
            "calibration_rounds": sorted(int(value) for value in validation_rounds),
            "apply_from_round": max(validation_rounds) + 1,
            "threshold_percentile": percentile}
