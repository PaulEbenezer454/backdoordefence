"""Transparent robust-z scoring and validation-only threshold selection."""
from __future__ import annotations
import json
from collections.abc import Sequence
import numpy as np
import pandas as pd


def robust_location_scale(values: Sequence[float]) -> tuple[float, float]:
    """Estimate median and a robust scale, with fallbacks for zero-MAD data."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("Cannot estimate a reference from empty/non-finite values")
    center = float(np.median(array))
    scale = float(1.4826 * np.median(np.abs(array - center)))
    if not np.isfinite(scale) or scale <= 1e-12:
        q25, q75 = np.percentile(array, [25, 75])
        scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = max(abs(center) * 1e-6, 1e-8)
    return center, scale


def bounded_robust_deviation(values, centers, scales):
    """Map absolute robust z deviations to [0, 1) via z/(1+z)."""
    z = np.abs((np.asarray(values, dtype=float) - np.asarray(centers, dtype=float)) /
              np.asarray(scales, dtype=float))
    return np.where(np.isfinite(z), z / (1.0 + z), 1.0)


def fit_robust_reference(frame: pd.DataFrame, dimensions: Sequence[str],
                         features: Sequence[str], benign_only: bool = False) -> pd.DataFrame:
    """Fit median/MAD references; if labels exist, only benign rows are retained."""
    data = frame
    if benign_only and "ground_truth" in frame:
        data = frame[frame.ground_truth == 0]
    if data.empty:
        raise ValueError("Reference fit set is empty")
    long = data.melt(id_vars=list(dimensions), value_vars=list(features),
                     var_name="score_feature", value_name="score_value")
    grouped = long.groupby([*dimensions, "score_feature"])["score_value"]
    records = []
    for key, values in grouped:
        center, scale = robust_location_scale(values.to_numpy())
        records.append({**dict(zip([*dimensions, "score_feature"], key, strict=True)),
                        "center": center, "scale": scale})
    return pd.DataFrame(records)


def _score_against_reference(frame: pd.DataFrame, reference: pd.DataFrame,
                             dimensions: Sequence[str], features: Sequence[str]) -> pd.Series:
    long = frame.melt(id_vars=list(dimensions), value_vars=list(features),
                      var_name="score_feature", value_name="score_value")
    joined = long.merge(reference, on=[*dimensions, "score_feature"], how="left")
    joined["normalized_deviation"] = bounded_robust_deviation(
        joined.score_value, joined.center, joined.scale)
    return joined.groupby(list(dimensions[:3]), dropna=False).normalized_deviation.mean()


def score_tclad(fingerprints: pd.DataFrame, layer_reference: pd.DataFrame,
                 temporal: pd.DataFrame, temporal_reference: pd.DataFrame,
                 cross_layer: pd.DataFrame, cross_reference: pd.DataFrame,
                 weights: dict[str, float], threshold: float | None = None,
                 suspicious_layer_z: float = 3.0) -> pd.DataFrame:
    """Return client-round component scores and interpretable suspect layers."""
    event = ["experiment_id", "round", "client_id"]
    fp = fingerprints.merge(layer_reference, on=["layer_name", "feature_name"], how="left")
    fp["raw_deviation"] = ((fp.feature_value - fp.center) / fp.scale).abs()
    fp["normalized_deviation"] = bounded_robust_deviation(
        fp.feature_value, fp.center, fp.scale)
    per_layer = fp.groupby([*event, "layer_name"]).agg(
        score=("normalized_deviation", "mean"), raw_score=("raw_deviation", "mean")).reset_index()
    layer_score = per_layer.groupby(event).score.mean().rename("layer_score").reset_index()
    suspicious = per_layer[per_layer.raw_score >= suspicious_layer_z].groupby(event).layer_name.apply(list)
    suspicious = suspicious.rename("suspicious_layers").reset_index()

    temporal_features_list = ["absolute_difference", "moving_std", "temporal_variance", "deviation_persistence",
                              "deviation_duration", "trend_slope", "ewma"]
    temporal_dims = [*event, "layer_name", "feature_name"]
    temporal_values = temporal.melt(id_vars=temporal_dims, value_vars=temporal_features_list,
                                    var_name="score_feature", value_name="score_value")
    temporal_joined = temporal_values.merge(temporal_reference, on=["layer_name", "feature_name", "score_feature"], how="left")
    temporal_joined["normalized_deviation"] = bounded_robust_deviation(
        temporal_joined.score_value, temporal_joined.center, temporal_joined.scale)
    temporal_score = temporal_joined.groupby(event).normalized_deviation.mean().rename("temporal_score").reset_index()

    cross_features = ["summary_cosine", "normalized_summary_distance", "relative_l2_magnitude"]
    cross_dims = [*event, "layer_a", "layer_b"]
    cross_values = cross_layer.melt(id_vars=cross_dims, value_vars=cross_features,
                                    var_name="score_feature", value_name="score_value")
    cross_joined = cross_values.merge(cross_reference, on=["layer_a", "layer_b", "score_feature"], how="left")
    cross_joined["normalized_deviation"] = bounded_robust_deviation(
        cross_joined.score_value, cross_joined.center, cross_joined.scale)
    cross_score = cross_joined.groupby(event).normalized_deviation.mean().rename("cross_layer_score").reset_index()

    result = layer_score.merge(temporal_score, on=event, how="outer").merge(cross_score, on=event, how="outer").merge(suspicious, on=event, how="left")
    result[["layer_score", "temporal_score", "cross_layer_score"]] = result[["layer_score", "temporal_score", "cross_layer_score"]].fillna(0.0)
    result["suspicious_layers"] = result.suspicious_layers.apply(lambda value: value if isinstance(value, list) else [])
    active_weights = {key: max(0.0, float(value)) for key, value in weights.items()}
    total = sum(active_weights.values())
    if total <= 0:
        raise ValueError("At least one anomaly component weight must be positive")
    for key in active_weights:
        active_weights[key] /= total
    result["overall_anomaly_score"] = (active_weights.get("layer", 0) * result.layer_score +
        active_weights.get("temporal", 0) * result.temporal_score +
        active_weights.get("cross_layer", 0) * result.cross_layer_score)
    result["layer_scores"] = result.apply(lambda row: json.dumps(dict(zip(per_layer[
        (per_layer.experiment_id == row.experiment_id) & (per_layer["round"] == row["round"]) &
        (per_layer.client_id == row.client_id)].layer_name,
        per_layer[(per_layer.experiment_id == row.experiment_id) & (per_layer["round"] == row["round"]) &
        (per_layer.client_id == row.client_id)].score))), axis=1)
    result["predicted_label"] = (result.overall_anomaly_score >= threshold).astype(int) if threshold is not None else 0
    return result


def select_threshold(validation_scores: Sequence[float], strategy: str,
                     fixed_value: float | None = None, percentile: float = 95.0,
                     target_false_positive_rate: float = 0.05) -> float:
    """Select from benign/validation scores only; caller owns the split."""
    scores = np.asarray(validation_scores, dtype=float)
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("Validation scores must be non-empty and finite")
    if strategy == "fixed":
        if fixed_value is None:
            raise ValueError("fixed strategy needs fixed_value")
        return float(fixed_value)
    if strategy == "percentile":
        return float(np.percentile(scores, percentile))
    if strategy == "validation_fpr":
        if not 0 < target_false_positive_rate < 1:
            raise ValueError("target FPR must be between zero and one")
        return float(np.quantile(scores, 1.0 - target_false_positive_rate))
    raise ValueError(f"Unknown threshold strategy: {strategy}")

