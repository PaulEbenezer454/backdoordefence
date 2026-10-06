"""Benign-only per-layer reference construction."""
from __future__ import annotations
import pandas as pd
from defense.anomaly_detector import robust_location_scale


def build_benign_reference(fingerprints: pd.DataFrame, fit_rounds: set[int],
                           method: str = "median") -> pd.DataFrame:
    """Fit feature medians and robust scales from explicitly allowed benign rows.

    Ground truth is used only for offline reference fitting. The resulting
    returned table contains no labels and is the only object passed downstream.
    """
    required = {"round", "layer_name", "feature_name", "feature_value", "ground_truth"}
    if not required.issubset(fingerprints.columns):
        raise ValueError(f"Missing required columns: {required - set(fingerprints.columns)}")
    if method not in {"median", "mean"}:
        raise ValueError("method must be median or mean")
    fit = fingerprints[fingerprints["round"].isin(fit_rounds) & (fingerprints["ground_truth"] == 0)]
    if fit.empty:
        raise ValueError("No benign rows available for reference fit")
    grouped = fit.groupby(["layer_name", "feature_name"])["feature_value"]
    rows = []
    for (layer, feature), values in grouped:
        center, scale = robust_location_scale(values.to_numpy())
        if method == "mean":
            center = float(values.mean())
        rows.append({"layer_name": layer, "feature_name": feature,
                     "center": center, "scale": scale})
    return pd.DataFrame(rows, columns=["layer_name", "feature_name", "center", "scale"])
