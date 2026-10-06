"""Per-client per-layer temporal features over observed communication rounds."""
from __future__ import annotations
import numpy as np
import pandas as pd


def temporal_features(fingerprints: pd.DataFrame, reference: pd.DataFrame | None = None,
                      deviation_threshold: float = 3.0) -> pd.DataFrame:
    """Compute first differences, rolling moments, trend, EWMA and persistence.

    Missing participation rounds remain gaps. If a fitted benign reference is
    supplied, deviation persistence/duration use its robust median/MAD scores;
    without one those fields are neutral (zero) rather than data-derived labels.
    """
    keys = ["experiment_id", "client_id", "layer_name", "feature_name"]
    frame = fingerprints.copy()
    if reference is not None:
        frame = frame.merge(reference, on=["layer_name", "feature_name"], how="left")
    rows = []
    for key, group in frame.groupby(keys, sort=False):
        group = group.sort_values("round")
        values = group["feature_value"].to_numpy(dtype=float)
        rounds = group["round"].to_numpy(dtype=float)
        diffs = np.diff(values, prepend=values[0])
        if reference is not None:
            center = group["center"].fillna(0).to_numpy(dtype=float)
            scale = group["scale"].fillna(1).clip(lower=1e-8).to_numpy(dtype=float)
            deviated = np.abs((values - center) / scale) >= deviation_threshold
        else:
            deviated = np.zeros(len(values), dtype=bool)
        duration = 0
        for index, (_, row) in enumerate(group.iterrows()):
            start = max(0, index - 2)
            window = values[start:index + 1]
            past_flags = deviated[:index + 1]
            duration = duration + 1 if deviated[index] else 0
            rows.append({**dict(zip(keys, key, strict=True)), "round": int(rounds[index]),
                "first_difference": float(diffs[index]), "absolute_difference": float(abs(diffs[index])),
                "elapsed_rounds": float(rounds[index] - rounds[index - 1]) if index else 0.0,
                "moving_mean": float(window.mean()), "moving_std": float(window.std()),
                "temporal_variance": float(values[:index + 1].var()),
                "deviation_persistence": float(past_flags.mean()), "deviation_duration": duration,
                "trend_slope": float(np.polyfit(rounds[:index + 1], values[:index + 1], 1)[0]) if index >= 1 else 0.0,
                "ewma": float(pd.Series(values[:index + 1]).ewm(alpha=0.5, adjust=False).mean().iloc[-1])})
    return pd.DataFrame(rows)

