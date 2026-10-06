"""Architecture-aware relational statistics between layer summaries."""
from __future__ import annotations
from itertools import combinations
import numpy as np
import pandas as pd

RELATIONAL_FEATURES = ("mean", "std", "l1", "l2", "max_abs", "sparsity", "skewness", "kurtosis")


def cross_layer_features(fingerprints: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise relations between same-client/round layer summaries.

    Uses shared scalar-statistic coordinates because parameter tensors in
    different layers generally have different sizes and cannot be directly
    compared elementwise.
    """
    rows = []
    keys = ["experiment_id", "round", "client_id"]
    for key, group in fingerprints[fingerprints.feature_name.isin(RELATIONAL_FEATURES)].groupby(keys):
        pivot = group.pivot_table(index="layer_name", columns="feature_name", values="feature_value")
        available = [name for name in RELATIONAL_FEATURES if name in pivot.columns]
        for left, right in combinations(sorted(pivot.index), 2):
            a = pivot.loc[left, available].fillna(0).to_numpy(dtype=float)
            b = pivot.loc[right, available].fillna(0).to_numpy(dtype=float)
            cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) if np.linalg.norm(a) and np.linalg.norm(b) else 0.0
            left_l2 = float(pivot.loc[left].get("l2", 0.0))
            right_l2 = float(pivot.loc[right].get("l2", 0.0))
            rows.append({**dict(zip(keys, key, strict=True)), "layer_a": left, "layer_b": right,
                         "summary_cosine": cosine,
                         "normalized_summary_distance": float(np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b) + 1e-8)),
                         "relative_l2_magnitude": float(left_l2 / (right_l2 + 1e-8))})
    return pd.DataFrame(rows)
