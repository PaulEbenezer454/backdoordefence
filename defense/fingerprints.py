"""Layer-wise parameter update fingerprints and isolated ground-truth labels."""
from __future__ import annotations
from collections.abc import Mapping
import numpy as np
import pandas as pd
import torch
from scipy.stats import kurtosis, skew

STATISTICS = ("mean", "std", "median", "min", "max", "l1", "l2", "max_abs",
              "sparsity", "skewness", "kurtosis")


def compute_layer_statistics(values: torch.Tensor | np.ndarray) -> dict[str, float]:
    """Calculate finite scalar statistics over a layer's flattened update."""
    array = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    array = np.asarray(array, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Layer update must be non-empty and finite")
    return {
        "mean": float(array.mean()), "std": float(array.std()), "median": float(np.median(array)),
        "min": float(array.min()), "max": float(array.max()),
        "l1": float(np.abs(array).sum()), "l2": float(np.linalg.norm(array)),
        "max_abs": float(np.abs(array).max()), "sparsity": float(np.mean(array == 0)),
        "skewness": float(skew(array, bias=False)) if array.size >= 3 and np.std(array) > 0 else 0.0,
        "kurtosis": float(kurtosis(array, fisher=True, bias=False)) if array.size >= 4 and np.std(array) > 0 else 0.0,
    }


def fingerprint_rows(updates: Mapping[str, torch.Tensor | np.ndarray], experiment_id: str,
                     round_number: int, client_id: int,
                     ground_truth: int | None = None,
                     reference_updates: Mapping[str, torch.Tensor | np.ndarray] | None = None) -> pd.DataFrame:
    """Return long-format feature rows; label is metadata, never a feature."""
    rows = []
    for layer_name, values in updates.items():
        stats = compute_layer_statistics(values)
        if reference_updates and layer_name in reference_updates:
            current = np.asarray(values.detach().cpu() if isinstance(values, torch.Tensor) else values).reshape(-1)
            reference = np.asarray(reference_updates[layer_name].detach().cpu() if isinstance(reference_updates[layer_name], torch.Tensor) else reference_updates[layer_name]).reshape(-1)
            stats["cosine_similarity_to_reference"] = float(np.dot(current, reference) / (np.linalg.norm(current) * np.linalg.norm(reference))) if np.linalg.norm(current) and np.linalg.norm(reference) else 0.0
            stats["relative_parameter_change"] = float(np.linalg.norm(current - reference) / (np.linalg.norm(reference) + 1e-8))
        for feature, value in stats.items():
            rows.append({"experiment_id": experiment_id, "round": int(round_number),
                         "client_id": int(client_id), "layer_name": layer_name,
                         "feature_name": feature, "feature_value": value,
                         "ground_truth": ground_truth})
    return pd.DataFrame(rows, columns=["experiment_id", "round", "client_id", "layer_name",
                                       "feature_name", "feature_value", "ground_truth"])
