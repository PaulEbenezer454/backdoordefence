"""Transparent client-update rejection and down-weighting."""
from __future__ import annotations
import numpy as np


def mitigation_decision(score: float, threshold: float, strategy: str,
                        minimum_weight: float = 0.05) -> tuple[bool, float]:
    """Return (rejected, aggregation weight) for one client update."""
    if strategy == "reject":
        suspicious = score >= threshold
        return suspicious, 0.0 if suspicious else 1.0
    if strategy == "down_weight":
        weight = float(np.clip(threshold / max(score, 1e-12), minimum_weight, 1.0))
        return False, weight
    if strategy == "none":
        return False, 1.0
    raise ValueError("strategy must be none, reject, or down_weight")
