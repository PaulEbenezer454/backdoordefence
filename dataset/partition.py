"""Deterministic IID and Dirichlet label-skew client partitions."""
from __future__ import annotations
import numpy as np


def partition_indices(labels: np.ndarray, num_clients: int, distribution: str = "iid",
                      alpha: float = 0.5, seed: int = 42) -> list[np.ndarray]:
    """Partition index positions so every example is assigned once.

    Dirichlet partitioning is label-wise; empty client outcomes are retried
    with deterministic child seeds up to 100 times, then rejected.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or not len(labels):
        raise ValueError("labels must be a non-empty 1-D array")
    if num_clients < 2:
        raise ValueError("num_clients must be at least 2")
    if distribution not in {"iid", "non_iid"}:
        raise ValueError("distribution must be 'iid' or 'non_iid'")
    if distribution == "non_iid" and alpha <= 0:
        raise ValueError("Dirichlet alpha must be positive")

    rng = np.random.default_rng(seed)
    for _ in range(100):
        buckets: list[list[int]] = [[] for _ in range(num_clients)]
        if distribution == "iid":
            shuffled = rng.permutation(len(labels))
            for cid, group in enumerate(np.array_split(shuffled, num_clients)):
                buckets[cid].extend(group.tolist())
        else:
            for label in np.unique(labels):
                label_indices = rng.permutation(np.flatnonzero(labels == label))
                proportions = rng.dirichlet(np.full(num_clients, alpha))
                counts = rng.multinomial(len(label_indices), proportions)
                start = 0
                for cid, count in enumerate(counts):
                    buckets[cid].extend(label_indices[start:start + count].tolist())
                    start += count
        if min(map(len, buckets)) > 0:
            return [rng.permutation(bucket).astype(np.int64) for bucket in buckets]
    raise RuntimeError("Could not produce non-empty client partitions after 100 attempts")
