"""Partition tests for IID and non-IID client assignment."""
import numpy as np
from dataset.partition import partition_indices


def test_iid_partition_assigns_each_index_once():
    parts = partition_indices(np.arange(103) % 10, 5, "iid", seed=7)
    combined = np.concatenate(parts)
    assert sorted(combined.tolist()) == list(range(103))
    assert max(map(len, parts)) - min(map(len, parts)) <= 1


def test_dirichlet_partition_is_reproducible_and_complete():
    labels = np.arange(600) % 10
    first = partition_indices(labels, 5, "non_iid", alpha=0.1, seed=19)
    second = partition_indices(labels, 5, "non_iid", alpha=0.1, seed=19)
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
    assert sorted(np.concatenate(first).tolist()) == list(range(len(labels)))
    assert all(len(part) > 0 for part in first)
