"""Isolation tests for controlled LSA-inspired update processing."""
import numpy as np
import pytest
from attacks.base_attack import TriggerConfig, apply_trigger
from attacks.lsa import smooth_malicious_update
import torch


def test_trigger_changes_only_configured_corner():
    images = torch.zeros(2, 1, 8, 8)
    changed = apply_trigger(images, TriggerConfig(size=2, value=1.0))
    assert torch.equal(changed[:, :, -2:, -2:], torch.ones(2, 1, 2, 2))
    assert changed[:, :, :-2, :-2].sum() == 0
    assert images.sum() == 0


def test_layer_smoothing_keeps_selected_and_uses_benign_elsewhere():
    global_values = [np.zeros((2,), dtype=np.float32), np.zeros((2,), dtype=np.float32)]
    local = [np.ones((2,), dtype=np.float32) * 3, np.ones((2,), dtype=np.float32) * 9]
    benign = [np.ones((2,), dtype=np.float32), np.ones((2,), dtype=np.float32) * 2]
    result = smooth_malicious_update(local, global_values, benign,
        ["conv1.weight", "output.weight"], ("output",), 1.0)
    assert np.allclose(result[0], 1.0)
    assert np.allclose(result[1], 9.0)
    scaled = smooth_malicious_update(local, global_values, benign,
        ["conv1.weight", "output.weight"], ("output",), 0.5)
    assert np.allclose(scaled[1], 5.5)
    with pytest.raises(ValueError):
        smooth_malicious_update(local, global_values[:1], benign,
            ["conv1.weight", "output.weight"], ("output",), 1.0)
