"""LSA-inspired, controlled layer-selective update smoothing approximation."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from attacks.base_attack import TriggerConfig


@dataclass(frozen=True)
class LSAConfig:
    malicious_fraction: float = 0.2
    target_class: int = 0
    poison_fraction: float = 0.3
    trigger: TriggerConfig = TriggerConfig()
    selected_layers: tuple[str, ...] = ("output",)
    smoothing_lambda: float = 1.0
    attack_rounds: tuple[int, ...] = ()

    def active(self, round_number: int) -> bool:
        return not self.attack_rounds or round_number in self.attack_rounds


def smooth_malicious_update(local_parameters: list[np.ndarray], global_parameters: list[np.ndarray],
                            benign_mean_delta: list[np.ndarray], parameter_names: list[str],
                            selected_layers: tuple[str, ...], smoothing_lambda: float) -> list[np.ndarray]:
    """Approximate LSA mask/smoothing: preserve selected-layer attacker deltas,
    replace other-layer deltas by the benign mean, and scale selected deltas.

    This is **not** an exact reproduction: BC layers are supplied/configured,
    not found by the paper's clean/malicious layer-substitution procedure.
    """
    if not (len(local_parameters) == len(global_parameters) == len(benign_mean_delta) == len(parameter_names)):
        raise ValueError("Parameter arrays and names must have equal lengths")
    result = []
    for local, global_value, benign_delta, name in zip(local_parameters, global_parameters,
                                                       benign_mean_delta, parameter_names, strict=True):
        layer = name.split(".", 1)[0]
        if layer in selected_layers:
            delta = smoothing_lambda * (local - global_value) + max(0.0, 1.0 - smoothing_lambda) * benign_delta
        else:
            delta = benign_delta
        result.append((global_value + delta).astype(local.dtype, copy=False))
    return result
