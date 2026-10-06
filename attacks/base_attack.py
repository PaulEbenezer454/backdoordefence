"""Shared local-only trigger helpers for controlled backdoor experiments."""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class TriggerConfig:
    size: int = 3
    value: float = 1.0


def apply_trigger(images: torch.Tensor, trigger: TriggerConfig) -> torch.Tensor:
    """Return a copy with a bottom-right square trigger in tensor space."""
    if images.ndim != 4 or trigger.size <= 0 or trigger.size > min(images.shape[-2:]):
        raise ValueError("Expected image batch and a valid positive trigger size")
    result = images.clone()
    result[:, :, -trigger.size:, -trigger.size:] = trigger.value
    return result


def poison_batch(images: torch.Tensor, labels: torch.Tensor, target_class: int,
                 fraction: float, trigger: TriggerConfig,
                 generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Poison a random subset of examples locally; never accesses external systems."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("poison fraction must be in [0, 1]")
    mask = torch.rand(len(labels), generator=generator, device="cpu") < fraction
    mask = mask.to(labels.device)
    poisoned = images.clone()
    if mask.any():
        poisoned[mask] = apply_trigger(images[mask], trigger)
    targets = labels.clone()
    targets[mask] = int(target_class)
    return poisoned, targets
