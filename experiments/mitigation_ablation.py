"""End-to-end mitigation ablation across TCLAD components."""
from __future__ import annotations

from pathlib import Path
import yaml

from experiments.defense_experiment import fit_detector_bundle
from federated.server import run as run_federated


COMPONENT_WEIGHTS = {
    "A_layer": {"layer": 1.0, "temporal": 0.0, "cross_layer": 0.0},
    "B_layer_temporal": {"layer": 1.0, "temporal": 1.0, "cross_layer": 0.0},
    "C_layer_cross_layer": {"layer": 1.0, "temporal": 0.0, "cross_layer": 1.0},
    "D_temporal_cross_layer": {"layer": 0.0, "temporal": 1.0, "cross_layer": 1.0},
    "E_full_tclad": {"layer": 1.0, "temporal": 1.0, "cross_layer": 1.0},
}


def run_mitigation_ablation(config_path: Path, reference_dir: Path,
                            mitigation: str, experiment_prefix: str) -> list[Path]:
    """Run the five predeclared component ablations under one mitigation rule."""
    if mitigation not in {"reject", "down_weight"}:
        raise ValueError("mitigation must be 'reject' or 'down_weight'")
    root = Path(__file__).resolve().parents[1]
    work_dir = root / "work"
    work_dir.mkdir(exist_ok=True)
    base = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base.setdefault("attack", {})["enabled"] = True
    outputs = []
    for component, weights in COMPONENT_WEIGHTS.items():
        config = yaml.safe_load(yaml.safe_dump(base))
        config.setdefault("defense", {})["component"] = component
        config["defense"]["weights"] = weights
        bundle = fit_detector_bundle(reference_dir, weights=weights)
        experiment_id = f"{experiment_prefix}-{mitigation}-{component}"
        temporary_config = work_dir / f"{experiment_id}.yaml"
        temporary_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        try:
            outputs.append(run_federated(temporary_config, experiment_id, bundle, mitigation))
        finally:
            temporary_config.unlink(missing_ok=True)
    return outputs
