"""Controlled local execution of the documented LSA-inspired approximation."""
from __future__ import annotations
from pathlib import Path
import yaml
from federated.server import run as run_federated


def run(config_path: Path, experiment_id: str | None = None) -> Path:
    """Run a configured attack, forcing its explicitly simulated clients active."""
    project_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config.setdefault("attack", {})["enabled"] = True
    work_dir = project_root / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    resolved = work_dir / f"attack-{experiment_id or 'temporary'}.yaml"
    resolved.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    try:
        return run_federated(resolved, experiment_id)
    finally:
        resolved.unlink(missing_ok=True)
