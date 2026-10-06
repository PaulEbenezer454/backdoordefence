"""Reproducible factor sweeps for attack fraction, partition and random seed."""
from __future__ import annotations
from itertools import product
import json
from pathlib import Path
import re
import pandas as pd
import yaml
from federated.server import run as run_federated


def run_factor_sensitivity(config_path: Path, output_csv: Path,
                           malicious_fractions: list[float],
                           distributions: list[str],
                           dirichlet_alphas: list[float],
                           seeds: list[int]) -> pd.DataFrame:
    """Run matched local experiments and save per-run outcomes to a new CSV.

    The server persists each resolved config, raw trajectory, checkpoints and
    plots under a unique result ID. With small client counts, nearby configured
    fractions can round to the same integer attacker count; realized fractions
    are therefore included in the returned table.
    """
    if output_csv.exists():
        raise FileExistsError(f"Refusing to overwrite {output_csv}")
    base = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    project_root = Path(__file__).resolve().parents[1]
    work_dir = project_root / "work" / "factor-sensitivity"
    work_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    alpha_grid = dirichlet_alphas or [0.5]
    original_root = config_path.resolve().parent.parent
    base_data_dir = Path(base["dataset"]["data_dir"])
    if not base_data_dir.is_absolute():
        base["dataset"]["data_dir"] = str((original_root / base_data_dir).resolve())
    base_output_root = Path(base["project"].get("output_root", "results"))
    if not base_output_root.is_absolute():
        base["project"]["output_root"] = str((original_root / base_output_root).resolve())
    factor_settings = []
    for fraction, distribution, seed in product(malicious_fractions, distributions, seeds):
        # Dirichlet alpha has no effect for IID and must not create duplicate runs.
        alphas = alpha_grid if distribution == "non_iid" else [alpha_grid[0]]
        factor_settings.extend((fraction, distribution, alpha, seed) for alpha in alphas)
    for fraction, distribution, alpha, seed in factor_settings:
        config = yaml.safe_load(yaml.safe_dump(base))
        config.setdefault("attack", {})["enabled"] = True
        config["attack"]["malicious_fraction"] = float(fraction)
        config.setdefault("dataset", {}).setdefault("partition", {})["type"] = distribution
        config["dataset"]["partition"]["dirichlet_alpha"] = float(alpha)
        config.setdefault("project", {})["seed"] = int(seed)
        suffix = re.sub(r"[^a-zA-Z0-9-]+", "-", f"{distribution}-a{alpha}-m{fraction}-s{seed}").strip("-")
        experiment_id = f"sensitivity-{suffix}"
        temporary_config = work_dir / f"{experiment_id}.yaml"
        temporary_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        try:
            output_dir = run_federated(temporary_config, experiment_id)
        finally:
            temporary_config.unlink(missing_ok=True)
        metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
        history = metrics["history"]
        rows.append({"experiment_id": experiment_id, "seed": int(seed),
            "partition": distribution, "dirichlet_alpha": float(alpha),
            "configured_malicious_fraction": float(fraction),
            "realized_malicious_fraction": metrics["realized_malicious_fraction"],
            "final_clean_accuracy": history[-1]["global_accuracy"],
            "final_asr": history[-1].get("attack_success_rate"),
            "runtime_seconds": metrics["runtime_seconds"], "result_dir": str(output_dir)})
    frame = pd.DataFrame(rows)
    frame.to_csv(output_csv, index=False)
    return frame
