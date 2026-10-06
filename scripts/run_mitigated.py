"""Run federated training with validation-calibrated TCLAD mitigation."""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.defense_experiment import fit_detector_bundle
from federated.server import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/mnist.yaml"))
    parser.add_argument("--reference", required=True, type=Path,
                        help="Separate benign run with saved fingerprints")
    parser.add_argument("--mitigation", choices=["reject", "down_weight"], required=True)
    parser.add_argument("--component", choices=["layer", "layer_temporal", "layer_cross",
                        "temporal_cross", "full"], default="full",
                        help="Detector components used for calibration and scoring")
    parser.add_argument("--experiment-id")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config.setdefault("attack", {})["enabled"] = True
    component_weights = {
        "layer": {"layer": 1.0, "temporal": 0.0, "cross_layer": 0.0},
        "layer_temporal": {"layer": 1.0, "temporal": 1.0, "cross_layer": 0.0},
        "layer_cross": {"layer": 1.0, "temporal": 0.0, "cross_layer": 1.0},
        "temporal_cross": {"layer": 0.0, "temporal": 1.0, "cross_layer": 1.0},
        "full": {"layer": 1.0, "temporal": 1.0, "cross_layer": 1.0},
    }
    bundle = fit_detector_bundle(args.reference, weights=component_weights[args.component])
    config.setdefault("defense", {})["component"] = args.component
    config["defense"]["weights"] = component_weights[args.component]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    temp = ROOT / "work" / f"mitigated-{args.mitigation}-{args.component}-resolved.yaml"
    temp.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    try:
        print(run(temp, args.experiment_id, bundle, args.mitigation))
    finally:
        temp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
