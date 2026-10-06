"""Run clean-accuracy and ASR ablations through federated mitigation."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mitigation_ablation import run_mitigation_ablation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True,
                        help="Separate benign run used for fit and threshold validation")
    parser.add_argument("--mitigation", choices=["reject", "down_weight"], required=True)
    parser.add_argument("--experiment-prefix", required=True)
    args = parser.parse_args()
    for output in run_mitigation_ablation(args.config, args.reference,
                                         args.mitigation, args.experiment_prefix):
        print(output)


if __name__ == "__main__":
    main()
