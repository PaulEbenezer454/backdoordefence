"""Run configured malicious-fraction, data-partition and seed sweeps."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.sensitivity import run_factor_sensitivity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "mnist_lsa_late.yaml")
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.2, 0.4])
    parser.add_argument("--distributions", choices=["iid", "non_iid"], nargs="+", default=["iid", "non_iid"])
    parser.add_argument("--dirichlet-alphas", type=float, nargs="+", default=[0.1, 0.5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 21])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = run_factor_sensitivity(args.config, args.output, args.fractions,
        args.distributions, args.dirichlet_alphas, args.seeds)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
