"""Run the controlled local LSA-inspired backdoor approximation."""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.lsa_experiment import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/mnist.yaml"))
    parser.add_argument("--experiment-id")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(run(args.config, args.experiment_id))


if __name__ == "__main__":
    main()
