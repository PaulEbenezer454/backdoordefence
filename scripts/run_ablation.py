"""Run held-out detector ablations against one reference/evaluation pair."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.ablation import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    args = parser.parse_args()
    print(run(args.reference, args.evaluation, args.output, args.threshold_percentile))


if __name__ == "__main__":
    main()
