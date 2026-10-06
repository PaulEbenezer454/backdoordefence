"""Threshold sensitivity analysis using fixed benign calibration splits."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.defense_experiment import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "processed" / "threshold_sensitivity")
    parser.add_argument("--percentiles", type=float, nargs="+", default=[90, 95, 97.5, 99])
    args = parser.parse_args()
    runs = []
    for percentile in args.percentiles:
        output = args.output_root / f"p{str(percentile).replace('.', '_')}"
        result = run(args.reference, args.evaluation, output, threshold_percentile=percentile)
        frame = pd.read_csv(result / "comparison.csv")
        frame.insert(0, "threshold_percentile", percentile)
        runs.append(frame)
    args.output_root.mkdir(parents=True, exist_ok=True)
    pd.concat(runs, ignore_index=True).to_csv(args.output_root / "sensitivity.csv", index=False)
    print(args.output_root / "sensitivity.csv")


if __name__ == "__main__":
    main()
