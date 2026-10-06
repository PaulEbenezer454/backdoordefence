"""Measure offline detector feature/scoring overhead on saved run artifacts."""
from __future__ import annotations
import json
import time
import tracemalloc
from pathlib import Path
import pandas as pd
from defense.anomaly_detector import score_tclad
from defense.cross_layer import cross_layer_features
from defense.temporal import temporal_features
from experiments.defense_experiment import _fit_components


def measure_detector_overhead(reference_dir: Path, evaluation_dir: Path) -> dict:
    """Measure wall time and Python-tracked peak allocation for feature scoring.

    This is a local diagnostic, not a controlled hardware benchmark. Native
    allocations outside tracemalloc's tracking are not included in peak memory.
    """
    tracemalloc.start()
    started = time.perf_counter()
    (ref_fp, clean_ref, layer_ref, ref_temporal, temporal_ref, ref_cross,
     cross_ref, fit_rounds, validation_rounds, geometry_features,
     geometry_reference) = _fit_components(reference_dir)
    eval_fp = pd.read_csv(evaluation_dir / "fingerprints.csv").drop(
        columns=["ground_truth"], errors="ignore")
    eval_temporal = temporal_features(eval_fp, layer_ref)
    eval_cross = cross_layer_features(eval_fp)
    scored = score_tclad(eval_fp, layer_ref, eval_temporal, temporal_ref,
                         eval_cross, cross_ref,
                         {"layer": 1.0, "temporal": 1.0, "cross_layer": 1.0})
    elapsed = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "reference_experiment": json.loads((reference_dir / "metrics.json").read_text()).get("experiment_id", reference_dir.name),
        "evaluation_experiment": json.loads((evaluation_dir / "metrics.json").read_text()).get("experiment_id", evaluation_dir.name),
        "runtime_seconds": elapsed,
        "python_tracemalloc_peak_bytes": int(peak_bytes),
        "input_fingerprint_rows": int(len(eval_fp)),
        "client_round_scores": int(len(scored)),
        "temporal_rows": int(len(eval_temporal)),
        "cross_layer_rows": int(len(eval_cross)),
        "fit_rounds": sorted(int(value) for value in fit_rounds),
        "threshold_validation_rounds": sorted(int(value) for value in validation_rounds),
        "measurement_note": "Single local CPU measurement; tracemalloc peak excludes some native allocations and is not total process RSS.",
    }
