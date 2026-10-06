"""Paired run-level confidence intervals and tests."""
from __future__ import annotations
import numpy as np
from scipy import stats


def paired_summary(control, treatment, confidence: float = 0.95) -> dict[str, float]:
    """Summarize paired seed-level deltas; do not pass client-round repeats."""
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    if control.shape != treatment.shape or control.ndim != 1 or len(control) < 2:
        raise ValueError("Need aligned one-dimensional run-level values from >=2 seeds")
    if not np.isfinite(control).all() or not np.isfinite(treatment).all():
        raise ValueError("Inputs must be finite")
    delta = treatment - control
    mean = float(delta.mean())
    sem = float(stats.sem(delta))
    interval = stats.t.interval(confidence, df=len(delta) - 1, loc=mean, scale=sem)
    test = stats.wilcoxon(delta, alternative="two-sided", zero_method="wilcox", method="auto")
    return {"n_paired_seeds": int(len(delta)), "mean_difference": mean,
            "std_difference": float(delta.std(ddof=1)), "ci_lower": float(interval[0]),
            "ci_upper": float(interval[1]), "wilcoxon_statistic": float(test.statistic),
            "wilcoxon_p_value": float(test.pvalue)}
