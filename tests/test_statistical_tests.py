"""Paired seed statistics require independent-run level inputs."""
import pytest
from evaluation.statistical_tests import paired_summary


def test_paired_summary_reports_seed_level_interval():
    result = paired_summary([0.2, 0.3, 0.4], [0.3, 0.35, 0.5])
    assert result["n_paired_seeds"] == 3
    assert result["mean_difference"] == pytest.approx(0.083333333)
    assert result["ci_lower"] <= result["mean_difference"] <= result["ci_upper"]


def test_paired_summary_rejects_single_run():
    with pytest.raises(ValueError):
        paired_summary([0.1], [0.2])
