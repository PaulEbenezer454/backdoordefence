"""Tests for detection, thresholding, ASR, and mitigation helpers."""
import pytest
from defense.mitigation import mitigation_decision
from defense.anomaly_detector import select_threshold
from evaluation.metrics import attack_success_rate, detection_metrics


def test_validation_threshold_and_detection_metrics():
    assert select_threshold([0.1, 0.2, 0.3, 0.4], "percentile", percentile=75) == pytest.approx(0.325)
    scores = [0.1, 0.9, 0.2, 0.8]
    result = detection_metrics([0, 1, 0, 1], [0, 1, 1, 1], scores)
    assert result["detection_rate"] == 1.0
    assert result["false_positive_rate"] == 0.5
    assert result["roc_auc"] == 1.0


def test_attack_success_rate_excludes_target_class_samples():
    assert attack_success_rate([0, 0, 1, 1], [1, 2, 0, 3], target_class=0) == pytest.approx(2 / 3)


def test_mitigation_decisions():
    assert mitigation_decision(2.0, 1.0, "reject") == (True, 0.0)
    rejected, weight = mitigation_decision(2.0, 1.0, "down_weight")
    assert not rejected and weight == pytest.approx(0.5)
    with pytest.raises(ValueError):
        mitigation_decision(1.0, 2.0, "other")


def test_fixed_and_validation_fpr_thresholds():
    assert select_threshold([0.1, 0.2, 0.3], "fixed", fixed_value=0.25) == 0.25
    assert select_threshold([0.1, 0.2, 0.3, 0.4], "validation_fpr",
                            target_false_positive_rate=0.25) == pytest.approx(0.325)
