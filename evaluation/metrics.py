"""Binary detection and backdoor evaluation metrics."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def detection_metrics(labels, predictions, scores=None) -> dict[str, float]:
    """Calculate detection metrics with explicit zero-division handling."""
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    result = {"detection_rate": float(tp / (tp + fn)) if tp + fn else 0.0,
              "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
              "precision": float(precision_score(labels, predictions, zero_division=0)),
              "recall": float(recall_score(labels, predictions, zero_division=0)),
              "f1": float(f1_score(labels, predictions, zero_division=0)),
              "benign_acceptance_rate": float(tn / (tn + fp)) if tn + fp else 0.0,
              "malicious_rejection_rate": float(tp / (tp + fn)) if tp + fn else 0.0}
    if scores is not None and len(np.unique(labels)) > 1:
        result["roc_auc"] = float(roc_auc_score(labels, scores))
        result["pr_auc"] = float(average_precision_score(labels, scores))
    return result


def attack_success_rate(predictions, true_labels, target_class: int) -> float:
    """ASR over triggered examples whose clean ground-truth differs from target."""
    predictions = np.asarray(predictions)
    labels = np.asarray(true_labels)
    eligible = labels != target_class
    if not np.any(eligible):
        raise ValueError("No eligible non-target samples for ASR")
    return float(np.mean(predictions[eligible] == target_class))
