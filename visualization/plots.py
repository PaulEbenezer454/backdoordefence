"""Publication-oriented plots for TCLAD-FL result tables and diagnostics."""
from __future__ import annotations
import json
from pathlib import Path
from collections.abc import Mapping
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc


def plot_anomaly_heatmap(predictions: pd.DataFrame, output: Path) -> None:
    """Plot average per-layer score over observed client rounds."""
    rows = []
    for _, row in predictions.iterrows():
        scores = json.loads(row.get("layer_scores", "{}"))
        for layer, value in scores.items():
            rows.append({"round": int(row["round"]), "layer": layer, "score": float(value)})
    if not rows:
        return
    matrix = pd.DataFrame(rows).pivot_table(index="layer", columns="round", values="score", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(max(8, matrix.shape[1] * 0.35), 4.8))
    sns.heatmap(matrix, cmap="mako", ax=ax, cbar_kws={"label": "Mean layer anomaly score"})
    ax.set(title="Layer anomaly by round", xlabel="Federated round", ylabel="Layer")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_score_distribution(predictions: pd.DataFrame, output: Path) -> None:
    """Compare client-round overall score distributions by evaluation label."""
    if "ground_truth" not in predictions or predictions.ground_truth.nunique() < 2:
        return
    frame = predictions.copy()
    frame["label"] = frame.ground_truth.map({0: "Benign", 1: "Malicious"})
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    sns.violinplot(data=frame, x="label", y="overall_anomaly_score", inner="quart", cut=0, ax=ax)
    ax.set(title="Client-round anomaly scores", xlabel="Evaluation label", ylabel="Overall anomaly score")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_roc_pr(predictions: pd.DataFrame, output: Path) -> None:
    """Plot ROC and precision-recall curves from held-out prediction scores."""
    if predictions.ground_truth.nunique() < 2:
        return
    labels = predictions.ground_truth.astype(int).to_numpy()
    scores = predictions.overall_anomaly_score.astype(float).to_numpy()
    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axes[0].legend(loc="lower right")
    axes[1].plot(recall, precision)
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_temporal_trajectories(temporal: pd.DataFrame, labels: pd.DataFrame, output: Path) -> None:
    """Plot mean absolute fingerprint changes per client across rounds."""
    if temporal.empty:
        return
    values = (temporal.groupby(["experiment_id", "round", "client_id"], as_index=False)
              .absolute_difference.mean().rename(columns={"absolute_difference": "mean_absolute_change"}))
    label_keys = labels[["experiment_id", "round", "client_id", "ground_truth"]].drop_duplicates()
    values = values.merge(label_keys, on=["experiment_id", "round", "client_id"], how="left")
    values["status"] = values.ground_truth.map({0: "Benign", 1: "Malicious"}).fillna("Unknown")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=values, x="round", y="mean_absolute_change", hue="status", style="client_id",
                 markers=False, dashes=False, alpha=0.72, ax=ax)
    ax.set(title="Temporal update-change trajectories", xlabel="Federated round",
           ylabel="Mean absolute fingerprint change")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_cross_layer_matrix(cross_layer: pd.DataFrame, output: Path) -> None:
    """Plot a symmetric matrix of mean pairwise summary cosine values."""
    if cross_layer.empty:
        return
    matrix = cross_layer.pivot_table(index="layer_a", columns="layer_b", values="summary_cosine", aggfunc="mean")
    layers = sorted(set(matrix.index) | set(matrix.columns))
    matrix = matrix.reindex(index=layers, columns=layers)
    for left in layers:
        for right in layers:
            if pd.isna(matrix.loc[left, right]) and not pd.isna(matrix.loc[right, left]):
                matrix.loc[left, right] = matrix.loc[right, left]
    for layer in layers:
        matrix.loc[layer, layer] = 1.0
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, vmin=-1, vmax=1, center=0, cmap="vlag", annot=True, fmt=".2f", ax=ax)
    ax.set(title="Cross-layer summary cosine consistency", xlabel="Layer", ylabel="Layer")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_metric_curves(series: Mapping[str, tuple[np.ndarray, np.ndarray]], output: Path,
                       x_label: str, y_label: str, title: str) -> None:
    """Plot multiple named x/y series using consistent publication styling."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, (x_values, y_values) in series.items():
        ax.plot(x_values, y_values, label=name)
    ax.set(xlabel=x_label, ylabel=y_label, title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)
