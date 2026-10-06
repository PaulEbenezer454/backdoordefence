"""Generate result tables and publication-style figures from raw artifacts."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from defense.cross_layer import cross_layer_features
from defense.reference import build_benign_reference
from defense.temporal import temporal_features
from visualization.plots import (plot_anomaly_heatmap, plot_cross_layer_matrix,
                                 plot_roc_pr, plot_score_distribution,
                                 plot_temporal_trajectories)


def _write_markdown(frame: pd.DataFrame, path: Path) -> None:
    """Write a simple Markdown table without an optional tabulate dependency."""
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_results(root: Path = Path("results")) -> None:
    """Derive baseline, detector comparison, and learning-curve artifacts."""
    table_dir, figure_dir = root / "tables", root / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for metrics_path in (root / "raw").rglob("metrics.json"):
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        config_path = metrics_path.parent / "config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        attack_config = config.get("attack") or {}
        aggregation_config = config.get("aggregation") or {}
        partition_config = data.get("partition", config.get("dataset", {}).get("partition", {})) or {}
        partition_name = partition_config.get("type") if isinstance(partition_config, dict) else partition_config
        configured_fraction = data.get("configured_malicious_fraction",
                                       float(attack_config.get("malicious_fraction", 0.0)))
        malicious_count = data.get("malicious_client_count")
        if malicious_count is None:
            malicious_count = (max(1, int(round(int(data.get("num_clients", 0)) * configured_fraction)))
                               if attack_config.get("enabled", False) else 0)
        realized_fraction = data.get("realized_malicious_fraction")
        if realized_fraction is None:
            realized_fraction = malicious_count / int(data["num_clients"]) if data.get("num_clients") else 0.0
        history = data.get("history", [])
        last = history[-1] if history else {}
        summary_rows.append({"experiment_id": data.get("experiment_id", metrics_path.parent.name),
            "result_type": metrics_path.parent.parent.name,
            "seed": data.get("seed"), "device": data.get("device"),
            "final_clean_accuracy": last.get("global_accuracy", data.get("test_accuracy")),
            "final_clean_loss": last.get("global_loss", data.get("test_loss")),
            "final_asr": last.get("attack_success_rate"),
            "aggregation": data.get("aggregation", aggregation_config.get("method", "fedavg")),
            "partition": partition_name,
            "attack": attack_config.get("name", "none") if attack_config.get("enabled", False) else "none",
            "mitigation": data.get("mitigation", "none"),
            "detector_component": (config.get("defense") or {}).get("component"),
            "malicious_client_count": malicious_count,
            "configured_malicious_fraction": configured_fraction,
            "realized_malicious_fraction": realized_fraction,
            "runtime_seconds": data.get("runtime_seconds")})
        if history and any("global_accuracy" in row for row in history):
            rounds = [row["round"] for row in history]
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].plot(rounds, [row["global_accuracy"] for row in history], marker="o", ms=3)
            axes[0].set(xlabel="Round", ylabel="Clean test accuracy", title="Federated clean accuracy")
            axes[1].plot(rounds, [row["global_loss"] for row in history], marker="o", ms=3)
            axes[1].set(xlabel="Round", ylabel="Test loss", title="Federated test loss")
            if any("attack_success_rate" in row for row in history):
                ax2 = axes[0].twinx()
                ax2.plot(rounds, [row.get("attack_success_rate") for row in history], color="tab:red", alpha=0.7)
                ax2.set_ylabel("ASR")
            fig.tight_layout()
            fig.savefig(figure_dir / f"{metrics_path.parent.name}_learning.png", dpi=180)
            plt.close(fig)
    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(table_dir / "baseline_and_attack_runs.csv", index=False)
        _write_markdown(summary, table_dir / "baseline_and_attack_runs.md")
        attacks = summary[summary["final_asr"].notna()].copy()
        if not attacks.empty:
            attacks.to_csv(table_dir / "attack_comparison.csv", index=False)
            _write_markdown(attacks, table_dir / "attack_comparison.md")
            robust = attacks[(attacks.aggregation.isin(["fedavg", "coordinate_median", "trimmed_mean", "multi_krum"])) &
                             attacks.attack.eq("lsa_inspired") & attacks.mitigation.eq("none")]
            robust = robust[robust.experiment_id.str.contains("late20|late-mnist", case=False, regex=True)]
            if not robust.empty:
                robust.to_csv(table_dir / "robust_aggregation_comparison.csv", index=False)
                _write_markdown(robust, table_dir / "robust_aggregation_comparison.md")
            factor_runs = attacks[(attacks.attack == "lsa_inspired") &
                                  (attacks.aggregation == "fedavg") &
                                  (attacks.mitigation == "none")].copy()
            if not factor_runs.empty:
                factor_runs.to_csv(table_dir / "attack_factor_sensitivity.csv", index=False)
                _write_markdown(factor_runs, table_dir / "attack_factor_sensitivity.md")
                fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
                sns.scatterplot(data=factor_runs, x="realized_malicious_fraction", y="final_asr",
                                hue="partition", style="seed", s=90, ax=axes[0])
                axes[0].set(xlabel="Realized malicious client fraction", ylabel="Final ASR",
                            title="Attacker fraction and partition")
                sns.scatterplot(data=factor_runs, x="realized_malicious_fraction", y="final_clean_accuracy",
                                hue="partition", style="seed", s=90, ax=axes[1], legend=False)
                axes[1].set(xlabel="Realized malicious client fraction", ylabel="Final clean accuracy",
                            title="Clean accuracy trade-off")
                fig.tight_layout()
                fig.savefig(figure_dir / "attack_factor_sensitivity.png", dpi=200)
                plt.close(fig)
            mitigation_ablation = attacks[(attacks.attack == "lsa_inspired") &
                (attacks.mitigation == "down_weight") &
                attacks.detector_component.isin(["A_layer", "B_layer_temporal", "C_layer_cross_layer",
                                                  "D_temporal_cross_layer", "E_full_tclad"])]
            if not mitigation_ablation.empty:
                columns = ["experiment_id", "seed", "detector_component", "final_clean_accuracy",
                           "final_asr", "realized_malicious_fraction", "runtime_seconds"]
                mitigation_ablation[columns].to_csv(table_dir / "mitigation_ablation.csv", index=False)
                _write_markdown(mitigation_ablation[columns], table_dir / "mitigation_ablation.md")
            plotted = attacks.set_index("experiment_id")[["final_clean_accuracy", "final_asr"]]
            ax = plotted.plot(kind="bar", figsize=(10, 5), rot=25)
            ax.set_ylabel("Rate")
            ax.set_title("Clean accuracy and attack success rate")
            plt.tight_layout()
            plt.savefig(figure_dir / "attack_comparison.png", dpi=180)
            plt.close()
    comparisons = list((root / "processed").rglob("comparison.csv"))
    if comparisons:
        combined = pd.concat([pd.read_csv(path).assign(experiment=path.parent.name) for path in comparisons], ignore_index=True)
        combined.to_csv(table_dir / "detector_comparison.csv", index=False)
        _write_markdown(combined, table_dir / "detector_comparison.md")
        for metric, ascending in [("f1", False), ("roc_auc", False)]:
            if metric in combined:
                plot = combined.pivot_table(index="method", columns="experiment", values=metric, aggfunc="mean")
                ax = plot.plot(kind="bar", figsize=(10, 5))
                ax.set_ylabel(metric.upper())
                ax.set_title(f"Detector comparison: {metric.upper()}")
                plt.tight_layout()
                plt.savefig(figure_dir / f"detector_{metric}.png", dpi=180)
                plt.close()
    seed_rows = []
    for comparison_path in (root / "processed").glob("tclad-lsa-late-seed*-bounded-*/comparison.csv"):
        run_metrics_path = comparison_path.parent / "metrics.json"
        if not run_metrics_path.exists():
            continue
        run_metadata = json.loads(run_metrics_path.read_text(encoding="utf-8"))
        evaluated_metrics_path = Path(run_metadata["evaluation_run"]) / "metrics.json"
        if not evaluated_metrics_path.exists():
            continue
        seed = json.loads(evaluated_metrics_path.read_text(encoding="utf-8")).get("seed")
        per_run = pd.read_csv(comparison_path)
        per_run = per_run[per_run.method.isin(["whole_update", "A_layer_only", "E_full_tclad"])]
        for row in per_run.to_dict(orient="records"):
            seed_rows.append({"seed": seed, "experiment": comparison_path.parent.name, **row})
    if seed_rows:
        per_seed = pd.DataFrame(seed_rows)
        per_seed.to_csv(table_dir / "detector_seed_metrics.csv", index=False)
        _write_markdown(per_seed, table_dir / "detector_seed_metrics.md")
        summary_rows = []
        for method, group in per_seed.groupby("method"):
            for metric in ["detection_rate", "false_positive_rate", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
                if metric not in group:
                    continue
                values = group[metric].dropna().to_numpy(dtype=float)
                if values.size == 0:
                    continue
                mean = float(values.mean())
                sd = float(values.std(ddof=1)) if values.size > 1 else float("nan")
                half = (float(stats.t.ppf(0.975, values.size - 1) * sd / np.sqrt(values.size))
                        if values.size > 1 else float("nan"))
                lower, upper = mean - half, mean + half
                if metric in {"detection_rate", "false_positive_rate", "precision", "recall", "f1", "roc_auc", "pr_auc"}:
                    lower, upper = max(0.0, lower), min(1.0, upper)
                summary_rows.append({"method": method, "metric": metric, "n_independent_seeds": int(values.size),
                                     "mean": mean, "sd": sd,
                                     "descriptive_t_ci95_lower": lower if np.isfinite(half) else float("nan"),
                                     "descriptive_t_ci95_upper": upper if np.isfinite(half) else float("nan")})
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(table_dir / "detector_seed_summary.csv", index=False)
        _write_markdown(summary, table_dir / "detector_seed_summary.md")
    sensitivity_files = list((root / "processed").rglob("sensitivity.csv"))
    if sensitivity_files:
        sensitivity = pd.concat([
            pd.read_csv(path).assign(experiment=path.parent.name)
            for path in sensitivity_files
        ], ignore_index=True)
        sensitivity.to_csv(table_dir / "threshold_sensitivity.csv", index=False)
        _write_markdown(sensitivity, table_dir / "threshold_sensitivity.md")
        for metric in ["false_positive_rate", "detection_rate"]:
            if metric in sensitivity and "threshold_percentile" in sensitivity:
                plot = sensitivity.pivot_table(index="threshold_percentile", columns=["method", "experiment"],
                                               values=metric, aggfunc="mean")
                ax = plot.plot(marker="o", figsize=(11, 6))
                ax.set_ylabel(metric.replace("_", " ").title())
                ax.set_xlabel("Benign calibration percentile")
                ax.set_title(f"Threshold sensitivity: {metric.replace('_', ' ').title()}")
                plt.tight_layout()
                plt.savefig(figure_dir / f"threshold_sensitivity_{metric}.png", dpi=180)
                plt.close()
    overhead_files = list((root / "processed").glob("overhead-*.json"))
    if overhead_files:
        overhead = pd.DataFrame([json.loads(path.read_text(encoding="utf-8"))
                                 for path in overhead_files])
        overhead.to_csv(table_dir / "overhead.csv", index=False)
        _write_markdown(overhead, table_dir / "overhead.md")
    diagnostic_runs = [
        "tclad-lsa-late-seed42-bounded-20261004",
        "tclad-lsa-late-seed7-bounded-20261004",
        "tclad-conventional-seed42-bounded-20261004",
        "tclad-noniid-seed42-bounded-20261004",
        "tclad-high-fraction-seed42-bounded-20261004",
    ]
    for run_name in diagnostic_runs:
        run_dir = root / "processed" / run_name
        prediction_path = run_dir / "predictions_E_full_tclad.csv"
        metrics_path = run_dir / "metrics.json"
        if not prediction_path.exists() or not metrics_path.exists():
            continue
        predictions = pd.read_csv(prediction_path)
        plot_anomaly_heatmap(predictions, figure_dir / f"{run_name}_layer_heatmap.png")
        plot_score_distribution(predictions, figure_dir / f"{run_name}_score_distribution.png")
        plot_roc_pr(predictions, figure_dir / f"{run_name}_roc_pr.png")
        analysis = json.loads(metrics_path.read_text(encoding="utf-8"))
        reference_dir = Path(analysis["reference_run"])
        evaluation_dir = Path(analysis["evaluation_run"])
        reference_fp = pd.read_csv(reference_dir / "fingerprints.csv")
        layer_reference = build_benign_reference(reference_fp, set(analysis["fit_rounds"]))
        evaluation_fp = pd.read_csv(evaluation_dir / "fingerprints.csv")
        truth = evaluation_fp[["experiment_id", "round", "client_id", "ground_truth"]].drop_duplicates()
        clean_fp = evaluation_fp.drop(columns=["ground_truth"], errors="ignore")
        temporal = temporal_features(clean_fp, layer_reference)
        cross = cross_layer_features(clean_fp)
        plot_temporal_trajectories(temporal, truth, figure_dir / f"{run_name}_temporal.png")
        plot_cross_layer_matrix(cross, figure_dir / f"{run_name}_cross_layer.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    generate_results(parser.parse_args().results)
