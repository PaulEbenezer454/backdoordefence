"""Local federated training orchestration using Flower FedAvg."""
from __future__ import annotations
import json
import logging
import platform
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from attacks.base_attack import TriggerConfig, apply_trigger
from attacks.lsa import smooth_malicious_update
from dataset.partition import partition_indices
from dataset.loader import load_torchvision_dataset
from defense.anomaly_detector import score_tclad
from defense.cross_layer import cross_layer_features
from defense.mitigation import mitigation_decision
from defense.temporal import temporal_features
from experiments.baseline import evaluate, seed_everything
from federated.client import FederatedClient
from defense.fingerprints import fingerprint_rows
from federated.strategy import aggregate_round, arrays_to_state, fedavg_strategy, state_to_arrays
from models.cnn import create_model
from evaluation.metrics import attack_success_rate

LOGGER = logging.getLogger("tclad.federated")


def evaluate_asr(model: torch.nn.Module, dataset, target_class: int, trigger: TriggerConfig,
                 batch_size: int, device: torch.device) -> float:
    """Evaluate triggered eligible test samples only; exclude target-class labels."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions, labels_all = [], []
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            mask = labels != target_class
            if not mask.any():
                continue
            images = apply_trigger(images[mask].to(device), trigger)
            labels_all.extend(labels[mask].numpy().tolist())
            predictions.extend(model(images).argmax(dim=1).cpu().numpy().tolist())
    return attack_success_rate(predictions, labels_all, target_class)


def run(config_path: Path, experiment_id: str | None = None,
        detector_bundle: dict | None = None, mitigation: str = "none") -> Path:
    """Run benign MNIST FedAvg and persist metrics, checkpoints, and plots."""
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_name = config["dataset"]["name"].lower()
    seed = int(config["project"]["seed"])
    seed_everything(seed, config["training"].get("deterministic", True))
    requested = config["training"].get("device", "auto")
    device = torch.device("cuda" if requested == "auto" and torch.cuda.is_available()
                          else "cpu" if requested == "auto" else requested)
    data_dir = Path(config["dataset"]["data_dir"])
    if not data_dir.is_absolute():
        data_dir = config_path.resolve().parent.parent / data_dir
    download = bool(config["dataset"].get("download", False))
    train_data = load_torchvision_dataset(dataset_name, data_dir, train=True, download=download)
    test_data = load_torchvision_dataset(dataset_name, data_dir, train=False, download=download)
    subset_size = config["dataset"].get("subset_size")
    eligible = np.arange(len(train_data))
    if subset_size is not None:
        eligible = eligible[:min(int(subset_size), len(eligible))]
    validation_count = int(round(len(eligible) * float(config["dataset"]["validation_fraction"])))
    rng = np.random.default_rng(seed)
    validation_indices = rng.permutation(eligible)[:validation_count]
    pool_indices = np.setdiff1d(eligible, validation_indices, assume_unique=False)
    partition_cfg = config["dataset"]["partition"]
    client_indices_local = partition_indices(np.asarray(train_data.targets)[pool_indices],
        int(partition_cfg["clients"]), partition_cfg["type"],
        float(partition_cfg.get("dirichlet_alpha", 0.5)), seed)
    client_indices = [pool_indices[positions] for positions in client_indices_local]
    clients = [FederatedClient(cid, train_data, client_indices[cid], device,
                int(config["training"]["batch_size"]), float(config["training"]["learning_rate"]),
                int(config["training"]["local_epochs"]), seed,
                config["model"]["name"], int(config["model"]["num_classes"]))
               for cid in range(len(client_indices))]

    model = create_model(config["model"]["name"], int(config["model"]["num_classes"])).to(device)
    global_arrays = state_to_arrays(model)
    strategy = fedavg_strategy(global_arrays, len(clients), float(config["training"]["client_fraction"]))
    test_loader = DataLoader(test_data, batch_size=int(config["training"]["batch_size"]), shuffle=False)
    history = []
    all_fingerprints = []
    update_geometry = []
    mitigation_rows = []
    attack_cfg = config.get("attack", {})
    malicious_count = max(1, int(round(len(clients) * float(attack_cfg.get("malicious_fraction", 0.2)))))
    malicious_clients = set(range(min(malicious_count, len(clients))))
    identifier = experiment_id or datetime.now(timezone.utc).strftime("fedavg-%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    experiment_name = identifier
    parameter_names = [name for name, _ in model.named_parameters()]
    for round_number in range(1, int(config["training"]["rounds"]) + 1):
        count = max(1, int(np.ceil(len(clients) * float(config["training"]["client_fraction"]))))
        round_rng = np.random.default_rng(seed + round_number)
        selected = np.sort(round_rng.choice(len(clients), size=count, replace=False))
        attack_active = bool(attack_cfg.get("enabled", False)) and (
            not attack_cfg.get("attack_rounds") or round_number in attack_cfg["attack_rounds"])
        updates = []
        active_malicious = set()
        for cid in selected:
            is_malicious = attack_active and int(cid) in malicious_clients
            poison_config = None
            if is_malicious:
                trigger_cfg = attack_cfg.get("trigger", {})
                poison_config = {"target_class": int(attack_cfg.get("target_class", 0)),
                    "poison_fraction": float(attack_cfg.get("poison_fraction", 0.3)),
                    "trigger_size": int(trigger_cfg.get("size", 3)),
                    "trigger_value": float(trigger_cfg.get("value", 1.0))}
                active_malicious.add(int(cid))
            updates.append(clients[cid].fit(global_arrays, poison_config))
        if active_malicious and attack_cfg.get("name", "lsa_inspired") == "lsa_inspired":
            benign_updates = [updates[pos][0] for pos, cid in enumerate(selected) if int(cid) not in active_malicious]
            selected_layers = tuple(attack_cfg.get("selected_layers") or ["output"])
            if benign_updates:
                benign_mean_delta = [np.mean([values[index] - global_arrays[index]
                    for values in benign_updates], axis=0) for index in range(len(global_arrays))]
                updated = list(updates)
                for pos, cid in enumerate(selected):
                    if int(cid) in active_malicious:
                        local, count, metrics = updated[pos]
                        local = smooth_malicious_update(local, global_arrays, benign_mean_delta,
                            parameter_names, selected_layers,
                            float(attack_cfg.get("smoothing_lambda", 1.0)))
                        updated[pos] = (local, count, metrics)
                updates = updated
        round_fingerprints = []
        for cid, (local_arrays, _, _) in zip(selected, updates, strict=True):
            grouped: dict[str, list[np.ndarray]] = {}
            flat_parts = []
            global_parts = []
            for name, local, global_value in zip(parameter_names, local_arrays, global_arrays, strict=True):
                layer_name = name.split(".", 1)[0]
                grouped.setdefault(layer_name, []).append((local - global_value).reshape(-1))
                flat_parts.append((local - global_value).reshape(-1))
                global_parts.append(global_value.reshape(-1))
            layer_deltas = {name: np.concatenate(parts) for name, parts in grouped.items()}
            client_fp = fingerprint_rows(layer_deltas, experiment_name,
                round_number, int(cid), ground_truth=int(int(cid) in active_malicious))
            round_fingerprints.append(client_fp)
            vector = np.concatenate(flat_parts)
            reference = np.concatenate(global_parts)
            update_geometry.append({"round": round_number, "client_id": int(cid),
                "experiment_id": experiment_name,
                "whole_update_l1": float(np.abs(vector).sum()),
                "whole_update_l2": float(np.linalg.norm(vector)),
                "whole_update_max_abs": float(np.abs(vector).max()),
                "ground_truth": int(int(cid) in active_malicious),
                "whole_update_cosine_to_global_parameters": float(np.dot(vector, reference) /
                    (np.linalg.norm(vector) * np.linalg.norm(reference) + 1e-8))})
        all_fingerprints.extend(round_fingerprints)
        client_weights = [1.0] * len(updates)
        round_decisions = []
        if (detector_bundle is not None and mitigation != "none" and
                round_number >= int(detector_bundle.get("apply_from_round", 1))):
            current_fp = pd.concat(round_fingerprints, ignore_index=True)
            trajectory = pd.concat(all_fingerprints, ignore_index=True).drop(columns=["ground_truth"], errors="ignore")
            temporal_all = temporal_features(trajectory, detector_bundle["layer_reference"])
            current_temporal = temporal_all[temporal_all["round"] == round_number]
            current_cross = cross_layer_features(current_fp.drop(columns=["ground_truth"], errors="ignore"))
            scores = score_tclad(current_fp.drop(columns=["ground_truth"], errors="ignore"),
                detector_bundle["layer_reference"], current_temporal, detector_bundle["temporal_reference"],
                current_cross, detector_bundle["cross_reference"], detector_bundle["weights"],
                threshold=detector_bundle["threshold"])
            score_by_client = dict(zip(scores.client_id.astype(int), scores.overall_anomaly_score.astype(float)))
            for pos, cid in enumerate(selected):
                score = score_by_client[int(cid)]
                rejected, weight = mitigation_decision(score, detector_bundle["threshold"], mitigation)
                client_weights[pos] = weight
                row = {"round": round_number, "client_id": int(cid), "score": score,
                       "threshold": detector_bundle["threshold"], "rejected": rejected,
                       "aggregation_weight": weight, "ground_truth": int(int(cid) in active_malicious)}
                mitigation_rows.append(row)
                round_decisions.append(row)
            if not any(weight > 0 for weight in client_weights):
                # Fail closed: preserve the current global model for this round.
                for row in round_decisions:
                    row["all_rejected_noop"] = True
            else:
                for row in round_decisions:
                    row["all_rejected_noop"] = False
        aggregation_cfg = config.get("aggregation", {})
        aggregation_method = aggregation_cfg.get("method", "fedavg")
        if any(weight > 0 for weight in client_weights):
            global_arrays, client_metrics = aggregate_round(strategy, round_number, updates, client_weights,
                method=aggregation_method, trim_count=int(aggregation_cfg.get("trim_count", 1)),
                byzantine_count=int(aggregation_cfg.get("byzantine_count", 1)))
        else:
            global_arrays = state_to_arrays(model)
            client_metrics = {
                key: float(np.mean([metrics[key] for _, _, metrics in updates]))
                for key in updates[0][2]
            }
        arrays_to_state(model, global_arrays)
        global_loss, global_accuracy = evaluate(model, test_loader, device)
        record = {"round": round_number, "selected_clients": selected.tolist(),
                  "global_loss": global_loss, "global_accuracy": global_accuracy,
                  "weighted_client_train_loss": client_metrics["train_loss"],
                  "weighted_client_accuracy": client_metrics["client_accuracy"],
                  "client_metrics": [{"client_id": int(cid), "num_examples": int(clients[cid].indices.size),
                                      "train_loss": float(metrics["train_loss"]),
                                      "client_accuracy": float(metrics["client_accuracy"])}
                                     for cid, (_, _, metrics) in zip(selected, updates, strict=True)]}
        if detector_bundle is not None:
            record["mitigation_decisions"] = round_decisions
        if attack_active:
            trigger_cfg = attack_cfg.get("trigger", {})
            record["attack_success_rate"] = evaluate_asr(model, test_data,
                int(attack_cfg.get("target_class", 0)),
                TriggerConfig(size=int(trigger_cfg.get("size", 3)), value=float(trigger_cfg.get("value", 1.0))),
                int(config["training"]["batch_size"]), device)
        history.append(record)
        LOGGER.info("round=%d global_loss=%.4f global_accuracy=%.4f", round_number,
                    global_loss, global_accuracy)

    root = Path(config["project"].get("output_root", "results"))
    if not root.is_absolute():
        root = config_path.resolve().parent.parent / root
    output = root / "raw" / "federated" / identifier
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    pd.concat(all_fingerprints, ignore_index=True).to_csv(output / "fingerprints.csv", index=False)
    pd.DataFrame(update_geometry).to_csv(output / "whole_update_features.csv", index=False)
    if mitigation_rows:
        pd.DataFrame(mitigation_rows).to_csv(output / "mitigation_decisions.csv", index=False)
    metrics = {"experiment_id": identifier, "seed": seed, "device": str(device),
               "runtime_seconds": time.perf_counter() - started,
                "environment": {"python": platform.python_version(), "torch": torch.__version__,
                               "torchvision": __import__("torchvision").__version__, "flower": __import__("flwr").__version__},
               "partition": partition_cfg, "aggregation": aggregation_method,
               "num_clients": len(clients),
               "malicious_client_count": int(malicious_count) if attack_cfg.get("enabled", False) else 0,
               "configured_malicious_fraction": float(attack_cfg.get("malicious_fraction", 0.0)),
               "realized_malicious_fraction": (float(malicious_count / len(clients))
                                               if attack_cfg.get("enabled", False) else 0.0),
               "attack_name": attack_cfg.get("name", "none") if attack_cfg.get("enabled", False) else "none",
               "history": history}
    metrics["mitigation"] = mitigation
    if detector_bundle is not None:
        metrics["calibration_rounds"] = detector_bundle["calibration_rounds"]
        metrics["detector_threshold"] = float(detector_bundle["threshold"])
        metrics["mitigation_apply_from_round"] = int(detector_bundle.get("apply_from_round", 1))
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"model_state_dict": model.state_dict(), "metrics": metrics}, output / "checkpoint.pt")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rounds = [row["round"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(rounds, [row["global_accuracy"] for row in history], marker="o")
    axes[0].set(xlabel="Round", ylabel="Test accuracy", title="Benign FedAvg accuracy")
    axes[1].plot(rounds, [row["global_loss"] for row in history], marker="o")
    axes[1].set(xlabel="Round", ylabel="Test loss", title="Benign FedAvg loss")
    fig.tight_layout()
    fig.savefig(output / "metrics.png", dpi=160)
    plt.close(fig)
    return output


