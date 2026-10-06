"""Federated server and Flower FedAvg strategy helpers."""
from __future__ import annotations
import numpy as np
import torch
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg


def fedavg_strategy(initial: list[np.ndarray], num_clients: int, fraction_fit: float = 1.0) -> FedAvg:
    """Construct Flower's standard example-count-weighted FedAvg strategy."""
    min_clients = max(2, int(np.ceil(num_clients * fraction_fit)))
    return FedAvg(fraction_fit=fraction_fit, min_fit_clients=min_clients,
                  min_available_clients=num_clients, initial_parameters=ndarrays_to_parameters(initial),
                  fit_metrics_aggregation_fn=aggregate_fit_metrics, inplace=False)


def aggregate_fit_metrics(metrics: list[tuple[int, dict]]) -> dict[str, float]:
    """Example-weight scalar client metrics for Flower's strategy interface."""
    total = sum(count for count, _ in metrics)
    return {key: sum(count * float(values[key]) for count, values in metrics) / total
            for key in metrics[0][1]} if total else {}


def aggregate_round(strategy: FedAvg, round_number: int,
                    updates: list[tuple[list[np.ndarray], int, dict[str, float]]],
                    client_weights: list[float] | None = None, method: str = "fedavg",
                    trim_count: int = 1, byzantine_count: int = 1) -> tuple[list[np.ndarray], dict[str, float]]:
    """Pass simulated client results through Flower's official FedAvg aggregator."""
    results = []
    if client_weights is None:
        client_weights = [1.0] * len(updates)
    if len(client_weights) != len(updates):
        raise ValueError("client_weights must align with updates")
    if method != "fedavg":
        if any(weight != 1.0 for weight in client_weights):
            raise ValueError("Robust aggregation baselines cannot be combined with TCLAD client weights")
        if method not in {"coordinate_median", "trimmed_mean", "multi_krum"}:
            raise ValueError(f"Unknown aggregation method: {method}")
        selected = list(range(len(updates)))
        if method == "multi_krum":
            n = len(updates)
            if byzantine_count < 0 or n < 2 * byzantine_count + 3:
                raise ValueError("Multi-Krum requires n >= 2f + 3")
            vectors = [np.concatenate([array.reshape(-1) for array in update[0]]) for update in updates]
            distances = np.stack([[np.sum((left - right) ** 2) for right in vectors] for left in vectors])
            neighbor_count = n - byzantine_count - 2
            scores = np.sort(distances, axis=1)[:, 1:neighbor_count + 1].sum(axis=1)
            selected_count = n - byzantine_count - 2
            selected = np.argsort(scores)[:selected_count].tolist()
        aggregated = []
        for layer_index in range(len(updates[0][0])):
            values = np.stack([updates[index][0][layer_index] for index in selected])
            if method == "coordinate_median":
                combined = np.median(values, axis=0)
            elif method == "trimmed_mean":
                if trim_count < 0 or 2 * trim_count >= len(values):
                    raise ValueError("trim_count must be non-negative and less than half the clients")
                ordered = np.sort(values, axis=0)
                retained = ordered[trim_count:len(values) - trim_count] if trim_count else ordered
                combined = retained.mean(axis=0)
            else:
                combined = values.mean(axis=0)
            aggregated.append(combined.astype(values.dtype, copy=False))
        chosen = [updates[index] for index in selected]
        total = sum(item[1] for item in chosen)
        metrics = {key: sum(item[1] * float(item[2][key]) for item in chosen) / total
                   for key in chosen[0][2]}
        return aggregated, metrics
    for (arrays, count, metrics), weight in zip(updates, client_weights, strict=True):
        if weight <= 0:
            continue
        effective_count = max(1, int(round(count * weight)))
        fit_res = FitRes(status=Status(code=Code.OK, message=""),
                         parameters=ndarrays_to_parameters(arrays), num_examples=effective_count,
                         metrics={key: float(value) for key, value in metrics.items()})
        results.append((None, fit_res))
    if not results:
        raise ValueError("No client updates remain after mitigation")
    parameters, metrics = strategy.aggregate_fit(round_number, results, [])
    if parameters is None:
        raise RuntimeError("FedAvg returned no aggregated parameters")
    return parameters_to_ndarrays(parameters), {key: float(value) for key, value in metrics.items()}


def state_to_arrays(model: torch.nn.Module) -> list[np.ndarray]:
    """Serialize model trainable parameters in stable named-parameter order."""
    return [parameter.detach().cpu().numpy().copy() for _, parameter in model.named_parameters()]


def arrays_to_state(model: torch.nn.Module, arrays: list[np.ndarray]) -> None:
    """Load ordered trainable arrays into a model."""
    with torch.no_grad():
        for (_, parameter), array in zip(model.named_parameters(), arrays, strict=True):
            parameter.copy_(torch.from_numpy(array).to(parameter.device, parameter.dtype))
