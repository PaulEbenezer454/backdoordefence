"""Factor-grid expansion tests without launching expensive training runs."""
import json
import yaml
from experiments.sensitivity import run_factor_sensitivity


def test_factor_grid_skips_irrelevant_iid_alpha_and_records_realized_fraction(tmp_path, monkeypatch):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "base.yaml"
    config_path.write_text(yaml.safe_dump({
        "project": {"seed": 42, "output_root": "results"},
        "dataset": {"data_dir": "data", "partition": {"type": "iid", "clients": 5}},
        "attack": {"enabled": False, "malicious_fraction": 0.2},
    }), encoding="utf-8")
    calls = []

    def fake_run(temporary_config, experiment_id):
        config = yaml.safe_load(temporary_config.read_text(encoding="utf-8"))
        calls.append(config)
        output = tmp_path / "runs" / experiment_id
        output.mkdir(parents=True)
        clients = config["dataset"]["partition"]["clients"]
        count = max(1, round(clients * config["attack"]["malicious_fraction"]))
        metrics = {"experiment_id": experiment_id, "num_clients": clients,
                   "malicious_client_count": count,
                   "realized_malicious_fraction": count / clients,
                   "runtime_seconds": 1.0,
                   "history": [{"global_accuracy": 0.5, "attack_success_rate": 0.1}]}
        (output / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        return output

    monkeypatch.setattr("experiments.sensitivity.run_federated", fake_run)
    output = tmp_path / "tables" / "sensitivity.csv"
    frame = run_factor_sensitivity(config_path, output, [0.1, 0.2], ["iid", "non_iid"],
                                   [0.1, 0.5], [42])
    assert len(frame) == 6
    iid_calls = [config for config in calls if config["dataset"]["partition"]["type"] == "iid"]
    assert len(iid_calls) == 2
    assert all(config["dataset"]["data_dir"] == str(tmp_path / "data") for config in calls)
    assert set(frame["realized_malicious_fraction"]) == {0.2}
    assert output.is_file()
