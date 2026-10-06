# TCLAD-FL

TCLAD-FL is a controlled research prototype for evaluating layer-wise update fingerprints, temporal behavior, and explicit cross-layer relations against backdoor behavior in simulated federated learning. It is an experiment, not a claim that the detector works.

## Research position

The initial literature review found direct overlap with ANODYNE's layerwise/spatial-temporal detection and with layer-criticality defenses such as FedSurrogate. We make no “first layer-aware defense” claim. The candidate contribution is an LSA-focused, held-out comparison of explicit cross-layer relations against layer-temporal and whole-update baselines. The LSA implementation here is a documented approximation, not an exact reproduction. Read [the literature review](research/LITERATURE_REVIEW.md), [research gap](research/RESEARCH_GAP.md), and [methodology](research/METHODOLOGY.md) before interpreting results.

## Current status

The repository contains the full source, configurations, tests, research documents, saved experiment outputs, metrics, and figures. It excludes the local Python environment, downloaded datasets, temporary working files, logs, and model checkpoints.

Executed small MNIST quick runs demonstrate that the prototype executes end-to-end, but not that TCLAD-FL is effective. Results are seed-sensitive, the attack often has weak efficacy, false positives are high, and the selected detector misses malicious clients in the tested non-IID setting. No paper-scale multi-seed, full-MNIST, CIFAR-10, or journal-quality comparison has completed. Review `research/FINAL_EXPERIMENT_REPORT.md` and `research/FAILURE_ANALYSIS.md` for actual findings and limitations.

## Install and run (Windows PowerShell)

Install Python 3.11 or 3.14, then from the cloned repository run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m pytest -q
```

On a CUDA-enabled computer, install the matching official PyTorch CUDA wheels instead of using the CPU wheel index above. Then try:

```powershell
python scripts/train_centralized.py --config configs/mnist.yaml
python scripts/train_federated.py --config configs/mnist.yaml
python scripts/run_lsa.py --config configs/mnist_lsa_late.yaml
python scripts/generate_results.py
```

Dataset downloads are configuration-controlled. Tests use synthetic data and do not download datasets. Set `dataset.download: true` in the chosen config when you want the public MNIST or CIFAR-10 data downloaded for an experiment. See the README command reference and experiment plan for defense, mitigation, ablation, sensitivity, and report-generation commands. Every run is assigned a unique output directory under `results/raw/`.

## Main folders

- `models/`, `dataset/`, `federated/`: models, data loading/partitioning, and local Flower FedAvg simulation.
- `attacks/`: controlled, simulated-only backdoor attack implementations.
- `defense/`, `evaluation/`: fingerprints, reference models, temporal/cross-layer features, detector, mitigation, and metrics.
- `experiments/`, `scripts/`, `configs/`: runnable experiments, command-line entry points, and settings.
- `tests/`: fast tests that use synthetic data.
- `research/`: literature, methodology, experiment plan, and honest result/failure reports.
- `results/tables/`, `results/figures/`, `results/raw/`, `results/processed/`: saved experiment evidence.

Clients are simulated in-process. Flower's strategy aggregation is used; Ray-backed network simulation is not used. Coordinate median, trimmed mean, and Multi-Krum are documented local reference implementations, not official package implementations. Individual-update server detection is incompatible with secure aggregation that hides client updates.
