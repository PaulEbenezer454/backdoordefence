# TCLAD-FL

TCLAD-FL is a controlled research prototype for evaluating layer-wise update fingerprints, temporal behavior, and explicit cross-layer relations against backdoor behavior in simulated federated learning. It is an experiment, not a claim that the detector works.

## Research position

The initial literature review found direct overlap with ANODYNE’s layerwise/spatial-temporal detection and with layer-criticality defenses such as FedSurrogate. We make no “first layer-aware defense” claim. The candidate contribution is an LSA-focused, held-out comparison of explicit cross-layer relations against layer-temporal and whole-update baselines. The LSA implementation here is a documented approximation, not an exact reproduction. Read [the literature review](research/LITERATURE_REVIEW.md), [research gap](research/RESEARCH_GAP.md), and [methodology](research/METHODOLOGY.md) before interpreting results.

## Current status

Implemented: centralized MNIST/CIFAR CNN definitions, data partitioning, local Flower FedAvg strategy aggregation, an LSA-inspired controlled attack, per-client layer fingerprints, temporal and cross-layer features, robust reference construction, scoring/thresholding, reject/down-weight mitigation, robust aggregation baselines, held-out ablation/sensitivity runners, tests, and automatic result generation.

Executed MNIST quick runs (CPU; protocol-specific seeds shown below):

- Centralized baseline: 4,500 train / 500 validation, 3 epochs; validation accuracy ended at 73.00%, test accuracy 74.24%.
- Benign IID FedAvg: 5 clients, one local epoch, 20 rounds; test accuracy increased to 85.54%, loss decreased to 0.4664, and continued to improve at round 20.
- Benign Dirichlet non-IID FedAvg (alpha=0.1): same 5-client/20-round quick protocol; test accuracy increased to 80.95%, loss decreased to 0.5808. This is a one-seed development run.
- LSA-inspired approximation, attack from round 16: final clean accuracy 78.72%, ASR 16.18%; ASR peaked at 23.34%. The attack has weak efficacy and reduces clean accuracy during poisoning.
- Paired seed-7 LSA-inspired run: final clean accuracy 83.32%, ASR 2.35%.
- Paired seed-21 LSA-inspired run: final clean accuracy 85.72%, ASR 2.51%. Across seeds 42/7/21, the descriptive mean clean accuracy was 82.59% (SD 3.56 pp), while mean ASR was 7.01% (SD 7.94 pp); n=3, exploratory only.
- Matched conventional-trigger control (same seed, partition, trigger and attack rounds, without smoothing): final clean accuracy 77.24%, ASR 18.29%. Both attack paths were weak in this quick configuration.
- Fail-closed rejection rerun (bounded score): final clean accuracy 78.18%, ASR 0.82%. It flagged 21/25 attack-period client-round updates (including 16/20 benign records) and held the global model for three rounds when every client was rejected. The low final ASR followed from stopping aggregation in those rounds and is one exploratory run, not evidence of reliable defense.
- Corrected offline ablation/threshold sensitivity at the 95th percentile: across three IID seeds, mean FPR was 24.9% for full TCLAD (SD 10.4 pp), 20.4% for whole-update (SD 7.4 pp), and 18.9% for layer-only (SD 14.8 pp). All methods had TPR/AUC 100% on this small late-round protocol; all evaluated detectors missed the attacker at the selected threshold in the non-IID split. Down-weighting ended at 81.93% clean accuracy / 9.66% ASR in one run. End-to-end down-weight ablation by components (seed 42): layer-only 82.34% accuracy / 8.90% ASR; layer+temporal 82.62% / 8.26%; layer+cross-layer 81.97% / 9.57%; temporal+cross-layer 81.59% / 10.29%; full TCLAD 81.93% / 9.66%. Single-run evidence only. See `research/FINAL_EXPERIMENT_REPORT.md`, `research/FAILURE_ANALYSIS.md`, and saved outputs.
- The corrected bounded-score offline processing measurement took 16.35 seconds and used about 9.65 MiB peak Python-tracked allocation for 100 client-round scores; this is a local diagnostic, not total memory or online overhead.
- Against the seed-42 approximate attack, coordinate median / trimmed mean / Multi-Krum ended at clean accuracy 84.13% / 83.83% / 84.44% and ASR 3.12% / 3.71% / 1.87%. These are single-seed exploratory comparisons against a weak attack.
- With 40% malicious clients (2/5), the IID run ended at 71.84% clean accuracy and 31.49% ASR; the non-IID alpha=0.1 run at 20% (1/5) ended at 80.55% and 4.59% ASR. Under the corrected non-IID detector test, all compared methods including full TCLAD missed all malicious client-rounds at the selected threshold. Both attacks were weak and these are single runs.

No paper-scale multi-seed, full-MNIST, CIFAR-10, or journal-quality comparison has completed. Three small-subset LSA-inspired IID seeds and one conventional-trigger control show weak, seed-sensitive attacks. Detector ranking also changes between seeds, so the current results do not establish an advantage.

## Environment and installation

Validated on Windows 11, Python 3.14.6, Torch 2.11.0+cpu, torchvision 0.26.0+cpu, Flower 1.39.0. This host has an RTX 3050, but its active Torch build is CPU-only. All reported runs used CPU. Dependency pins and environment details are in [requirements.txt](requirements.txt) and [research/ENVIRONMENT.md](research/ENVIRONMENT.md).

Use the project-local environment. For CPU Torch, install the official CPU wheel pair from the index command in `requirements.txt`, then run `pip install -r requirements.txt`. For a CUDA machine, select the matching official PyTorch wheel index. Dataset download is controlled by `dataset.download`; tests use synthetic data and do not download full datasets.

## Commands

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/train_centralized.py --config configs/mnist.yaml
.\.venv\Scripts\python.exe scripts/train_federated.py --config configs/mnist.yaml
.\.venv\Scripts\python.exe scripts/run_lsa.py --config configs/mnist_lsa_late.yaml
.\.venv\Scripts\python.exe scripts/run_lsa.py --config configs/mnist_conventional_late.yaml
.\.venv\Scripts\python.exe scripts/run_defense.py --reference results/raw/federated/<benign-run> --evaluation results/raw/federated/<attack-run>
.\.venv\Scripts\python.exe scripts/run_mitigated.py --config configs/mnist_lsa_late.yaml --reference results/raw/federated/<benign-run> --mitigation reject
.\.venv\Scripts\python.exe scripts/run_ablation.py --reference results/raw/federated/<benign-run> --evaluation results/raw/federated/<attack-run> --output results/processed/<run-id>
.\.venv\Scripts\python.exe scripts/run_sensitivity.py --reference results/raw/federated/<benign-run> --evaluation results/raw/federated/<attack-run>
.\.venv\Scripts\python.exe scripts/run_factor_sensitivity.py --config configs/mnist_lsa_late.yaml --fractions 0.1 0.2 0.4 --distributions iid non_iid --dirichlet-alphas 0.1 0.5 --seeds 42 7 21 --output results/processed/factor-sensitivity-<run-id>.csv
.\.venv\Scripts\python.exe scripts/generate_results.py
.\.venv\Scripts\python.exe scripts/package_project.py
.\.venv\Scripts\python.exe scripts/measure_overhead.py --reference results/raw/federated/<benign-run> --evaluation results/raw/federated/<attack-run> --output results/processed/overhead-<run-id>.json
```

Every run gets a unique directory under `results/raw/`; an existing ID is refused. Results include resolved config, metrics, checkpoints and plots. FL runs save per-client fingerprint and whole-update CSVs.

The portable ZIP includes source, configuration, tests, research documents, and generated results. It excludes `.venv/`, downloaded datasets in `data/`, transient work files, and logs. Set `dataset.download: true` in the selected config to fetch missing public data before training.

## Federated implementation note

Clients are simulated in-process. Their local updates are wrapped in Flower `FitRes` objects and aggregated with Flower’s `FedAvg` strategy. Flower’s Ray-backed simulation transport is not used. Coordinate median, trimmed mean, and Multi-Krum are local reference implementations documented as such, not official package implementations.

## Integrity and limitations

The LSA seed paper uses layer substitution to identify backdoor-critical layers, iterative critical-layer selection, fine-tuning, and smoothed layer updates. This prototype instead uses a configured layer mask (default `fc1`/`output` in the late config), trigger-poisoned local examples, and benign-mean replacement for unselected-layer updates. It must be reported as **LSA-inspired approximation**. The small experiment showed weak ASR and high benign rejection; do not conceal these results.

Server-side individual-update detection is incompatible with secure aggregation that hides client updates. Threshold calibration requires a separate benign run and is sensitive to training-time drift. The current 20-round, 5-client metrics repeat the same client identities across rounds; they are not independent observations and do not support significance claims. Larger, matched multi-seed studies are required.

