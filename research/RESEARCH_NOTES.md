# Research Notes and Stage Log

## Stage 0/1 — scaffold and scoping review (2026-10-01)

- Inspected Windows, Python 3.14.6, RTX 3050 hardware, initially no PyTorch.
- Created configs, research files, directories, requirements, README, license.
- Found direct conceptual overlap in ANODYNE and FedSurrogate; no first-method claim.
- Wrote no experimental values at this stage.

## Stage 2 — centralized MNIST (2026-10-04)

- Implemented named-layer CNN, deterministic seeds, central trainer, checkpoint/metric/config/plot output, and named-parameter extraction/difference/restoration helpers.
- Quick CPU run, seed 42, 4,500 train / 500 validation, 3 epochs, 10,000 test. Validation accuracy 39.80%, 71.20%, 73.00%; test accuracy 74.24%, test loss 0.7851.
- Smoke tests passed. This verifies the quick code path only.

## Stage 3 — benign FedAvg (2026-10-04)

- Implemented local in-process clients and used Flower 1.39.0 `FedAvg.aggregate_fit` for example-weighted server aggregation. It is a local simulator; Flower's Ray simulation transport is not used.
- Deterministic IID and Dirichlet partitions, same MNIST subset, per-client metrics and round metrics.
- Quick IID seed-42 run with 5 clients, one local epoch, 20 rounds: accuracy rose monotonically from 10.56% to 85.54%; loss declined from 2.2853 to 0.4664. This is clear learning but no plateau was established.
- Added per-client layer fingerprints and whole-update geometry. One-round integration smoke produced both structured files.
- Dirichlet non-IID alpha=0.1 run completed with 5 clients and 20 rounds: test accuracy rose from 10.72% to 80.95%; loss fell from 2.2896 to 0.5808. It is one seed and does not establish a matched IID/non-IID causal comparison.

## Stage 4 — controlled backdoor code (2026-10-04; initial scoring values superseded below)

- Seed-paper full text reviewed. The LSA publication says accepted to IEEE ICC 2026. It uses layer substitution between clean and poisoned local models, ranks layers by BSR loss under clean-layer substitution, accumulates critical layers until BSR reaches τ times malicious BSR, then fine-tunes selected substituted layers. It constructs a smoothed update from selected-layer fine-tuned update plus benign average; reported evaluation used τ=0.8, λ=1, 100 users, 10% malicious, 10% sampled each round, Fashion-MNIST and CIFAR-10.
- Implemented a controlled *LSA-inspired approximation*: fixed/configured selected layer mask (default output layer), trigger-poisoned local examples, benign update mean on non-selected layers, λ scaling on selected-layer updates. It does not implement layer-substitution analysis, iterative BC-layer search, model replacement evaluation, or official reference model architectures. It is explicitly not an exact LSA reproduction.
- Two-round smoke run completed. Clean accuracy was about 10.5% and ASR 0.00. With this untrained model and two rounds, the backdoor did not work. No attack-success claim is supported.
- Late seed-42 attack run reached final ASR 16.18%, peak 23.34%, and clean accuracy 78.72%; matched seed 7 reached ASR 2.35% and clean accuracy 83.32%. The seed-42 conventional trigger control reached ASR 18.29%, peak 26.76%, and clean accuracy 77.24%. Attack strength is weak and seed-sensitive.
- Detector FPR ranking reversed across the two LSA runs (seed 42: whole-update 13.7%, full TCLAD 28.4%; seed 7: whole-update 26.3%, full TCLAD 7.4%). This is unstable exploratory evidence, not an efficacy result.
- Matched seed-42 robust aggregation runs: coordinate median 84.13% clean accuracy / 3.12% ASR; trimmed mean 83.83% / 3.71%; Multi-Krum (f=1) 84.44% / 1.87%. FedAvg under the approximate attack was 78.72% / 16.18%; benign FedAvg was 85.54%. This is one seed and a weak attack.
- High-fraction IID run (2/5 malicious): 71.84% clean accuracy, 31.49% ASR. Non-IID alpha=0.1 run (1/5): 80.55% clean accuracy, 4.59% ASR; full detector TPR 40%, whole-update and cross-layer-only TPR 0% against the matched benign non-IID reference. Both are single weak-attack runs.

## Detector implementation and exploratory evaluation status (initial scoring; superseded below)

Fingerprints, benign reference, temporal and cross-layer descriptors, threshold selection, score combination, mitigation, ASR/detection metrics, and an offline held-out ablation runner now exist. A zero-MAD scale issue was fixed using IQR/std fallbacks and is covered by a unit test. Two late-round LSA-inspired runs completed; attack success and detector FPR rankings vary strongly between seeds. A rejection mitigation rejected all malicious but also 80% of benign updates during attack rounds. Down-weighting produced 85.13% clean accuracy and 2.44% ASR in one seed-42 run, which cannot support efficacy claims. Corrected threshold sensitivity on seed 42 showed full TCLAD FPR 26.3–30.5%, versus 11.6–14.7% for whole-update. These are exploratory, repeated client-round observations, not independent samples.

## Open work

- Improve attack reproduction and detector calibration; quantify attack strength without selecting settings based on detector test labels.
- Add more independent attack and mitigation seeds and create paired run-level confidence intervals.
- The conventional trigger control exists for one seed; repeat it across seeds and assess detector transfer beyond the LSA-inspired approximation.
- Five-client quick mode rounds both configured 10% and 20% malicious fractions to one attacker; use more clients for meaningful low-versus-medium fraction comparisons.
- Improve calibration and component weighting: the corrected full score still has high false-positive rates.
- Robust aggregators were run once against the seed-42 approximate attack; repeat over independent seeds and non-IID settings. A first offline timing/memory diagnostic exists and must be repeated under controlled conditions.
- CIFAR-10, published defense reproductions, thorough literature expansion, and paper-scale seeds remain incomplete.

## Final verification checkpoint (2026-10-04)

- `pytest -q`: 22 passed, 2 upstream Typer deprecation warnings.
- `compileall` passed for all project Python modules and tests.
- Results generator completed after adding layer heatmap, score distribution, ROC/PR, temporal trajectory, cross-layer matrix, threshold sensitivity, attack comparison, and overhead table outputs.
- One offline overhead diagnostic recorded 16.79 seconds and about 9.61 MiB Python-tracked peak memory for 100 client-round scores; total native process memory and online overhead remain unmeasured.


## Final verification checkpoint (2026-10-04, bounded-score correction)

- Fixed scoring to bound robust deviations by `z/(1+z)` and reran held-out comparisons. IID seed-42 95th-percentile FPR: whole update 12.6%, full TCLAD 36.8%; seed 7 ordering reverses (27.4% vs 17.9%). Full TCLAD non-IID TPR is 0% at the selected threshold. Prior unbounded-score comparison numbers above are historical and superseded.
- Reran mitigation: down-weight seed 42 ended at 81.93% clean accuracy / 9.66% ASR. Rejection first exposed an all-client fail-open bug; implementation now holds the global model on those rounds. Fail-closed rerun ended at 78.18% / 0.82% ASR and paused aggregation for three rounds, so it is not evidence of reliable defense.
- Added `research/FAILURE_ANALYSIS.md`; corrected README and final report; regenerated CSV/Markdown tables and 70 figures from raw results. Bounded-score overhead diagnostic saved separately.
- Final checks: `compileall` passed; pytest: 23 passed, 2 upstream Typer deprecation warnings.
- Remaining paper-critical work: faithful LSA reproduction, full MNIST/CIFAR-10 coverage, more independent seeds, robust uncertainty estimates, and direct reproduction of selected published defenses. The repository is a completed quick research prototype, not a completed journal-grade study.


## Continuation checkpoint (2026-10-06)

- Added matched seed-21 benign/attack MNIST runs and held-out detector evaluation. The three-seed IID detector summary is in `results/tables/detector_seed_metrics.csv` and `detector_seed_summary.csv`; intervals are descriptive only.
- Added and ran the end-to-end down-weight mitigation ablation for A–E using seed 42 and the separate benign calibration run. Results are in `results/tables/mitigation_ablation.csv`. B (layer+temporal) had the lowest ASR in this one run; no general component claim is justified.
- Attempted a 2-round CIFAR-10 smoke run. Python download was denied by the host network policy and CIFAR files were not present, so there is no CIFAR result.
- Updated final report, README, experiment-plan execution status, and literature/research-gap files; added two primary-source entries, including the 2026 SoK and online-first FedVigil.
- Added benign-reference leakage/round tests and fixed-threshold / target-FPR threshold tests. Final compileall succeeded; pytest: 26 passed, 2 upstream Typer deprecation warnings.
- Created a portable project ZIP with source, configs, tests, research documents, and raw/processed outputs. The archive omits the local virtual environment and downloaded data.
