# Final Experiment Report — TCLAD-FL

**Status: complete executable prototype; research study remains incomplete.** This report records only work executed in this repository. The small-subset, CPU results are exploratory evidence, not journal-quality conclusions. TCLAD-FL has not demonstrated an advantage over whole-update detection.

## Abstract

This project implements a reproducible local federated-learning prototype for studying per-layer update statistics, temporal change, and cross-layer relationships against a controlled Layer Smoothing Attack (LSA)-inspired backdoor. The environment, centralized CNN, Flower FedAvg simulation, attack approximation, feature extraction, offline detector comparisons, mitigation paths, and unit tests are implemented. MNIST quick runs learned in IID and Dirichlet non-IID settings. The approximate late-round attack is weak and seed sensitive; under the selected non-IID protocol, full TCLAD misses the malicious client-rounds at the 95th-percentile threshold. In IID seed 42, whole-update scoring has a lower false-positive rate than full TCLAD. Fail-closed rejection can pause aggregation when every client is flagged; component-specific down-weighting reduces ASR in this one seed but does not establish efficacy. No claim of detector novelty or effectiveness is warranted.

## Introduction and problem statement

Federated clients send model updates that may contain targeted backdoor behavior while preserving clean task performance. LSA emphasizes selected backdoor-critical layers and attempts to make malicious updates less distinguishable. The study question is whether layer-wise statistical, temporal, and explicit cross-layer signals add useful held-out detection value beyond whole-update statistics.

## Research gap and positioning

The targeted literature review identifies direct overlap with ANODYNE’s layer/sub-vector and spatial-temporal anomaly analysis, plus layer-criticality methods such as FedSurrogate. The reviewed evidence does not establish that explicit cross-layer relations are novel. This work is positioned as an empirical evaluation question: test whether such relations add out-of-sample utility under a matched LSA protocol. The implemented attack is an approximation and the test matrix is incomplete. See `LITERATURE_REVIEW.md` and `RESEARCH_GAP.md`.

## Research questions and hypotheses

RQ1: Can layer fingerprints detect LSA behavior? RQ2: Does temporal information add value? RQ3: Does cross-layer analysis add value? RQ4: Does TCLAD outperform whole-update detection? RQ5: How does non-IID drift affect false positives? RQ6: How does malicious fraction affect results? RQ7: What computational overhead is introduced?

H1–H4 (layer signal, temporal gain, cross-layer gain, and useful combined mitigation) remain hypotheses and are not accepted based on the current data.

## Threat model

All clients and data are simulated locally. The server can inspect per-client model updates; secure aggregation is therefore outside the supported threat model. Attacker clients poison a configured fraction of local examples toward one target and submit an LSA-inspired layer-selective update. No real-world system is targeted.

## Methodology and mathematical formulation

For client (i), round (t), let (W_i^t) be its locally trained parameters and (W^t) the incoming global parameters. The update is \(\Delta W_i^t=W_i^t-W^t\); layer \(l\) has \(\Delta W_{i,l}^t\). A fingerprint \(F_{i,l}^t\) contains mean, standard deviation, median, extrema, L1/L2 norms, max absolute value, sparsity, skewness, and kurtosis, with reference similarity/distance where defined.

Temporal differences are \(\delta F_{i,l}^t=F_{i,l}^t-F_{i,l}^{t-1}\), with absolute change, rolling summaries, variance, persistence/duration, and slope features. Cross-layer terms \(C_{i,l,m}^t\) include pairwise cosine similarity, normalized distances, relative magnitudes, and correlations of changes. Robust benign medians/MAD-like scales form references; a separate benign validation window selects thresholds. The implementation combines normalized layer, temporal, and cross-layer components using configurable weights. Exact feature and scoring steps are documented in `METHODOLOGY.md` and implemented under `defense/`.

The implemented LSA approximation uses a fixed configured layer mask, triggered local poisoning, benign-mean replacement for unselected layer deltas, and configured smoothing. It omits the seed paper’s clean-layer substitution, iterative backdoor-critical layer search, and fine-tuning procedure. Do not call it an exact reproduction.

Mitigation applies a reject or down-weight decision to client aggregation weights. Metrics include clean accuracy, ASR on triggered non-target inputs, TPR/FPR, precision/recall/F1, benign acceptance, malicious rejection, ROC-AUC, and PR-AUC where defined.

## System architecture and algorithms

Implemented components include named MNIST/CIFAR CNNs, deterministic data partitioning, local Flower client training and strategy aggregation, attack isolation, update fingerprints, reference construction, temporal and cross-layer features, transparent scores and thresholds, robust aggregation baselines (coordinate median, trimmed mean, Multi-Krum), reject/down-weight mitigation, offline ablation and threshold sensitivity runners, synthetic unit tests, a synthetic-data end-to-end FL integration test, and automatic tables/figures. Flower executes in-process; the distributed Ray transport is not used.

## Experimental setup

Windows 11; Python 3.14.6; CPU PyTorch 2.11.0; torchvision 0.26.0; Flower 1.39.0. An RTX 3050 is present but the installed Torch build is CPU-only. MNIST experiments use a 5,000-example training subset; tests use synthetic data. All research FL runs use five clients, one local epoch, and 20 rounds unless noted. Attack seeds 42, 7, and 21 were run under the matched IID quick protocol. Configurations and per-run artifacts are saved with results.

## Datasets and baselines

Executed: MNIST centralized quick baseline, IID FedAvg, Dirichlet non-IID FedAvg (alpha 0.1), three LSA-inspired seeds, a matched conventional-trigger control, coordinate median/trimmed mean/Multi-Krum against the seed-42 LSA-inspired setup, offline detector component comparisons, threshold sensitivity, reject mitigation, and down-weight mitigation. A CIFAR-10 run was attempted, but the environment blocked Python network access while torchvision tried to download the dataset; no CIFAR-10 metrics are claimed. ANODYNE, FedSurrogate, FedDLAD, and FLTrust are reviewed but not reproduced.

## Results

### Learning baselines

- Centralized quick training used 4,500 train and 500 validation examples for three epochs. Validation accuracy was 39.80%, 71.20%, and 73.00%; test accuracy was 74.24%, test loss 0.7851.
- IID FedAvg (seed 42) improved test accuracy from 10.56% in round 1 to 85.54% in round 20; test loss decreased from 2.2853 to 0.4664.
- Dirichlet non-IID FedAvg, alpha=0.1 (seed 42), improved test accuracy from 10.72% to 80.95%; test loss decreased from 2.2896 to 0.5808.

These are single-seed subset runs; the IID/non-IID results are not replicated comparisons.

### Attack and detection

The late-round LSA-inspired runs finished at 78.72% clean accuracy / 16.18% ASR (seed 42), 83.32% / 2.35% (seed 7), and 85.72% / 2.51% (seed 21). Across these three independent seeds, clean accuracy averaged 82.59% (SD 3.56 percentage points) and ASR averaged 7.01% (SD 7.94 points). With n=3, these are descriptive summaries only. The large spread and weak ASR in two runs show that this attack approximation is unstable, so detector findings may reflect training-stage or client-specific signals rather than reliable LSA behavior.

The seed-42 matched conventional-trigger control had final clean accuracy 77.24% and ASR 18.29%, peaking at 26.76%. It used the same seed, IID client partition, trigger, poison fraction, and attack rounds, with the LSA-inspired smoothing step disabled. The similarly low ASRs show that this quick setup did not produce a strong backdoor for either arm. On the conventional run, whole-update TPR/FPR was 1.00/0.137, full TCLAD 1.00/0.263, layer-only 1.00/0.379, and cross-layer-only 1.00/0.126.

### Aggregation baselines against the seed-42 LSA-inspired run

With the same five-client IID partition, seed, local-training schedule, and approximate attack configuration, FedAvg ended at 78.72% clean accuracy / 16.18% ASR; coordinate median at 84.13% / 3.12%; trimmed mean at 83.83% / 3.71%; and Multi-Krum (f=1) at 84.44% / 1.87%. The benign FedAvg run ended at 85.54%. This one-seed result suggests these aggregators limited this approximate attack in this run, but it does not establish performance against exact LSA or other partitions.

### Attacker fraction and partition sensitivity

At 40% configured/realized malicious clients (2 of 5) on IID seed 42, final clean accuracy was 71.84% and ASR 31.49%, compared with 78.72% / 16.18% at 20% (1 of 5). The approximate attack became stronger while clean performance fell. In quick mode, configured fractions of 10% and 20% both round to one malicious client and therefore cannot be separated with five clients.

For Dirichlet non-IID alpha=0.1 at 20% configured/realized malicious clients (1 of 5), final clean accuracy was 80.55% and ASR 4.59%. Against a matched non-IID benign reference, at the 95th-percentile threshold whole-update detection had TPR 0.00/FPR 0.074, layer-only 0.00/0.074, cross-layer-only 0.00/0.042, and full TCLAD 0.00/0.147. The low ASR makes this a weak attack test but exposes a clear missed-detection failure. Cross-layer-only ranked examples better than chance in this split (ROC-AUC 0.815) while its selected operating threshold detected none, illustrating the gap between ranking and threshold performance.

Offline held-out comparison used separate benign reference and validation rounds per seed. The score was corrected to bound individual robust deviations as z/(1+z), preventing extreme temporal features from dominating due to near-zero robust scales. Across three matched IID seeds at the 95th percentile, whole-update detection had mean TPR 1.00 and FPR 0.204 (SD 0.074); full TCLAD had TPR 1.00 and FPR 0.249 (SD 0.104); layer-only had TPR 1.00 and FPR 0.189 (SD 0.148). All methods ranked the attack records perfectly in these three small runs (ROC-AUC 1.00), but the calibrated full detector rejected about one quarter of benign client-rounds. The three-seed t intervals are descriptive, small-sample summaries only; the repeated client-rounds within a seed are not independent observations.

### Mitigation and threshold sensitivity

The initial bounded-score rejection run exposed a fail-open bug and is superseded. After changing all-client rejection to preserve the current global model for that round, the corrected run finished at 78.18% clean accuracy and 0.82% ASR. It flagged 21/25 attack-period client-round updates, including 16/20 benign records, and held the global model for three rounds when all clients were rejected. The low ASR followed from withholding aggregation in those rounds; this single result does not show reliable attack mitigation. The new end-to-end down-weight ablation (seed 42) measured clean accuracy / ASR as follows: A layer-only 82.34% / 8.90%; B layer+temporal 82.62% / 8.26%; C layer+cross-layer 81.97% / 9.57%; D temporal+cross-layer 81.59% / 10.29%; E full TCLAD 81.93% / 9.66%. The matched no-mitigation attack was 78.72% / 16.18%. Component-specific mean client weights for full TCLAD were 0.965 benign and 0.679 malicious. This is one deterministic seed, and the differences between ablations are not evidence of general component contributions; the mild reduction in ASR requires replication. It supersedes earlier unbounded-score mitigation numbers.

Across 90th, 95th, 97.5th, and 99th percentile thresholds, corrected full TCLAD FPR ranged from 31.6% to 44.2%; whole-update FPR ranged from 11.6% to 14.7%. Cross-layer-only FPR ranged from 24.2% to 33.7%. Raising the threshold did not make full TCLAD outperform the whole-update baseline on FPR in this run.

## Ablation and sensitivity analysis

The corrected component comparison suggests cross-layer-only or full scoring can look useful on an individual seed, but their ranking changes across seeds. This does not demonstrate an incremental or generalizable benefit. Coordinate median, trimmed mean, and Multi-Krum were also compared once against the seed-42 approximate attack. Threshold sensitivity was executed on seed 42 only. Malicious fraction and IID/non-IID sensitivity each have a single small-run comparison; Dirichlet alpha, smoothing strength, and number of clients have not been varied systematically.

## Statistical analysis

No inferential statistical tests are reported. Three seeds were run for the IID LSA-inspired arm, and detector rows are temporally correlated, sharing just five client identities per seed. A valid paper study needs more independent seeds/runs as the unit of analysis, paired configurations, confidence intervals, and predeclared comparisons. `evaluation/statistical_tests.py` provides basic supported tests but does not justify testing these correlated rows as independent samples.

## Computational overhead

A corrected bounded-score offline detector-processing measurement on the saved 20-round run took 16.35 seconds for reference fitting, temporal/cross-layer extraction, and scoring of 4,400 fingerprint rows (100 client-round scores); `tracemalloc` reported 10,117,539 bytes (about 9.65 MiB) peak Python-tracked allocations. The corresponding benign reference FL run took 119.18 seconds. This is not an online overhead estimate or total process memory; `tracemalloc` does not capture every native allocation, and the measurement includes file I/O. Repeated controlled profiling is required.

## Failure analysis

Observed failure modes: the attack approximation has low and seed-sensitive ASR; the full detector misses all malicious examples at the selected threshold in the non-IID split; full TCLAD has higher false-positive rates than whole-update scoring in IID seed 42; seed 7 reverses that comparison; and rejection can stall learning when all clients are flagged (the implementation now fails closed by retaining the current global model for that round). Training-stage shift between benign calibration and attack rounds may contribute to false positives. The end-to-end ablation shows layer-only and layer+temporal weighting had lower ASR in the seed-42 run than the full three-component score, but one seed cannot establish which component caused the difference. The present outputs do not isolate whether temporal change, non-IID drift, or attack behavior caused detection errors. Additional layer-level residual diagnostics and same-stage benign controls are required.

## Limitations

Three seeds for the approximate LSA-inspired IID arm, one seed for conventional trigger, robust aggregation, malicious-fraction and non-IID attack comparisons; reduced MNIST subset, small model, CPU-only execution, approximate attack, one late-round setting, no CIFAR-10, no recent published-defense reproduction, no inferential statistics, and only a rough offline overhead measurement. Existing literature overlaps with the broad temporal/layerwise framing. Results do not support TCLAD efficacy, general robustness, or novelty claims.

## Future work

Implement or obtain a faithful LSA reproduction; establish stronger attack efficacy independently of detector labels; run non-IID attack controls; refine score calibration to control FPR; evaluate robust aggregation and recent defenses; replicate across seeds; profile overhead in a controlled benchmark; and expand systematic literature review. Preserve current negative findings.

## Conclusion and contribution statement

The codebase demonstrates a working, tested quick MNIST FL research prototype. Current experiments do not show a consistent advantage for TCLAD over whole-update scoring and reveal a non-IID detection failure plus a rejection-induced training stall when all clients are flagged. One down-weighted run modestly reduced ASR, but the evidence is too limited to attribute that effect reliably. The supported contribution is **a reproducible prototype, a three-seed exploratory detection comparison, and an end-to-end component ablation that documents failures**, not a demonstrated new defense. No final journal contribution claim is justified yet.
