# Experiment Plan

All cells below are planned work, not completed results. Begin with the centralized MNIST baseline. Do not proceed to federated or attack experiments until each preceding stage passes its own run and review gate.

## Stage gates

1. **Centralized MNIST:** synthetic model checks, then subset quick run, then full MNIST; save config, checkpoint, metrics, and plot. Confirm learning before FL.
2. **Benign FedAvg:** same model/data partitions across methods; compare IID and non-IID; confirm convergence.
3. **Attack protocol:** audit LSA paper/code, record exact/approximate status; add correct clean/triggered evaluation and ASR.
4. **Detector:** validate fingerprint, reference, temporal, cross-layer modules independently before joining them.
5. **Mitigation/baselines:** add rejection/down-weighting and whole-update/robust aggregation comparisons.
6. **Ablation/sensitivity/multiple seeds:** run only after all components are stable.
7. **CIFAR-10:** after MNIST pipeline functional; dataset download must be deliberate.

## Minimum matrix (paper mode)

| Factor | Levels |
|---|---|
| Dataset | MNIST; CIFAR-10 if compute allows |
| Partition | IID; Dirichlet non-IID with reported alpha |
| Clients | quick 5; paper 20 (plus sensitivity) |
| Malicious fraction | 0.1, 0.2, 0.4, constrained by valid attack assumptions |
| Attack | no attack; audited/explicitly approximate LSA; conventional fixed-trigger backdoor if feasible |
| Defense | FedAvg; whole-update detector; layer-only; temporal-only; cross-layer-only; full TCLAD-FL; feasible median/trimmed mean/Multi-Krum/FLTrust/published comparators |
| Seeds | at least 3 independent seeds where feasible; each defense shares partition and attack seed |
| Outcomes | clean accuracy, ASR, TPR/FPR, precision/recall/F1, benign acceptance, malicious rejection, ROC/PR AUC where defined, runtime/memory |

## Mandatory ablation

A: layer statistics only; B: layer + temporal; C: layer + cross-layer; D: temporal + cross-layer; E: full model. Evaluate detection measures and clean accuracy/ASR on the same held-out clients/rounds. Include uncertainty estimates. The D variant must state exactly how temporal and cross-layer signals are normalized without relying on layer-score leakage.

## Sensitivity

Vary threshold (including benign validation quantiles), attacker fraction, Dirichlet alpha/non-IID severity, attack strength, communication rounds, clients, and detector weights. Predeclare candidate ranges and selection rule. Keep final test seeds out of tuning.

## Fairness and reproducibility

Each comparison shares model initialization, client partitions, selected clients, attack trigger/target, local-training budget, and random seeds where meaningful. Every run has a unique identifier, resolved config, software/hardware versions, wall-clock duration, raw per-client/round outputs, and no overwrite. Tables/figures are generated from raw outputs only. Record unsuccessful runs and protocol deviations.


## Execution status (updated 6 October 2026)

The executable quick path is implemented. MNIST has centralized training, 20-round IID and non-IID FedAvg, three IID LSA-inspired seeds, one conventional-trigger control, one non-IID attack and one 40% attacker-fraction run. Detector-only ablation and threshold sensitivity are available; end-to-end down-weight ablation across A–E was run for seed 42. Coordinate median, trimmed mean, and Multi-Krum were each run once. A CIFAR-10 smoke test was attempted but could not download data because the host blocks Python network access; no CIFAR-10 results are claimed. The 20-client/50-round paper matrix, further fractions/severities, faithful LSA reproduction, recent defense reproductions, and inferential multi-seed analysis remain incomplete. See `FINAL_EXPERIMENT_REPORT.md` for measured results and limitations.
