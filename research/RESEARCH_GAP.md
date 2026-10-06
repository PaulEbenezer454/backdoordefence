# Research Gap and Positioning

## Evidence-based provisional conclusion (6 October 2026)

The initial targeted literature review does **not** support claiming that TCLAD-FL is the first layer-aware or temporal federated backdoor defense. ANODYNE (2025, Expert Systems with Applications) already analyzes low-dimensional update sub-vectors with spatial-temporal metrics and anomaly scoring. FedSurrogate (2026 preprint) uses layer-criticality selection and adaptive anomaly detection. FedVigil (online-first 2026, Future Generation Computer Systems 2027 volume) adds penultimate-representation distance, clustering, and temporal trust-weighted aggregation. These works create direct overlap with the broad framing “temporal + layer-wise anomaly detection.”

The original LSA paper itself argues for layer-aware detection and evaluates attack behavior against existing defenses. Thus, the project’s most defensible initial position is an **LSA-specific controlled evaluation of layer, temporal, and explicit cross-layer relational features**, with matched whole-update and existing-defense baselines, ablations, and honest negative-result reporting. The review has not established that the explicit cross-layer component is novel; it is a hypothesis to test.

## Candidate gap (not a novelty claim)

The reviewed sources do not establish whether explicit pairwise/relational weight-layer update behavior adds useful held-out detection information against LSA beyond (a) whole-update anomaly scores and (b) layerwise temporal features such as those used in ANODYNE. Nor do they settle the effect under matched IID and non-IID client partitions, varying attacker fractions, and equal threshold-selection protocols. This creates a practical empirical question suitable for a student prototype. A broader, systematic search and full-text comparison are required before claiming this is an unaddressed literature gap.

## Research questions to retain

- RQ1: Do layer-wise fingerprints contain out-of-sample signal for identifying LSA clients?
- RQ2: Does temporal information add signal beyond same-round layer features?
- RQ3: Does explicit cross-layer relational analysis add signal beyond layer-wise temporal features?
- RQ4: How does the combined detector compare with a whole-update baseline and feasible published defenses under identical partitions and attack protocols?
- RQ5: How do heterogeneity, malicious fraction, and detector calibration affect false positives, clean accuracy, and ASR?
- RQ6: What computation and storage overhead does the analysis introduce?

## Overlap risks and response

| Risk | Consequence | Required response |
|---|---|---|
| ANODYNE overlaps on layer/sub-vector plus spatial-temporal anomaly detection | Combining these ingredients cannot be presented as novel | Extract the ANODYNE algorithm from the full paper; implement if reliable and compare, or document a precise protocol limitation. Test incremental contribution of explicit cross-layer relations. |
| FedSurrogate overlaps on layer criticality/adaptive detection | Layer selection or anomaly scoring may duplicate known ideas | Treat as a recent defense comparator, not inspiration for an unqualified novelty claim; verify preprint status and assumptions. |
| LSA uses backdoor-critical layers | Layer-aware detector may share the attack’s representation and could be adaptively evaded | Evaluate selected-layer ablations, matched non-LSA attacks, and false positives under heterogeneous benign training. Avoid assuming layer-specific separability. |
| Non-IID client drift resembles malicious deviations | Thresholds can reject useful benign clients | Use held-out benign validation, report FPR and benign acceptance, never tune on test labels. |
| Paper metadata and evidence are time-sensitive | Publication status, code, and related work can change | Refresh search before submission; distinguish proceedings publication from preprint claims. |

## Go/no-go criterion for the proposed detector

Continue as a detector contribution only if a pre-specified validation procedure and independent test seeds show that cross-layer features add measurable performance or utility over the layer-temporal baseline, while maintaining acceptable false-positive rate, clean accuracy, and runtime. Use confidence intervals and an appropriate paired analysis. If not, report the null/negative result and reposition the outcome as an LSA evaluation/benchmarking study. Do not select weights or thresholds based on the final evaluation set.

## Initial research contribution wording

> “We conduct a controlled study of layer-wise, temporal, and explicit cross-layer update statistics for detecting LSA behavior in federated learning, comparing their incremental value against whole-update and relevant published baselines under matched IID and non-IID protocols.”

This is a statement of study scope, **not an established novelty or efficacy claim**. Final contribution wording must await literature expansion and executed experiments.

## Environment and implementation gate

The Windows environment now has a working Python 3.14.6 virtual environment with CPU PyTorch and Flower, and centralized MNIST plus short IID/non-IID federated runs have executed. See `ENVIRONMENT.md`. This validates the local quick path, not full-scale or GPU performance.

