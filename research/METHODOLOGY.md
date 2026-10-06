# Methodology

## Problem formulation and system model

At round `t`, the server broadcasts global parameters `W^t`; client `i` returns `W_i^t` after local training. The update is `ΔW_i^t = W_i^t - W^t`, decomposed by named trainable layer `l` as `ΔW_{i,l}^t`. A client-round is the detection unit. The server sees individual updates and prior observed updates, but not private examples or labels. This visibility assumption is incompatible with secure aggregation.

## Threat model and attack protocol

Experiments use simulated local clients only. A configurable fraction poison a subset of local examples by setting a bottom-right square trigger and target label. ASR excludes samples whose clean label already equals the target class. The seed LSA paper (IEEE ICC 2026) performs clean/poison local training, layer substitution to rank backdoor-critical layers, compact critical-layer selection to a τ·BSR criterion, fine-tuning of selected layers, and a benign-average-smoothed layer mask.

This prototype does **not** reproduce that full procedure. Its `lsa_inspired` approximation uses a configured layer mask, local trigger poisoning, scales selected-layer deltas with λ, and substitutes a benign-client mean delta for unselected layers. The current late schedule attacks in rounds 16–20 after a clean warm-up. Manual layer selection and different model/dataset are important deviations. Do not label this exact LSA.

## Layer fingerprint

For flattened update vector `x = ΔW_{i,l}^t`, compute mean, standard deviation, median, min, max, L1/L2 norms, maximum absolute value, zero sparsity, skewness, and excess kurtosis. Degenerate skew/kurtosis return zero. Features are emitted in long format with ground-truth metadata in a separate column. Ground truth is dropped before score calculation.

A benign reference stores per-layer/feature medians and robust scales from fit rounds of an independent attack-free run. Scale is estimated as 1.4826·MAD; when MAD is zero, the implementation falls back to IQR/1.349, then sample standard deviation, then a small center-relative floor. Robust score is `|x - median| / scale`.

## Temporal features

For each client/layer/feature observed at ordered rounds, compute first and absolute differences, elapsed observed-round gap, rolling mean/std over the latest three observations, temporal variance, trend slope against actual round indices, EWMA, reference-based deviation persistence, and consecutive deviation duration. Missing rounds are not imputed. Persistence/duration compare robust scores with a configured three-scale cutoff.

## Cross-layer features

Layer parameter tensors can have different shapes, so direct elementwise cosine is not used. Each layer is summarized in shared scalar statistic coordinates (mean, std, L1/L2, max abs, sparsity, skewness, kurtosis). For each available pair, compute summary cosine, normalized summary distance, and L2 magnitude ratio. Pairwise feature deviations use benign pair-specific reference values.

## Scoring and thresholding

For each feature, compute the absolute robust deviation `z=|x-center|/scale`, then map it to `g(z)=z/(1+z)`. This bounds every feature contribution to `[0,1)` so a zero-MAD fallback or one extreme temporal feature cannot overwhelm the other components. Component scores are means of these bounded deviations over layer, temporal, or pairwise features. The combined score is a configurable weighted sum after normalizing nonnegative weights to sum to one:

`A_TCLAD = w_L·A_layer + w_T·A_temporal + w_C·A_cross`.

The validation threshold is the selected benign validation-score percentile (95% in current runs). The fit rounds, threshold rounds, and apply-from round are stored. No final-test label is used by score/threshold functions. `score_tclad` outputs component scores, per-layer scores, most-suspicious layers, overall score, and prediction.

## Mitigation and aggregation

Reject removes updates with score at/above threshold. Down-weight scales their effective example count by `min(1, threshold/score)` with a 0.05 floor. Unflagged clients retain weight 1. If all clients are rejected, the server keeps the current global model for that round (fail-closed no-op) and logs `all_rejected_noop`; it does not aggregate flagged updates. This can stall learning and is reported explicitly. Decisions are saved per client/round. Compare no defense, reject, and down-weight under identical seeds and partitions.

Flower FedAvg performs example-count-weighted averaging through its `FedAvg.aggregate_fit`. Coordinate median, trimmed mean, and Multi-Krum are separately implemented experimental baselines, with assumptions/deviations described in code and citations.

## Evaluation and statistical protocol

Clean accuracy is measured on untouched test data. ASR is the fraction of eligible triggered non-target test samples classified as target. Detection metrics use client-round labels only after prediction. Current summaries are exploratory because repeated rounds for the same five clients are correlated. Journal claims require multiple independent seeds and paired run-level confidence intervals/tests, not treating client-rounds as independent. Current results show threshold drift and high benign rejections; this is a failure finding, not detector evidence.

Formally, for triggered test examples \(D_{trig}\) and target class \(y^*\), the eligible set is \(E=\{(x,y)\in D_{trig}:y\ne y^*\}\), and \(ASR=|E|^{-1}\sum_{(x,y)\in E}\mathbf{1}[f(x_{trig})=y^*]\). Clean accuracy is computed separately on unmodified test inputs.
