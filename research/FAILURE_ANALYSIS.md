# Failure Analysis — Current Quick Experiments

This note summarizes observed failures in the corrected bounded-score implementation. All values below come from the saved 2026-10-04 results; they are exploratory client-round measurements from five clients and must not be treated as independent statistical samples.

## Detection failures

- **Non-IID missed detection:** with Dirichlet alpha=0.1, one malicious client of five, seed 42, and the 95th-percentile benign threshold, whole-update, layer-only, cross-layer-only, and full TCLAD had TPR 0.00. Full TCLAD FPR was 0.147; cross-layer-only FPR was 0.042. Cross-layer-only ROC-AUC was 0.815 despite zero detections at the selected threshold. This indicates calibration/operating-point failure, not absence of all ranking signal. The attack itself was weak (ASR 0.0459).
- **IID false positives vary by seed:** at the 95th percentile, IID seed 42 FPR was 0.126 for whole-update, 0.326 for layer-only, 0.263 for cross-layer-only, and 0.368 for full TCLAD. At seed 7, the corresponding whole-update/layer-only/cross-layer-only/full values were 0.274/0.032/0.189/0.179. The ordering reverses; two seeds do not support a stable winner.
- **Threshold sensitivity:** across calibration percentiles 90, 95, 97.5, and 99 in seed 42, full TCLAD FPR ranged 0.316–0.442, compared with 0.116–0.147 for whole-update. No tested threshold made full TCLAD better on this measure.
- **Weak and variable attack signal:** the approximate LSA-inspired ASR was 0.1618 in seed 42 and 0.0235 in seed 7 at the same configured protocol; non-IID ASR was 0.0459. Detection success against such a weak attack does not establish realistic LSA robustness.

## Mitigation failures

Reject mitigation flagged 21/25 client-round updates in attack rounds, including 16/20 benign updates. On 15 client-round records (three full rounds) every client was flagged. The implementation was corrected to fail closed and retain the previous global model for those rounds. The corrected run finished at 0.7818 clean accuracy / 0.0082 ASR, but this is one short run and the low ASR followed from freezing aggregation; it is not evidence of reliable mitigation. Down-weighting assigned mean weights 0.965 to benign and 0.679 to malicious client-round updates; final clean accuracy/ASR were 0.8193/0.0966. This is one exploratory run and the modest ASR change may be due to training variability.

## Layer and component interpretation

Per-layer heatmaps and score distributions for seed 42, seed 7, conventional-trigger, non-IID, and high-fraction cases are generated under `results/figures/`. The current feature aggregation averages bounded robust deviations across layer/temporal/cross-layer features. Component scores are interpretable, but these experiments do not isolate which specific named layer or feature causes the non-IID miss. The `suspicious_layers` field uses a separate robust-z diagnostic cutoff and can be empty even when the aggregate score crosses a threshold. Do not infer a causal layer explanation from these plots alone.

## Required follow-up before a paper claim

1. Validate detector thresholds on independent benign seeds and use per-run, not per-client-round, statistical units.
2. Compare IID and non-IID benign controls at matching training rounds and client label distributions.
3. Reproduce the seed paper's layer substitution/search/fine-tuning procedure or clearly keep the work scoped to an approximation.
4. Verify attack efficacy independently before interpreting detection rates.
5. Study the fail-closed reject policy across seeds and tune calibration on independent benign runs; all-client rejection currently pauses global learning for that round.
6. Replicate mitigation and component ablations over more independent seeds.
