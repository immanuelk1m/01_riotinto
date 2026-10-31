# Research Report: Validation-to-next-regime transfer

**Session ID:** `regime-transfer-20261031`  
**Date:** 2026-10-31  
**Status:** Complete; no direction met the full predeclared transfer contract

## Executive summary

The 2024+ literature does not support treating a single validation win as evidence of improvement in an unseen temporal regime. The strongest applicable work instead changes one of three boundaries: it estimates risk across multiple historical transitions, adapts only after delayed labels mature, or separates a shared predictor from a small regime-specific component.

The repository audit found an additional local confound: E013/E014 evaluate one model and scaler on the gate, then discard it and deploy a newly seeded refit model with a recomputed scaler. A failed gate-to-test transfer therefore mixes market change with model identity change. E015 isolates that boundary before adding more complex adaptation.

## Primary evidence

- [Model Assessment and Selection under Temporal Distribution Shift](https://arxiv.org/abs/2402.08672) motivates adaptive historical risk windows, but its current-period labeled-data setting is not a next-block guarantee.
- [D3A](https://arxiv.org/abs/2403.14949) detects mature forecast-error deterioration before adapting; only its delayed-feedback evidence is admissible here.
- [Proceed](https://arxiv.org/abs/2412.08435) directly identifies the horizon-induced gap between the current input and the newest fully labeled training sample.
- [FOIL](https://arxiv.org/abs/2406.09130) shows that time-series OOD generalization needs temporal environment inference and a surrogate objective; E013's scalar environment-risk variance is not a full reproduction.
- [SKADA-Bench](https://arxiv.org/abs/2407.11676) shows that realistic source-only model selection can erase much of an adaptation method's reported gain.
- [RainShift](https://arxiv.org/abs/2507.04930) provides evidence for past-only distribution alignment, while also reporting domains where the same alignment hurts.
- [GEPS](https://arxiv.org/abs/2410.23889) and [Neural Context Flows](https://arxiv.org/abs/2405.02154) separate shared dynamics from small environment-specific contexts and evaluate on trajectories distinct from adaptation data.
- [Stress-testing coupled hybrid climate simulations](https://arxiv.org/abs/2401.02098) shows that model rankings in the current climate need not persist in a warmer climate.
- [Online Linear Regression in Dynamic Environments via Discounting](https://arxiv.org/abs/2405.19175) supports delayed, discounted lightweight correction rather than full-model refitting.

## Rejected evidence patterns

- Target-domain labeled validation or oracle `t-1` labels.
- Updating an already issued forecast after partial future labels arrive.
- Random temporal train/validation/test splits used as evidence of next-block transfer.
- Coverage, regret, or best-model-set guarantees presented as point-MAPE guarantees.
- Selecting a direction or hyperparameter after reading the same outer result.

## Selected experiments

### E015 — model-consistent gate transfer

Compare the current fresh-refit deployment with the exact gate-evaluated model and a warm continuation of that same model. This isolates the repository's model/scaler identity change.

### E016 — causal delayed affine adapter

Fit horizon-specific scale/bias corrections from only forecasts whose target dates have matured. Compare 52-week, 26-week, and mature-error drift-triggered windows. This is a lightweight D3A/Proceed/GEPS-inspired ablation, not a reproduction.

### E017 — past-only rolling quantile alignment

Compare identity input processing with 52- and 104-week past-only quantile alignment before GRU prediction. The transform cannot read a future input row or target.

## Predeclared success contract

A direction is accepted only if all of the following hold on common origins:

1. Shared-five deploy MAPE is below both the matching E014 control and persistence.
2. Blocks with positive internal validation gain have a greater than 50% next-block win rate, computed at the block level rather than treating overlapping horizons as independent.
3. The improvement is not explained by near-total persistence fallback; deploy coverage must remain at least 10% of blocks.
4. No commodity worsens by more than 0.05 MAPE points relative to its matching control.
5. All updates are invariant to mutation of labels whose target dates are later than the forecast origin.

Failure of any item is reported as non-transfer, even if aggregate candidate MAPE improves.

## Colab experiment results

All nine directions completed in independent tmux panes. Metrics below use the shared-five commodity set and aggregate each commodity/block over all eight horizons before computing the transfer rate.

| Experiment | Direction | Deploy MAPE | Candidate MAPE | Validation-win next-block rate | Deploy block coverage | Worst commodity deploy delta | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| E015 | fresh | 6.086963 | 6.083137 | 37.5% | 12.9% | +0.0000 | Fail: no deploy improvement; transfer below 50% |
| E015 | frozen | 6.114062 | 6.078799 | 54.2% | 12.9% | +0.0899 | Fail: deploy gate reverses candidate gain |
| E015 | warm | 6.176649 | 6.103090 | 45.8% | 12.9% | +0.3776 | Fail |
| E016 | static52 | 6.291040 | 6.494221 | 42.1% | 91.9% | +0.4493 | Fail |
| E016 | recent26 | 6.264191 | 6.571109 | 29.4% | 88.7% | +0.4045 | Fail |
| E016 | drift26 | 6.317238 | 6.508784 | 42.1% | 88.7% | +0.4623 | Fail |
| E017 | identity | 6.086963 | 6.083137 | 37.5% | 12.9% | +0.0000 | Control reproduced exactly |
| E017 | qmap52 | 6.090719 | 6.114334 | 26.3% | 9.7% | +0.0877 | Fail |
| E017 | qmap104 | 6.074828 | 6.085102 | 45.0% | 11.3% | +0.0812 | Fail: mean improves, transfer and safety do not |

Matching E014 ERM deploy MAPE is 6.086963 and persistence is 6.084932. The fifth contract item, future-label mutation invariance, passed for E015-E017. E016 is a deterministic causal replay and has no GRU training stage.

## Bottleneck diagnosis

The strongest finding is E015 frozen. Reusing the exact gate-evaluated model and scaler produces a candidate MAPE of 6.078799, a 54.2% next-block win rate among validation-win blocks, and a worst commodity candidate delta of only +0.0398 MAPE points. Those candidate-level values meet the accuracy, transfer, and commodity-safety thresholds.

The existing deploy gate then selects only 8 of 62 shared-five blocks. Those deployed blocks win only 50.0% of the time and have mean next-block relative MAPE change of +0.49%, while the 16 validation-win blocks rejected by the significance gate win 56.3% of the time and improve next-block relative MAPE by 3.29% on average. The immediate bottleneck is therefore the current significance/deploy rule, after the model/scaler identity mismatch is removed.

Warm continuation does not fix this boundary and materially worsens coffee. Delayed affine correction also fails, so the evidence does not support adding a downstream calibrator before repairing the gate. Quantile alignment at 104 weeks improves the aggregate mean but does not make validation wins predict the next block reliably.

## Training-curve audit

- E015: all 18 commodity-direction curve files and 216 blocks were present; 0 non-finite values; every selected epoch matched the validation minimum within the configured `1e-5` early-stopping tolerance. Baseline gate histories had 53/72 early-stopped blocks, 19/72 full-epoch blocks, and 3 epoch-zero fallbacks. Mean and median selected validation-minus-train gaps were 0.0733 and 0.0265. Fresh and warm refit losses remained finite and decreased on every block; mean final/initial loss ratios were 0.867 and 0.904.
- E017: all 18 commodity-direction curve files and 216 blocks were present; 0 non-finite values. Identity reproduced E014 exactly. Qmap52 and qmap104 each had two numerically near-tied selected epochs whose logged minimum differed by less than `1e-5`; this is consistent with the configured minimum-improvement tolerance. Their mean selected validation-minus-train gaps were 0.0889 and 0.0884 on the shared-five set.
- These curves show stable optimization and working early stopping. The principal failure is temporal selection/deployment transfer, not numerical divergence or uncontrolled within-block training overfit.

## tmux allocation

- `E015-model-consistency`: panes `fresh`, `frozen`, `warm`
- `E016-delayed-adapter`: panes `static52`, `recent26`, `drift26`
- `E017-qmap`: panes `identity`, `qmap52`, `qmap104`

All panes exited with code 0 and remain open at their shell prompts. All Colab sessions were stopped after local artifact validation.

## Verification evidence

- `uv run python experiment/E015/test_e015.py`
- `uv run python experiment/E016/test_e016.py`
- `uv run python experiment/E017/test_e017.py`
- Python compile checks, `bash -n` for all pane launchers, and `git diff --check`
- Final Colab state: zero active sessions; balance 46.87 compute units

## Limitations

- E014's outer interval has already been inspected repeatedly; these are retrospective benchmark experiments, not a new untouched production holdout.
- Forced `dt`-lag-0 sheet variables remain non-causal timing risks.
- Physics and climate results are analogical evidence, not financial-market guarantees.
- No paper guarantees improvement under every unseen regime; the experiments must establish transfer empirically.

[PROMISE:RESEARCH_COMPLETE]
