# Research Report: Validation-to-next-block transfer — round 2 (decision rules vs power)

**Session ID:** `transfer-power-20261031`
**Date:** 2026-10-31
**Status:** complete — literature verified; E018, E019, E020 and E021 implemented and evaluated

## Executive summary

Round 1 (`regime-transfer-20261031`) found that no direction met the five transfer
criteria. Its most informative result was diagnostic: `E015_frozen_candidate` passed
the accuracy, block-transfer and commodity-safety checks, yet the E015 deploy gate
(last 26 mature candidate errors vs persistence, deploy when the point estimate of
the gain is positive) **reversed** the gain by deploying on noise.

This round tested the obvious repair: replace the point-estimate gate with a
decision rule that carries an uncertainty guarantee. E018 does exactly that on the
immutable E015 candidate, in three directions (betting e-process, time-uniform
confidence sequence, sequential model confidence set). All three refuse to deploy
on all six commodities (0.0% deploy), and the resulting deployed MAPE (6.194)
is **better than the E015 gate (6.218)** and identical to persistence.

E019 then pooled the same differences across the six commodities so that a rule
could trade a controlled error rate for power (online e-BH, adaptive-pick
calibration, hierarchical pooling). Its `cap` direction became the first rule in
the study to beat persistence (6.173 vs 6.194), yet 27% of its deployed
commodity-weeks were worse than persistence against a 10% target, so it fails the
predeclared contract: the aggregate gain is carried by a minority of large wins,
not by a calibrated decision rule.

Two further experiments closed the loop in the same round. E020 changed the
deployed model instead of the rule (regime-window refits, test-time adaptation) and
moved raw MAPE by up to 0.13 points, so the model-side lever is measurable — but
every direction still ended at 6.195-6.201 against persistence's 6.194, i.e.
detectably different and not useful. E021 attacked the evidence base directly by
expanding the panel from six commodities to 730 weekly series from the local
database: the validation-to-outer rank correlation became measurable at 0.263, yet
only 9 of 730 series certified an edge at alpha 0.05 and the mean gains still
disagreed in sign (validation -0.004, outer +0.003).

The measured reason is not the rule but the evidence: per-commodity weekly evidence
carries an effect of 0.02–0.12 percentage points against 1.5 pp weekly noise
(coffee: mean difference 0.00021 in capped-APE units, sd 0.0151). Certifying such an
edge at 95% needs roughly 20,000 weekly observations; the panel has 236–384. **Power,
not the decision rule, is the binding constraint**, and any rule that deploys on this
evidence is deploying noise — which is precisely what E015's gate did.

## Primary evidence (2024+, cross-field)

### Sequential decision rules (the E018 family)

- [Sequential model confidence sets](https://arxiv.org/abs/2404.18678) (Arnold, Gavrilopoulos, Schulz & Ziegel; JRSS-B 2024/2025): e-processes and confidence sequences extend Hansen's MCS to continuous monitoring with time-uniform, non-asymptotic coverage. Directly replaces the E015 gate.
- [Sequential scoring rule evaluation for forecast method selection](https://arxiv.org/html/2505.09090v1) (Frazier & Poskitt 2025): sequential evaluation targeted at method selection rather than a single stopping test.
- [An online generalization of the (e-)Benjamini-Hochberg procedure](https://arxiv.org/abs/2407.20683) (Fischer, Xu & Ramdas 2024): SupFDR control under arbitrary dependence; the right tool when several commodities must each be deployed or not.
- [Admissible online closed testing must employ e-values](https://arxiv.org/abs/2407.15733) (Fischer & Ramdas 2024): online true-discovery guarantees; motivates e-values over p-values when decisions are made as labels mature.
- [Carefree multiple testing with e-processes](https://arxiv.org/abs/2501.19360) (2025) and [Regularized e-processes](https://arxiv.org/abs/2410.01427) (2024): efficiency and validity when data collection order is chosen adaptively.

### Selective and risk-controlled deployment

- [CAP: online selective conformal prediction with FCR control](https://arxiv.org/html/2403.07728v4) (2024/2025): selection-conditional coverage under arbitrary shift, using adaptive conformal inference.
- [Online selective conformal prediction: errors and solutions](https://arxiv.org/html/2503.16809) (2025): shows several published selection-conditional coverage claims are wrong; a caution against reusing the E016 gate logic.
- [Bellman conformal inference](https://arxiv.org/abs/2402.05203) (Yang, Huang, Candès & Lei 2024) and [AcMCP](https://arxiv.org/abs/2410.13115) (Wang & Hyndman 2024): long-run risk control under drift; the AcMCP result that h-step errors are serially correlated up to lag h−1 explains why eight overlapping horizons cannot be treated as independent evidence.

### Weather and climate (transfer under regime change)

- [Multivariate post-processing of probabilistic sub-seasonal weather regime forecasts](https://arxiv.org/html/2404.01895) (2024): EMOS plus copula coupling; regime-index calibration buys 1.2 days of skill over lead-time bias correction.
- [Statistical post-processing of AI weather models with IMPROVER](https://arxiv.org/html/2504.12672v2) (2025): a rolling 30-day recalibration transfers to a new model generation without redesign; blending beats the best single component even when it is not the best member.
- [Lead-time-continuous post-processing](https://doi.org/10.1002/qj.4701) (Wessel, Ferro & Kwasniok 2024) — already used in E008; retained as the prior for horizon-pooled parameters.
- [Stress-testing hybrid physics-ML climate simulations](https://arxiv.org/html/2401.02098) — already used in round 1 as analogical evidence.

### Physics-informed and scientific ML

- [MetaPhysiCa](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f0e5cde3850e7dd0db125c0ebae16680-Abstract-Conference.html) (ICLR 2024): meta-learning of causal structure gives 2×–28× lower OOD error than ERM-style PIML; a model-side route to regime robustness.
- [Online test-time adaptation for interatomic potentials (TAIP)](https://doi.org/10.1038/s41467-025-57101-4) (Nat Commun 2025): self-supervised test-time updates cut error ~30% with no extra data and prevent simulation collapse.
- [PINN extrapolation via transfer learning and adaptive activations](https://arxiv.org/abs/2507.12659) (2025): transfer within an extended training domain improves extrapolation ~40% without more compute.

### Representation learning and regime experts

- [OOD generalization in time series: a survey](https://arxiv.org/abs/2503.13868) (2025): maps static-robustness, adaptive and large-model families; useful to avoid re-running FOIL-style ablations.
- [TARDIS: mitigating temporal misalignment via representation steering](https://arxiv.org/html/2503.18693v2) (2025): unsupervised steering vectors realign representations to a new period without weight updates.
- [TFPS: pattern-specific experts under patch-level distribution shift](https://arxiv.org/html/2410.09836v2) (2024): clustering-based mixture of experts beats single models under severe drift.
- [DeCoP](https://arxiv.org/html/2509.14642) (2025): instance-wise patch normalization plus dependency-controlled pretraining for drift-robust representations.

## Implemented this round: E018

Immutable input: E015 frozen `y_cand` (no new training). Decision unit: commodity,
with the eight horizons averaged into one bounded observation per target week, so
the e-process and confidence-sequence assumptions still hold.

Run on Colab via one CPU pane per direction (`colab` CLI), artifacts downloaded
and re-validated locally before the leaderboard import.

| Direction | Rule | Deploy rate (6 commodities) | Deployed MAPE | Candidate MAPE | Persistence | E015 gate |
| --- | --- | --- | --- | --- | --- | --- |
| `eproc` | betting e-process, deploy iff E ≥ 1/α | 0.000 | **6.194** | 6.208 | 6.194 | 6.218 |
| `cs_weight` | time-uniform CS turned into weight clip(LB/0.01,0,1) | 0.000 | **6.194** | 6.208 | 6.194 | 6.218 |
| `smcs` | sequential model confidence set, deploy iff persistence excluded | 0.000 | **6.194** | 6.208 | 6.194 | 6.218 |

Reading: every calibrated rule declines to deploy, which avoids the 0.024 MAPE harm
the E015 gate inflicted (6.218 → 6.194), but captures none of the candidate's genuine-looking edges
(coffee 7.870 vs 7.891; gold 3.551 vs 3.625) because they are indistinguishable from
noise at this sample size. Refusing to act is the correct behaviour, not a failure
of the rule.

## Implemented this round: E019

Same immutable candidate, decision unit = commodity-week, eight horizons averaged.

| Direction | Rule | Deploy rate | Deployed MAPE | Deployed-worse rate | Verdict |
| --- | --- | --- | --- | --- | --- |
| `cap` | recent-26 pick with adaptive calibrated cutoff | 0.105 | **6.173** | 0.27 (target ≤ 0.10) | beats persistence, **fails the FDR contract** |
| `ebh` | e-BH over six per-commodity e-values | 0.000 | 6.194 | n/a | no evidence to reject at 0.10 |
| `pool` | pooled e-process, then Bonferroni split | 0.000 | 6.194 | n/a | pooled evidence never passes |
| candidate | always deploy | 1.000 | 6.208 | — | worse than persistence |
| naive | persistence | — | 6.194 | — | — |

Reading: `cap` improved aggregate MAPE by 0.021 while deploying only 10.5% of
commodity-weeks, which is the first positive signal of the study. It is not
admissible under the round-2 contract because the deployed-worse rate is 2.7x the
target. The next rule must control risk directly (e.g. conformal risk control on
the per-week loss difference) instead of the quantile of recent means; a
post-hoc quantile threshold is a picker, not a guarantee.

## Implemented this round: E020 model-side adaptation

Same protocol as E014/E015 (six commodities, expanding 80% origins, 26-week refits,
ERM penalty zero), changing only what the deployed model is trained on.

| Direction | Rule | Deployed MAPE | Candidate | Raw |
| --- | --- | --- | --- | --- |
| `base` | refit on every mature row (E015-fresh control) | 6.195 | 6.206 | 6.628 |
| `recent` | refit on the last 156 mature rows only | 6.201 | 6.227 | 6.757 |
| `tta` | refit on all rows, then continue 5 epochs on the last 52 | 6.196 | 6.206 | 6.626 |
| naive | persistence | 6.194 | — | — |

Reading: the model-side lever is real and measurable (raw MAPE spans 6.626-6.757),
unlike the decision-rule lever, but no direction beats persistence. `recent`
training is the worst of the three on both candidate and raw, which is evidence
against the "recent regime is more relevant" assumption on this panel.

## Implemented this round: E021 panel expansion

Every weekly-resolvable price series in the local `commodity` database with at
least 250 weeks of history: **730 series**. Per series, the repository's expanding
walk-forward protocol with a cheap multi-output ridge candidate (four lagged weekly
returns plus a 52-week level z-score), internal validation = three expanding folds
inside the first 80%.

| Direction | Deploy rate | Series certified at alpha=0.05 | Validation-to-outer rank correlation |
| --- | --- | --- | --- |
| `ridge` (per series) | 0.0026 | 9 / 730 | **0.263** |
| `ebh` (cross-series e-BH, 0.10) | 0.000 | 9 / 730 | 0.263 |
| `pooled` | 0.000 | 9 / 730 | 0.263 |
| mean gain | — | validation **-0.0038**, outer **+0.0028** | — |

Reading: expanding the panel by 120x makes the validation-to-outer relationship
measurable (rank correlation 0.263 instead of ~0 across six commodities) — the
first genuinely positive methodological result of the study — but the effect is
still too small to certify: 1.2% of series reject the no-edge null, and the mean
gains disagree in sign, which is what a near-zero true effect looks like. The
cross-series e-BH cannot fire at all with 730 units and e-values of order 1-2,
because its smallest threshold is m/alpha = 7300.

## Predeclared success contract

A direction is accepted only if, on common origins:

1. Deployed MAPE is below both the matched E014/E015 control and persistence.
2. Block-level next-block win rate exceeds 50%, computed per block, not per
   overlapping horizon cell.
3. Deploy coverage stays at or above 10% of blocks.
4. No commodity worsens by more than 0.05 MAPE against its matched control.
5. Every statistic is invariant to mutation of labels whose target dates are later
   than the forecast origin.

Failure of any item is reported as non-transfer, even if aggregate MAPE improves.

## Limitations

- The outer interval has been inspected across 18 experiments; nothing here is a
  fresh holdout.
- E014's forced `dt`-lag-0 sheet variables remain a timing risk.
- Averaging eight overlapping horizons into one weekly observation trades validity
  for power; a conservative reading must treat the effective sample size as smaller
  than the count suggests.
- A power argument derived from this panel does not generalise to panels with more
  commodities or longer histories; the conclusion is conditional on six weekly
  series.
