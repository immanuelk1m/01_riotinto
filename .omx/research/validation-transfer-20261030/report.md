# Validation-to-Future Transfer Under Regime Shift

Date: 2026-10-30

## Decision

The E010 result isolates the current bottleneck: regularization fixed the internal
learning-curve pathology, but the improvement did not transfer consistently to the
later walk-forward evaluation. The next experiments therefore target the transfer
mechanism, not additional generic regularization.

Selected directions:

1. **E011 — delay-aware local correction.** Freeze the GRU forecast and update only
   horizon-specific low-dimensional correctors from labels whose target dates have
   already matured. Maintain several predeclared memory lengths.
2. **E012 — delayed expert aggregation.** Combine only GRU-family forecasts and the
   naive forecast with horizon-specific discounted weights. Update weights only from
   matured forecast losses.
3. **E013 — invariant-risk GRU ablation.** Add a penalty on the dispersion of losses
   across contiguous train-only temporal environments. This is a FOIL-inspired
   ablation, not a reproduction of FOIL.

All three preserve the issued-at-origin forecast. Later observations may update the
state for future origins, but never revise an already issued forecast.

## Strongest evidence

- **Delay-aware TTA audit (2026):** enforcing label release at or after the forecast
  horizon removed most published adaptation gains on four of five benchmarks; simple
  RLS/linear correctors were strong baselines. <https://arxiv.org/abs/2610.12232>
- **Proceed (2024):** formulates the gap created by horizon-delayed labels and adapts
  proactively to the current input concept. Its reported results motivate the problem,
  but do not establish commodity transfer. <https://arxiv.org/abs/2412.08435>
- **MELO (2026):** hedges several forgetting horizons rather than selecting one memory
  length from a validation split. The large load-forecast gain used ex-post realized
  temperature, while highly correlated residual tasks improved by less than 0.5%.
  <https://arxiv.org/abs/2605.06541>
- **Forecast-error change detection (2025):** monitors mean/variance changes in
  forecast errors; detection itself is not evidence of better forecasts.
  <https://arxiv.org/abs/2502.14173>
- **FOIL (2024):** learns temporally coherent environments and penalizes environment
  risk variation for OOD forecasting. Its exact normalized-horizon surrogate is not
  directly applicable to a scalar one-step target. <https://arxiv.org/abs/2406.09130>
- **Adaptive temporal model assessment (2024):** estimates current risk from a rolling
  window, but uses a labeled current-period batch and does not guarantee the next
  unseen period. <https://arxiv.org/abs/2402.08672>
- **Weather/climate transfer:** learned analog distances can improve forecasting when
  the historical library contains related dynamics, while gray-swan and warmer-climate
  studies show that aggregate in-distribution skill can hide support failure and drift.
  <https://arxiv.org/abs/2404.15419>, <https://arxiv.org/abs/2410.14932>,
  <https://arxiv.org/abs/2409.18529>

## Evaluation contract

- At origin `t`, a row for horizon `h` may update an adaptive component only when its
  `target_date <= t`; any source publication lag must be added when available.
- Hyperparameters and memory grids are fixed before the outer stream. Outer outcomes
  update only predeclared online state.
- Report the same origin/horizon rows for GRU, corrected/aggregated prediction, and
  naive. Primary metrics are MAPE and RMSE for repository compatibility; squared log
  return loss is used for causal online updates.
- Report per-horizon and per-time-block results, validation-to-next-block rank transfer,
  and worst-block loss. An internal improvement alone is not a success.
- Existing historical results are a retrospective pseudo-OOS diagnostic because prior
  experiments have already inspected the period. A prospective forecast ledger is
  still required for a genuinely untouched confirmation.

## Explicitly rejected as first-line experiments

- Immediate-label or suffix-revision TTA scored as if it were an origin-time forecast.
- Pure conformal/interval methods when the goal is point-forecast improvement.
- Change-point detection scored only by alarm accuracy.
- A large representation/foundation model before low-dimensional causal corrections
  and aggregation beat their baselines.
- Physics-labelled regularization without a defensible commodity identity.

## Completed experiment result

All directions were run through Colab CLI and downloaded locally. Commodity-mean
MAPE (unweighted mean across the five repository commodities) was:

| Direction | Mean MAPE | Delta vs naive |
| --- | ---: | ---: |
| Naive | 6.0849 | 0.0000 pp |
| E011 delay-aware correction | 6.2774 | +0.1925 pp |
| E012 adaptive GRU hedge | 6.1585 | +0.0735 pp |
| E012 fixed equal GRU mixture | 6.0731 | -0.0118 pp |
| E013 matched ERM | 6.1001 | +0.0152 pp |
| E013 invariant penalty 0.1 | 6.1008 | +0.0159 pp |
| E013 invariant penalty 1.0 | 6.1541 | +0.0691 pp |

E012's validation winner beat naive in the following 26-week block only 27.7% of
411 commodity/horizon/cutoff comparisons; median validation-to-future expert-rank
correlation was 0.120. In E013, median validation rebound stayed below 0.8% for all
three directions, so classical epoch overfitting remained controlled. Raising the
invariance penalty reduced median environment-loss variance from 0.003217 to
0.002964, but did not improve outer MAPE. Mean validation-to-test rank correlation
was -0.022 (ERM), -0.017 (0.1), and 0.030 (1.0).

Conclusion: the remaining bottleneck is not optimization overfit. It is weak and
unstable predictive signal across regimes, compounded by delayed feedback and highly
correlated GRU experts. Equal diversification produced a very small retrospective
gain; adaptive correction, recent-winner weighting, and stronger invariance did not
transfer consistently. A genuinely untouched prospective ledger is required before
promoting even the equal-weight result.

