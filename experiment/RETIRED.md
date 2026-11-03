# Retired experiments (2026-11-03)

## The rule

> A direction can be ranked against persistence only if its forecast comes from a
> model whose definition is fixed by `PROTOCOL.md` or by a **closed form with no free
> parameter**. Everything else is retired.

This is an *a priori* rule about model definitions, applied to every experiment at
once, without reference to how well each scored. It removes the round's nominal
winner (`E024 gbrt_agri`), its nominal runner-up (`E024 combo_agri`), its worst
(`E026 mlp_broad`) and the program's best legacy number (`E019 cap`, 6.160) in the
same pass. That symmetry is the evidence that the rule is not results-driven.

A constant is allowed to survive only if it is registered in `_exo/params.py` with
a `source` (`protocol` / `definitional` / `declared`) **before** a run, and a
`declared` one has to be shown not to matter by `docs/SENSITIVITY.md`. Nothing
already observed can be rescued retroactively, which is why no legacy experiment
survives.

## What was retired

| experiment | removed model | removed constants | reason |
|---|---|---|---|
| `E001` | LightGBM + genetic feature search | LGBMRegressor; array_split; rng.choice; GENERATIONS; PRUNE_FRAC; N_KEEP | the forecast comes from boosted trees whose depth, learning rate and tree count were hand-set, on top of a stochastic feature search |
| `E002` | LightGBM with a top-k feature screen | LGBMRegressor; TOP_K = 30 | TOP_K decides which features enter the model and was never chosen by a rule |
| `E003` | LightGBM + Lasso/LassoCV knockoff screen | LGBMRegressor; Lasso; LassoCV; array_split; rng.choice | two estimator families with free penalties screen the features before the fit |
| `E004` | LightGBM, direct/recursive/mimo variants | LGBMRegressor; TOP_K = 30 | every variant inherits the top-k screen and the boosted-tree settings |
| `E005` | Ridge over a GridSearchCV penalty grid | Ridge; GridSearchCV; alpha grid | the penalty is selected by a grid search, so the reported number is the best point of a searched family |
| `E006` | LightGBM ensembles with four weighting schemes | LGBMRegressor; n_estimators = 300/200; learning_rate = 0.05 | the boosting settings and the ensemble weights are hand-set constants |
| `E007` | LightGBM robustness variants | LGBMRegressor; max_depth; n_estimators; learning_rate | each robustness variant is a different set of boosted-tree constants, so the comparison ranks constants rather than models |
| `E008` | LightGBM loss/averaging variants | LGBMRegressor; huber/dro/global variants | the ranking across variants is decided by the loss and averaging constants |
| `E009` | GRU (torch) candidate | GRU; hidden size; learning rate; epochs; patience | the architecture and the training schedule are free parameters |
| `E010` | GRU (torch) with registration | GRU; hidden size; learning rate; epochs; patience | same GRU core as E009 with a different regularisation setting |
| `E011` | delayed EWLS correction of E010 GRU forecasts | EWLS decay and memory-horizon constants on top of E010 | it post-processes GRU forecasts, so it inherits E010's free parameters and adds its own decay constant |
| `E012` | delayed fixed-share aggregation of E009/E010 forecasts | fixed-share learning rate and delay constants | the aggregation rate is a hand-set constant applied to knob-carrying inputs |
| `E013` | invariant-risk GRU core | GRU; environment split; penalty weight | the invariance penalty and the network are both free parameters |
| `E014` | sheet-pool wrapper around the E013 GRU core | E013 constants; pooling weights | it wraps the E013 GRU core and adds pooling weights |
| `E015` | GRU pools with fresh/frozen/warm starts | GRU; start-up schedule | the start-up schedule and the network are free parameters |
| `E016` | affine calibration of E014 forecasts | recent-26 and static-52 calibration windows on top of E014 | it calibrates E014's GRU output and chooses the calibration window by hand |
| `E017` | sheet-pool wrapper around the E013 GRU core | E013 constants; pooling weights | same wrapper as E014 over the same knob-carrying core |
| `E018` | e-process with shrunk cross-sectional weights | ALPHA = 0.05; MIN_LABELS = 13; SHRINK_MIN = 13; SHRINK_FRACTION = 0.5; EFFECT_SCALE = 0.01 | five hand-set constants define the rule, including a shrinkage schedule |
| `E019` | adaptive-conformal (ACI) deployment rule | ACI_GAMMA = 0.10; LAMBDA_MAX = 0.5; MIN_LABELS = 13; RECENT = 26 | the step size of the adaptive level is a hand-set constant; it is this rule that held the program's best legacy number (6.160) |
| `E020` | GRU with the E014 snapshot | GRU; snapshot constants | it runs the GRU core against a fixed snapshot |
| `E021` | multi-output ridge over 730 series | RIDGE_LAMBDA = 1.0; LAGS = 4; MIN_WEEKS = 250 | the ridge penalty is a constant, and this single experiment owns 53 % of the leaderboard rows |

## What survives

| model | why it is allowed |
|---|---|
| `persistence` | forecasts a zero log-return; no parameter |
| `adl` | closed-form OLS on a registered design matrix; `np.linalg.lstsq`, no penalty |

The leaderboard rows that survive are the persistence baselines plus the ADL
directions of E023 and E024 re-run under `PARAMS_VERSION = 2.0`.
