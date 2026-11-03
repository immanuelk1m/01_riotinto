# Sensitivity of the declared constants

**Status: pending the E023/E024 re-run.** Gate F 7.3 requires a `declared` constant that
could decide a conclusion to be shown not to matter. This file records the grid that is
registered *before* the numbers exist, so the analysis cannot be chosen after the fact.

## Registered grid

`LAGS` is the only declared constant that changes the design matrix, so it is the one
that gets a swing. Everything else in `_exo/params.py` is either a protocol clause
(`CAP`, `BLOCK`, `SEALED_ORIGINS`, `ALPHA`, `MIN_COVERAGE`, `SAFETY_TOL`), forced by the
object (`H`, `Z_MIN_OBS`, `PCT`, `SEED`) or a truncation rule that only decides which
series are eligible at all (`MIN_WEEKS`, `MAX_WEEKS`, `MIN_TRAIN`, `TRAIN_FRAC`,
`REFIT_EVERY`, `BOOT`) — those are reported as fixed design, not as sensitivity.

| run | `LAGS` | expectation |
|---|---|---|
| registered | 4 | the reported result |
| short | 2 | direction of the effect must not flip |
| long | 8 | direction of the effect must not flip |

Panels: E024 agriculture, three directions (`adl_parity_fx`, `adl_enso`, `adl_rain`).
Nine runs, executed on Colab under the same `PARAMS_VERSION = 2.0` registry.

## What will be reported

For each direction and each `LAGS` value: `n_cells`, mean paired difference, the 26-week
block interval, and the eligible flag. A constant "matters" if the sign of the mean
difference flips between the registered value and a swing value, or if the eligibility
verdict changes.

If the sign flips, that is the finding: it means the result is a statement about the lag
length rather than about the exogenous information, and it is reported as such.

## Preview (local, not authoritative)

A local run of the three E024 directions with `LAGS = 4` — identical data, a different
pandas build from the Colab VM — gives:

| direction | cells | MAPE persistence → candidate | mean d | 95 % interval | eligible |
|---|---|---|---|---|---|
| `adl_enso` | 1074 | 6.438 → 6.499 | −0.00061 | [−0.00337, +0.00256] | no |
| `adl_parity_fx` | 1536 | 5.659 → 5.561 | +0.00098 | [−0.00081, +0.00298] | no |
| `adl_rain` | 1792 | 5.639 → 5.695 | −0.00056 | [−0.00310, +0.00193] | no |

These are a smoke test of the pipeline, not a result: they are not in the leaderboard and
must not be quoted. The authoritative numbers come from the Colab re-run.
