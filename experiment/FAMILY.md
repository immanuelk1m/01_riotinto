# Exogenous-information round (E022–E026) — family declaration

**Registered:** 2026-10-31, before any direction was run.
**Protocol:** [`PROTOCOL.md`](../PROTOCOL.md). The full Gate C registration for the round is
[`E022/README.md`](E022/README.md); this file fixes the **family size** that Gate D requires.

## What this round is

A Gate A audit of exogenous-information candidates. Every direction is a
(candidate, persistence) pair evaluated on identical walk-forward origins with the
repository's existing cell definition. **No deployment rule is evaluated here**, so
Gate B's restriction on rule-only experiments does not bind; the axes changed are
*data* (published exogenous series with real-time availability lags) and *model*
(ADL/ARIMAX, Lasso, 3-pass regression filter, gradient boosting, forest, MLP,
combination).

## Family size: 25

| experiment | directions | count |
|---|---|---|
| E022 energy | adl_gpr_vix, adl_stocks_cot, lasso_macro, pcr3_macro, gbrt_multisource, combo_energy | 6 |
| E023 metals | adl_real_yield, adl_basis_macro, lasso_metal, pcr3_metal, gbrt_metal, combo_metal | 6 |
| E024 agriculture | adl_parity_fx, adl_enso, arimax_rain, lasso_agri, gbrt_agri, combo_agri | 6 |
| E025 risk embeddings | lasso_gpr, pcr3_gpr, gbrt_gpr | 3 |
| E026 combination / neural | adl_broad, rf_broad, combo_broad, mlp_broad | 4 |

With 25 directions, a single direction's 5% Gate A pass is **not** a finding:
Holm at α = 0.05 across the 25 is applied before anything is called eligible, and
experiments that share a target panel are additionally reported with the
per-experiment count. The previously inspected outer interval (18 experiments on
the six-commodity panel) is declared as retrospective; the new panels
(energy, metals, agriculture) have not been used for direction selection.

## Sealed block

The most recent 52 origins of each panel are sealed for every direction. They are
excluded from the eligibility computation and reported once, separately, in each
`results/<direction>_audit.md`.

## Decision rule

A direction is **Eligible** only if, on the development window, the mean paired
difference (persistence minus candidate, capped APE) is positive with a 26-week
moving-block bootstrap 95% interval excluding zero. Everything else is reported as
"no detectable difference" or as a loss, and is not described as a result.

## Leaderboard

Every direction writes `<EXP>_<direction>` rows and the direction-free baseline writes
`<EXP>_persistence` (one per series and horizon 1-8, plus an `all` row) through
`_exo/leaderboard.py`:

- `mape` - dev-window mean capped APE in percent (the number the audit reports)
- `max_ape_pct` - worst single dev-window cell in percent, uncapped, so a tail is visible
  instead of being clipped at 100 %

The sealed 52 origins are in neither column. `--limit` runs do not write the leaderboard
unless `EXO_LEADERBOARD` points at a scratch database.

## Plan revision — 2026-11-01: the panel is coffee and corn

The E022–E026 round is closed as run (25 directions, all fail Holm; reference:
[`LEADERBOARD_exogenous_2026.md`](LEADERBOARD_exogenous_2026.md)). Its per-series table left
two series with a positive, consistent candidate gap and nothing else, so **from E027 the
research panel is two series: `corn` and `coffee_r`.** No other commodity is researched until
a two-series result clears the gates.

| round | panel | directions | tests under Holm |
|---|---|---|---|
| E022–E026 (closed) | 15 series in 5 sector panels | 25 | 25 |
| E027 (open) | `corn`, `coffee_r` | 3 | 9 (direction × statement) |

Accepted costs, stated before E027 was run:

- **Power falls with the panel.** Per-statement MDE at the E027 grid is 0.0125 for coffee and
  0.0130 for corn (capped-APE units), against dev-window effects of 0.0073 and 0.0012. The
  registered stop rule in [`E027/README.md`](E027/README.md) therefore returns "no detectable
  difference" for both series. That is the honest outcome, not a failed run.
- **The panel choice is retrospective.** It was made after reading a development-window table,
  so the E027 dev result cannot carry a claim; the confirmatory read is the forward evaluation
  registered in `E027/README.md`.
- **Family size shrinks but so does the denominator of any multiple-testing story.** Nine tests
  is not "fewer chances"; it is the same evidence budget spent on two series.

A future round that widens the panel again must declare the widening in this file before
running, and must not average two-series and six-series results into one ranking
(PROTOCOL 5.3).

## Provenance

Directions are taken from the 2026 exogenous-variable literature listed in
[`E022/README.md`](E022/README.md) and in `.omx/research/exogenous-2026/`.
