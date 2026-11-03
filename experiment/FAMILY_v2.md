# Re-registration — E023/E024 under the constant-free pipeline (Gate C)

**Registered:** 2026-11-03, before the re-run.
**Protocol:** [`PROTOCOL.md`](../PROTOCOL.md) v1.1, including the new Gate F.
**Supersedes for this panel:** [`FAMILY.md`](FAMILY.md) (the 25-direction E022–E026 round).
**Pipeline:** `PARAMS_VERSION = 2.0`, `PARAMS_HASH` in `_exo/params.py`, evaluator `cell-v2`.

## 1. Why the family changed

The E022–E027 round was audited against Gate F and 20 of its 25 directions were
removed together with all 21 legacy experiments (E001–E021). The rule, stated once:

> A direction may be ranked against persistence only if its forecast comes from a
> model whose definition is fixed by the protocol or by a closed form with **no free
> parameter**.

This is a statement about model definitions, not about results. It was applied to the
whole repository in one pass and it removed, in the same commit:

- the round's nominal winner `E024 gbrt_agri` (5.711 vs 6.076, p = 0.084) — five
  hand-set boosting constants;
- its nominal runner-up `E024 combo_agri` (5.920, p = 0.087) — an unvalidated
  equal-weight blend;
- its worst direction `E026 mlp_broad` (MAPE 50.23) — a diverged network;
- the program's best legacy number `E019 cap` (6.160) — an ACI rule with a hand-set
  step size `γ = 0.10`;
- every legacy experiment, because E001–E008 run LightGBM (with a genetic feature
  search, a top-`k` screen or a ridge grid), E009/E010/E013/E015/E020 are GRU cores,
  E011/E012/E014/E016/E017/E018 post-process those forecasts, and E021 is a ridge
  penalised at a constant `λ = 1.0`.

That the rule deletes the best and the worst directions at once is the evidence that
it is not results-driven. The full list is in [`RETIRED.md`](RETIRED.md).

**Retrospectivity (§Gate C 4.4).** This panel was already inspected under v1. The
re-registration below is retrospective for the *direction list*, and says so; what is
new is the pipeline. No direction was added after a result was seen, and no direction
survives because it scored well.

## 2. The registered family — m = 5

| experiment | direction | targets | exogenous | model |
|---|---|---|---|---|
| E023 metals | `adl_real_yield` | gold, silver | real_yield, dollar, vix | ADL (closed-form OLS) |
| E023 metals | `adl_basis_macro` | copper_cash, aluminium, nickel | copper_3m, dollar, bdi | ADL |
| E024 agriculture | `adl_parity_fx` | corn, soybean, wheat | brl, dollar | ADL |
| E024 agriculture | `adl_enso` | coffee_r, sugar, rice | oni, soi | ADL |
| E024 agriculture | `adl_rain` | corn, soybean, wheat, rice | rain, crude | ADL |

`adl_rain` was previously named `arimax_rain`; it always fitted an OLS, and Gate F 7.5
requires the name to say so.

Two v1 defects are repaired rather than inherited:

- `zinc` is dropped from `adl_basis_macro`. The v1 regex resolved nothing for it and
  the panel never contained a zinc series (the database has zinc *inventory* and a
  World Bank monthly price, not a daily LME cash price). Recorded under `unavailable`
  in `E023/config.json`.
- `copper_3m` was listed in that direction's exog set but was never prepared into the
  exog panel, so the basis regressor silently never loaded. It is now prepared
  explicitly and the direction runs the information set that was originally
  registered.

Family size **m = 5**, so Holm at α = 0.05 gives α₁ = 0.0100.

## 3. Detection floor of the registered design

Estimated from the v1 development windows of the same panels (`sd(d)` of the paired
capped-APE difference, 80 % power):

| direction | cells | series | origins | blocks | MDE (d) | MDE (MAPE pp) |
|---|---|---|---|---|---|---|
| `adl_real_yield` | 1024 | 2 | 64 | 11 | 0.00130 | 0.13 |
| `adl_basis_macro` | 1178 | 3 | 84 | 15 | 0.00173 | 0.17 |
| `adl_parity_fx` | 1536 | 3 | 64 | 11 | 0.00131 | 0.13 |
| `adl_enso` | 1096 | 3 | 138 | 21 | 0.00265 | 0.27 |
| `adl_rain` | 1820 | 4 | 163 | 17 | 0.00155 | 0.16 |

The design is registered as **underpowered for effects below 0.13–0.27 MAPE points**.
An interval that includes zero here is "no detectable difference", not "no effect".
The v2 pipeline changes the design matrix (expanding z instead of a 52-week window,
no 4-week difference, no forecast clip), so these floors are an estimate and are
recomputed from the new cells in the report.

## 4. Decision rule (unchanged, Gate A)

A direction is **eligible** only if, on the development window, the mean paired
difference (persistence − candidate, capped APE) is positive with a 26-week
calendar-block bootstrap 95 % interval excluding zero, **and** the smallest p-value
survives Holm at α₁ = 0.0100 across the five.

Everything else is reported as a loss or as no detectable difference. Three of the
five directions lost on the v1 cells by more than their own MDE (`adl_enso` −0.293,
`adl_rain` −0.212, `adl_basis_macro` −0.106 MAPE points), so a null result here is the
expected outcome and is not evidence of a broken pipeline.

## 5. Sealed block

The most recent 52 origins of each panel are sealed for every direction: excluded
from the eligibility computation and from the leaderboard, and reported once in each
`results/<direction>_audit.md`.

## 6. What this re-run can and cannot establish

- It **can** establish the first leaderboard whose every row is produced by a
  parameter-free model under registered constants — a property the previous 12,485
  rows do not have.
- It **cannot** establish that exogenous information is useless: with m = 5 and a
  floor of 0.13–0.27 MAPE points, only an effect larger than that is visible.
- It **cannot** be used to argue for `gbrt`/`combo` by pointing at the v1 numbers.
  Those rows are `retired-heuristic` and are not comparable to anything.
