# Leaderboard — exogenous-variable round

Generated from W&B artifact `riotinto/ledger:latest` by `tools/leaderboard_report.py` at 2026-11-10T00:31:14. Do not hand-edit.

## What may be ranked

Only `status = 'ok'` rows are compared to anything. A row is `ok` when its forecast
comes from a model with no free parameter — the naive forecast, or closed-form ADL — produced
under registered constants (`_exo/params.py`). Rows that a hand-set constant or an
unregistered pipeline produced (the v1 `legacy` pipeline) were removed from the ledger;
see `experiment/RETIRED.md` for the rule.

## Row accounting

| status | rows | meaning |
|---|---|---|
| `ok` | 360 | rankable: a candidate whose model has no free parameter |
| `retired-heuristic` | 0 | model carries an unregistered free parameter, or the v1 pipeline produced it — kept for the record, never compared |
| `invalid` | 0 | the stored metric is not a MAPE (negative value, or `n = 0` cells) |
| `diverged` | 0 | the fit blew up; the number is an artefact of divergence |
| **total** | **360** | |

## Rankable candidates

| experiment | direction | n_series | params_hash | rows |
|---|---|---|---|---|
| `E022_adl_gpr_vix` | `adl_gpr_vix` | 5 | `7b9c91677bdc2917` | 45 |
| `E022_adl_stocks_cot` | `adl_stocks_cot` | 5 | `7b9c91677bdc2917` | 45 |
| `E023_adl_basis_macro` | `adl_basis_macro` | 3 | `7b9c91677bdc2917` | 27 |
| `E023_adl_real_yield` | `adl_real_yield` | 2 | `7b9c91677bdc2917` | 18 |
| `E024_adl_enso` | `adl_enso` | 3 | `7b9c91677bdc2917` | 27 |
| `E024_adl_parity_fx` | `adl_parity_fx` | 3 | `7b9c91677bdc2917` | 27 |
| `E024_adl_rain` | `adl_rain` | 4 | `7b9c91677bdc2917` | 36 |
| `E026_adl_broad` | `adl_broad` | 7 | `7b9c91677bdc2917` | 63 |
| `E027_adl_cc` | `adl_cc` | 2 | `7b9c91677bdc2917` | 18 |
| `E028_struct` | `struct` | 6 | `7b9c91677bdc2917` | 54 |
