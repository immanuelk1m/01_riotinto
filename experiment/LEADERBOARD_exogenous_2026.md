# Leaderboard — exogenous-variable round

Generated from `leaderboard.sqlite` by `tools/leaderboard_report.py` at 2026-11-03T14:56:33. Do not hand-edit.

## What may be ranked

Only `status = 'ok'` rows are compared to anything. A row is `ok` when its forecast
comes from a model with no free parameter — persistence, or closed-form ADL — produced
under registered constants (`_exo/params.py`). Every row that a hand-set constant or an
unregistered pipeline produced is kept in the same table under `retired-heuristic` and
is excluded from every table below. See `experiment/RETIRED.md` for the rule.

## Row accounting

| status | rows | meaning |
|---|---|---|
| `ok` | 315 | rankable: persistence baseline, or a candidate whose model has no free parameter |
| `retired-heuristic` | 5544 | model carries an unregistered free parameter, or the v1 pipeline produced it — kept for the record, never compared |
| `invalid` | 6590 | the stored metric is not a MAPE (negative value, or `n = 0` cells) |
| `diverged` | 36 | the fit blew up; the number is an artefact of divergence |
| **total** | **12485** | |

## Rankable candidates

| experiment | direction | n_series | params_hash | rows |
|---|---|---|---|---|
| — | — | — | — | 0 |

**No rankable candidate rows exist yet.** The v1 pipeline produced none that survive
the constant-free rule; the E023/E024 re-run under `PARAMS_VERSION = 2.0` is what
creates them. This is the honest state of the program: there is no certified
result to report.

## Persistence reference, one panel at a time (§5.3)

Series sets of different sizes are never averaged together, so each panel is its own
table. These are the only numbers a future candidate is compared against.

### `E022_persistence` — 5 series, n_series=— (v1 row)

| commodity | horizon | MAPE | n |
|---|---|---|---|
| `brent` | 1 | 3.854 | 64 |
| `brent` | 2 | 5.353 | 64 |
| `brent` | 3 | 5.930 | 64 |
| `brent` | 4 | 7.626 | 64 |
| `brent` | 5 | 8.606 | 64 |
| `brent` | 6 | 9.585 | 64 |
| `brent` | 7 | 9.690 | 64 |
| `brent` | 8 | 11.146 | 64 |
| `gas_hh` | 1 | 12.483 | 64 |
| `gas_hh` | 2 | 14.791 | 64 |
| `gas_hh` | 3 | 15.197 | 64 |
| `gas_hh` | 4 | 18.434 | 64 |
| `gas_hh` | 5 | 21.748 | 64 |
| `gas_hh` | 6 | 23.588 | 64 |
| `gas_hh` | 7 | 23.988 | 64 |
| `gas_hh` | 8 | 23.578 | 64 |
| `gasoline` | 1 | 4.360 | 64 |
| `gasoline` | 2 | 5.664 | 64 |
| `gasoline` | 3 | 6.648 | 64 |
| `gasoline` | 4 | 8.388 | 64 |
| `gasoline` | 5 | 9.425 | 64 |
| `gasoline` | 6 | 9.785 | 64 |
| `gasoline` | 7 | 10.689 | 64 |
| `gasoline` | 8 | 12.023 | 64 |
| `heating` | 1 | 4.727 | 64 |
| `heating` | 2 | 6.314 | 64 |
| `heating` | 3 | 7.816 | 64 |
| `heating` | 4 | 10.655 | 64 |
| `heating` | 5 | 11.693 | 64 |
| `heating` | 6 | 12.914 | 64 |
| `heating` | 7 | 13.187 | 64 |
| `heating` | 8 | 15.053 | 64 |
| `wti` | 1 | 3.877 | 64 |
| `wti` | 2 | 5.284 | 64 |
| `wti` | 3 | 5.980 | 64 |
| `wti` | 4 | 7.698 | 64 |
| `wti` | 5 | 8.813 | 64 |
| `wti` | 6 | 9.510 | 64 |
| `wti` | 7 | 9.634 | 64 |
| `wti` | 8 | 11.198 | 64 |

| commodity | all horizons | MAPE | n |
|---|---|---|---|
| `brent` | all | 7.724 | 512 |
| `gas_hh` | all | 19.226 | 512 |
| `gasoline` | all | 8.373 | 512 |
| `heating` | all | 10.295 | 512 |
| `wti` | all | 7.749 | 512 |

### `E023_persistence` — 5 series, n_series=— (v1 row)

| commodity | horizon | MAPE | n |
|---|---|---|---|
| `aluminium` | 1 | 2.378 | 20 |
| `aluminium` | 2 | 3.426 | 20 |
| `aluminium` | 3 | 4.393 | 20 |
| `aluminium` | 4 | 4.767 | 19 |
| `aluminium` | 5 | 5.142 | 19 |
| `aluminium` | 6 | 5.526 | 19 |
| `aluminium` | 7 | 6.991 | 19 |
| `aluminium` | 8 | 6.874 | 18 |
| `copper_cash` | 1 | 1.907 | 64 |
| `copper_cash` | 2 | 2.905 | 64 |
| `copper_cash` | 3 | 3.427 | 64 |
| `copper_cash` | 4 | 4.162 | 64 |
| `copper_cash` | 5 | 4.738 | 64 |
| `copper_cash` | 6 | 5.193 | 64 |
| `copper_cash` | 7 | 5.726 | 64 |
| `copper_cash` | 8 | 6.259 | 64 |
| `gold` | 1 | 1.587 | 64 |
| `gold` | 2 | 2.273 | 64 |
| `gold` | 3 | 2.993 | 64 |
| `gold` | 4 | 3.512 | 64 |
| `gold` | 5 | 3.505 | 64 |
| `gold` | 6 | 3.990 | 64 |
| `gold` | 7 | 4.663 | 64 |
| `gold` | 8 | 4.698 | 64 |
| `nickel` | 1 | 2.932 | 64 |
| `nickel` | 2 | 4.658 | 64 |
| `nickel` | 3 | 5.594 | 64 |
| `nickel` | 4 | 6.264 | 64 |
| `nickel` | 5 | 6.495 | 64 |
| `nickel` | 6 | 7.089 | 64 |
| `nickel` | 7 | 8.280 | 64 |
| `nickel` | 8 | 9.013 | 64 |
| `silver` | 1 | 2.685 | 64 |
| `silver` | 2 | 4.164 | 64 |
| `silver` | 3 | 5.229 | 64 |
| `silver` | 4 | 5.955 | 64 |
| `silver` | 5 | 5.840 | 64 |
| `silver` | 6 | 6.522 | 64 |
| `silver` | 7 | 7.396 | 64 |
| `silver` | 8 | 7.746 | 64 |

| commodity | all horizons | MAPE | n |
|---|---|---|---|
| `aluminium` | all | 4.895 | 154 |
| `copper_cash` | all | 4.290 | 512 |
| `gold` | all | 3.403 | 512 |
| `nickel` | all | 6.291 | 512 |
| `silver` | all | 5.692 | 512 |

### `E024_persistence` — 6 series, n_series=— (v1 row)

| commodity | horizon | MAPE | n |
|---|---|---|---|
| `coffee_r` | 1 | 3.105 | 64 |
| `coffee_r` | 2 | 5.665 | 64 |
| `coffee_r` | 3 | 7.108 | 64 |
| `coffee_r` | 4 | 7.935 | 64 |
| `coffee_r` | 5 | 8.927 | 64 |
| `coffee_r` | 6 | 10.609 | 64 |
| `coffee_r` | 7 | 11.725 | 64 |
| `coffee_r` | 8 | 12.447 | 64 |
| `corn` | 1 | 3.060 | 64 |
| `corn` | 2 | 3.635 | 64 |
| `corn` | 3 | 4.931 | 64 |
| `corn` | 4 | 5.173 | 64 |
| `corn` | 5 | 6.413 | 64 |
| `corn` | 6 | 6.938 | 64 |
| `corn` | 7 | 8.165 | 64 |
| `corn` | 8 | 8.209 | 64 |
| `rice` | 1 | 1.740 | 33 |
| `rice` | 2 | 2.986 | 33 |
| `rice` | 3 | 3.776 | 33 |
| `rice` | 4 | 5.041 | 33 |
| `rice` | 5 | 6.098 | 32 |
| `rice` | 6 | 6.467 | 32 |
| `rice` | 7 | 7.560 | 32 |
| `rice` | 8 | 8.120 | 32 |
| `soybean` | 1 | 2.220 | 64 |
| `soybean` | 2 | 3.095 | 64 |
| `soybean` | 3 | 4.248 | 64 |
| `soybean` | 4 | 4.356 | 64 |
| `soybean` | 5 | 5.370 | 64 |
| `soybean` | 6 | 6.094 | 64 |
| `soybean` | 7 | 6.821 | 64 |
| `soybean` | 8 | 6.817 | 64 |
| `sugar` | 1 | 2.489 | 41 |
| `sugar` | 2 | 3.907 | 41 |
| `sugar` | 3 | 3.856 | 41 |
| `sugar` | 4 | 5.081 | 41 |
| `sugar` | 5 | 5.068 | 40 |
| `sugar` | 6 | 5.540 | 40 |
| `sugar` | 7 | 5.559 | 40 |
| `sugar` | 8 | 6.264 | 40 |
| `wheat` | 1 | 3.439 | 64 |
| `wheat` | 2 | 3.914 | 64 |
| `wheat` | 3 | 6.148 | 64 |
| `wheat` | 4 | 6.527 | 64 |
| `wheat` | 5 | 6.876 | 64 |
| `wheat` | 6 | 7.146 | 64 |
| `wheat` | 7 | 9.201 | 64 |
| `wheat` | 8 | 8.955 | 64 |

| commodity | all horizons | MAPE | n |
|---|---|---|---|
| `coffee_r` | all | 8.440 | 512 |
| `corn` | all | 5.816 | 512 |
| `rice` | all | 5.195 | 260 |
| `soybean` | all | 4.877 | 512 |
| `sugar` | all | 4.709 | 324 |
| `wheat` | all | 6.526 | 512 |

### `E025_persistence` — 4 series, n_series=— (v1 row)

| commodity | horizon | MAPE | n |
|---|---|---|---|
| `copper_cash` | 1 | 1.907 | 64 |
| `copper_cash` | 2 | 2.905 | 64 |
| `copper_cash` | 3 | 3.427 | 64 |
| `copper_cash` | 4 | 4.162 | 64 |
| `copper_cash` | 5 | 4.738 | 64 |
| `copper_cash` | 6 | 5.193 | 64 |
| `copper_cash` | 7 | 5.726 | 64 |
| `copper_cash` | 8 | 6.259 | 64 |
| `gold` | 1 | 1.587 | 64 |
| `gold` | 2 | 2.273 | 64 |
| `gold` | 3 | 2.993 | 64 |
| `gold` | 4 | 3.512 | 64 |
| `gold` | 5 | 3.505 | 64 |
| `gold` | 6 | 3.990 | 64 |
| `gold` | 7 | 4.663 | 64 |
| `gold` | 8 | 4.698 | 64 |
| `soybean` | 1 | 2.220 | 64 |
| `soybean` | 2 | 3.095 | 64 |
| `soybean` | 3 | 4.248 | 64 |
| `soybean` | 4 | 4.356 | 64 |
| `soybean` | 5 | 5.370 | 64 |
| `soybean` | 6 | 6.094 | 64 |
| `soybean` | 7 | 6.821 | 64 |
| `soybean` | 8 | 6.817 | 64 |
| `wti` | 1 | 3.877 | 64 |
| `wti` | 2 | 5.284 | 64 |
| `wti` | 3 | 5.980 | 64 |
| `wti` | 4 | 7.698 | 64 |
| `wti` | 5 | 8.813 | 64 |
| `wti` | 6 | 9.510 | 64 |
| `wti` | 7 | 9.634 | 64 |
| `wti` | 8 | 11.198 | 64 |

| commodity | all horizons | MAPE | n |
|---|---|---|---|
| `copper_cash` | all | 4.290 | 512 |
| `gold` | all | 3.403 | 512 |
| `soybean` | all | 4.877 | 512 |
| `wti` | all | 7.749 | 512 |

### `E026_persistence` — 7 series, n_series=— (v1 row)

| commodity | horizon | MAPE | n |
|---|---|---|---|
| `brent` | 1 | 3.854 | 64 |
| `brent` | 2 | 5.353 | 64 |
| `brent` | 3 | 5.930 | 64 |
| `brent` | 4 | 7.626 | 64 |
| `brent` | 5 | 8.606 | 64 |
| `brent` | 6 | 9.585 | 64 |
| `brent` | 7 | 9.690 | 64 |
| `brent` | 8 | 11.146 | 64 |
| `coffee_r` | 1 | 3.105 | 64 |
| `coffee_r` | 2 | 5.665 | 64 |
| `coffee_r` | 3 | 7.108 | 64 |
| `coffee_r` | 4 | 7.935 | 64 |
| `coffee_r` | 5 | 8.927 | 64 |
| `coffee_r` | 6 | 10.609 | 64 |
| `coffee_r` | 7 | 11.725 | 64 |
| `coffee_r` | 8 | 12.447 | 64 |
| `copper_cash` | 1 | 1.907 | 64 |
| `copper_cash` | 2 | 2.905 | 64 |
| `copper_cash` | 3 | 3.427 | 64 |
| `copper_cash` | 4 | 4.162 | 64 |
| `copper_cash` | 5 | 4.738 | 64 |
| `copper_cash` | 6 | 5.193 | 64 |
| `copper_cash` | 7 | 5.726 | 64 |
| `copper_cash` | 8 | 6.259 | 64 |
| `gas_hh` | 1 | 12.483 | 64 |
| `gas_hh` | 2 | 14.791 | 64 |
| `gas_hh` | 3 | 15.197 | 64 |
| `gas_hh` | 4 | 18.434 | 64 |
| `gas_hh` | 5 | 21.748 | 64 |
| `gas_hh` | 6 | 23.588 | 64 |
| `gas_hh` | 7 | 23.988 | 64 |
| `gas_hh` | 8 | 23.578 | 64 |
| `gold` | 1 | 1.587 | 64 |
| `gold` | 2 | 2.273 | 64 |
| `gold` | 3 | 2.993 | 64 |
| `gold` | 4 | 3.512 | 64 |
| `gold` | 5 | 3.505 | 64 |
| `gold` | 6 | 3.990 | 64 |
| `gold` | 7 | 4.663 | 64 |
| `gold` | 8 | 4.698 | 64 |
| `soybean` | 1 | 2.220 | 64 |
| `soybean` | 2 | 3.095 | 64 |
| `soybean` | 3 | 4.248 | 64 |
| `soybean` | 4 | 4.356 | 64 |
| `soybean` | 5 | 5.370 | 64 |
| `soybean` | 6 | 6.094 | 64 |
| `soybean` | 7 | 6.821 | 64 |
| `soybean` | 8 | 6.817 | 64 |
| `wti` | 1 | 3.877 | 64 |
| `wti` | 2 | 5.284 | 64 |
| `wti` | 3 | 5.980 | 64 |
| `wti` | 4 | 7.698 | 64 |
| `wti` | 5 | 8.813 | 64 |
| `wti` | 6 | 9.510 | 64 |
| `wti` | 7 | 9.634 | 64 |
| `wti` | 8 | 11.198 | 64 |

| commodity | all horizons | MAPE | n |
|---|---|---|---|
| `brent` | all | 7.724 | 512 |
| `coffee_r` | all | 8.440 | 512 |
| `copper_cash` | all | 4.290 | 512 |
| `gas_hh` | all | 19.226 | 512 |
| `gold` | all | 3.403 | 512 |
| `soybean` | all | 4.877 | 512 |
| `wti` | all | 7.749 | 512 |

### `E027_persistence` — 2 series, n_series=— (v1 row)

| commodity | horizon | MAPE | n |
|---|---|---|---|
| `coffee_r` | 1 | 2.654 | 116 |
| `coffee_r` | 2 | 4.743 | 116 |
| `coffee_r` | 3 | 5.759 | 116 |
| `coffee_r` | 4 | 6.488 | 116 |
| `coffee_r` | 5 | 7.473 | 116 |
| `coffee_r` | 6 | 8.604 | 116 |
| `coffee_r` | 7 | 9.291 | 116 |
| `coffee_r` | 8 | 9.841 | 116 |
| `corn` | 1 | 2.804 | 116 |
| `corn` | 2 | 3.510 | 116 |
| `corn` | 3 | 4.372 | 116 |
| `corn` | 4 | 4.834 | 116 |
| `corn` | 5 | 6.103 | 116 |
| `corn` | 6 | 6.358 | 116 |
| `corn` | 7 | 7.328 | 116 |
| `corn` | 8 | 7.652 | 116 |

| commodity | all horizons | MAPE | n |
|---|---|---|---|
| `coffee_r` | all | 6.856 | 928 |
| `corn` | all | 5.370 | 928 |

### `naive` — 6 series, n_series=— (v1 row)

| commodity | horizon | MAPE | n |
|---|---|---|---|
| `coffee` | 1 | 3.247 | 261 |
| `coffee` | 2 | 5.022 | 261 |
| `coffee` | 3 | 6.465 | 261 |
| `coffee` | 4 | 7.828 | 260 |
| `coffee` | 5 | 8.815 | 259 |
| `coffee` | 6 | 9.783 | 258 |
| `coffee` | 7 | 10.611 | 257 |
| `coffee` | 8 | 11.499 | 256 |
| `copper` | 1 | 2.093 | 269 |
| `copper` | 2 | 2.916 | 268 |
| `copper` | 3 | 3.529 | 267 |
| `copper` | 4 | 4.095 | 266 |
| `copper` | 5 | 4.645 | 265 |
| `copper` | 6 | 5.058 | 264 |
| `copper` | 7 | 5.539 | 263 |
| `copper` | 8 | 6.038 | 262 |
| `corn` | 1 | 2.352 | 258 |
| `corn` | 2 | 3.180 | 258 |
| `corn` | 3 | 4.124 | 258 |
| `corn` | 4 | 4.799 | 257 |
| `corn` | 5 | 5.451 | 256 |
| `corn` | 6 | 5.911 | 255 |
| `corn` | 7 | 6.327 | 254 |
| `corn` | 8 | 6.725 | 253 |
| `gold` | 1 | 1.682 | 383 |
| `gold` | 2 | 2.436 | 382 |
| `gold` | 3 | 3.036 | 381 |
| `gold` | 4 | 3.505 | 380 |
| `gold` | 5 | 3.926 | 379 |
| `gold` | 6 | 4.435 | 378 |
| `gold` | 7 | 4.814 | 377 |
| `gold` | 8 | 5.217 | 376 |
| `wheat` | 1 | 3.457 | 236 |
| `wheat` | 2 | 4.814 | 235 |
| `wheat` | 3 | 5.651 | 234 |
| `wheat` | 4 | 6.438 | 233 |
| `wheat` | 5 | 7.165 | 232 |
| `wheat` | 6 | 7.929 | 231 |
| `wheat` | 7 | 8.703 | 230 |
| `wheat` | 8 | 9.236 | 229 |
| `wti_crude_oil` | 1 | 4.781 | 383 |
| `wti_crude_oil` | 2 | 6.906 | 382 |
| `wti_crude_oil` | 3 | 8.071 | 381 |
| `wti_crude_oil` | 4 | 9.266 | 380 |
| `wti_crude_oil` | 5 | 10.786 | 379 |
| `wti_crude_oil` | 6 | 11.789 | 378 |
| `wti_crude_oil` | 7 | 13.007 | 377 |
| `wti_crude_oil` | 8 | 14.187 | 376 |

| commodity | all horizons | MAPE | n |
|---|---|---|---|
| `coffee` | all | 7.891 | 2073 |
| `copper` | all | 4.228 | 2124 |
| `corn` | all | 4.849 | 2049 |
| `gold` | all | 3.625 | 3036 |
| `wheat` | all | 6.656 | 1860 |
| `wti_crude_oil` | all | 9.831 | 3036 |

## Retired rows, kept for the record

Counted, never ranked. The numbers are in `leaderboard_archive` exactly as v1 wrote them.

| experiment | directions | rows |
|---|---|---|
| `E001` | 1 | 45 |
| `E002` | 1 | 45 |
| `E003_knockoff` | 1 | 45 |
| `E003_stability` | 1 | 45 |
| `E004_direct` | 1 | 45 |
| `E004_dirmo2` | 1 | 45 |
| `E004_dirmo4` | 1 | 45 |
| `E004_dirrec` | 1 | 45 |
| `E004_mimo` | 1 | 45 |
| `E004_recursive` | 1 | 45 |
| `E005_combo` | 1 | 45 |
| `E005_ridge` | 1 | 45 |
| `E006_aalf` | 1 | 45 |
| `E006_boa` | 1 | 45 |
| `E006_ecw` | 1 | 45 |
| `E006_rewts` | 1 | 45 |
| `E007_anomaly` | 1 | 45 |
| `E007_purged` | 1 | 45 |
| `E007_robust` | 1 | 45 |
| `E007_shallow` | 1 | 45 |
| `E007_tsfm` | 1 | 45 |
| `E008_anytime` | 1 | 45 |
| `E008_avg` | 1 | 45 |
| `E008_dro` | 1 | 45 |
| `E008_global` | 1 | 45 |
| `E008_huber` | 1 | 45 |
| `E009_gru` | 1 | 45 |
| `E009_gru_candidate` | 1 | 45 |
| `E009_gru_raw` | 1 | 45 |
| `E010_gru_reg` | 1 | 45 |
| `E010_gru_reg_candidate` | 1 | 45 |
| `E010_gru_reg_raw` | 1 | 45 |
| `E011_delay` | 1 | 45 |
| `E011_delay_equal` | 1 | 45 |
| `E011_delay_raw` | 1 | 45 |
| `E012_gru_hedge` | 1 | 45 |
| `E012_gru_hedge_equal` | 1 | 45 |
| `E013_erm` | 1 | 45 |
| `E013_erm_candidate` | 1 | 45 |
| `E013_erm_raw` | 1 | 45 |
| `E013_inv01` | 1 | 45 |
| `E013_inv01_candidate` | 1 | 45 |
| `E013_inv01_raw` | 1 | 45 |
| `E013_inv1` | 1 | 45 |
| `E013_inv1_candidate` | 1 | 45 |
| `E013_inv1_raw` | 1 | 45 |
| `E014_sheet_pool_erm` | 1 | 54 |
| `E014_sheet_pool_erm_candidate` | 1 | 54 |
| `E014_sheet_pool_erm_raw` | 1 | 54 |
| `E014_sheet_pool_inv01` | 1 | 54 |
| `E014_sheet_pool_inv01_candidate` | 1 | 54 |
| `E014_sheet_pool_inv01_raw` | 1 | 54 |
| `E014_sheet_pool_inv1` | 1 | 54 |
| `E014_sheet_pool_inv1_candidate` | 1 | 54 |
| `E014_sheet_pool_inv1_raw` | 1 | 54 |
| `E015_fresh` | 1 | 54 |
| `E015_fresh_candidate` | 1 | 54 |
| `E015_fresh_raw` | 1 | 54 |
| `E015_frozen` | 1 | 54 |
| `E015_frozen_candidate` | 1 | 54 |
| `E015_frozen_raw` | 1 | 54 |
| `E015_naive` | 1 | 54 |
| `E015_warm` | 1 | 54 |
| `E015_warm_candidate` | 1 | 54 |
| `E015_warm_raw` | 1 | 54 |
| `E016_drift26` | 1 | 54 |
| `E016_drift26_candidate` | 1 | 54 |
| `E016_recent26` | 1 | 54 |
| `E016_recent26_candidate` | 1 | 54 |
| `E016_static52` | 1 | 54 |
| `E016_static52_candidate` | 1 | 54 |
| `E018_cs_weight` | 1 | 54 |
| `E018_cs_weight_candidate` | 1 | 54 |
| `E018_eproc` | 1 | 54 |
| `E018_eproc_candidate` | 1 | 54 |
| `E018_smcs` | 1 | 54 |
| `E018_smcs_candidate` | 1 | 54 |
| `E019_cap` | 1 | 54 |
| `E019_cap_candidate` | 1 | 54 |
| `E019_ebh` | 1 | 54 |
| `E019_ebh_candidate` | 1 | 54 |
| `E019_pool` | 1 | 54 |
| `E019_pool_candidate` | 1 | 54 |
| `E020_base` | 1 | 54 |
| `E020_base_candidate` | 1 | 54 |
| `E020_base_raw` | 1 | 54 |
| `E020_recent` | 1 | 54 |
| `E020_recent_candidate` | 1 | 54 |
| `E020_recent_raw` | 1 | 54 |
| `E020_tta` | 1 | 54 |
| `E020_tta_candidate` | 1 | 54 |
| `E020_tta_raw` | 1 | 54 |
| `E022_adl_gpr_vix` | 1 | 45 |
| `E022_adl_stocks_cot` | 1 | 45 |
| `E022_combo_energy` | 1 | 45 |
| `E022_gbrt_multisource` | 1 | 45 |
| `E022_lasso_macro` | 1 | 45 |
| `E023_adl_basis_macro` | 1 | 27 |
| `E023_adl_real_yield` | 1 | 18 |
| `E023_combo_metal` | 1 | 27 |
| `E023_gbrt_metal` | 1 | 45 |
| `E023_lasso_metal` | 1 | 45 |
| `E023_pcr3_metal` | 1 | 45 |
| `E024_adl_enso` | 1 | 27 |
| `E024_adl_parity_fx` | 1 | 27 |
| `E024_arimax_rain` | 1 | 36 |
| `E024_combo_agri` | 1 | 27 |
| `E024_gbrt_agri` | 1 | 45 |
| `E024_lasso_agri` | 1 | 45 |
| `E025_gbrt_gpr` | 1 | 36 |
| `E025_lasso_gpr` | 1 | 36 |
| `E025_pcr3_gpr` | 1 | 36 |
| `E026_adl_broad` | 1 | 63 |
| `E026_combo_broad` | 1 | 63 |
| `E026_rf_broad` | 1 | 63 |
| `E027_adl_cc` | 1 | 18 |
| `E027_combo_cc` | 1 | 18 |
| `E027_gbrt_cc` | 1 | 18 |
