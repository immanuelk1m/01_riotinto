# Registered constants

Generated from `_exo/params.py` by `tools/params_inventory.py`. Every number the
pipeline uses appears here with its source; anything not in this table cannot run.

| constant | value | source | rationale |
|---|---|---|---|
| `BOOT` | `1000` | declared | Bootstrap draws; 1000 resolves the 2.5 % quantile used by the interval. |
| `COST_BPS` | `5` | declared | One-way trading cost in basis points for the E028 P&L check (liquid futures, with slippage). Swing over COST_SWING. |
| `COST_SWING` | `(0, 5, 25)` | declared | Registered cost grid for the E028 P&L check; 25 bp stands in for a physical basis trade. |
| `CSCV_SPLITS` | `16` | declared | Number of time partitions S for CSCV / PBO, the value used by Bailey, Borwein, Lopez de Prado and Zhu (2014). |
| `LAGS` | `4` | declared | One month of weekly lags of the target log-return (and of the exogenous changes). Swing-tested over {2,4,8} in docs/SENSITIVITY.md. |
| `MAX_WEEKS` | `1040` | declared | History is truncated at 20 years so that a panel from 1970 and one from 2005 are scored on comparable sample sizes. |
| `MIN_TRAIN` | `100` | declared | Minimum training window in weeks before the first forecast is issued. |
| `MIN_WEEKS` | `200` | declared | A target panel needs at least 200 weekly observations (≈ 4 years) to be in the study at all. |
| `PLANTED_R2` | `0.1` | declared | Share of one-week return variance planted as a known linear signal in leakage detector D2. |
| `REFIT_EVERY` | `4` | declared | Refit cadence in weeks. Weekly refits add no information at weekly resolution and multiply the compute by four. |
| `SENSITIVITY_LAGS` | `(2, 4, 8)` | declared | The registered swing grid for LAGS (docs/SENSITIVITY.md). A run may only use a lag count from this tuple, and a swing run is written outside the leaderboard by construction, so a sensitivity value can never become a ranking. |
| `TRAIN_FRAC` | `0.7` | declared | Warm-up fraction of the walk-forward: the first 70 % of the usable origin grid is in-sample, the rest is walk-forward. Declared in the round's README before the run. |
| `CARRY_DAYCOUNT` | `360` | definitional | USD money-market rates (EFFR) accrue actual/360, so a carry over d days is rate * d / 360. |
| `CORN_MONTHS` | `(3, 5, 7, 9, 12)` | definitional | CBOT corn contract months H K N U Z. Barchart's nearby rolls on the first trading day of each (experiment/E028/CEILING.md §4: 32 of the 40 largest basis-index jumps, all 9 above 6 %). |
| `DAYS_PER_WEEK` | `7` | definitional | Calendar days in a week. |
| `H` | `8` | definitional | The repository's registered horizon grid h = 1..8 weeks (PROTOCOL.md §5.1 cell definition); a different grid is a different experiment, not a tuning knob. |
| `PCT` | `100.0` | definitional | Percent conversion of a fraction. |
| `SEED` | `0` | definitional | Bootstrap and estimator seed; fixed so a re-run reproduces the cells instead of being a second draw. |
| `TOL_ABS` | `1e-06` | definitional | Float comparison tolerance when a tool re-derives a stored mean; it decides whether arithmetic reproduces, not what is estimated. |
| `TOL_REL` | `1e-09` | definitional | Relative twin of TOL_ABS. |
| `WEEKS_3M` | `13` | definitional | The LME 3-month prompt is three calendar months ahead, 52 / 4 = 13 weeks. |
| `WEEKS_PER_YEAR` | `52` | definitional | Annualisation of a weekly Sharpe ratio. |
| `Z_MIN_OBS` | `2` | definitional | A standard deviation is undefined below two observations, so an expanding z-score cannot start earlier. |
| `ALPHA` | `0.05` | protocol | PROTOCOL.md §5.5: the declared family-wise level. |
| `BLOCK` | `26` | protocol | PROTOCOL.md §5.4: the paired interval must come from a moving-block bootstrap with 26-week blocks. |
| `CAP` | `1.0` | protocol | PROTOCOL.md §2 Gate A: err = min(1.0, |expm1(forecast - log_return_target)|). |
| `MIN_COVERAGE` | `0.1` | protocol | PROTOCOL.md §5.6 and §6.1 item 3: deploy coverage floor. |
| `SAFETY_TOL` | `0.05` | protocol | PROTOCOL.md §6.1 item 4: no single series may regress by more than 0.05 MAPE. |
| `SEALED_ORIGINS` | `52` | protocol | PROTOCOL.md §4.2: the most recent 52 weeks of mature targets are sealed. |

`PARAMS_VERSION = 2.1`, `PARAMS_HASH = a4c2dbada7829812`
