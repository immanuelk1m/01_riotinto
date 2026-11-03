# E003


## Retired 2026-11-03

This experiment cannot run. Model: LightGBM + Lasso/LassoCV knockoff screen. Removed constants: LGBMRegressor; Lasso; LassoCV; array_split; rng.choice. Reason: two estimator families with free penalties screen the features before the fit. Implementation preserved at commit `2ca48e9`; rows kept in the leaderboard under `status = 'retired-heuristic'`. See [`../RETIRED.md`](../RETIRED.md).
