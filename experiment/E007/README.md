# E007


## Retired 2026-11-03

This experiment cannot run. Model: LightGBM robustness variants. Removed constants: LGBMRegressor; max_depth; n_estimators; learning_rate. Reason: each robustness variant is a different set of boosted-tree constants, so the comparison ranks constants rather than models. Implementation preserved at commit `2ca48e9`; rows kept in the leaderboard under `status = 'retired-heuristic'`. See [`../RETIRED.md`](../RETIRED.md).
