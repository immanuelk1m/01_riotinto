# E001


## Retired 2026-11-03

This experiment cannot run. Model: LightGBM + genetic feature search. Removed constants: LGBMRegressor; array_split; rng.choice; GENERATIONS; PRUNE_FRAC; N_KEEP. Reason: the forecast comes from boosted trees whose depth, learning rate and tree count were hand-set, on top of a stochastic feature search. Implementation preserved at commit `2ca48e9`; rows kept in the leaderboard under `status = 'retired-heuristic'`. See [`../RETIRED.md`](../RETIRED.md).
