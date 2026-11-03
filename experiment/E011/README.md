# E011


## Retired 2026-11-03

This experiment cannot run. Model: delayed EWLS correction of E010 GRU forecasts. Removed constants: EWLS decay and memory-horizon constants on top of E010. Reason: it post-processes GRU forecasts, so it inherits E010's free parameters and adds its own decay constant. Implementation preserved at commit `2ca48e9`; rows kept in the leaderboard under `status = 'retired-heuristic'`. See [`../RETIRED.md`](../RETIRED.md).
