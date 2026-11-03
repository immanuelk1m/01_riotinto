# E016 delayed affine forecast calibration

Uses immutable E014 ERM raw forecasts for six commodities and preserves their
origin/horizon grids. This is a D3A/Proceed/GEPS-inspired diagnostic ablation,
not a reproduction of their models. No new GRU training is performed.

Directions: `static52` fits on the last 52 observed targets per horizon;
`recent26` uses 26; `drift26` uses 26 only when the latest 26 mature raw absolute
percentage errors have mean greater than 1.5 times the preceding 26, otherwise
52. Errors are capped at 1 for this detector. The minimum fit size is 13.

The affine map fits log return with ridge 0.01 toward identity, slope clipped to
[0,1], intercept to +/-0.05, and log-return correction to +/-0.10. Cold starts
produce persistence. The deploy gate compares the last 26 previously issued
candidate errors with persistence, requires 13 labels, and activates only on
positive average MAPE gain. It never uses fitted in-sample calibration error.

At origin t and horizon h, only prior rows with target_date <= t and observed
targets may update fit, detector or gate. Predictions already issued are never
revised. The target-week close is assumed available at that origin, consistent
with E014. E014's retrospective feature selection and forced lag0 limitations
remain. Parameters above are fixed before replay; no outer score tunes them.

Transfer logs average all horizons per 26-week block and compare the preceding
26 mature predictions per horizon against the next block. They are diagnostic
only; evaluation labels are read after replay. Cold-start blocks are retained
in overall metrics but excluded from validation-win transfer rates.

Run local tests with `uv run python experiment/E016/test_e016.py`.
Run a direction in its assigned tmux pane with
`S=e016-static52 DIRECTION=static52 bash experiment/E016/pane.sh`.
The runner uses a CPU Colab VM, bounded CLI calls and retries. Failed transfer
or validation retains the VM; successful download and local validation stops
it. Do not edit a running runner. Leader owns tmux allocation.

Source references: https://arxiv.org/abs/2403.14949,
https://arxiv.org/abs/2412.08435, https://arxiv.org/abs/2410.23889.

## Retired 2026-11-03

This experiment cannot run. Model: affine calibration of E014 forecasts. Removed constants: recent-26 and static-52 calibration windows on top of E014. Reason: it calibrates E014's GRU output and chooses the calibration window by hand. Implementation preserved at commit `2ca48e9`; rows kept in the leaderboard under `status = 'retired-heuristic'`. See [`../RETIRED.md`](../RETIRED.md).
