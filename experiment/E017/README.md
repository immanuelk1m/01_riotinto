# E017 causal marginal quantile alignment

RainShift-inspired ablation, not a reproduction:
https://arxiv.org/html/2507.04930v1

Directions are identity, qmap52, and qmap104. All use the existing E014 sheet
snapshots, six commodities, seed 42, AdamW, the E013 GRU training core with
penalty zero, the original early/gate purges and 26-week refit schedule.
The identity direction is the E014 ERM control. No live sheet or DB refresh.

For each feature, qmap maps the empirical recent-window CDF onto the source
training-prefix CDF. Ties use midrank probabilities; interpolation clamps at
the boundary quantiles. Features with fewer than 16 finite values or constant
marginals use identity. Target log-price and one-week return remain unchanged.

Mapping statistics are fixed separately for early validation, gate, and outer
evaluation: fit_end is the first origin of each phase, recorded in curves.
The early/gate source prefix ends at train.max(); the outer source prefix ends
at the last mature training row. No rows after a phase's first origin enter its
mapping fit. Future features are transformed independently using those fixed
knots. Training inputs retain their original distribution; only evaluation
inputs are aligned. This avoids applying outer-cutoff feature information to
earlier validation origins.

The existing gate-to-fresh-refit transition is retained as a controlled part of
E014. Gate kappa and significance share a calibration sample, and the E014
forced-lag0 features and repeated historical test inspection remain limitations.
Quantile alignment can destroy useful level signals; all directions must be
reported. Good validation loss alone does not establish next-block improvement.

Run local checks:

```bash
uv run python experiment/E017/test_e017.py
uv run python experiment/E017/E017.py identity copper
```

For Colab, one pane per direction runs:

```bash
S=e017-identity DIRECTION=identity bash experiment/E017/pane.sh
S=e017-qmap52 DIRECTION=qmap52 bash experiment/E017/pane.sh
S=e017-qmap104 DIRECTION=qmap104 bash experiment/E017/pane.sh
```

The runner uploads the existing E014 data/features and E013 source as
e013_core.py. Remote GPU training follows torch smoke. Downloads include curves,
transfer CSV, summaries and six prediction parquets. The local leaderboard is
experiment-local, preserving earlier experiment and naive rows.

## Retired 2026-11-03

This experiment cannot run. Model: sheet-pool wrapper around the E013 GRU core. Removed constants: E013 constants; pooling weights. Reason: same wrapper as E014 over the same knob-carrying core. Implementation preserved at commit `2ca48e9`; rows kept in the leaderboard under `status = 'retired-heuristic'`. See [`../RETIRED.md`](../RETIRED.md).
