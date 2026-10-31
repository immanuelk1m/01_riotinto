# E012 GRU memory-horizon aggregation

This is a fixed-recipe retrospective prequential experiment using immutable
E009/E010 GRU forecasts, inspired by delayed online aggregation and multiple
memory horizons. It is not an implementation of the full MELO or ELF papers.

Seven experts: naive plus raw, candidate and gated variants of both GRU runs.
Six fixed-share aggregators use half-lives 13/52/104 observations and shares
0.01/0.05. A meta aggregator with half-life 52 and share 0.01 combines them.
All start uniformly; learning rate 2 and squared log-return loss divided by
0.01 and capped at 1 are fixed before replay. Aggregation occurs in log-return
space. Each horizon has independent state and consumes a saved forecast's
label once, only when `target_date <= origin`. Meta losses use the component
forecast saved at its original issuance. Missing labels are skipped.

The archive supplies target dates but no additional publication timestamps;
zero additional target release lag is assumed. This cannot validate external
feature vintages or remove the source GRUs' original selection limitations.
The historical outer data have already been inspected in previous experiments,
so results are not an untouched holdout or prospective proof.

`*_hedge_transfer.csv` compares recent mature 52-observation expert losses with
the following 26-week block, including ranking correlation and sign of gain
against naive. These are post-replay diagnostics and never feed the selector.
All origins, including initial uniform-weight warm-up, remain in results.
The equal-weight GRU family is scored as an explicit ablation.

Local commands:

```sh
uv run experiment/E012/E012.py hedge
uv run experiment/E012/E012.py leaderboard
uv run python -m unittest discover -s experiment/E012 -p 'test_e012.py' -v
```

Default local input lookup first checks `E012/inputs/E009` and `E012/inputs/E010`,
then existing sibling `E009/results/gru` and `E010/results/gru_reg`. An explicit
`--input-dir` uses only that path. Each directory must contain the five
`predictions_<commodity>.parquet` files. No GBM predictions are used.

For Colab upload `E012.py` to `/content/E012/E012.py` and the ten source files to
`/content/E012/inputs/E009/` and `/content/E012/inputs/E010/`. Install numpy,
pandas and pyarrow in the runtime. Then `S=e012 bash experiment/E012/pane.sh`
launches the remote process, polls, downloads results/diagnostics and writes
only the E012 and E012-equal leaderboard entries locally. The caller owns
session allocation and shutdown. Use a fresh remote workspace for a new run.
