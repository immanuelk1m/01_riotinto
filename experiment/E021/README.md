# E021 panel expansion: is the evidence base the binding constraint?

E018 showed that every anytime-valid gate refuses to deploy on six commodities
(a 0.0002 capped-APE edge against 0.015 weekly noise), and E019's only positive
rule failed its deployed-worse contract. E021 attacks the constraint itself: every
weekly-resolvable price series in the local `commodity` database with at least
`MIN_WEEKS=250` weeks of history goes through the repository's expanding
walk-forward protocol.

Protocol per series: weekly `W-SUN` log price, internal validation = three
expanding folds inside the first 80% scored as capped-APE gain over persistence,
outer evaluation = expanding refits from the 80% point at step one week for
horizons 1-8, candidate = multi-output ridge on four lagged weekly returns plus a
52-week level z-score. No GPU and no sheet features are involved.

| Direction | Rule |
| --- | --- |
| `ridge` | per-series e-process on the outer weekly differences |
| `ebh` | cross-series e-BH at `ALPHA_FDR=0.10` |
| `pooled` | one pooled e-process across all series |

The headline diagnostic is the rank correlation between the internal validation
gain and the realised outer gain across series.

## Run

Preparation reads Postgres and therefore runs on the Mac mini:

```bash
uv run python experiment/E021/E021.py prepare
uv run python experiment/E021/E021.py ridge
uv run python experiment/E021/E021.py ebh
uv run python experiment/E021/E021.py pooled
```

`--limit N` restricts the panel for a quick check. Summaries land in `logs/`.

## Retired 2026-11-03

This experiment cannot run. Model: multi-output ridge over 730 series. Removed constants: RIDGE_LAMBDA = 1.0; LAGS = 4; MIN_WEEKS = 250. Reason: the ridge penalty is a constant, and this single experiment owns 53 % of the leaderboard rows. Implementation preserved at commit `2ca48e9`; rows kept in the leaderboard under `status = 'retired-heuristic'`. See [`../RETIRED.md`](../RETIRED.md).
