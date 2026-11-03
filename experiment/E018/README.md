# E018 anytime-valid deployment gates

E016 deployed the calibrated candidate only when the last 26 mature candidate
errors beat persistence on a point estimate (`gain > 0`). That rule has no
uncertainty control. In round 1 of the regime-transfer study the E015 gate
reversed a candidate that had already passed the accuracy, block-transfer and
commodity-safety checks. E018 changes **only the decision rule**: the candidate
forecast stays the immutable E015 frozen `y_cand`.

Directions (one Colab pane each):

| Direction | Rule | Source |
| --- | --- | --- |
| `eproc` | betting e-process on the mature loss difference, deploy while `E ≥ 1/α` | Waudby-Smith & Ramdas betting; Arnold et al. 2024 (e-process MCS) |
| `cs_weight` | time-uniform sub-Gaussian confidence sequence, converted to weight `clip(LB/0.01, 0, 1)` | Howard et al. confidence sequences |
| `smcs` | sequential model confidence set over {persistence, candidate, half-shrunk candidate}, deploy only after persistence is excluded | Arnold, Gavrilopoulos, Schulz & Ziegel 2024, JRSS-B |

Decision unit is the commodity: the eight horizons are averaged into one bounded
observation per target week, because a per-horizon test has no power on this panel.
Every statistic reads only labels whose `target_date` is already realised at the
origin, and issued forecasts are never revised.

## Result (six commodities, horizons 1-8, 48 cells per variant)

| Variant | MAPE |
| --- | --- |
| `E018_eproc` / `_cs_weight` / `_smcs` (deployed) | **6.194** |
| `E018_*_candidate` (always deploy) | 6.208 |
| E015 gate (`E015_frozen`) | 6.218 |
| naive persistence | 6.194 |

All three calibrated rules decline to deploy anywhere on any commodity, which
scores exactly persistence and avoids the 0.024 MAPE harm the E015 gate caused.
The binding constraint is power, not the rule: coffee's edge is 0.0002 in capped
APE units against a weekly standard deviation of 0.015, so certifying it would
need roughly 20,000 weekly observations against the 236-384 available.

## Run

Local checks and replay:

```bash
uv run python experiment/E018/test_e018.py
uv run python experiment/E018/E018.py eproc coffee wheat
uv run python experiment/E018/E018.py leaderboard eproc
```

Colab panes (CPU is sufficient):

```bash
S=e018-eproc   DIRECTION=eproc     bash experiment/E018/pane.sh
S=e018-cs      DIRECTION=cs_weight bash experiment/E018/pane.sh
S=e018-smcs    DIRECTION=smcs      bash experiment/E018/pane.sh
```

The runner uploads the immutable E015 frozen archive, runs the smoke and contract
tests inside Colab, launches the direction, polls, downloads artifacts, imports the
leaderboard only after local validation, then stops the VM. Verify a failure with
`colab exec -s <name> -f remote/poll.py`: `colab exec` does not propagate exit codes.

## Retired 2026-11-03

This experiment cannot run. Model: e-process with shrunk cross-sectional weights. Removed constants: ALPHA = 0.05; MIN_LABELS = 13; SHRINK_MIN = 13; SHRINK_FRACTION = 0.5; EFFECT_SCALE = 0.01. Reason: five hand-set constants define the rule, including a shrinkage schedule. Implementation preserved at commit `2ca48e9`; rows kept in the leaderboard under `status = 'retired-heuristic'`. See [`../RETIRED.md`](../RETIRED.md).
