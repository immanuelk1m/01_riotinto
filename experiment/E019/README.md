# E019 cross-sectional deployment with online FDR control

E018 proved that per-commodity anytime-valid gates cannot deploy anything on this
panel: a 0.0002 capped-APE edge against 0.015 weekly noise is not certifiable from
236-384 weeks. E019 keeps the evidence and changes the **family**: the six
commodities are treated as simultaneous hypotheses, so power can come from allowing
a controlled fraction of deployed-but-worse commodities.

Candidate: immutable E015 frozen `y_cand`. Decision unit: commodity-week, with the
eight horizons averaged into one bounded observation per target week.

| Direction | Rule | Source |
| --- | --- | --- |
| `ebh` | e-BH over the six per-commodity e-values every week at `ALPHA_FDR=0.10` | Wang & Ramdas e-BH; Fischer, Xu & Ramdas 2024 |
| `cap` | pick positive recent-26 mean gain, keep picks above a cutoff calibrated on the historical selected set with `alpha_{t+1} = alpha_t + gamma(alpha - err_t)` | CAP / adaptive conformal inference (Bao et al. 2024-2025) |
| `pool` | one pooled e-process across commodities must pass `1/alpha`, then each commodity needs `m/alpha` | hierarchical evidence split |

## Result (six commodities, horizons 1-8, 48 cells per variant)

| Variant | MAPE | Deploy rate | Deployed-worse rate |
| --- | --- | --- | --- |
| `E019_cap` | **6.173** | 10.5% | **0.27** (target ≤ 0.10) |
| `E019_ebh` | 6.194 | 0.0% | n/a |
| `E019_pool` | 6.194 | 0.0% | n/a |
| `E019_*_candidate` (always deploy) | 6.208 | 100% | — |
| naive persistence | 6.194 | — | — |

`cap` is the first direction in this study to beat persistence (6.173 vs 6.194,
-0.021 MAPE) and it does so with a 10.5% deploy rate, but **it fails the
predeclared FDR contract**: 27% of its deployed commodity-weeks were worse than
persistence against a 10% target. Under the round-2 contract that is non-transfer,
not a win: the aggregate gain comes from a minority of large wins while the
deployed-worse rate says the decision rule is not calibrated. A follow-up should
control the deployed-worse rate directly (risk control) rather than the quantile of
recent means.

## Run

```bash
uv run python experiment/E019/test_e019.py
uv run python experiment/E019/E019.py cap
uv run python experiment/E019/E019.py fdr cap      # retrospective deployed-worse rate
uv run python experiment/E019/E019.py leaderboard cap
```

Colab panes (CPU is sufficient):

```bash
S=e019-ebh  DIRECTION=ebh  bash experiment/E019/pane.sh
S=e019-cap  DIRECTION=cap  bash experiment/E019/pane.sh
S=e019-pool DIRECTION=pool bash experiment/E019/pane.sh
```

## Retired 2026-11-03

This experiment cannot run. Model: adaptive-conformal (ACI) deployment rule. Removed constants: ACI_GAMMA = 0.10; LAMBDA_MAX = 0.5; MIN_LABELS = 13; RECENT = 26. Reason: the step size of the adaptive level is a hand-set constant; it is this rule that held the program's best legacy number (6.160). Implementation preserved at commit `2ca48e9`; rows kept in the leaderboard under `status = 'retired-heuristic'`. See [`../RETIRED.md`](../RETIRED.md).
