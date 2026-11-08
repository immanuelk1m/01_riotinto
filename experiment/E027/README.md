# E027 — coffee and corn only (Gate C registration)

**Registered:** 2026-11-01, before any direction of this round was run.
**Protocol:** [`PROTOCOL.md`](../../PROTOCOL.md) gates A–E. Round-level registration and
family size live in [`../FAMILY.md`](../FAMILY.md).
**Supersedes for scope:** the two-series panel replaces the six-commodity panel used in
E001–E021 and the sector panels used in E022–E026. Those rounds stand as run and are not
re-analysed here.

## Why this round exists

The E022–E026 report (`../LEADERBOARD_exogenous_2026.md`) shows every one of its 25
directions failing Holm, but the per-series table inside the audit files shows a
consistent, positive candidate-over-persistence gap on exactly two series:

| series | E024 `gbrt_agri` mean d | other directions positive |
|---|---|---|
| `coffee_r` | +0.00733 (MAPE 8.543 → 7.810) | 7 of 7 |
| `corn` | +0.00118 (MAPE 5.263 → 5.145) | 4 of 5 |

Everywhere else the sign was mixed or negative. This round therefore fixes the panel to
those two series and asks a narrower question.

## Hypothesis

On coffee (CEPEA/ESALQ robusta, USD per 60 kg bag) and on corn (Barchart cash national corn
index), a pre-registered exogenous model beats persistence at 1–8 weeks on the development
window by more than this panel's detection floor.

Two series → three statements: `coffee`, `corn`, and `pooled`.

## Declared retrospectivity

The panel was chosen **after** reading the per-series development-window table of E024/E026.
Under PROTOCOL 4.4 the development-window result of this round is therefore retrospective and
cannot be the sole evidence for a claim. Two consequences, both accepted in advance:

1. The confirmatory evidence is the **forward evaluation** registered below, not the dev window.
2. The panel-level sealed aggregates were already printed in the E024/E026 audit files
   (mean d over all series). Series-level sealed values for coffee and corn have **not** been
   read; they are unsealed once, here, and reported separately.

## Gate B — axes changed

| axis | change vs E024 |
|---|---|
| data | panel narrowed from 3–5 agricultural series to 2 (coffee, corn); the exogenous set is unchanged |
| model | unchanged families (gbrt, combo, adl) — no model-axis change |
| evaluation grid | walk-forward now starts at 50 % of the sample instead of 70 %, to buy blocks (see Power) |

Changing the evaluation grid is a design change, not a tuning knob: it is fixed here, once,
and never searched. It is declared because it breaks cell-for-cell comparability with the
E024 numbers above.

## Design (fixed, no search)

- `cell` = (series, horizon, origin); horizons 1–8; `train_frac = 0.50`; refit every 4 weeks.
- primary metric: cell MAPE (capped APE) against persistence, paired per cell
  `d = ape_persistence − ape_candidate`.
- uncertainty: 26-week **calendar**-block bootstrap (`origin // 26`), 2000 draws, blocks
  resampled with their cells pooled; one-sided p = P(mean d ≤ 0).
- family size: 3 directions × 3 statements = **9** tests → Holm α₁ = 0.00556.
- sealed block: the most recent 52 origins, excluded from every number above.

## Power (registered before running)

`MAX_WEEKS = 1040` caps each series at 20 years, so `train_frac = 0.50` walks forward over
~468 weeks and produces **18** calendar blocks (E024's grid gave 11). Block-level dispersion
of `d` is measured on the E024/E026 dev cells. MDE is one-sided at Holm α₁ = 0.00556
(z = 2.54) with 80 % power, i.e. 3.38 × SE:

| statement | sd(block mean d) | MDE @ 18 blocks | effect that motivated it |
|---|---|---|---|
| pooled coffee+corn | 0.00948 | **0.0076** | 0.0043 |
| coffee only | 0.01567 | **0.0125** | 0.0073 |
| corn only | 0.01625 | **0.0130** | 0.0012 |

**Stop rule, stated now:** a statement whose MDE exceeds the effect that motivated it is
reported as **no detectable difference**, whatever the point estimate. At the registered
design that is the expected outcome for coffee (0.0073 < 0.0125) and a certainty for corn
(0.0012 ≪ 0.0130). This round cannot certify either series. It can only report the point
estimate, the interval and the floor — and it fixes the design for the forward test.

## Forward evaluation (the part that can certify)

Registered now, evaluated once:

- from 2026-11-02, archive one forecast per week per series for `gbrt_cc` and `combo_cc`
  (same specification, no refit of hyperparameters);
- evaluate when 52 new origins have matured (target **2027-11-01**), on the 52-week block
  alone, with the same statistic;
- family size for that evaluation: 2 (one candidate per series) → Holm α₁ = 0.025;
- at 2 blocks per series the forward test is descriptive, not confirmatory. The
  confirmatory read is the accumulating 26-week block mean, reported as it moves.

## Directions (family size for this experiment: 3)

| direction | model | exogenous set |
|---|---|---|
| `gbrt_cc` | gradient boosting | oni, soi, rain, brl, dollar, crude, bdi, fertilizer |
| `combo_cc` | mean of ADL/GBRT/RF | oni, brl, crude, bdi |
| `adl_cc` | ADL/ARIMAX (linear) | oni, brl, crude, bdi |

`adl_cc` is the linear contrast: `combo_cc` contains it, so a positive `combo_cc` with a flat
`adl_cc` says the gain needs the nonlinear terms.

## What this round will not claim

- no deployment rule, no position sizing, no "transfer" verdict (Gate E is not run);
- no statement about any commodity other than coffee and corn;
- no combination of these two series into a portfolio-level claim;
- no rewording of a failing direction as a finding.

## Engine changes carried by this registration

- `_exo/engine.py`: a config `design` block may override `train_frac`, `refit_every`,
  `block`, `sealed_origins`; without the block the E022–E026 settings are reproduced exactly.
- `_exo/engine.py`: `block_bootstrap` now blocks on `origin // 26`. The old version counted
  distinct origins and sliced them as consecutive weeks, which reported 6 blocks where 18
  exist and made `gbrt_agri` a false PASS in its `*_audit.md`. Re-running any direction
  changes its md CI (not its leaderboard rows); the E022–E026 report already used the
  calendar convention and remains the reference for those rounds.

## Reproduction

```bash
# Mac mini (data preparation only)
uv run python experiment/E027/E027.py prepare
# Colab, one pane per direction
uv run python -m _colab.job run --exp E027 --cmd "gbrt_cc" --include _exo
```

## Amendment 1 — evaluator (declared 2026-11-08, before any forward target matured)

The forward evaluation above is to be scored with evaluator **cell-v3**, not the cell-v2 code that
produced the development and sealed numbers in `RESULTS.md`.

- cell-v2 (`_exo/engine.py` until 2026-11-08) scored persistence as `|expm1(r)| = |P_{t+h}/P_t − 1|`
  while scoring the candidate as `|P̂/P_{t+h} − 1|`; PROTOCOL §2.1/§5.1 define both against the
  target, so persistence must be `|expm1(−r)|`. The mismatch inflated `d` in rising samples
  (on `gbrt_cc` cells: mean d +0.00433 under cell-v2, +0.00390 under §5.1).
- The forecasts, models, family size and decision rule of the forward test are unchanged. Only the
  persistence APE formula changes, to the one the protocol already stated.
- Timing: the first forward origin is 2026-11-02 and its h=1 target week ends 2026-11-08; no forward
  target had been observed when this amendment was written.
- The development and sealed numbers in `RESULTS.md` stay as run (cell-v2) and are not compared with
  cell-v3 rows (§5.3).
