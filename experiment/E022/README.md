# E022 — exogenous predictors for the energy panel (pre-registration)

> **Revision (2026-11-09):** the dev/sealed split and the 52-origin seal are gone. Evaluation is the monthly-first-week TSCV, 2025-01..2026-09, h = 1..8 (PROTOCOL.md v1.2 §4). The dev, sealed and `window` text below records the old rule (cell-v3).

**Round:** exogenous-information axis, 2026-10-31
**Protocol:** [`PROTOCOL.md`](../../PROTOCOL.md) gates A–E. This file is the Gate C registration;
it was written before any direction was run.

## Hypothesis

Weekly energy prices (WTI, Brent, Henry Hub, gasoline, heating oil) contain a
component predictable from **published, lag-annotated exogenous series** —
geopolitical risk, macro-financial conditions, inventories, positioning, freight —
and a candidate using them beats the naive forecast at 1–8 weeks by more than the
detection floor.

## Gate B — axes changed

| axis | change |
|---|---|
| data | new: 9 exogenous series with real-time availability lags (GPR, EPU, VIX, 10y, 10y-2y, dollar index, EIA crude stocks, CFTC WTI net positions, Baltic Dry) |
| model | new families: ADL/ARIMAX, Lasso, 3-pass regression filter, gradient boosting, forest, combination |

Two axes change, so Gate B is satisfied. The target panel and the evaluation
protocol are unchanged from E021 (`capped APE`, cell = series × horizon × origin,
26-week moving-block bootstrap).

## Candidate, control, metric

- **Candidate:** the direction's exogenous model forecast.
- **Matched control:** the E021 ridge candidate (price-only, same walk-forward).
- **Baseline:** naive forecast (zero log-return).
- **Primary metric:** cell MAPE of the candidate against the naive forecast.
- **Decision rule:** Gate A — mean paired difference `d = ape_naive − ape_candidate`
  must be > 0 with a 26-week moving-block bootstrap 95% interval excluding 0.
  No deployment rule is evaluated in this experiment.

## Sealed block

The most recent 52 origins are sealed. They are excluded from eligibility
(`sealed_cut = max(origin) − 52`) and reported once, separately, in each audit file.
No direction, exogenous set or hyperparameter was chosen by looking at them.

## Directions (family size for this experiment: 6)

| direction | model | exogenous set |
|---|---|---|
| `adl_gpr_vix` | ADL/ARIMAX | GPR, VIX, 10y, dollar |
| `adl_stocks_cot` | ADL/ARIMAX | EIA crude stocks, CFTC net positions, BDI |
| `lasso_macro` | Lasso | all 9 |
| `pcr3_macro` | 3-pass regression filter | all 9 |
| `gbrt_multisource` | gradient boosting | all 9 |
| `combo_energy` | mean of ADL/GBRT/RF | 6 core |

Round family size across E022–E026 is declared in `../FAMILY.md`; Holm
adjustment is applied there, not per experiment.

## Provenance

Directions follow the 2026 exogenous-variable literature selected in
`.omx/research/exogenous-2026/selection.md`: Baumeister et al. (2026, JAE) for
real-time fundamentals and MCS combination; Ma, Wang & Wang (2026, Financial
Innovation) for multi-source screening plus combination; Ekiz Yılmaz & Zehir
(2026, Entropy) for GPR/VIX/10y on Brent; Olaniran (2026, Energy Research
Letters) for GPR at 4/8/16-week horizons; Vasilakis & Wang (2026, JORS) for a
global risk factor; Zhou, Ma & Wang (2026, Economic Modelling) for iterated
combination.

## Expected outcome (stated in advance)

From the Gate A audit of the price-only candidate (E019–E021) and the detection
floor of ~0.35 MAPE points at 236–384 weekly observations, the prior is that most
directions fail Gate A and that the honest output is the audit table plus a
statement of which exogenous axes carry measurable information. A direction that
fails Gate A is not a result and is not reported as one.

## Reproduction

```bash
# Mac mini (data preparation only; Postgres lives here)
uv run python experiment/E022/E022.py prepare
# one direction per run
uv run python experiment/E022/E022.py adl_gpr_vix
```
