# E026 — pre-registration (exogenous-information round)

> **Revision (2026-11-09):** the dev/sealed split and the 52-origin seal are gone. Evaluation is the monthly-first-week TSCV, 2025-01..2026-09, h = 1..8 (PROTOCOL.md v1.2 §4). The dev, sealed and `window` text below records the old rule (cell-v3).

Gate C registration for this experiment. The round-level registration, the Gate B
axis statement, the sealed-block rule and the family size (23 directions) are in
[../FAMILY.md](../FAMILY.md); the full rationale and the literature provenance are
in [../E022/README.md](../E022/README.md).

**Candidate:** the direction's exogenous model forecast.
**Matched control:** E021 ridge (price-only), same walk-forward.
**Baseline:** naive forecast (zero log-return).
**Primary metric:** cell MAPE against the naive forecast; Gate A on the paired capped-APE
difference with a 26-week moving-block bootstrap 95% interval.
**Sealed block:** most recent 52 origins, excluded from eligibility, reported once.
**No deployment rule** is evaluated in this experiment.

## Directions

- `adl_broad` — model adl; targets wti, brent, gas_hh, gold, copper_cash, coffee_r, soybean; exogenous vix, dgs10, slope, dollar, gpr, stocks, bdi, oni
- `rf_broad` — model rf; targets wti, brent, gas_hh, gold, copper_cash, coffee_r, soybean; exogenous vix, dgs10, slope, dollar, gpr, stocks, bdi, oni
- `combo_broad` — model combo; targets wti, brent, gas_hh, gold, copper_cash, coffee_r, soybean; exogenous vix, dgs10, dollar, gpr, stocks, bdi
- `mlp_broad` — model mlp; targets wti, gold, copper_cash, soybean; exogenous vix, dgs10, dollar, gpr, stocks, bdi

## Expected outcome (stated in advance)

Prior from the 2026 literature (Souza et al. 2026: ex-ante exogenous gains vanish
in the holdout; Vairo et al. 2026: weather adds nothing at monthly frequency;
Zhang et al. 2026: macro augmentation loses to the random walk at 1–5 days): most
directions fail Gate A. The honest output is the audit table plus a statement of
which exogenous axes carry measurable information. A failing direction is not a
result and is not reported as one.
