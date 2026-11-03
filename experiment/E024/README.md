# E024 — pre-registration (exogenous-information round)

Gate C registration for this experiment. The round-level registration, the Gate B
axis statement, the sealed-block rule and the family size (23 directions) are in
[../FAMILY.md](../FAMILY.md); the full rationale and the literature provenance are
in [../E022/README.md](../E022/README.md).

**Candidate:** the direction's exogenous model forecast.
**Matched control:** E021 ridge (price-only), same walk-forward.
**Baseline:** persistence (zero log-return).
**Primary metric:** cell MAPE against persistence; Gate A on the paired capped-APE
difference with a 26-week moving-block bootstrap 95% interval.
**Sealed block:** most recent 52 origins, excluded from eligibility, reported once.
**No deployment rule** is evaluated in this experiment.

## Directions

- `adl_parity_fx` — model adl; targets corn, soybean, wheat; exogenous brl, dollar
- `adl_enso` — model adl; targets coffee_r, cocoa, sugar, rice; exogenous oni, soi
- `arimax_rain` — model adl; targets corn, soybean, wheat, rice; exogenous rain, crude
- `lasso_agri` — model lasso; targets corn, soybean, wheat, coffee_r, cocoa, sugar; exogenous oni, soi, rain, brl, dollar, crude, bdi, fertilizer
- `gbrt_agri` — model gbrt; targets corn, soybean, wheat, coffee_r, cocoa, sugar; exogenous oni, soi, rain, brl, dollar, crude, bdi, fertilizer
- `combo_agri` — model combo; targets corn, soybean, coffee_r; exogenous oni, brl, crude, bdi

## Expected outcome (stated in advance)

Prior from the 2026 literature (Souza et al. 2026: ex-ante exogenous gains vanish
in the holdout; Vairo et al. 2026: weather adds nothing at monthly frequency;
Zhang et al. 2026: macro augmentation loses to the random walk at 1–5 days): most
directions fail Gate A. The honest output is the audit table plus a statement of
which exogenous axes carry measurable information. A failing direction is not a
result and is not reported as one.

---

## Under the constant-free pipeline (v2, 2026-11-03)

This experiment was re-registered in [`../FAMILY_v2.md`](../FAMILY_v2.md) after Gate F
removed every direction whose model carries a free parameter. What remains here is the
ADL subset; `lasso`, `pcr`, `gbrt` and `combo` are listed under
`retired_directions` in `config.json` and in [`../RETIRED.md`](../RETIRED.md).

The series are no longer resolved by a regex at prepare time: `config.json` now names
the `col_name` that the v1 selector happened to pick, and a direction refuses to run if
a declared series is missing from the panel.
