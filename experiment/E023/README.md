# E023 — pre-registration (exogenous-information round)

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

- `adl_real_yield` — model adl; targets gold, silver; exogenous real_yield, dollar, vix
- `adl_basis_macro` — model adl; targets copper_cash, aluminium, nickel, zinc; exogenous copper_3m, dollar, bdi
- `lasso_metal` — model lasso; targets gold, silver, copper_cash, aluminium, nickel, zinc; exogenous real_yield, dgs10, dollar, vix, cu_inv, gold_inv, gpr, bdi
- `pcr3_metal` — model pcr; targets gold, silver, copper_cash, aluminium, nickel, zinc; exogenous real_yield, dgs10, dollar, vix, cu_inv, gold_inv, gpr, bdi
- `gbrt_metal` — model gbrt; targets gold, silver, copper_cash, aluminium, nickel, zinc; exogenous real_yield, dgs10, dollar, vix, cu_inv, gold_inv, gpr, bdi
- `combo_metal` — model combo; targets gold, silver, copper_cash; exogenous real_yield, dollar, vix, cu_inv, bdi

## Expected outcome (stated in advance)

Prior from the 2026 literature (Souza et al. 2026: ex-ante exogenous gains vanish
in the holdout; Vairo et al. 2026: weather adds nothing at monthly frequency;
Zhang et al. 2026: macro augmentation loses to the random walk at 1–5 days): most
directions fail Gate A. The honest output is the audit table plus a statement of
which exogenous axes carry measurable information. A failing direction is not a
result and is not reported as one.
