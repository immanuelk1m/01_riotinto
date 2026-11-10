# E027 — results (coffee and corn only)

> **Revision (2026-11-09):** the dev/sealed split and the 52-origin seal are gone. Evaluation is the monthly-first-week TSCV, 2025-01..2026-09, h = 1..8 (PROTOCOL.md v1.2 §4). The dev, sealed and `window` text below records the old rule (cell-v3).

Registered in [`README.md`](README.md) before the run; executed 2026-11-01 13:10–13:27 on a
Colab CPU runtime (16.7 min, session stopped). 2,072 cells per direction (1,856 development +
216 sealed), 116 origins × 8 horizons × 2 series, 19 calendar blocks of 26 weeks.

## 1. Hypothesis and sealed block

As registered: on `coffee_r` and on `corn`, a pre-registered exogenous model beats the naive forecast
at 1–8 weeks on the development window by more than this panel's detection floor; statements
are `pooled`, `coffee`, `corn`, and the family is 3 directions × 3 statements = 9 tests.
The most recent 52 origins of each direction are sealed and are **not** part of any number
below except the row labelled *sealed*.

## 2. Gate A audit — development window

26-week calendar-block bootstrap, 2000 draws, one-sided p = P(mean d ≤ 0).

| direction | statement | mean d | 95 % CI | p | MAPE pers → cand | sealed mean d |
|---|---|---|---|---|---|---|
| `gbrt_cc` | pooled | +0.00487 | [+0.00047, +0.00926] | 0.0130 | 6.113 → 5.626 | −0.00032 |
| `gbrt_cc` | coffee | +0.00659 | [+0.00128, +0.01160] | 0.0070 | 6.856 → 6.197 | −0.00057 |
| `gbrt_cc` | corn | +0.00315 | [−0.00401, +0.01097] | 0.2110 | 5.370 → 5.055 | −0.00007 |
| `combo_cc` | pooled | +0.00302 | [−0.00118, +0.00697] | 0.0860 | 6.113 → 5.812 | −0.00202 |
| `combo_cc` | coffee | +0.00451 | [−0.00007, +0.00902] | 0.0285 | 6.856 → 6.406 | −0.00321 |
| `combo_cc` | corn | +0.00153 | [−0.00526, +0.00810] | 0.3295 | 5.370 → 5.218 | −0.00083 |
| `adl_cc` | pooled | −0.00142 | [−0.00650, +0.00350] | 0.7030 | 6.113 → 6.255 | −0.00714 |
| `adl_cc` | coffee | −0.00082 | [−0.00717, +0.00575] | 0.6015 | 6.856 → 6.938 | −0.01271 |
| `adl_cc` | corn | −0.00202 | [−0.00947, +0.00506] | 0.7020 | 5.370 → 5.573 | −0.00157 |

Block win rate (pooled): `gbrt_cc` 0.79, `combo_cc` 0.63, `adl_cc` 0.53.

## 3. Family decision

Holm over the 9 pre-registered tests: α₁ = 0.00556. The smallest p is `gbrt_cc` coffee at
0.0070 — **0 of 9 pass**. `gbrt_cc` pooled would pass an unadjusted Gate A, and that is
exactly the reading the family size exists to prevent: with three directions and three
statements, a 0.013 p-value is the expected best-of-nine.

## 4. Power check against the registration

| statement | registered MDE | observed mean d | power at the observed effect |
|---|---|---|---|
| pooled coffee+corn | 0.0076 | 0.0049 | ≈ 38 % |
| coffee only | 0.0125 | 0.0066 | ≈ 24 % |
| corn only | 0.0130 | 0.0031 | ≈ 5 % |

The design was registered as unable to certify these effects, and it did not. The pooled
interval clearing zero is a low-power event, not evidence on its own.

## 5. Sealed block

`gbrt_cc` drops from +0.0049 on the development window to **−0.0003** on the 52 origins that
were never used (MAPE 5.903 → 5.935). `combo_cc` −0.0020, `adl_cc` −0.0071. The development
gain does not reappear out of sample, which is the same pattern as E022–E026.

## 6. Oracle bound (Gate B 3.4)

A rule that knew per cell whether the naive forecast or the candidate was better would reach MAPE
4.648 against the naive forecast 6.113 on the `gbrt_cc` panel — **1.465 pp available**. `gbrt_cc`
captured 0.487 pp of it (33 %). No deployment rule is evaluated in this round.

## 7. Limitations specific to this data and design

- 19 blocks is the ceiling this history allows at 26-week blocks; every MDE above is larger
  than the effect it was meant to catch, so a null is the registered outcome.
- The panel was chosen from an E024 development-window table, so the development numbers here
  are retrospective (PROTOCOL 4.4) and carry no standalone claim.
- The sealed block is 216 cells per direction. A mean of −0.0003 is "no measurable effect at
  this n", not "no effect".
- E027's leaderboard rows live on a different walk-forward grid (50 % start) than E024's
  (70 %); the two are not comparable cell for cell, and the naive forecast itself moves (coffee
  6.856 here vs 8.440 there) because the window moved.

## 8. Verdict

**No detectable difference, both series.** The gradient-boosting candidate is the only one that
came close, it failed the registered Holm rule, and its development-window gain did not appear
in the sealed block. The forward evaluation registered in `README.md` (52 new origins, target
2027-11-01) is the only remaining path to a claim on these two series.
