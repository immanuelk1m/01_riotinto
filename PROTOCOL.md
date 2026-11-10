# Research Protocol

**Version:** 1.2 (2026-11-09) — §4 replaces the sealed block with the fixed monthly-TSCV
evaluation origins; the leaderboard drops `window` (evaluator cell-v4).
**Previous:** 1.1 (2026-11-03) — added Gate F (constant discipline) and the v2
leaderboard schema.
**Superseded:** 1.0 (2026-10-31).
**Scope:** every experiment under `experiment/`, every report under `.omx/research/`.
**Status:** normative. Where this document conflicts with an experiment README or a
report narrative, this document wins.

This protocol exists because four experiments (E018–E021) ran to completion without
producing a single result that beat the naive forecast in a way an outside reader would
accept. The failures were not of causal discipline — that held up — but of
**ordering and search space**: effect sizes, power and the external reference point
were checked after implementation, and three of the four rounds varied only the
decision rule on a candidate that was already worse than the baseline. Appendix A
records the evidence.

Comparisons in this repository are stated as **margins over the naive forecast and over a
matched control, with confidence intervals**. No experiment may be described as
state of the art, best in class, or otherwise superior without satisfying every gate
below; those words are not used as claims anywhere in the repository.

---

## 1. Purpose and scope

1.1 Every experiment passes Gates A–E in order. A gate that fails stops the work at
that point; later gates are not run.

1.2 Gates are checked at **design time** where stated (A, B, C) and at **evaluation
time** (D, E). Retrospective compliance is not accepted: an experiment that skips a
design-time gate is reported as non-compliant, not as a result.

1.3 The unit of a round is one pre-registered hypothesis (Gate C). A round may run
several directions only if they are declared together with a stated family size
(Gate D).

---

## 2. Gate A — Baseline gap audit (mandatory, before implementation)

2.1 Before any deployment rule, calibration, aggregation or adapter is written, audit
the candidate itself. For every (model, series set) pair intended for use, compute the
paired per-cell difference

```
d_c = err_naive(c) - err_candidate(c),   c = (series, horizon, origin)
err = min(1.0, |expm1(forecast - log_return_target)|)      # capped APE
```

2.2 Require **mean(d) > 0 and a 95% interval excluding 0**. Because horizons overlap,
the interval must come from a **moving-block bootstrap with 26-week blocks**, not from
an i.i.d. formula.

2.3 A candidate that fails 2.2 is **ineligible** for rule, calibration or adaptation
experiments on that series set. Record the audit table in the report:

| model | series set | n cells | mean d | 95% CI | eligible |
| --- | --- | --- | --- | --- | --- |

2.4 If no candidate is eligible, the round's honest output is the audit table plus a
statement that no transferable edge exists on this data. That is a valid result and
must not be dressed up as a method contribution.

---

## 3. Gate B — Search-space policy

3.1 Three axes exist: **data** (series, panel, vintage), **features** (inputs and
their timing), **model** (architecture, training objective, adaptation).

3.2 A round changes **at least two axes**. Changing only the model, or only the
features, is allowed when the other axis was changed in a recent round and the report
states which.

3.3 **Rule-only experiments are restricted.** A gate, weighting scheme, FDR procedure,
conformal step or aggregation layer over an unchanged forecast may be run only if that
forecast passes Gate A on the same series set.

3.4 Every deployment-rule experiment reports the **oracle-gate bound**: the MAPE of a
rule that knows, per cell, whether the candidate or the naive forecast was better, and the
fraction of that bound the rule captured. If the bound is inside the noise band from
Gate D, the rule cannot produce a reportable gain and the experiment should not be
started.

---

## 4. Gate C — Pre-registration and the evaluation origins

4.1 Before implementation, write the hypothesis in the experiment README:

- candidate and its Gate A audit row;
- matched control and baseline;
- primary metric and decision rule;
- the declared family size (number of variants that will be evaluated).

4.2 **Evaluation is one fixed rule** (revision 2026-11-09; it replaces the dev/sealed
split of 1.1). The evaluation origins are the **first week of each month** from
`TSCV_START` (2025-01) to `TSCV_END` (2026-09), where the first week of a month is the
W-SUN week-end that first falls inside it. At each origin the model is refit on the
information mature at that origin (expanding window) and forecasts `h = 1..8` weeks. An
origin counts for a series only when all eight targets have matured. Nothing is held
back: every cell the leaderboard reports is a cell of these origins.

4.3 Because no block is sealed, a result is no longer protected from the researcher who
saw it. Register the hypothesis and decision rule before the run, and report every
direction that was tried (the family size in 4.1). A rule changed after the cells were
seen is exploratory and labelled so.

4.4 Reusing a previously inspected interval is allowed but must be declared; such a
result is retrospective and cannot be the sole evidence for a claim. The 2026-11-09
revision itself is retrospective: the E028 sealed result was known when the seal was
removed, so E028 under 4.2 is exploratory, not confirmatory.

4.5 With about twenty origins 4–5 weeks apart, an 8-week horizon overlaps the next
origin's targets. Origin-level tests use the HAC lag `ORIGIN_LAG` (origins), and the
26-week bootstrap blocks (§5.4) give only a handful of blocks, so the interval is
reported but is not the decision statistic.

---

## 5. Gate D — Evaluation rules

5.1 **Cell** means `(series, horizon, origin)`. A cell MAPE is
`mean(|forecast/target - 1|) * 100`. Aggregate rows (for example `horizon = 'all'`)
**must not enter a cell mean**; they may be reported separately and labelled.

5.2 **Primary metric:** cell MAPE of the deployed forecast. **Secondary:** 26-week
block win rate against the matched control and against the naive forecast, plus deploy
coverage.

5.3 **Comparability.** Two experiments may be compared only when `n_series`, cell
count and `evaluator_version` match. Series sets of five and six commodities are never
averaged into one ranking; mixed comparisons are reported as separate tables with the
coverage stated in the caption.

5.4 **Uncertainty.** Report a paired 26-week block bootstrap interval for the primary
metric against its control. A margin whose interval includes 0 is reported as "no
detectable difference", regardless of sign.

5.5 **Family size.** State how many variants were evaluated. With more than one, apply
Holm or Benjamini–Hochberg at the declared level and report both raw and adjusted
values. Selecting the best of many variants post hoc is not a finding.

5.6 **Deploy coverage floor.** A rule that wins by falling back to the naive forecast almost
everywhere reports its coverage; coverage below 10% of blocks is non-transfer.

---

## 6. Gate E — Contract in code

6.1 Five items are machine-checked by `experiment/transfer_contract.py`. Each returns
pass or fail with the numbers behind it.

1. **Margin** — deployed cell MAPE is below both the matched control and the naive forecast,
   with a bootstrap interval excluding 0.
2. **Block transfer** — the next-block win rate, computed per 26-week block, exceeds
   50%.
3. **Coverage** — deploy coverage is at least 10% of blocks.
4. **Safety** — no individual series regresses by more than 0.05 MAPE against its
   matched control.
5. **Mutation invariance** — recomputing with every label whose target date is later
   than a cutoff perturbed leaves all decisions and issued forecasts unchanged.

6.2 Failure of any item labels the direction **non-transfer**, even if the aggregate
MAPE improved. The label goes in the report and in the leaderboard row.

6.3 The leaderboard records `n_series`, `cells`, `evaluator_version` and
`protocol_version` so that comparisons can be checked mechanically rather than
remembered.

---

## 7. Gate F — Constant discipline

Gate E made the contract executable. Gate F makes the *search space* executable: a
number that changes what is estimated is a degree of freedom, and an unregistered
degree of freedom cannot be audited after the fact.

7.1 **Every constant carries a source.** A number may enter the pipeline only through
`_exo/params.py`, with a `source` of `protocol` (a clause of this document fixes it),
`definitional` (forced by the meaning of the object) or `declared` (written down
before the run, with a rationale). There is no fourth category. A constant whose
origin is unknown cannot be represented in the registry at all, so it cannot be
imported by accident.

7.2 **No model with a free parameter may be ranked.** A direction may appear in the
leaderboard as eligible only if its forecast comes from a model whose definition is
fixed by this protocol or by a closed form with no penalty, no architecture and no
search. A model whose result depends on a hand-set `alpha`, `k`, depth, learning rate
or clip bound is **retired**: its rows are kept under `status = 'retired-heuristic'`
and are never compared against the naive forecast.

7.3 **Declared constants must be shown not to matter.** A `declared` constant that
could decide a conclusion requires a sensitivity table over a pre-declared grid,
published with the result. A constant that does move the conclusion is a finding
about fragility and is reported as one.

7.4 **Nothing is rescued retroactively.** A constant can be promoted to `declared`
only before the run it governs. Once a number has been observed, a post-hoc rationale
is not a substitute for registration.

7.5 **Direction names must name the estimator that ran.** A direction called
`arimax_*` that fits an OLS is a false label and is renamed or removed.

7.6 **The leaderboard records provenance.** In addition to §6.3, every row carries
`params_hash`, `params_version`, `pipeline`, `aggregate` (1 for `horizon = 'all'`, so
§5.1 is a filter rather than a memory) and `status`.
`tools/leaderboard_lint.py` fails when an eligible row lacks any of them, when a
metric is not a MAPE, or when the migrated table lost a row.

---

## 8. Reporting template

Every report contains:

1. Hypothesis as registered, and the evaluation origins (4.2).
2. Gate A audit table for the candidate, including the ineligible entries.
3. Gate B statement: which axes changed, and the oracle-gate bound with the fraction
   captured.
4. A variant table: direction, rule, deploy coverage, primary metric with interval,
   block win rate, contract verdict.
5. Family size and adjusted values.
6. Limitations that are specific to this data and this design, not generic.
7. A one-line verdict: transfer, non-transfer, or no detectable difference.

There is no field for state-of-the-art claims.

---

## 9. Anti-patterns

- Choosing a direction after reading the evaluation block.
- Quoting a mechanism's published gain from a different signal regime without a Gate A
  audit on this data.
- Reporting an aggregate improvement produced by a minority of large wins while the
  deployed-worse rate is uncontrolled.
- Treating overlapping horizons as independent observations.
- Averaging cells across different series sets into a single ranking.
- Declaring a rule "calibrated" because a quantile was fitted, without a coverage
  check.
- Letting a fallback dominate the deployed series and calling the difference a gain.
- Re-running a launch that already wrote artifacts without inspecting them first.

---

## Appendix A — Evidence from E018–E021 (descriptive, not normative)

Computed from the E015 frozen candidate archive with equal weight per commodity,
horizons 1–8:

| Quantity | Value |
| --- | --- |
| naive-forecast MAPE | 6.180 |
| candidate (always deployed) MAPE | 6.194 |
| oracle gate MAPE (knows per-cell winner) | 5.976 |
| oracle bound on any deployment rule | **0.204 MAPE** |
| best rule observed (E019 `cap`) | 6.160 |
| fraction of the bound captured | **10%** |

Other facts recorded during the audit:

- E019 `cap` reached 6.173 on the leaderboard, but 54 of 200 deployed commodity-weeks
  (0.27) were worse than the naive forecast against a 0.10 target — a Gate E item 1/4 failure
  that aggregate MAPE hid.
- Across 71 earlier comparisons, none survived a Bonferroni threshold of 0.0007; the
  best uncorrected p-value was 0.0032 for `E007_robust`, itself the maximum of five
  directions chosen on the same outer interval.
- The per-commodity effect that E018 was trying to certify was 0.0002 in capped-APE
  units against a weekly standard deviation of 0.0151, which is why every calibrated
  gate declined to deploy.
- E021 expanded the panel from 6 to 730 series. The validation-to-outer rank
  correlation became measurable (0.263) but only 9 of 730 series certified an edge at
  0.05, and the mean gains disagreed in sign (validation −0.0038, outer +0.0028).

These numbers are the reason for Gates A and B: the rule was never the binding
constraint, and re-gating a weaker-than-baseline candidate could not have exceeded the
0.204 bound while capturing only a tenth of it.

---

## 한국어 요약

- **목적**: 실험 착수 전에 통과해야 하는 5개 게이트를 고정한다. "SOTA"라는 표현은
  이 저장소에서 주장으로 쓰지 않고, 모든 비교는 지속예측·매칭 대조 대비 margin과
  신뢰구간으로만 서술한다.
- **Gate A (착수 전)**: 후보 자체가 baseline보다 나은지 paired 26주 블록 부트스트랩
  CI로 먼저 감사한다. 0을 포함하면 그 후보로 규칙·보정 실험을 시작하지 않는다.
- **Gate B (설계)**: 데이터·특징·모델 3축 중 최소 2축을 바꾼다. 규칙만 바꾸는 실험은
  후보가 Gate A를 통과했을 때만 허용하고, 오라클 상한과 회수율을 반드시 보고한다.
- **Gate C (사전등록)**: 라운드당 주가설 1개. 평가는 dev/sealed 없이 단일 규칙 —
  2025-01~2026-09 각 월 첫 주를 origin으로 하는 TSCV(h=1..8). 봉인이 없으므로 가설·판정 규칙을
  실행 전에 등록하고, 본 뒤에 바꾼 규칙은 탐색적으로 표기한다.
- **Gate D (평가)**: 'all' 행을 셀 평균에서 제외, 시리즈 수·셀 수·evaluator_version이
  같을 때만 비교, family size와 보정 p값 공개, 커버리지 10% 미만은 non-transfer.
- **Gate E (계약 코드화)**: 5개 항목을 `experiment/transfer_contract.py`로 자동 판정,
  하나라도 실패하면 집계 MAPE와 무관하게 non-transfer로 표기한다.
- **부록**: E018~E021 감사 수치(오라클 상한 0.204 중 10% 회수, deployed-worse 0.27,
  Bonferroni 0/71, 730종 중 9종만 인증)를 규범이 아닌 근거로 기록.
