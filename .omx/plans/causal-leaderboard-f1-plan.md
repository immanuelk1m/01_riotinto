# 인과발견 리더보드에 F1 추가 계획

작성 2026-11-15. 상태: **완료**(2026-11-15). 선행: `.omx/plans/causal-leaderboard-wandb-plan.md` 1단계(완료).
결정(사용자 확정): **주별 F1@3 과 집합 F1 둘 다** 올린다.

## 1. 요구사항

- `sonet_krr/causal` 리더보드의 방법 변형마다 F1 을 함께 본다.
- 재적합하지 않는다. 저장된 적합 결과(`contributions.parquet`)와 답지로만 계산한다.
- 순위의 기준(주지표)은 지금처럼 확인 구간 top-3 적중률이다. F1 은 옆에 붙는 지표다.

## 2. 사실 (근거)

- 답지는 인용이 있는 주마다 원인 변수를 **정확히 1개** 인용한다(세 원자재 모두, 2023-07 이후 인용 크기 분포 `{0, 1}`).
  - 그래서 주별 F1@3 = 2·TP / (|T| + |C|) = top-3 적중 × 2/(|T|+1)이다. 고른 후보가 1개면 1.0, 2개면 0.67, 3개면 0.5.
    즉 "적게 골라 맞혔는가"를 본다. 주별 recall@3 은 top-3 적중률과 같다.
  - top-1 F1 은 top-1 적중률과 같으므로 넣지 않는다.
- 저장된 결과로 계산할 수 있다. 적합 하나마다 `contributions.parquet`(채점 구간의 주 × 패널 변수, 답지 주만 담김)가 있다
  (`analysis/causal_study.py:129`, `:173`). 열은 채점 패널의 변수다(`analysis/causal_study.py:128`).
- 순위 규칙은 재사용한다: `cc._top`(|기여| 상위 k, 0 제외, 동점은 열 순서), `cc.load_key`, `cc._cited`
  (`analysis/causal_common.py:275-314`, `:369-372`). 짝 비교 구간은 `analysis.causal_study.paired_ci`(4주 블록 bootstrap, `:191-203`).
- 이 파일들은 **import 만 한다**. 수정하면 fingerprint 가 바뀌어 캐시가 무효가 된다(`analysis/causal_study.py:140-150`).
  `tools/` 는 fingerprint 에 들지 않는다.
- 시제품 계산(확인 구간, 세 원자재 평균):

  | 변형 | top-3 | F1@3 | 집합 P | 집합 R | 집합 F1 |
  | --- | --- | --- | --- | --- | --- |
  | robust_release | 0.222 | 0.111 | 0.83 | 0.19 | 0.305 |
  | robust | 0.182 | 0.119 | 0.89 | 0.10 | 0.181 |
  | baseline | 0.178 | 0.130 | 1.00 | 0.10 | 0.183 |
  | robust_release_no_direction | 0.008 | 0.008 | 0.67 | 0.04 | 0.076 |

  F1@3 으로는 순위가 뒤집힌다(baseline 1위). `robust_release` 는 후보를 3개씩 고르고, baseline 은 부모가 1~2개라 적게 고른다.
  시제품의 top-3 재계산값은 저장된 값과 일치했다(예: robust_release 커피 0.200, baseline 커피 0.100).

## 3. 정의

채점 주 W = 그 구간에서 인용 변수가 패널에 있는 답지 주(top-3 적중률과 같은 분모). 주 w 의 인용 변수 C_w, 후보 T_w = `cc._top(contrib, W, 3)`.

- **주별 F1@3** (`f1_3`): F1_w = 2·|T_w ∩ C_w| / (|T_w| + |C_w|), W 에서 평균. 후보가 없는 주는 0.
  - 함께 올리는 값: `precision_3` = |T∩C|/|T|(후보가 없으면 0)의 주 평균, `recall_3` = |T∩C|/|C| 의 주 평균.
  - `f1_3_chance`: 무작위 3개를 고를 때의 기대값 2·3·|C_w|/N / (3 + |C_w|)의 주 평균. N = 패널 변수 수. `top3_chance` 와 같은 자리다.
- **집합 F1** (`set_f1`): S = 그 구간에서 기여가 0 이 아닌 적이 있는 변수(= 채점 패널에 든 그래프 부모), G = ∪_w C_w.
  P = |S∩G|/|S|(S 가 비면 0), R = |S∩G|/|G|, F1 = 2PR/(P+R). `set_precision`·`set_recall` 도 올린다.
  - 읽을 때 주의: P 는 대부분 1.0 이다(부모가 언젠가는 거의 인용된다). 차이는 R(부모 수)에서 난다.
- 평균은 top-3 와 같게 **원자재별 값 → 세 원자재가 모두 끝났을 때만 평균**.

## 4. 구현

1. `tools/log_causal_leaderboard.py`
   - `f1_scores(contrib, cited) -> (dict, Series)`: 순수 함수. 위 정의의 값 7개와 주별 F1 Series 를 돌려준다.
   - `rows(directory, keys=None)`: 완료된 (phase, name, series)마다 `contributions.parquet` 를 읽어 `f1_scores` 를 붙인다.
     `keys` 를 주면 그 답지를 쓰고(테스트), 없으면 `cc.load_key`.
   - 답지 확인: `docs/commodity_weekly_drivers.xlsx` 의 sha256 이 결과 폴더 `inputs.json` 의 값과 다르면 멈춘다
     (실행 뒤 답지를 고쳤으면 F1 이 top-3 와 다른 답지로 계산된다).
   - 일치 확인: 재계산한 top-3(= 주별 hit 평균)가 `summary.csv` 의 `top3` 와 다르면 멈춘다(정의가 어긋나지 않았다는 증거).
   - summary 키 추가 (`p` = `selection`·`confirmation`):
     - 평균: `p/f1_3`, `p/precision_3`, `p/recall_3`, `p/f1_3_chance`, `p/set_f1`, `p/set_precision`, `p/set_recall`.
     - 원자재별: `p/f1_3/<series>`, `p/set_f1/<series>`.
     - 짝 비교(`comparisons.csv` 와 같은 쌍): `p/f1_3_delta_vs_<parent|baseline>`, `p/f1_3_ci_low_vs_…`, `p/f1_3_ci_high_vs_…`,
       `confirmation/f1_3_improved_vs_…`(확인 구간에서 하한 > 0). 집합 F1 은 원자재당 값이 하나라 구간을 붙이지 않는다.
   - `leaderboard` Table: `f1_3`, `set_f1` 열과 `rank_f1`(구간 안 `f1_3` 순위)을 추가한다. 정렬은 그대로 `top3`.
     `comparisons` Table 에 F1@3 비교 행을 `metric` 열(`top3`|`f1_3`)로 구분해 붙인다.
2. `tests/test_causal_leaderboard.py` 에 `test_f1_scores` 1개:
   후보 1개 적중 → 1.0, 3개 중 적중 → 0.5, 빗나감 → 0, 인용 없는 주는 분모에서 빠짐, 후보 없는 주는 0, 집합 P·R·F1 값.
   기존 `test_rows` 는 `keys` 와 작은 `contributions.parquet` 를 넣어 F1 키가 생기는지까지 본다.
3. `AGENTS.md` 9절에 키 이름과 정의 한 줄씩, "F1@3 은 후보 수 벌점이 붙은 top-3 다, 집합 P 는 대부분 1.0" 을 적는다.
4. `uv run python tools/log_causal_leaderboard.py --wandb-mode disabled` 로 표를 보고 2절 표와 대조 → `online` 으로 다시 올린다.

## 5. 인수 기준

- 완료된 적합 41건 모두 재계산 top-3 = `summary.csv` 의 `top3`(차이 < 1e-12).
- `robust_release` `confirmation/f1_3` = 0.111, `confirmation/set_f1` = 0.305. `baseline` 0.130, 0.183(소수 셋째 자리).
- 모든 변형에서 `p/recall_3` = `p/top3`(답지 인용이 주당 1개라서). 다르면 답지 구조가 바뀐 것이므로 보고한다.
- `incomplete` 변형(`lpcmci`·`matched_pcmci`·`varlingam`)은 F1 평균 키가 없고, 끝난 원자재의 `p/f1_3/<series>` 는 있다.
- W&B `sonet_krr/causal` run 수 12 그대로(같은 id 덮어쓰기), 이름 중복 0.
- `fingerprint()` 실행 전후 `943bf478abc4c32dece6`.
- `uv run pytest` 통과.

## 6. 위험과 대응

| 위험 | 대응 |
| --- | --- |
| F1@3 순위가 top-3 와 뒤집혀 "어느 쪽이 맞나" 혼동 | 정렬은 top-3 그대로. AGENTS.md 에 F1@3 = top-3 × 후보 수 벌점이라고 적는다. `rank_f1` 은 별도 열 |
| 실행 뒤 답지가 바뀌어 다른 답지로 F1 을 잰다 | `inputs.json` 의 답지 sha 와 비교해 다르면 멈춘다 |
| 순위 규칙을 다시 구현하다 동점 처리가 어긋난다 | `cc._top` 을 그대로 쓰고, 재계산 top-3 일치를 실행마다 확인한다 |
| 집합 F1 의 P 가 대부분 1.0 이라 정보가 적다 | P·R 을 같이 올리고, 해석 주의를 AGENTS.md 에 적는다 |
| fingerprint 파일을 건드린다 | `analysis/*` 는 import 만. 인수 기준에 fingerprint 불변 |

## 7. 하지 않는 것

- top-1 F1(top-1 적중률과 같다), 원자재 간 micro F1, 그래프 간선 F1(간선 정답이 없다).
- F1 로 선택 규칙(`choose`)을 바꾸는 것 — 연구 코드가 바뀌어 재실행이 필요하다. 원하면 별도 연구로.
- 노트북 3개(기존 계획 2단계, 보류 유지).
