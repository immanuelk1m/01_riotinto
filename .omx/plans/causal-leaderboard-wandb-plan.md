# 인과발견 방법론 ablation 종합 리더보드 (W&B) 계획

작성 2026-11-15. 상태: **1단계 완료**(2026-11-15, 프로젝트는 `sonet_krr/causal` 로 확정). 2단계(노트북 3개)는 보류.

## 1. 요구사항

- 원자재(커피·구리·금) 인과발견 방법론 ablation 결과를 한곳에서 순위로 본다.
- 관리 위치는 W&B. CSV 원장은 만들지 않는다(로컬 `summary.csv`·`comparisons.csv` 는 입력으로만 읽는다).
- 진행 중인 연구이므로 다시 올려도 중복 run 이 생기지 않아야 한다.

## 2. 현재 상태 (근거)

- 실행 중인 프로세스 없음. 마지막 실행 `data/07_reporting/causal_study/943bf478abc4c32dece6/` 는 `status.json` = `incomplete`.
  현재 코드·데이터의 `fingerprint()` 도 `943bf478abc4c32dece6` 로 같다 — 결과가 지금 코드와 일치한다.
- `summary.csv`: 48행 = 방법 변형 11개 × 구간(선택 10·확인 6) × 원자재 3. 완료 41, 실패 6, 시간 초과 1.
  - 실패 6건은 모두 `matched_pcmci`·`varlingam` 의 커피·구리: `Insufficient contiguous complete training weeks`
    (`analysis/causal_study.py:119-120`).
  - 시간 초과 1건은 `lpcmci` 구리(45분 한도).
- `comparisons.csv`: 15행 — 부모·baseline 대비 top-3 차이와 4주 블록 bootstrap 95% 구간(`analysis/causal_study.py:191-203`, `:266-276`).
- 주지표는 top-3 적중률(`analysis/causal_common.py:348-366`). 세 원자재가 모두 끝난 변형의 평균:

  | 구간 | 변형 | top-3 | 우연 |
  | --- | --- | --- | --- |
  | 확인 | robust_release | 0.222 | 0.127 |
  | 확인 | robust | 0.182 | 0.127 |
  | 확인 | baseline | 0.178 | 0.127 |
  | 확인 | robust_release_no_direction | 0.008 | 0.127 |
  | 선택 | robust_release | 0.318 | 0.127 |
  | 선택 | robust | 0.283 | 0.127 |
  | 선택 | robust_spline | 0.278 | 0.127 |
  | 선택 | robust_tau8 | 0.227 | 0.127 |
  | 선택 | gpdc | 0.202 | 0.127 |
  | 선택 | baseline | 0.196 | 0.127 |
  | 선택 | robust_window260 | 0.181 | 0.127 |

- W&B `sonet_krr/riotinto`: run 65개, 전부 예측 실험(지표 `mape`·`gain_pct`). 인과 연구 run 은 0개.
  인과 연구 코드에는 W&B 기록이 없다(`analysis/causal_study.py`, `tools/run_causal_study.py` 에 `wandb` 없음).
- 노트북 3개의 결과는 표로 저장돼 있지 않다 — 출력 셀에만 있다.
  - `ATTRIBUTION-ABLATION.ipynb`: `d0..d6`, `e1..e3`, `c1..c4`, `p0..p3` 약 20개 설정. `lru_cache` 만 쓴다.
  - `PCMCI-LPCMCI.ipynb`·`RCBNB-MB.ipynb`: 그래프 캐시(`data/07_reporting/pcmci_lpcmci/`, `rcbnb_mb/`)만 있고 점수표는 없다.

## 3. 결정 필요 (추천안 먼저)

1. **W&B 위치** — 추천: 새 프로젝트 `sonet_krr/riotinto-causal`.
   - 지표(top-3 적중률)·타깃(커피는 CEPEA 헤알 가격)·데이터 지문이 예측 실험과 달라 같은 표에 섞으면 빈 열만 생긴다.
   - `baseline` 이 이미 예측 쪽 group 이름이다. 2026-11-12 에 맞춘 "이름 중복 0"(`.omx/plans/wandb-dedupe-plan.md`)도 깨진다.
   - 대안: 기존 `riotinto` 에 `job_type="causal"` 로 넣고 저장된 보기에서 걸러 본다. `AGENTS.md` 5절 "원장 = `sonet_krr/riotinto`" 와는 이쪽이 맞는다.
2. **범위** — 추천: 1단계는 `causal_study` 실행 결과(변형 11개)만, 노트북 3개는 2단계.
   - 1단계는 입력 파일이 이미 있어 재적합 없이 끝난다. 2단계는 노트북을 다시 돌려 점수표를 내보내야 한다.
3. **run 단위** — 추천: 방법 변형 하나 = run 하나(선택·확인 구간은 같은 run 의 summary 키). 예측 실험의 규칙과 같다.

## 4. 설계

### run (변형 하나 = run 하나)

- id = `sha1(fingerprint + name)[:12]`. 같은 결과를 다시 올리면 덮어쓰고, 코드·데이터·답지가 바뀌면(fingerprint 변경) 새 run 이 된다.
- name = 변형 이름, group = `causal_study.<axis>`(예: `causal_study.release`), job_type = `causal`.
- config: `Spec` 7개 필드(`selection.json` 의 `specs`), `parent`, `axis`, `fingerprint`, 입력 파일 sha(`inputs.json`),
  `protocol`(= `split_v1`: 선택 2023-07~2024-12·학습 <2023-01, 확인 2025-01~·학습 <2025-01, `analysis/causal_study.py:110`, `:126`),
  `uses_answer_key`(= false).
- summary (구간 `p` ∈ `selection`·`confirmation`):
  - `p/top3`, `p/top1`, `p/top3_chance`, `p/top3_lift`(= top3 − 우연), `p/top3_all_weeks`, `p/direction` — **세 원자재가 모두 끝났을 때만** 평균을 적는다.
  - `p/top3/<series>`, `p/top1/<series>`, `p/status/<series>`, `p/seconds`, `p/candidate_count/<series>`.
  - `p/delta_vs_parent`, `p/ci_low`, `p/ci_high`, `p/delta_vs_baseline`, `confirmation/improved`.
  - `complete`(구간별 3개 원자재 완료 여부), `selected`(선택된 변형), `winner`.
- tags: `incomplete`(실패·시간 초과 포함), `selected`, `matched_window`(VARLiNGAM 대조군 — 전체 구간 모델과 순위를 매기지 않는다).
- 한글 열 이름(`top3 (전체 주)`, `방향 일치 (top3 적중 주)`)은 W&B 키에서 위의 영문 키로 바꾼다.

### 리더보드

- 리더보드 = 프로젝트의 **Runs 표**. 정렬 `confirmation/top3` 내림차순, 2차 `selection/top3`.
  끝나지 않은 변형은 평균이 비어 아래로 내려가고 `incomplete` 태그로 구분된다 — 0점으로 읽히지 않는다.
- 한 링크로 보는 표: id 가 고정된 run `leaderboard`(job_type `leaderboard`)에 Table 2개를 매번 덮어쓴다.
  - `leaderboard`: 변형 × 구간 × 원자재 48행(+ 평균 행), 순위 열 포함.
  - `comparisons`: 15행(차이, 95% 구간, `improved`).

### 코드

- 새 파일 `tools/log_causal_leaderboard.py` 하나(약 80줄). 결과 폴더의 `summary.csv`·`comparisons.csv`·`selection.json`·`inputs.json` 만 읽는다.
  - `rows(directory) -> (runs, table, comparisons)`: 순수 함수, W&B 없이 테스트한다.
  - `main`: `--path`(생략 시 `causal_study/latest.json`), `--wandb-mode online|offline|disabled`, `--project`.
- `tools/run_causal_study.py` 끝에 `--wandb-mode`(기본 `online`)를 붙여 연구가 끝나면 같은 함수를 부른다.
- **`analysis/causal_study.py`·`causal_common.py`·`uv.lock` 은 건드리지 않는다.** 이 셋은 fingerprint 에 들어간다
  (`analysis/causal_study.py:140-150`). 바꾸면 결과 폴더가 새로 잡히고 캐시된 적합(lpcmci 금 24분, gpdc 구리·금 9분 등)을 다시 돌린다.
  새 의존성도 같은 이유로 넣지 않는다 — `wandb` 는 이미 있다(`pyproject.toml:14`).

### 2단계 — 노트북 3개 (1단계 승인·확인 뒤)

- 각 노트북 끝에 셀 하나: 같은 열(`name, phase, series, parent, axis, status, top1, top3, 우연, …`)의 표를 만들어
  `data/07_reporting/causal_study/notebooks/<노트북>/summary.csv` 로 쓰고, 같은 `log_causal_leaderboard.py` 로 올린다.
- 비교 가능성은 `protocol` 로 가른다. 서로 다른 `protocol` 은 같은 순위에 넣지 않는다.
  - `ATTRIBUTION-ABLATION`: 확인 구간은 `split_v1` 과 같은 조건(학습 <2025-01, 채점 2025-01~)이라 **확인 구간만** 직접 비교된다.
    선택 구간은 적합에 그 구간이 들어가므로(`CONFIRM` 이전 전체로 적합) `protocol = fit_lt_2025`.
  - 트랙 ②(`p0_prior`, `p1_*`, `p2_*`, `p3_*`)는 `uses_answer_key = true`. 인과 방법 순위와 따로 본다.
  - `PCMCI-LPCMCI`·`RCBNB-MB`: 전체 기간으로 그래프를 찾고 같은 기간을 채점한다 → `protocol = in_sample_full`.

## 5. 구현 단계

1. `tools/log_causal_leaderboard.py` 작성 — `rows()` + `main`.
2. `tests/test_causal_leaderboard.py` 1개: 작은 `summary.csv`(완료 2 변형 + 실패 1 변형)로
   평균은 3개 원자재가 모두 끝난 변형에만 생기는지, id 가 fingerprint·이름으로 정해지는지, 한글 열이 영문 키로 바뀌는지 확인.
3. `tools/run_causal_study.py` 에 `--wandb-mode` 추가.
4. `--wandb-mode disabled` 로 `943bf478abc4c32dece6` 를 읽어 표를 출력하고 2절의 표와 대조.
5. `online` 으로 올리고, Runs 표의 열·정렬을 맞춘 보기를 기본 보기로 저장(UI 에서 한 번).
6. `AGENTS.md` 에 절 하나 추가: 실행 명령, 프로젝트 이름, 키 이름, `protocol` 규칙.

## 6. 인수 기준

- W&B 대상 프로젝트에 `job_type=causal` run 이 정확히 11개, 이름 중복 0, `leaderboard` run 1개.
- `robust_release` 의 `confirmation/top3` = 0.222, `baseline` = 0.178(소수 셋째 자리까지 `summary.csv` 평균과 일치).
- `lpcmci`·`matched_pcmci`·`varlingam` 은 `incomplete` 태그가 있고 평균 키가 없다. 원자재별 값과 `status` 는 남아 있다.
- `robust_release` 의 `confirmation/improved` = true, `confirmation/delta_vs_parent` = 0.041 [0.017, 0.068].
- 같은 명령을 두 번 실행해도 run 수가 12개 그대로다.
- `fingerprint()` 가 실행 전후 모두 `943bf478abc4c32dece6` 이다(캐시가 살아 있다).
- `uv run pytest` 통과.

## 7. 위험과 대응

| 위험 | 대응 |
| --- | --- |
| fingerprint 에 드는 파일을 건드려 12시간 예산의 캐시가 무효가 된다 | 기록 코드는 `tools/` 의 새 파일에만 둔다. 인수 기준에 fingerprint 불변을 넣었다 |
| 코드·데이터가 바뀌면 옛 fingerprint 의 run 이 표에 남아 순위가 섞인다 | 올릴 때 현재 fingerprint 의 run 에 `current` 태그를 달고 나머지에서는 뗀다. 기본 보기는 `current` 만 |
| 끝나지 않은 변형이 0점처럼 읽힌다 | 평균 키를 쓰지 않고 `incomplete` 태그와 원자재별 `status` 로 남긴다 |
| 서로 다른 평가 조건(노트북)이 한 순위에 섞인다 | `protocol`·`uses_answer_key` 를 config 에 적고 기본 보기는 `split_v1` + `uses_answer_key=false` |
| 선택 구간 점수로 고른 변형을 선택 구간 순위로 다시 자랑한다 | 1차 정렬은 확인 구간. 두 구간 모두 이미 본 기간이라 holdout 이라 부르지 않는다(`report.md` 문구 유지) |
| W&B 기록은 외부 전송이다 | 올리는 것은 점수표·설정·파일 해시뿐. 답지·패널·모델 파일은 올리지 않는다 |

## 8. 하지 않는 것

- CSV·엑셀 원장, 로컬 HTML 리더보드.
- W&B Report 자동 생성(`wandb-workspaces` 의존성이 필요하고 `uv.lock` 이 바뀐다) — 보기 저장은 UI 에서 한 번이면 된다.
- 실패한 적합의 재시도·예산 조정(`lpcmci` 구리, 대조군 창 길이) — 별도 작업.
