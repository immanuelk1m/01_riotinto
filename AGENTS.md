# AGENTS.md — `01_riotinto` 실행 정책

## 1. 구조 (Kedro)

```
conf/base/
  parameters.yml               # 모든 실험 공통 평가 설정 (타겟·horizon·origin)
  parameters_experiments.yml   # ablation 목록 — 실험 하나 = 항목 하나
  parameters_exogenous.yml     # 외생변수: 원자재별 시작 날짜(start) + 주간 원인 변수 목록(drivers)
  catalog.yml                  # 산출물 경로
src/                           # Kedro 패키지 자체(package_name = "src") — import 는 `src.features` 처럼 쓴다
  features/                    # 피처 블록: 파일 하나 = 블록 하나, __init__.py 의 FEATURES 에 등록
  models/                      # 모델: 파일 하나 = 모델 하나, __init__.py 의 MODELS 에 등록
    statistical/               #   통계 모델 (naive, seasonal_naive, ols)
    ml/                        #   머신러닝 모델 (xgb)
    dl/                        #   딥러닝 모델 (NeuralForecast)
  pipelines/<이름>/            # Kedro modular pipeline: nodes.py(노드 함수) + pipeline.py(create_pipeline)
    data_ingestion/            #   load_prices, load_exogenous — 등록 이름 `data`, __default__ 에서 빠진다
    data_processing/           #   make_targets
    feature_engineering/       #   build_features, make_tabular, make_timeseries
    modeling/                  #   select_input, make_model, fit, predict — 실험마다 namespace
    reporting/                 #   evaluate, log_run — 실험마다 namespace / summarize — 공유
  pipeline_registry.py         # find_pipelines() 로 pipelines/ 아래를 모두 등록
  experiments.py               # 실험 목록(OmegaConfigLoader 로 읽음)·namespace 규칙 — modeling·reporting 이 함께 쓴다
  utils.py                     # 여러 파이프라인이 쓰는 열 규칙(KEYS, block_keys, xy_columns)
data/                          # 01_raw … 07_reporting, 실행 산출물(git 미추적)
tests/
```

단계: `data → preprocessing → feature → model input → model → learning → inference → evaluation`.
`data`(Postgres 스냅샷)는 별도 파이프라인이다. preprocessing·타겟 격자·피처·모델 입력
(`make_targets → build_features → make_tabular·make_timeseries`)은 모든 실험이 공유하고,
`select_input → make_model → fit → predict`(modeling) `→ evaluate → log_run`(reporting) 은 실험마다 같은 namespace 로 한 벌씩 돈다.
새 단계는 `pipelines/<이름>/` 폴더 하나로 추가한다(`create_pipeline()` 이 있으면 자동 등록, `__default__` 에 들어간다).

실험 namespace 는 `<입력 형식>.<모델 카테고리>.<실험>` 이고(예: `tabular.statistical.ols_lags`), 앞 두 마디가 05·06 의 폴더다:
`data/05_models/<tabular|timeseries>/<statistical|ml|dl>/<실험>.pkl`,
`data/06_model_output/<tabular|timeseries>/<statistical|ml|dl>/<실험>_forecasts.parquet`. 07 은 평평하게 둔다.
- 모델 카테고리 = 모델 파일이 들어 있는 `models/` 하위 폴더.
- 입력 형식 = 모델 클래스의 `input` 속성, 없으면 `tabular`. timeseries 를 읽는 모델은 `input = "timeseries"` 를 적는다.

모델 입력(`data/04_model_input/`)은 **모든 실험이 공유하는 두 파일**이다. 목록의 실험들이 쓰는 (블록, 인자) 조합을
한 번씩만 계산해 옆으로 붙인 넓은 표이고, 실험은 `select_input` 에서 자기 열만 고른다
(`data/03_feature/feature_columns.json` = `{"블록:인자": [열 이름]}`).
- `tabular.parquet`: 행 = (series, week), `logp`·`is_origin`·`y1..yH` + 피처 열. 지금의 `fit`·`predict` 가 읽는다.
- `timeseries.parquet`: Nixtla long 형식 `unique_id, ds, y`(= `logp`) + 피처 열. 빠진 주는 `y` 가 NaN.
  아직 읽는 모델이 없다 — 첫 timeseries 모델을 넣을 때 그 모델의 walk-forward(`ds <= origin`)와 함께 붙인다.
- **열 이름이 같으면 값도 같아야 한다.** 인자가 값을 바꾸는 블록은 인자를 열 이름에 드러낸다(안 그러면 `build_features` 가 멈춘다).

## 2. 실험·모델·피처 추가

- **실험**: `conf/base/parameters_experiments.yml` 에 항목 추가.
  `study`(W&B group), `model`(필수), `features`(블록 목록), `model_params`, `feature_params: {블록: {인자: 값}}`,
  `target`(`level|diff|log|logdiff`, 생략 시 `logdiff` — `src/utils.py` 의 `TARGETS`),
  `train_start`(그 주부터의 행만 학습 — `exog` 실험과 학습 구간을 맞춘 대조군용, 생략 시 전체).
- **모델**: `src/models/<statistical|ml|dl>/<이름>.py` 에 `fit(X, Y) -> self`, `predict(X) -> (n, H)` 클래스 + `MODELS` 한 줄.
  Y·예측은 실험의 `target` 공간이다. 출력 공간이 고정된 모델(naive 류)은 클래스 속성 `target` 을 적는다.
- **피처**: `src/features/<이름>.py` 에 `build(logp, horizon, exog, **params) -> DataFrame` + `FEATURES` 한 줄.
  행 t 에는 **t 까지의 정보만** 넣는다. `exog` 는 그 시리즈의 외생변수 long 프레임(안 쓰는 블록은 무시).
- 평가 설정(`parameters.yml`)은 실험별로 바꾸지 않는다 — 실험 간 비교가 깨진다.

## 3. 실행

```bash
uv run kedro run --pipeline data                 # Postgres `commodity` → data/01_raw (스냅샷 갱신할 때만)
                                                 #   가격 = `ml.price_week`(끝난 W-SUN 주의 마지막 양수 관측값, 앞값 채움·지연 없음)
uv run kedro run --pipeline data --nodes load_exogenous   # 외생변수 스냅샷만 (가격·data_sha 는 그대로)
uv run kedro run                                 # 목록의 모든 실험 + W&B 기록 + data/07_reporting/summary.parquet
uv run kedro run --namespaces tabular.statistical.ols_lags12   # 실험 하나만 — 04_model_input 은 다시 만들지 않는다.
uv run kedro run --namespaces tabular.statistical              # 그 묶음 전체
                                                 #   실험 목록·피처·스냅샷을 바꿨으면 먼저 전체 실행
uv run kedro run --runner ThreadRunner           # 실험 병렬
WANDB_MODE=disabled uv run kedro run             # W&B 기록 없이 (시험 삼아 돌릴 때)
uv run pytest
```

## 4. 평가 규칙

- 타겟: 커피·구리·금 주간(`W-SUN`) 가격 h=1..8 (MIMO). 모델이 학습하는 공간은 실험별 `target`
  (`level` P_{t+h}, `diff` P_{t+h}−P_t, `log` log P_{t+h}, `logdiff` log(P_{t+h}/P_t)), 예측은 가격으로 되돌려 평가한다.
  `06_model_output` 의 `fc`·`actual`·`base` 는 가격(P̂_{t+h}, P_{t+h}, P_t)이다.
- origin: 2024-01 이후 **매월 첫 주**, H개 타겟이 모두 성숙한 origin 만. 모델은 origin 마다
  라벨이 성숙한 행(`week + H <= origin`)만으로 expanding-window 재학습한다.
- `data/07_reporting/{실험}_metrics.parquet` (series × horizon):
  `mape`(셀 APE 100% 캡 후 평균, %), `max_ape_pct`(캡 없는 **최악 단일 APE%** — 위험도 지표로만 읽는다),
  `mape_naive`(같은 셀에서의 naive forecast MAPE), `gain_pct`(naive forecast 대비 개선율, 양수가 좋음).

## 5. 실험 원장 = W&B (`sonet_krr/riotinto`)

- 실험이 돌 때마다 마지막 노드 `log_run` 이 run 하나를 남긴다. **끝난 실험은 YAML 에서 지운다 — 기록은 W&B 에 남는다.**
  `parameters_experiments.yml` 은 "지금 돌릴 실험"만 담는다.
- run 에 담기는 것: config(실험 spec + 공통 평가 설정 + `data_sha`·`data_last_week`·`features_sha`), summary(`mape`, `mape_naive`,
  `gain_pct`, `max_ape_pct`, 시리즈별 `mape/<series>`·`gain_pct/<series>`), Table `metrics`·`cells`.
  parquet·모델 파일은 올리지 않는다.
- run id 는 (실험 이름 + spec + 평가 설정 + 데이터 스냅샷 + 실제 피처 값 `features_sha`)의 해시다. 같은 조건의 재실행은
  같은 run 을 덮어쓰고, spec·스냅샷·피처 값(외생 스냅샷, `start`, 피처 코드)이 바뀌면 새 run 이 된다.
  **모델 코드**만 바꾼 재실행은 덮어쓰므로, 따로 남기려면 실험 이름을 바꾼다.
- `study` 키가 W&B group 이다 — ablation 한 묶음마다 이름을 준다.
- 서로 다른 `data_sha` 의 run 은 직접 비교하지 않는다(스냅샷이 다르면 평가 셀이 다르다).
- 인증은 `~/.netrc`(`uv run wandb login`) — 키를 레포에 넣지 않는다.

## 6. 외생변수

- 스냅샷 `data/01_raw/exogenous.parquet` (series, col_name, week, value, code, drivers, ml) 은 Postgres
  `ml.series_week_lagged` 를 그대로 받은 것이다 — 일요일 마감 주간 값, 앞값 채움, 발표 지연이 이미 적용돼 있다.
  갱신이 끊긴 변수도 마지막 값이 계속 이어진다(걸러내지 않는다).
- 두 묶음: `ml` = 그 원자재의 `ml.feature` 변수(스냅샷 때 DB 에서 읽음), `drivers` = `parameters_exogenous.yml` 에 적은
  주간 원인 변수. 뷰에 없는 drivers 는 경고를 남기고 건너뛴다.
- `start`(원자재별): 그 주부터의 외생 데이터만 쓴다. 늦게 시작하는 변수도 빼지 않는다(시작 전은 NaN).
  `exog` 를 쓰는 실험의 학습 구간은 `start` 부터로 고정된다(`fit` 이 자른다).
- 결측: `allow_nan = True` 모델(xgb)은 X 의 NaN 을 그대로 받는다. 아닌 모델(ols)은 X 가 모두 있는 행만 쓰므로,
  늦게 시작하는 변수가 있으면 학습 구간이 그 변수의 시작부터로 줄어든다.
- 고르기(`exog` 블록): `feature_params: {exog: {source: drivers|ml|all, codes: [...], exclude: [...]}}`.
  `codes`·`exclude` 는 `value_manifest.code` 접두어다 — 대 `MAC`, 중 `MAC.INF`, 소 `MAC.INF.PRODUCER`.
- 값은 기본으로 수준 그대로 들어간다(표준화 없음). `feature_params: {exog: {diff: k}}` 면 k주 변화량이고 열 이름은 `<col>__d<k>`.

## 7. 용어

- 기준선은 **naive forecast**(로그수익률 0, 즉 `P_t` 유지)라고 부른다. 개선은 "naive forecast 대비"로 쓴다.

## 8. 추적 가능한 연구 실행

- `conf/studies/p0_p1_v1.yml`은 `base_xgb`부터의 고정 ablation 명세다. `lineage`의 부모·변경 축·실제 설정 차이를 실행 전에 검증한다.
- `uv run python tools/run_study.py`로 전체 연구를 실행한다. `--phase anchor|features|adaptation`은 해당 단계와 필요한 조상·비교 기준선을 함께 실행한다. `--wandb-mode disabled`는 합성 검증용이다.
- 일반 `parameters_experiments.yml`과 기존 03~07 산출물은 보존한다. 연구 산출물은 `data/07_reporting/studies/<study_id>/<execution_fingerprint>/`에 입력·소스·모델·예측·계보·비교표와 함께 저장한다.
- 완료 run은 설정·입력·코드·환경 지문과 산출물 무결성이 일치할 때만 재사용한다. 연구 run은 코드 변경도 다른 ID로 기록하며, 기존 일반 실행의 run ID 규칙은 유지한다.
- 연구 spec의 `train_window_weeks`는 origin 기준 최근 달력 주의 범위다. rolling에서도 `week + H <= origin`을 유지한다. `prediction` 파생 실험은 부모 예측으로 계산하며 재학습하지 않는다.
- `columns_by_series`는 원자재별 정확한 외생변수 목록이다. 빈 목록은 사용하지 않는다는 뜻이며, 요청한 변수가 없으면 실패한다. 선택 범위가 다른 피처는 열 이름도 구분한다.
- 평가에는 naive forecast·직접 부모·루트 대비 같은 셀의 성과, 연도별 결과와 origin 블록 bootstrap을 남긴다. 현재 연구는 역사적 탐색이며 이미 본 기간을 미사용 holdout으로 부르지 않는다.
- GP 기준선 연구는 `uv run python tools/run_study.py --study conf/studies/gp_baseline_v1.yml`로 실행한다. `base_xgb → base_xgb_ffill → base_xgb_ffill_scaled → base_gaussian_process` 순서로 변경 축을 분리한다.
- `preprocessing: {missing: ffill, scaling: standard}`는 선택한 입력 X만 원자재별 순방향으로 채우고, 성숙 학습 행으로만 StandardScaler를 적합한다. 선두 결측 학습 행은 제외하며 추론 결측은 실패시킨다. 타겟·실제 가격은 채우지 않는다.
- GP의 origin별 체크포인트는 해당 run ID 경로에 저장하고 무결성 확인 후 재사용한다. 첫 origin 비용 측정에 사용한 모델도 전체 실행에서 재사용한다.
