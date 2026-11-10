`conf/base/parameters_experiments.yml`의 `ols_drivers` 실험(`model: ols`, `features: [lags, exog]`, `exog.source: drivers`) 파이프라인 정리 (namespace: `tabular.statistical.ols_drivers`).

파이프라인은 `src/pipelines/<이름>/` 다섯 개로 나뉨 (각 폴더 = `nodes.py` + `pipeline.py`, `src/pipeline_registry.py`의 `find_pipelines()`가 자동 등록).

| 파이프라인 | 노드 | 실행 단위 |
| :--- | :--- | :--- |
| `data_ingestion` (등록 이름 `data`) | `load_prices`, `load_exogenous` | 스냅샷 갱신할 때만 (`__default__`에서 빠짐) |
| `data_processing` | `to_weekly`, `make_targets` | 모든 실험 공통 1회 |
| `feature_engineering` | `build_features`, `make_tabular`, `make_timeseries` | 모든 실험 공통 1회 |
| `modeling` | `select_input`, `make_model`, `fit`, `predict` | 실험마다 namespace 한 벌 |
| `reporting` | `evaluate`, `log_run` / `summarize` | 실험마다 namespace 한 벌 / 전체 1회 |

---

### 0. 스냅샷: `data_ingestion` (별도 실행: `uv run kedro run --pipeline data`)
* **`load_exogenous` (`src/pipelines/data_ingestion/nodes.py:25`)**
  * 원자재별 외생변수 목록 생성: `ml.feature` 변수(`ml=True`)와 `parameters_exogenous.yml`의 `drivers`(`drivers=True`)를 outer merge함.
  * 타겟 자기 가격은 목록에서 제외함(:42).
  * `ml.series_week_lagged`에서 값 읽고 `value_manifest` 붙여 code 매핑함 (W-SUN 주간 집계, ffill, 발표 지연 이미 반영된 뷰임).
  * 뷰에 없는 변수는 경고 로그 남기고 스킵함.
  * 결과물: `data/01_raw/exogenous.parquet` (`series, col_name, week, value, code, drivers, ml`).

---

### 1. 공유 단계 (모든 실험 공통 1회 실행)
**`data_processing` (`src/pipelines/data_processing/nodes.py`)**
* **`to_weekly` (:8)**: 일별 가격의 주간 마지막 값 사용, 0 이하는 제거함.
* **`make_targets` (:17)**: 달력 격자 위에 `logp`, $y_1 \dots y_8 = \log p(t+h) - \log p(t)$, `is_origin`(2024-01 이후 매월 첫 주) 생성함.

**`feature_engineering` (`src/pipelines/feature_engineering/nodes.py`)**
* **`build_features` (:26)**: 전체 실험의 (블록, 인자) 조합 모아서 시리즈별 1회 계산함. 블록 키는 `src/utils.py`의 `block_keys`가 만듦.
  * `_exog_from` (:12): 해당 원자재 행만 필터링 후 `start(2015-01-04)` 규칙 적용함.
    * 첫 관측값이 `start`보다 늦으면 통째로 드랍함(:21).
    * 통과한 변수도 `week >= start` 행만 남김.
  * `exog.build` (`src/features/exog.py:14`): `source=drivers` 처리 후 week × col_name 으로 pivot 해서 `logp.index`에 reindex함(:30).
  * `lags.build` (`src/features/lags.py`): 최근 8주 주간 로그수익률 `lag0..lag7`.
  * 결과 컬럼은 `feature_columns.json`에 `{"exog:{\"source\": \"drivers\"}": [39개], "lags:{}": [8개]}` 형태로 기록됨.
  * 3개 원자재 drivers의 합집합이라 총 39개이며, 타 원자재 전용 열은 해당 시리즈에서 전부 NaN 처리됨.
* **`make_tabular` (:49)**: 타겟 격자 + 피처 병합하여 `04_model_input/tabular.parquet`로 저장함.

---

### 2. 실험별 단계 (`tabular.statistical.ols_drivers`)
namespace 는 `src/experiments.py`의 `namespace()`가 `<입력 형식>.<모델 카테고리>.<실험>` 으로 만들고, `modeling`과 `reporting`이 같은 namespace 를 씀.

**`modeling` (`src/pipelines/modeling/nodes.py`)**
* **`select_input` (:11)**: 공유 tabular에서 `lags:{}`와 `exog:{source: drivers}` 열만 추출함.
* **`fit` (:24)**: 시리즈별 walk-forward 학습 진행함.
  * `xcols = dropna(axis=1, how="all")`로 타 원자재 전용 NaN 열 제거함(:34).
  * `has_x`(모든 x가 존재하는 행) 필터링: `start` 이전 주는 외생변수가 NaN이라 자동 탈락하며, 이로 인해 학습 시작점이 2015-01-04로 고정됨.
  * origin $t$마다 라벨이 성숙한 행(`complete & position + 8 <= t`)만 모아서 expanding window 방식으로 재학습함.
  * 모델: `OLS.fit` (`src/models/statistical/ols.py`) 사용함. 상수항 추가 후 `np.linalg.lstsq`로 8개 horizon 동시 추정(MIMO)함 (rank 부족 시 최소노름 해 사용).
  * 결과물: `{(series, origin): 모델}` 형태로 `05_models/tabular/statistical/ols_drivers.pkl`에 저장됨.
* **`predict` (:43)**: 8개 타겟이 모두 성숙한 origin 행에서 타 원자재 열 드랍 후 $\beta$ 곱해서 예측함 (`06_model_output/tabular/statistical/ols_drivers_forecasts.parquet`).

**`reporting` (`src/pipelines/reporting/nodes.py`)**
* **`evaluate` (:13)**: series × horizon별 지표 산출함 (`07_reporting/ols_drivers_metrics.parquet`).
  * `mape`: 단일 셀 APE 100% 캡 적용 후 평균.
  * `max_ape_pct`: 캡 없는 단일 셀 최악 오차.
  * `mape_naive`: 동일 평가 셀 기준 naive forecast MAPE.
  * `gain_pct`: naive forecast 대비 개선율.
* **`log_run` (:84)**: W&B `exog` 그룹에 로깅함. `features_sha`가 외생변수 실제 값의 해시값이라 데이터 스냅샷이나 `start` 바뀌면 새 run으로 생성됨.
* **`summarize` (:36)**: 모든 실험의 평가표를 `07_reporting/summary.parquet` 한 장으로 합침 (namespace 없음, 전체 실행 때만 돎).

실험 하나만 돌리기: `WANDB_MODE=disabled uv run kedro run --namespaces tabular.statistical.ols_drivers` (modeling 4개 + reporting 2개 = 노드 6개).

---

### 3. 현재 스냅샷 기준 수치 요약
| 시리즈 | yml drivers | 실제 사용 외생 열 | 학습 시작일 | origin 수 |
| :--- | :--- | :--- | :--- | :--- |
| **coffee** | 20 | 13 | 2015-01-04 | 34 |
| **copper** | 30 | 24 | 2015-01-04 | 34 |
| **gold** | 24 | 22 | 2015-01-04 | 34 |

---

### 4. 주의할 점
* **설정 변수 누락 건**: yml에 적었다고 다 들어가지 않음. DB 뷰에 없거나 2015-01-04 기준 과거 데이터가 없는 변수는 자동 탈락함 (coffee는 20개 중 7개 탈락). 빠진 변수는 kedro 실행 로그(`외생변수 N개 중 M개 사용`)나 `load_exogenous` 경고 메시지 확인 필요함.
* **naive와 학습 구간 불일치**: naive는 전체 과거 이력을 다 쓰지만 OLS+exog는 2015년부터만 학습함. 단, `mape_naive`를 동일 셀 기준으로 뽑기 때문에 성능 비교 자체는 공정함.
* **변수 스케일링 부재**: 외생변수가 raw 레벨 그대로 들어감. 고용지표 같은 대형 수치와 수익률 lag가 한 OLS에 섞여서 들어가므로, 계산 자체는 `lstsq`로 풀리더라도 회귀계수 크기 비교 해석은 의미 없음.
