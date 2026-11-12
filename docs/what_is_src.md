### features, pipelines의 역할

src/features/: 피처 계산 함수 모음

- 파일 하나가 블록 하나입니다. 지금은 lags.py, seasonal.py, exog.py 세 개가 있습니다.
- 모든 블록은 같은 형태를 따릅니다: build(logp, horizon, exog, **params) -> DataFrame
  - logp는 한 시리즈의 주간 로그가격, exog는 그 시리즈의 외생변수입니다.
  - 행 t에는 t까지의 정보만 넣습니다(미래 정보 누설 금지).
- __init__.py의 FEATURES dict에 이름과 함수를 등록합니다. 실험 YAML의 features: [lags, exog]는 이 이름으로 블록을 찾습니다.
- Kedro를 전혀 모르는 순수 함수입니다. 파일 입출력, 시리즈 순회, 실험 목록 같은 건 다루지 않습니다.

src/pipelines/: Kedro 실행 흐름

폴더 하나가 Kedro modular pipeline 하나이고, 각각 두 파일로 이뤄집니다.
- nodes.py: 노드 함수. 실제 일을 하거나 features/, models/를 불러옵니다.
- pipeline.py: create_pipeline(). 어떤 데이터셋(catalog.yml)과 파라미터를 받아 어떤 데이터셋을 내보낼지 연결합니다.

┌─────────────────────┬─────────────────────────────────────────────────────────────────────┐
│     파이프라인      │                               하는 일                               │
├─────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ data_ingestion      │ Postgres → 01_raw 스냅샷 (--pipeline data로만 실행)                 │
├─────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ data_processing     │ 타겟 격자(y1..yH) 만들기 (주간 가격은 DB 뷰 ml.price_week)          │
├─────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ feature_engineering │ 피처 블록 계산 → 04_model_input의 두 공유 파일 생성                 │
├─────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ modeling            │ 실험마다 select_input → make_model → fit → predict                  │
├─────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ reporting           │ 실험마다 evaluate → log_run(W&B 기록), 마지막에 공유 노드 summarize │
└─────────────────────┴─────────────────────────────────────────────────────────────────────┘

둘의 연결 지점

pipelines/feature_engineering/nodes.py의 build_features가 둘을 잇습니다.
1. 실험 목록 전체에서 필요한 (블록, 인자) 조합을 중복 없이 모읍니다.
2. 시리즈마다 FEATURES[name](logp, horizon, exog, **params)로 블록을 호출합니다.
3. 결과를 옆으로 붙여 넓은 표 하나로 만듭니다. 같은 열 이름에 다른 값이 나오면 멈춥니다.
4. make_tabular와 make_timeseries가 이 표를 tabular.parquet, timeseries.parquet로 저장합니다.

models/와 pipelines/modeling/도 같은 관계입니다. models/는 fit/predict 클래스만 갖고 있고, 실험별 namespace를 만들어 돌리는 일은 modeling 파이프라인이 맡습니다.

그래서 새 피처를 추가할 때는 features/에 파일 하나와 FEATURES 등록 한 줄만 쓰면 됩니다. pipelines/는 고칠 필요가 없습니다.