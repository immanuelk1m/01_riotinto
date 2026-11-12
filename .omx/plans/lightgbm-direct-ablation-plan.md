# LightGBM Direct 축 계획

## 요구사항·범위

- LightGBM을 horizon별 독립 Direct 회귀로 추가한다. h=1..8 각각 X_t→y_h 모델 하나이며, 앞 horizon의 예측을 뒤 모델 입력으로 넣지 않는다.
- 모든 신규 실험은 base_xgb까지 부모 경로를 유지한다. GP 축이나 제거된 종목별 외생 실험을 복원하지 않는다.
- 이번 턴은 계획만 작성한다. 패키지 설치·소스 수정·학습은 구현 단계에 수행한다.
- 현재 모델은 fit(X,Y), predict(X)→(n,H)이고 XGB는 multi_output_tree 전략을 사용한다 (`src/models/ml/xgb.py:4`, `src/pipelines/modeling/nodes.py:110,165`). 이 외부 인터페이스를 유지한다.

## 1. 계보와 첫 실행 3종

기존 기준 경로:
`base_xgb → xgb_logdiff → xgb_ret_matched → xgb_ret8 → xgb_ret8_depth2 → xgb_ret8_depth2_lr003 → xgb_ret8_reg → xgb_ret8_reg_2015`

분기점은 logdiff 타겟·수익률 lag 0..7·학습 시작 2015-01-04인 현재 XGB 대조군이다 (`conf/studies/p0_p1_v1.yml:270-298`). GP의 가격 수준 타겟을 새 축의 기본값으로 가져오지 않는다.

| 신규 실험 | 부모 | 변경 축 | 비교 목적 |
|---|---|---|---|
| xgb_ret8_reg_2015_ffill | xgb_ret8_reg_2015 | native NaN → 입력 ffill | 결측 처리 효과 |
| xgb_direct | xgb_ret8_reg_2015_ffill | MIMO → horizon별 독립 회귀 | 예측 전략 효과 |
| base_lightgbm_direct | xgb_direct | XGBoost → LightGBM | Direct 조건에서 모델 계열 효과 |

- 세 실험 모두 logdiff, 동일 수익률 lag 8개, 동일 2015년 시작, expanding, 동일 origin·H=8 성숙 조건이다.
- ffill 뒤 최초 관측 이전 결측은 남기고 불완전 학습 행을 제외한다. 추론 행의 잔여 결측은 실패한다. Y·실제 가격은 채우지 않는다. 기존 공통 prepare_input을 사용한다 (`src/pipelines/modeling/nodes.py:36-46,96-109`).
- 표준화는 사용하지 않는다. 스케일링·외생변수·rolling·튜닝을 첫 모델 변경과 동시에 섞지 않는다.
- 기록 단위는 신규 3종이며, 조상 8종·naive·drift를 포함한 연구 명세는 총 13종이다. 기존 조상 결과 재사용은 코드·환경·데이터·spec·산출물 일치 증거가 있을 때만 허용하며, 원래 run ID와 출처를 보존한다. 검증에 실패하면 같은 실행 조건으로 필요한 조상만 재학습한다.

## 2. Direct와 라벨 성숙을 분리

현재 fit은 모든 horizon에 공통으로 완전 성숙한 행, 즉 week+8≤origin만 사용한다 (`src/pipelines/modeling/nodes.py:76-109`).

### 1차 필수: Direct-common

- XGB Direct와 LightGBM Direct 모두 동일한 X_train과 Y_train 행을 사용한다.
- 모델 h에는 Y_train[:,h-1]만 전달한다. 각 head의 fit 표본 수·학습 구간·cutoff가 MIMO 부모와 일치해야 한다.
- 따라서 첫 비교는 전략/모델 효과이며, 더 최근 라벨을 쓰는 효과는 포함하지 않는다.
- 기존 one_output_per_tree 실험도 동일 행을 쓰지만 이번에는 공통 Direct wrapper에서 명시적인 scalar 모델 8개를 보존해 head별 추적을 지원한다 (`conf/base/parameters_experiments.yml:48`).

### 후속 선택 축: Direct-h-mature

- 1차 실행 후 별도로 정의할 수 있다. h별 week+h≤origin과 해당 y_h의 관측 여부로 학습 행을 고른다.
- 단순히 모델 클래스만 교체해서는 구현되지 않는다. 현재 complete=all(y1..y8)와 공통 fit 행 선택을 별도로 확장해야 한다.
- 평가 origin·horizon·actual은 기존과 동일하게 유지한다. 이 후속 축은 이번 첫 실행에 포함하지 않는다.

## 3. 모델·설정 설계

### 공통 Direct wrapper

- 신규 `src/models/direct.py`에 clone 가능한 scalar estimator를 horizon별 8개 학습하는 DirectRegressor를 둔다. 기존 sklearn clone 또는 deepcopy 패턴을 활용한다.
- spec의 선택적 `forecast_strategy: mimo|direct`를 추가한다. 생략한 기존 실험은 기존 동작이다 (`src/experiments.py:11`, `src/pipelines/modeling/nodes.py:31-33`).
- make_model에서 direct이면 scalar base estimator를 wrapper로 감싼다. XGB Direct는 multi_strategy를 scalar 모델에 불필요하므로 제거하고, 나머지 XGB 하이퍼파라미터는 부모와 동일하게 유지한다.
- 이 제거는 forecast_strategy 축에 허용되는 명시적 변경으로 기록한다. 모델 키 xgb는 유지한다. LightGBM 분기에서만 모델 키와 모델 고유 model_params를 바꾼다.
- wrapper.predict는 head 1..8 순서로 column_stack하여 (n,8)을 반환한다. 재귀 예측이나 chain 입력은 사용하지 않는다.

### LightGBM 기준 설정안

- 신규 `src/models/ml/lightgbm.py`, MODELS 키 `lightgbm`, LGBMRegressor 기반 scalar 회귀.
- objective=regression, boosting_type=gbdt, n_estimators=100, learning_rate=0.03, max_depth=2, num_leaves=4.
- min_child_samples=20, min_child_weight=1e-3, reg_alpha=0, reg_lambda=1, subsample=1, colsample_bytree=1.
- CPU, random_state=0, n_jobs=1, deterministic=True, force_col_wise=True. early stopping과 하이퍼파라미터 탐색은 첫 실행에서 사용하지 않는다.
- XGB와 LightGBM은 트리 성장·규제 의미가 다르다. min_child_samples=20을 XGB min_child_weight=20과 동등한 규제라고 설명하지 않는다. 위 값은 사전 고정한 LightGBM 기준 설정이며 모델 계열 변경에 포함된다.
- Direct wrapper와 base estimator의 NaN 허용 속성을 일관되게 전달한다. 이번 연구는 ffill 뒤 완전관측 행을 두 Direct 모델 모두에 제공하므로 native missing 차이를 섞지 않는다.

## 4. 설치와 환경

- 현재 pyproject.toml에는 LightGBM이 없고 로컬 import 검사에서도 설치되지 않은 상태였다 (`pyproject.toml:5-15`). 사용자가 명시한 LightGBM 추가 범위에서 구현 시 의존성을 추가한다.
- uv로 Python 3.14 환경의 실제 호환 버전을 resolve하고 pyproject.toml·uv.lock에 기록한다. 설치 전 버전이나 wheel 가용성을 추측해서 고정하지 않는다.
- macOS에서 OpenMP/libomp 로딩 오류가 발생하면 실제 오류와 설치된 런타임을 확인한 후 필요한 복구만 수행한다. 환경을 이유 없이 광범위하게 변경하지 않는다.
- LightGBM 버전·effective parameters·CPU/스레드 설정을 실행 환경 지문과 각 head audit에 기록한다. 새 환경에서는 이전 run을 동일 실행이라고 표시하지 않는다 (`src/study_runner.py:57-65`).

## 5. 추적·평가와 구현 순서

| 단계 | 대상 | 완료 산출물 |
|---|---|---|
| 1 | pyproject.toml, uv.lock | 호환 LightGBM 설치와 import 확인 |
| 2 | src/models/direct.py, src/models/ml/lightgbm.py, src/models/__init__.py | 공통 Direct wrapper와 등록 |
| 3 | src/experiments.py, src/studies.py, modeling/nodes.py | forecast_strategy 필드·축 검증과 make_model 분기. 기존 MIMO 호환 |
| 4 | 신규 conf/studies/lightgbm_direct_v1.yml | 조상·benchmark·신규 3종과 정확한 changed_paths |
| 5 | src/study_runner.py | 환경 LightGBM 버전·head별 params/표본수/학습 범위·head 순서 audit. 기존 한 run당 한 실험 구조 유지 |
| 6 | 관련 신규 테스트 | 아래 기능·누수·계보 검증과 합성 Kedro 통합 실행 |
| 7 | 고정 스냅샷 실행 | 신규 3종 순차 학습, 동일 셀 parent/root/naive/drift 비교, W&B·로컬 계보와 결과표 |

- 현재 validator에 예측 전략 축은 없고 모델 계열 축은 있다. forecast_strategy 축만 필요한 범위로 확장한다 (`src/studies.py:11-28`).
- MIMO→Direct는 forecast_strategy 및 scalar XGB의 multi_strategy 제거만 허용한다. 모델 교체 축에서는 model/model_params 외 변경을 거부한다.
- W&B는 head별 8개 run으로 나누지 않는다. 한 실험 run 안에서 horizon별 모델 정보를 저장하고, root_run_id/parent_run_id/strategy/changed_paths를 config·Table에 남긴다. 현재 audit는 단일 get_booster를 가정하므로 Direct의 head별 audit를 추가해야 한다 (`src/study_runner.py:80-94`).
- 기준 성과는 기존 가격 공간 capped MAPE·naive 대비 gain이다. 원자재×horizon, 전체·연도별 성과, paired origin-block CI를 함께 보고한다. Direct 학습과 평가를 구분해 새로운 origin을 임의로 추가하지 않는다 (`src/pipelines/modeling/nodes.py:153-173`).

## 6. 테스트 가능한 수용 기준

- [ ] 원자재·origin마다 독립 estimator가 정확히 8개이고, head h에는 해당 Y[:,h-1]만 전달한다.
- [ ] 한 horizon의 Y만 바꾸면 다른 head의 학습 결과는 바뀌지 않는다. head별 estimator 객체를 공유하지 않는다.
- [ ] 출력은 항상 (n,8), 저장 모델을 복원해도 head 순서·예측이 유지된다.
- [ ] Direct-common의 XGB·LightGBM 학습 행·입력 지문이 같고 week+8≤origin을 만족한다.
- [ ] 미래 X/Y 변경은 이전 origin의 전처리·학습·예측에 영향을 주지 않는다. 입력만 ffill한다.
- [ ] 부모와 자식의 실제 spec 차이가 선언된 missing/forecast_strategy/model_family 축과 일치한다.
- [ ] 모든 신규 실험에서 base_xgb까지 경로가 복원되고 부모·루트 run ID를 검증할 수 있다.
- [ ] 기존 MIMO·GP 테스트와 관련 Direct 테스트, 작은 W&B-disabled Kedro 실행이 통과한다. 기존 검사 도구를 사용하고 불필요한 새 lint/typecheck 도구를 도입하지 않는다.
- [ ] 본 실행의 예상 평가 셀 완전성·actual/base 일치·산출물 무결성을 확인하고 W&B 원격 finished 상태를 확인한다.

## 7. 비용·위험과 종료 조건

- 현재 3종×33 origins에서는 Direct 설정 하나당 792개 scalar head를 학습한다. 신규 Direct 두 설정은 총 1,584 head이며, ffill MIMO 대조군은 별도 99개 MIMO 모델이다. 작은 origin에서 시간을 측정한 뒤 순차 실행한다.
- 학습은 실험/origin/head 중 한 층만 병렬화하도록 제한한다. 첫 실행은 n_jobs=1과 순차 head로 시작한다.
- 라이브러리 추가와 wrapper 변경으로 코드·환경 지문이 달라지므로 과거 결과를 무조건 캐시로 복사하지 않는다. 과학 코드 변경이 영향을 주는 조상은 재학습하고, 입력·실제 코드·환경이 검증된 비교 기록만 출처를 표시해 재사용한다.
- LightGBM Direct가 좋아져도 두 축을 한 번에 바꾼 XGB MIMO 대 LightGBM Direct 차이만으로 모델 효과를 주장하지 않는다. Direct XGB 부모와의 비교가 모델 효과다.
- 종료: 구현·회귀/통합 검증 → 사전 고정 신규 3종 학습 → head audit/계보·W&B 검증 → 같은 셀의 결과표. 커널·외생변수·추가 튜닝·h별 성숙 표본 축은 후속 작업으로 남긴다.

## 참고 문서

- [LightGBM LGBMRegressor API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html)
- [LightGBM Parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html)
- [LightGBM Installation Guide](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html)

커널·트리 설정의 비교값은 본 연구의 설계안이다. 구현 시 설치 버전의 공식 API와 실제 동작을 확인한다.
