# P0–P1 원자재 예측 실험 파이프라인 계획

- 작성일: 2026-11-11
- 모드: 계획. 이 문서 작성 단계에서는 실험 설정·소스·W&B 기록을 변경하거나 학습하지 않는다.
- 연구 ID: `p0_p1_v1`
- 목표: 모든 XGB ablation과 파생 예측에서 `base_xgb`까지 경로를 복원하고, naive forecast·직접 부모·루트 대비 성능을 동일 셀에서 비교한다.
- 완료 경계: 계보·평가 기반 구현 → 기준 경로 재현 → 사전 정의한 P1 실행 → 비교표 작성. 미래 데이터 확인은 별도 후속 단계이며 이번 역사적 실험 완료를 막지 않는다.

## 1. 요구사항과 현재 근거

| 현재 확인한 사실 | 설계에 반영할 결정 | 근거 |
|---|---|---|
| 현재 XGB 대조군은 logdiff + 수익률 lag 8개 + 강한 규제 | 이를 원형으로 부르지 않고, 원형부터 변경 경로를 복원한다 | `conf/base/parameters_experiments.yml:19-25` |
| 과거 `baseline_xgb`는 level + 가격 lag [1,2,4,12,52] + 기본 XGB | 새 연구의 루트 이름을 `base_xgb`로 고정하고 과거 이름을 출처로 남긴다 | `wandb/run-20261110_214417-162eb8e3ee99/files/config.yaml` |
| 저장된 원형 모델의 유효 값은 depth=6, eta≈0.3, min_child_weight=1, trees=100, seed=0, CPU | 이 값과 공통 설정을 명시해 라이브러리 암묵적 기본값을 제거한다 | `data/05_models/tabular/ml/baseline_xgb.pkl`의 첫 모델 `get_booster().save_config()`를 읽어 확인 |
| 현재 spec에는 부모·루트·변경 축이 없다 | 기존 spec 검증에 선택적 계보 메타데이터를 추가한다 | `src/experiments.py:11-34` |
| 현재 run ID는 spec·가격 타겟·실제 피처 지문에 의존하며 코드 지문은 없다 | 신규 연구 run에 코드·환경·프로토콜 지문을 추가한다 | `src/pipelines/reporting/nodes.py:49-66`, `AGENTS.md:95-97` |
| 완료 실험은 실행 YAML에서 지우고 W&B에 기록을 남긴다 | 고정 연구 명세와 일회성 실행 대기열을 분리한다 | `AGENTS.md:88-99` |
| 외생 실험은 2015년부터 학습하고 현재 학습은 expanding이다 | 학습 시작일 변경 자체를 독립 노드로 기록한 뒤 외생·rolling 실험을 분기한다 | `src/pipelines/modeling/nodes.py:35-59` |
| 가격 lag는 타겟 기준 offset이고 수익률 lag는 현재부터 연속 8개다 | 피처 표현과 lag 배치 효과를 분리하는 중간 노드를 둔다 | `src/features/price_lags.py:5-8`, `src/features/lags.py:4-7` |
| 입력은 공유 표이고 실험은 자기 열만 선택한다 | 공유 입력 구조를 유지하고 이름별 중복 계산을 피한다 | `src/pipelines/feature_engineering/nodes.py:14-39`, `src/pipelines/modeling/nodes.py:17-23` |
| 현재 평가에는 평균·최대 APE·n·naive 대비 개선만 있다 | P0에 셀 정합성·paired 비교·연도별 결과·불확실성을 추가한다 | `src/pipelines/reporting/nodes.py:13-37` |

위 파일·행은 계획 작성 시점 기준이다. 과거 모델은 설정 확인에만 사용했다. 신규 코드로 재생성되지 않은 과거 점수를 신규 통제 실험과 섞지 않는다.

## 2. 공통 실험 계약

1. 타겟 3종, W-SUN, H=8, 2024년 이후 월 첫 origin, 기존 capped MAPE를 유지한다. `conf/base/parameters.yml`의 평가 조건을 실험별로 바꾸지 않는다 (`AGENTS.md:60,79-86`).
2. 현재 가격·외생 스냅샷을 고정한다. 가격 타겟의 현재 지문 `549e62438023`은 구현 시작 시 다시 계산하고 외생 스냅샷 지문도 함께 고정한다. DB 갱신은 본 연구 범위에 넣지 않는다.
3. 모든 학습은 `week + 8 <= origin`을 유지한다. rolling도 이 조건을 먼저 만족시킨다. horizon별 라벨 성숙 변경은 이번 P1에 포함하지 않는다 (`src/pipelines/modeling/nodes.py:51-58`).
4. 후보의 `features_sha`는 피처 변경에 따라 달라도 정상이다. 비교 계약은 동일 가격·외생 스냅샷, 평가 프로토콜, 셀과 올바른 부모 설정이다.
5. 원자재별 학습이 독립인 현재 구조를 유지한다. 피처 실험의 주 판정 원자재를 미리 지정하되 세 원자재 결과를 모두 남긴다 (`src/pipelines/modeling/nodes.py:45-59`).
6. random_state=0과 CPU·100 trees를 공통 고정한다. 여러 seed·하이퍼파라미터 조합 탐색은 하지 않는다.
7. 2024–2026 결과는 이미 확인한 역사적 탐색 결과다. 이를 새 미사용 holdout이라고 부르지 않는다. α·창 길이는 아래 목록을 사전 고정하고 최종 구간에서 자동 최적화하지 않는다.

## 3. `base_xgb`부터의 기준 경로

공통 XGB: `objective=reg:squarederror`, `tree_method=hist`, `multi_strategy=multi_output_tree`, `n_estimators=100`, `random_state=0`, `device=cpu`. 나머지 유효 모델 설정·실행 환경도 manifest에 기록한다.

| ID / 실험 이름 | 직접 부모 | 바꾸는 한 축 | 결과 설정 |
|---|---|---|---|
| B00 `base_xgb` | 없음 | 원형 | target=level, price_lags=[1,2,4,12,52], depth=6, lr=0.3, child=1, 전체 학습 이력 |
| B01 `xgb_logdiff` | B00 | 타겟 공간 | target=logdiff, 나머지 동일 |
| B02 `xgb_ret_matched` | B01 | 피처 표현 | 같은 시점의 가격을 수익률로 변경: return offsets=[0,1,3,11,51] |
| B03 `xgb_ret8` | B02 | lag 배치 | return offsets=[0,1,2,3,4,5,6,7] |
| B04 `xgb_ret8_depth2` | B03 | 트리 깊이 | max_depth=2 |
| B05 `xgb_ret8_depth2_lr003` | B04 | 학습률 | learning_rate=0.03 |
| B06 `xgb_ret8_reg` | B05 | 최소 자식 가중치 | min_child_weight=20. 현재 규제 대조군과 같은 모델 조건 |
| B07 `xgb_ret8_reg_2015` | B06 | 학습 시작일 | train_start=2015-01-04. 이하 모든 외생 실험의 부모 |

- B02는 현재 `lags` 블록에 선택적 offsets를 추가해야 한다. 기본 n=8 동작은 유지한다.
- 표현 변경은 features와 해당 feature_params를 함께 바꾸는 하나의 의미상 축이다. 실제 변경 경로 목록을 모두 기록한다. B03에서 lag 배치 효과를 별도로 측정한다.
- 규제 세 값을 한꺼번에 바꾼 뒤 하나의 효과라고 해석하지 않도록 B04–B06을 분리한다.
- 이 순서의 효과는 해당 부모 조건에 대한 조건부 효과다. 전체 factorial 또는 순서와 무관한 기여도라고 주장하지 않는다.

## 4. P1 피처 실험: 사전 고정한 12종

모든 노드는 B07에서 시작하거나 그 후손이다. 외생 실험은 train_start를 명시적으로 B07과 동일하게 고정한다. 종목별 정확한 열 선택을 위해 exog에 `columns_by_series`를 추가하고 요청 열 부재는 실패시킨다. 다른 종목은 빈 목록을 명시해 임의의 다른 피처가 섞이지 않게 한다 (`src/features/exog.py:14-32` 확장).

| 축 / 실험 | 직접 부모 | 추가·변경 내용 | 주 판정 |
|---|---|---|---|
| G01 `gold_usd` | B07 | 달러지수 1개 | gold |
| G02 `gold_vix` | B07 | VIX 1개 | gold |
| G03 `gold_usd_vix` | G01 | VIX 추가. G02는 달러 추가의 보조 비교 대상 | gold |
| C01 `copper_spread` | B07 | LME Cu spread | copper |
| C02 `copper_stock` | B07 | LME closing stock 수준 | copper |
| C03 `copper_spread_stock` | C01 | LME closing stock 추가. C02도 보조 비교 | copper |
| C04 `copper_stock_d4` | C02 | 같은 재고의 4주 차분 | copper |
| K01 `coffee_brl` | B07 | BRL/USD | coffee |
| K02 `coffee_brl_conilon` | K01 | Conilon 생산 추정치 추가 | coffee |
| K03 `coffee_brl_conilon_exports` | K02 | Conillon 수출 추가 | coffee |
| K04 `coffee_robusta_core` | K03 | 유럽 Robusta 재고 추가 | coffee |
| X01 `drivers_all` | B07 | 현재 drivers 전체 바스켓 | 세 종목, 바스켓 단위 참고 비교 |

정확한 열 이름은 `conf/base/parameters_exogenous.yml:9-28,31-61,64-89`와 현재 스냅샷에서 검증한다. X01은 여러 변수를 추가하는 **바스켓 축**이며 개별 변수의 효과로 해석하지 않는다. K01–K04도 추가 순서에 따른 조건부 비교이며 제거식 전체 조합 탐색은 하지 않는다.

커피·구리 신호 선택의 문헌 근거: [Holmes & Otero (2020)](https://doi.org/10.1108/SEF-09-2019-0356), [Buncic & Moretto (2015)](https://www.danielbuncic.com/pdf/foreCopper.pdf). 논문의 시장·주기가 현재 타겟과 같다는 가정은 하지 않는다. 신규 ICE Robusta 선물·기상 수집은 후속 연구로 남기고, 이번에는 이미 확보한 스냅샷의 변수로 실행한다.

## 5. P1 학습창·축소: 재학습 6종 + 파생 예측 9종

결과를 보고 유리한 부모를 고르지 않고 B07, G03, C03을 사전 고정한 참조 3종으로 쓴다. 원자재별 주 판정은 각각 세 종목, gold, copper다.

### 5.1 학습창

- 각 참조에 최근 156주와 260주를 적용해 2종씩 생성한다. 주 수는 달력상의 기간이며 결측 제거 후 행 수가 아니다.
- origin=t의 최근 W주: `t-W <= week < t`와 `week+8 <= t`, 기존 train_start를 모두 만족하는 행만 사용한다. H=8 때문에 실제 성숙 학습 행 수는 W보다 적다는 점을 기록한다.
- expanding은 원래 참조를 그대로 쓴다. expanding·156주·260주 예측의 가격 공간 단순 평균을 참조별 1종씩 생성한다.
- ensemble의 직접 부모는 expanding 참조이고 components에는 세 run ID를 모두 기록한다. 실제 예측은 components 산출물이 모두 완료된 뒤 계산한다.
- 근거: [Pesaran & Pick (2011)](https://www.tandfonline.com/doi/abs/10.1198/jbes.2010.09018). 156·260주는 본 연구의 사전 선택값이다.

### 5.2 naive forecast 쪽으로의 축소

- 같은 참조 3종에 `P_hat_alpha = P_t * exp(alpha * log(P_hat_parent/P_t))`를 적용한다.
- α 후보는 {0,0.25,0.5,1}. 0.25·0.5만 각각 새 파생 실험으로 기록한다. α=1은 부모 재사용, α=0은 동일 셀의 naive forecast와의 동등성 검사로 처리한다.
- 학습은 반복하지 않는다. 부모 예측의 지문·α·후처리 코드 지문으로 파생 run을 구분한다. 부모 예측이 양수·유한인지 검사한다.
- 근거: [Beckmann & Schüssler (2014)](https://www.wiwi.uni-muenster.de/oeew/sites/oeew/files/CQE_paper/cqe_wp_32_2014.pdf). 원자재 직접 검증이 아닌 방법론적 실험 가설이다.

### 5.3 참고 기준선

- naive forecast와 drift 2종을 별도 benchmark로 둔다. drift는 각 origin에서 B07과 같은 시작일·완전 성숙 학습 행의 h별 누적 로그수익률 평균을 사용한다.
- 이 둘은 XGB의 한 축을 바꾼 ablation이라고 가장하지 않는다. `kind=benchmark`, 연결된 `study_root=base_xgb`를 기록한다.
- 개수: 기준 경로 8 + 피처 12 + 학습창 6 = **XGB 설정 26종**, 파생 예측 9종, benchmark 2종 = 총 **37개 기록 단위**. 현재 3종×33 origins 기준 XGB 재학습은 최대 2,574회이며 실행시간은 측정 전 추정하지 않는다.

## 6. P0: 연구 명세·계보·비교 파이프라인

### 6.1 설정과 실행 대기열

- 새 `conf/studies/p0_p1_v1.yml`: 기준 경로·모든 후보·부모·축·단계·주 판정 종목·비교 규칙의 고정 명세. 모델·피처 spec은 완전히 펼친 형태로 저장한다. YAML anchors는 재사용하되 별도 상속 언어는 만들지 않는다.
- 새 `src/studies.py`: 명세·계보 검증과 단계별 실행 목록 구성만 담당한다.
- 새 `tools/prepare_study.py`: 명세에서 현재 단계와 필요한 조상을 선택해 기존 `parameters_experiments.yml` 형식의 대기열을 만든다. 실행용 파일에는 완전히 펼친 spec이 들어가 현재 Kedro params 바인딩을 유지한다 (`src/pipelines/modeling/pipeline.py:14-25`).
- 현재 대기열은 작업 중 변경 내용이 있으므로 무조건 덮어쓰지 않는다. 구현 시 정확한 원본 내용을 보존하고 연구용 대기열 생성본과 구분한다. 새 연구 명세가 spec의 기준이며 생성된 대기열은 직접 수정하지 않는다.
- 완료 노드는 검증된 로컬 기록·W&B 참조로 남긴 뒤 대기열에서 뺄 수 있다. 조상 정의·전체 spec·parent 연결은 고정 명세와 연구 manifest에 남는다.

### 6.2 기록 필드와 검증

각 연구 run에 아래를 남긴다. 기존 비연구 실험은 선택적 신규 필드 없이 동작해야 한다 (`src/experiments.py:23-34` 확장).

| 묶음 | 필수 필드 / 규칙 |
|---|---|
| 계보 | study_id, experiment_name, kind, root_experiment, parent_experiment, axis, changed_paths, primary_series, comparator_experiments |
| 재현성 | resolved_spec, spec_sha, data_sha, exogenous_sha, features_sha, protocol_sha, source_sha, uv.lock 지문, 라이브러리 버전 |
| 참조 | root_run_id, parent_run_id, 파생 실험은 component_run_ids와 forecasts_sha |
| 실행 | 명세의 experiment 이름과 실제 run ID를 구분. 기준 조건이 바뀌면 새 run ID·새 로컬 산출물 디렉터리 |
| 변화 검증 | 메타데이터를 제외한 실제 부모→자식 spec 차이가 선언한 axis·changed_paths와 일치. 미선언 변경·부모 부재·순환·다른 root는 실행 전 실패 |

run ID는 모델·피처·모델링·평가·명세 해석에 실제 사용한 소스 바이트와 환경 지문을 포함한다. git commit만으로 작업 중 변경 내용을 대신하지 않는다. 부모 run ID를 먼저 결정해 위상 순서로 연결한다. artifact 경로·실행시간 같은 산출 정보로 run ID가 매번 달라지지 않게 한다.

새 연구 산출물은 `data/07_reporting/studies/<study_id>/<execution_fingerprint>/`에 manifest, lineage, comparisons, origin_losses를 보존한다. 05·06·개별 07 파일도 연구 실행 지문이 포함된 로컬 경로에 보존하고 기존 평평한 경로는 필요할 때 최신 출력으로만 취급한다. 코드 변경 후 이전 결과를 조용히 덮어쓰지 않는다 (`conf/base/catalog.yml:26-43` 확장).

W&B는 기존처럼 실험당 한 run을 쓴다. `study` group에 축을 표시하고 config에 계보·지문을 추가한다. 기존 metrics/cells와 함께 parent/root/naive 비교 Table을 기록한다. 모델·parquet를 W&B에 업로드하지 않는 계약을 유지한다 (`src/pipelines/reporting/nodes.py:83-95`, `AGENTS.md:92-94`).

### 6.3 데이터·실행 흐름

`고정 연구 명세 → 계보/한 축 변경 검사 → 스냅샷·코드·평가 셀 고정 → 공유 입력 생성 → 학습 또는 부모 예측으로 파생 → 기존 evaluate → root/parent/naive paired 비교 → P0 통계 → 로컬·W&B 기록`

- 기존 modeling/reporting 파이프라인을 확장한다. 학습형은 현재 select_input→make_model→fit→predict를 유지한다 (`src/pipelines/modeling/pipeline.py:8-30`).
- 파생형은 명시적 부모/component forecasts 입력을 받아 forecasts만 만든다. make_model·fit을 호출하지 않는다. 이후 evaluate·기록은 학습형과 공유한다.
- 계보상의 parent는 비교 기준이며, 모든 자식이 부모 모델을 이어 학습한다는 의미는 아니다. 파생형만 실제 부모 예측에 데이터 의존성이 있다.
- 필요한 부모 산출물이 현재 지문으로 완료되지 않았으면 비교·파생 단계를 차단한다. 기존 기록 재사용은 run ID·모든 입력 지문·완료 상태가 일치할 때만 허용한다.
- pipeline 생성 시 raw config를 읽는 현재 방식과 Kedro 런타임 params가 어긋나지 않게 대기열을 펼쳐 저장한다. 이번 작업에서 `--env` 전반을 재설계하지 않는다 (`src/experiments.py:14-20`).

### 6.4 비교·불확실성

1. 키는 `(series,origin,horizon)`이다. 후보·직접 부모·루트·naive의 키 유일성, actual·base 일치와 예상 셀 완전성을 검사한다. 셀이 빠지면 기본 비교는 실패하며 교집합으로 조용히 순위를 만들지 않는다.
2. 종목×horizon, 종목 전체, 전체 macro 평균에서 MAPE와 naive/parent/root 대비 상대 손실 감소율을 기록한다. 기존 gain_pct 정의를 유지하고 각 비교의 분모를 명시한다.
3. 부모·루트도 후보와 같은 셀에서 다시 집계한다. `features_sha`나 학습기간이 다르다는 이유만으로 비교를 금지하지는 않되, 그 차이가 선언된 축인지 검사한다.
4. 연도별 결과, worst uncapped APE, 셀·origin 수, 학습 행 수 범위를 함께 남긴다. 최대 APE를 평균 정확도로 해석하지 않는다.
5. 원자재·horizon을 묶은 origin 벡터를 월 순서대로 함께 재표집하는 moving-block bootstrap을 쓴다. 기본 block=3 origins, 민감도=2·4, 반복=2,000, seed=0. 상대 gain의 점추정은 유지하고 paired loss 차이의 95% 구간을 보고한다.
6. 단순 CI는 pointwise라고 명시한다. 여러 후보의 유의성을 주장할 때만 사전 정의한 축별 후보 집합에 대해 중심화한 block-bootstrap 단측 검정과 Holm 보정을 적용한다. 작은 표본에서의 한계도 남긴다. block 길이를 유리한 결과에 맞춰 고르지 않는다.
7. 세 원자재 전체 성능과 특정 원자재의 주 판정 결과를 구분한다. 기존 root가 약하므로 root 개선만으로 후보를 승격하지 않는다. naive·직접 부모 대비 개선, 연도별 일관성·불확실성을 함께 판단한다. α=0 선택은 XGB 예측 신호 확보 성공으로 세지 않는다.
8. 역사적 탐색 후보와 통계적 근거가 있는 후보를 구분한다. 실제 미래 성능 확인은 계획 고정 이후 도래하는 월 origin에 예측을 저장하고 H개 타겟 성숙 후 별도 보고한다. 현재 이미 본 2026년 구간을 미사용 테스트로 재명명하지 않는다.

평가 근거: [Hansen (2005)](https://www.tandfonline.com/doi/abs/10.1198/073500105000000063), [Diebold & Mariano (1995)](https://doi.org/10.1080/07350015.1995.10524599). SPA 전체 구현은 이번 P0의 필수 조건이 아니다.

## 7. 구현·실행 순서

| 단계 | 변경 대상 | 산출물 / 종료 조건 |
|---|---|---|
| 1. P0 계보 기반 | `src/experiments.py`, 신규 `src/studies.py`, `conf/studies/p0_p1_v1.yml`, `tools/prepare_study.py` | 명세와 실제 차이 검증, 위상 순서, 단계별 완전 spec 대기열. 35개 비benchmark 기록이 루트까지 연결 |
| 2. P0 보존·비교 | `src/pipelines/reporting/{nodes,pipeline}.py`, `conf/base/catalog.yml`, 필요 시 `src/pipeline_registry.py` | 입력 지문별 로컬 보존, root/parent/naive 비교, W&B 계보·코드 지문. 일반 실험은 호환 유지 |
| 3. P1 기능 | `src/features/lags.py`, `src/features/exog.py`, `src/pipelines/modeling/{nodes,pipeline}.py`, 신규 `src/models/statistical/drift.py`, 해당 레지스트리 | offsets, 종목별 정확한 열 선택, rolling, fit 없는 축소·평균, drift. 기능별 필요한 회귀 테스트 추가 |
| 4. 작은 통합 검증 | `tests/test_forecast.py`, `tests/test_pipeline.py`, 신규 `tests/test_studies.py` | 합성 시계열에서 계보→학습/파생→평가→기록 스키마 확인. W&B disabled |
| 5. 기준 경로 | B00–B07 + benchmark 2종 | 새 코드·고정 스냅샷에서 루트부터 재현. 과거 결과는 출처 참고만 사용 |
| 6. 정보 축 | G01–G03, C01–C04, K01–K04, X01 | 직접 부모·루트·naive·연도별 비교표. 현재 스냅샷의 열을 사용해 외부 수집 없이 실행 |
| 7. 적응·축소 | 사전 고정 3개 참조의 rolling 6종, 평균 3종, 축소 6종 | 실제 추가 fit 없이 파생 결과 생성, endpoint·component 검증, P0 통계 포함 |
| 8. 결과 정리 | 로컬 연구 summary와 W&B | 전체 계보·설정 차이·성과·불확실성 표. 후속 후보와 관측된 한계를 기록하고 종료 |

학습은 기본 순차 실행을 사용한다. 실행시간·메모리를 실제 확인한 뒤 독립 원자재/실험 병렬화를 판단한다. 계획만으로 동시 학습을 기본 설정하지 않는다. 완료 기록은 고정 명세·manifest에 보존한 뒤 대기열에서 정리한다.

## 8. 테스트 가능한 완료 기준

- [ ] `base_xgb`가 정확한 원형 설정으로 명시되고 과거 `baseline_xgb` 출처가 기록된다.
- [ ] 모든 26개 XGB 설정과 9개 파생 실험은 root/parent 경로가 존재한다. benchmark 2개는 별도 분류된다.
- [ ] 누락·순환 부모, 선언하지 않은 추가 변경, 다른 root, 잘못된 정확한 열 이름은 학습 전에 실패한다.
- [ ] 부모→자식의 실제 차이를 기계적으로 출력하고 B04–B06은 각각 한 하이퍼파라미터만 바꾼다.
- [ ] 모든 비교에서 snapshot·protocol과 `(series,origin,horizon,actual,base)` 정합성을 확인하며 결측 셀을 숨기지 않는다.
- [ ] 모델 코드/모델링 코드·spec·데이터·피처가 달라지면 해당 run ID가 달라지고 같은 조건의 재실행은 같다. 이전 로컬 연구 기록은 남는다.
- [ ] rolling 경계와 H=8 성숙 조건을 테스트한다. 미래 가격·외생 피처 값 변경이 과거 origin의 예측을 바꾸지 않는다.
- [ ] α=0은 naive와 일치하고 α=1은 부모와 일치한다. 축소·평균 실험에서 fit 호출 횟수는 0이다.
- [ ] 동일 forecasts를 비교하면 Δloss=0, gain=0이고 bootstrap이 가짜 우위를 만들지 않는다. 모든 h·종목을 같은 origin 블록으로 재표집한다.
- [ ] 실행 YAML에서 완료 항목을 제거해도 고정 명세와 저장 manifest/W&B로 계보와 실제 부모 run을 복원할 수 있다.
- [ ] 기존 전체 테스트와 작은 통합 실행이 통과한 후에만 전체 37개 기록 단위 연구를 실행한다. 라이브 데이터 갱신·신규 패키지 설치는 하지 않는다.
- [ ] 최종 표에 naive·parent·root 대비 개선, 연도별 성과·불확실성·coverage·설정 차이가 모두 포함된다.

## 9. 위험과 대응 / 최소 검증

- 기존 작업 중 파일이 많다: 구현은 해당 변경 위에 최소 diff로 적용하고 원본 대기열·산출물을 보존한다. 이번 계획 작성은 해당 파일을 수정하지 않는다.
- 피처/모델 이름만으로는 재현이 안 된다: 실제 소스·resolved spec·환경 지문과 run ID별 로컬 저장을 사용한다.
- 원자재에 관련 있어 보이는 변수가 꼭 예측에 유효한 것은 아니다: 사전 고정 추가 순서와 같은 학습기간 대조군으로 판단한다.
- 외생 forward fill의 값 변화 시점은 실제 공표일이 아니다: age나 발표 지연을 임의 추정하지 않고 DB의 현재 계약을 따른다. vintage 검증·수집기 보수는 별도 범위다 (`AGENTS.md:104-115`).
- 짧은 rolling 창과 33 origins는 분산이 크다: 표본수와 block 민감도를 공개하고 통계적 확신을 과장하지 않는다.
- 문헌의 선물·월간 결과와 현재 현물·주간 결과가 다르다: 문헌은 축 선정 근거로만 사용한다.
- 검증은 추가한 동작의 작은 회귀 테스트와 기존 `uv run pytest`, W&B-disabled 합성 통합 실행으로 한정한다. 현재 저장소에는 별도 lint/typecheck 설정이 없으므로 불필요한 도구를 새로 도입하지 않는다 (`pyproject.toml`). 실험 성공을 주장하기 전 실제 산출물·W&B 기록을 확인한다.
