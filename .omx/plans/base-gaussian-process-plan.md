# base_gaussian_process 실행 계획

## 요구사항과 범위

- 요청: Gaussian Process 기준 모델 추가, 결측치는 forward fill(ffill) 처리.
- 이번 단계는 계획 작성이며 구현·학습은 실행하지 않는다.
- 기존 base_xgb 계보·스냅샷·평가 조건을 유지한다. 현재 연구 명세는 축소된 14종이며 제거된 종목별 실험을 복원하지 않는다 (`conf/studies/p0_p1_v1.yml:1-36`).
- 세 원자재 모두 같은 변경을 적용한다. 새 패키지는 필요 없다. scikit-learn이 이미 설치돼 있다 (`pyproject.toml:5`).

## 첫 실행: 변경 효과를 분리하는 3개 신규 실험

| 실험 | 부모 | 변경 축 | 고정 조건 |
|---|---|---|---|
| base_xgb_ffill | base_xgb | 입력 결측 처리: native NaN → ffill | 가격 수준 타겟, 가격 lag [1,2,4,12,52], 기존 XGB 설정·전체 학습기간 |
| base_xgb_ffill_scaled | base_xgb_ffill | 입력 표준화: 없음 → StandardScaler | 동일 피처·학습 행·타겟·모델 |
| base_gaussian_process | base_xgb_ffill_scaled | 모델 계열: XGB → GaussianProcessRegressor | 동일 ffill·표준화·피처·학습 행·가격 수준 타겟 |

원형 조건은 현재 base_xgb 정의를 따른다 (`conf/studies/p0_p1_v1.yml:36`). base_gaussian_process를 처음부터 logdiff·수익률 lag·최근 학습창으로 바꾸지 않는다. 그러면 모델 교체 효과와 표현·기간 효과가 섞이기 때문이다. 위 3종이 이번 기본 범위다. logdiff, 수익률 lag, 2015년 시작, 외생변수는 이후 별도 한 축 ablation으로 추가할 수 있다.

## ffill·표준화 계약

1. 선택한 X 피처만 원자재별 week 오름차순으로 ffill한다. raw 가격·logp·y1..y8·actual·base·origin 표시는 변경하지 않는다. 현재 select_input은 피처 선택만 담당한다 (`src/pipelines/modeling/nodes.py:17`).
2. 각 origin까지의 피처 이력을 사용해 순방향으로만 채운다. bfill, 보간, 미래 값, 다른 원자재 값은 사용하지 않는다. origin 추론 행은 바로 앞주 값을 사용할 수 있으며, 학습 라벨 cutoff와 피처 관측 시점을 구분한다.
3. 최초 관측 이전의 NaN은 남긴다. ffill 후에도 X가 불완전한 학습 행은 제외한다. 해당 origin의 추론 X에 결측이 남으면 조용히 셀을 삭제하지 않고 실패한다. 전체가 미관측인 열의 선택은 origin까지의 정보로만 결정한다 (`src/pipelines/modeling/nodes.py:60-76`).
4. 같은 전처리 노드를 부모 XGB와 GP에 적용하고 동일 학습 행을 검증한다. 기존 native-NaN base_xgb와의 직접 차이는 결측 처리 축의 결과로 해석한다.
5. rolling을 나중에 붙일 경우 ffill은 origin까지의 과거 이력에서 수행하고, 이후 학습창을 자른다. 창 이전 마지막 관측을 쓰는 정책을 명시한다.
6. StandardScaler는 실제 성숙 학습 행만으로 fit한다. 같은 scaler로 학습 X와 origin X를 transform하며 fitted model에 보존한다. Y의 미래·평가 값으로 표준화하지 않는다.
7. spec에 preprocessing={missing:ffill, scaling:none|standard}를 명시하고 raw/processed features 지문, 채운 셀 수·잔여 결측·제외 행 수를 기록한다. 기존 spec은 필드가 없으면 기존 동작을 유지한다 (`src/experiments.py:11`, `src/study_runner.py:67`).

## GP 기본 설정

- 파일: 신규 `src/models/ml/gaussian_process.py`, MODELS 키 `gaussian_process`, 실험 이름 `base_gaussian_process` (`src/models/__init__.py:1`).
- 기존 인터페이스 fit(X,Y)->self, predict(X)->(n,H)를 유지한다. sklearn GPR의 다중 타겟 입력으로 H=8을 처리한다. 이는 출력 간 상관을 학습하는 별도 multi-task GP라는 뜻은 아니다.
- 커널: ConstantKernel(1.0, [1e-3,1e3]) × Matern(length_scale=1.0, bounds=[1e-2,1e2], nu=1.5) + WhiteKernel(noise_level=0.1, bounds=[1e-5,1e1]). 첫 기준선은 isotropic 하나만 사용한다.
- normalize_y=True, alpha=1e-6, optimizer=fmin_l_bfgs_b, n_restarts_optimizer=0, random_state=0. Y 정규화는 학습 Y로만 계산하는 GP 내부 처리이며 예측은 원래 타겟 단위로 반환한다.
- 커널·bounds·정규화·안정화·학습된 kernel_·경고·학습 시간과 표본수를 기록한다. 커널 선택은 기준선 설계안이며 성능 우위를 전제하지 않는다.
- 첫 실행은 예측 평균만 기존 가격 평가에 연결한다. GP posterior std와 origin-block bootstrap CI는 다른 종류의 불확실성이므로 섞지 않는다.
- API·설정 근거: [scikit-learn Gaussian Processes 공식 문서](https://scikit-learn.org/stable/modules/gaussian_process.html), 설치된 GaussianProcessRegressor.fit 문서의 Y=(n_samples,n_targets) 계약 확인.

## 구현 순서

| 순서 | 대상 | 산출물 |
|---|---|---|
| 1 | src/experiments.py, src/studies.py | preprocessing 필드 검증, missing/scaling/model_family 축 허용. 모델 계열 변경은 model 및 해당 model_params 교체만 허용하고 실제 changed_paths 기록 |
| 2 | src/pipelines/modeling/nodes.py | 학습/추론에 동일한 시간순 ffill 적용. 잔여 결측과 origin coverage 검사. 학습 행 선택은 전처리 후 수행 |
| 3 | 신규 src/models/ml/gaussian_process.py, src/models/__init__.py | GPR wrapper 및 학습 행 전용 scaler. XGB 비교군도 동일 scaler 경로 사용; 공통 전처리 wrapper로 중복 구현 방지 |
| 4 | src/study_runner.py | 새 전처리 노드 연결·처리 피처 지문·audit 보존. 기존 코드/환경 지문과 W&B parent/root 비교 유지 |
| 5 | 신규 conf/studies/gp_baseline_v1.yml | 현재 14종 정의를 보존하고 별도 연구에 필요한 base_xgb·naive·drift 및 신규 3종 명시. 불필요한 기존 실험을 다시 돌리지 않도록 phase 정의 |
| 6 | tests/test_study_models.py, tests/test_studies.py, tests/test_study_runner.py | 아래 회귀·작은 통합 테스트 실행 |
| 7 | 고정 스냅샷 실행 | 원자재별 첫 성숙 origin에서 비용 측정 → 전체 월별 origin 순차 실행 → parent/root/naive/drift 대비 결과표 |

기존 validator는 모델 계열 변경 축을 허용하지 않으므로 명시적 확장이 필요하다 (`src/studies.py:11,162`). runner는 공통 modeling 노드를 재사용하므로 별도 비추적 학습 스크립트는 만들지 않는다 (`src/study_runner.py:137,164`).

## 검증·완료 조건

- [ ] 중간 결측은 직전 과거 값으로 채워지고 선두 결측은 그대로 남는다. 다른 원자재 값은 유입되지 않는다.
- [ ] 미래 X·가격 값을 변경해도 그 이전 origin의 처리 X·scaler·예측은 동일하다.
- [ ] y/actual/base 결측과 평가 셀 정의는 ffill로 바뀌지 않는다.
- [ ] base_xgb_ffill_scaled와 base_gaussian_process의 학습 행·피처 값·스케일러 통계가 같다.
- [ ] GP predict 형상은 (n,8), 값은 유한하며 normalize_y의 역변환 후 가격 단위가 맞는다.
- [ ] 세 신규 실험의 부모·root_run_id·changed_paths가 모두 base_xgb까지 연결된다.
- [ ] 기존 테스트, 관련 회귀 테스트, W&B-disabled 합성 통합 실행을 통과한다. 불필요한 새 검사 도구는 설치하지 않는다.
- [ ] 전체 실행은 동일 예상 평가 셀을 사용하고 학습/전처리 실패를 숨기지 않는다. naive·직접 부모·root·drift 대비 MAPE·연도별 결과·bootstrap 구간을 기록한다.

## 비용·위험과 대응

- Exact GPR은 큰 학습 이력에서 연산·저장 비용이 크다. 특히 전체 금 이력을 쓰는 기준선은 첫 origin의 실제 시간·메모리·pickle 크기를 측정하고 전체 실행 비용을 추정한다. 첫 측정 결과를 버리지 않고 해당 origin 모델로 재사용한다. 표본을 임의로 줄이거나 rolling으로 조용히 바꾸지 않는다.
- 비용 때문에 그대로 실행하기 어려우면 측정 결과와 한계를 보고하고, 2015년 시작 또는 rolling을 별도 비교 축으로 설계한다. 이 대안은 원형 GP와 동일 실험으로 기록하지 않는다.
- 선두 결측 제거로 원형 XGB와 학습 행이 달라지는 효과는 첫 ffill 노드에 함께 표시한다. GP와 모델 교체 부모 사이에서는 행이 같아야 한다.
- GP 모델 변경에는 모델 고유 커널·정규화 설정이 포함된다. 이를 개별 커널 효과로 해석하지 않는다. 새로운 커널·재시작 탐색·종목별 설정은 이번 범위 밖이다.
- 기존 결과는 이미 본 기간의 역사적 탐색이다. 새 학습도 미래 holdout 검증이라고 부르지 않는다.
