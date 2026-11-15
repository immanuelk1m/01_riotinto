# 후속 인과발견 방법 추가 계획 (개정 2)

작성 2026-11-15, 개정 2026-11-15. 상태: **승인 대기**. 선행: 리더보드(`sonet_krr/causal`) 완료, 전체 재실행 진행 중(fingerprint `d6ed5f07977f4b180c46`).

사용자 결정(2026-11-15):
- 환경은 하나만 쓴다. 옛 버전에 묶인 코드는 **지금 환경(Python 3.14, numpy 2.5)에 맞게 옮긴다**.
- **딥러닝·신경망 모델은 뺀다.**
- **CaReBayes 는 뺀다.**
- Causal-TS 3종은 **torch 를 허용하고 포함**한다. torch 는 행렬 계산 라이브러리로만 쓰고 신경망(GRACE)은 쓰지 않는다.

## 1. 대상

| 방법 | 코드 | 지금 환경으로 옮기는 방법 | 신경망 |
| --- | --- | --- | --- |
| RCBNB-MB | `analysis/RCBNB-MB.ipynb` 자체 구현 | 노트북 구현을 모듈로 옮긴다 | 없음 |
| Regime-PCMCI | `tigramite.rpcmci.RPCMCI` | `ortools` 9.15 추가(3.14 휠 있음) | 없음 |
| CD-NOD | `causal-learn` 0.1.4.8 | 그대로 설치 | 없음 |
| CD-NOTS · CDNOTS+ · CEDAR | `causalts` 0.28.1 | 그대로 설치(3.14 지원 명시) | 방법은 없음 ※ |
| SPACETIME | `causalchange` 0.1.1 | numpy 고정만 푼다(`numpy<2` → 프로젝트 버전). 코드에 numpy 1 전용 API 없음, torch 없음 | 없음 |
| CASTOR (선형) | `arahmani1/CASTOR` | 필요한 부분만 옮긴다(아래) | 없음 |

제외:

| 방법 | 사유 |
| --- | --- |
| FlowMSM | 조건부 normalizing flow(신경망)로 레짐을 추정한다 |
| FANTOM | 가역 그래프 신경망·하이퍼넷(`_fantom` 에 `nn.Module` 10개 파일) |
| CASTOR (비선형) | MLP 기반(`CASTOR_nonlinear.py`, `MLP_lag.py`) |
| CaReBayes | 사용자 결정 |
| Causal-TS 의 GRACE | 신경망 게이트 — 원래 목록에도 없다 |

※ Causal-TS 는 import 하는 순간 torch 가 필요하다(`causalts/__init__.py` → `cdnots/phase3_utils.py` 맨 위 `import torch`, 커밋 `80e923c`).
CD-NOTS 는 torch 를 표준화·역행렬 계산에만 쓰고, CEDAR 는 쓰지 않는다. 신경망 모델(GRACE)은 쓰지 않는다 — 사용자 확정.

### CASTOR 선형판 옮기기

- 구조: EM 반복 — (M) 레짐별 그래프를 DYNOTEARS(`dynotears.py`, scipy 최적화)로 추정, (E) 잔차의 정규 우도로 주를 레짐에 재배정.
- torch 를 쓰는 곳은 한 군데다: 레짐 혼합 비율 π(t) 를 `nn.Linear(1, 레짐 수)` + softmax 로 맞춘다(`time_varying_weights.py`, `utils.train_pi`).
  이것은 시간에 대한 다항 로지스틱 회귀이므로 `sklearn.linear_model.LogisticRegression` 으로 바꾼다. 그러면 torch 가 필요 없다.
- 옮기는 파일: `CASTOR.run_linear`(약 70줄) + `dynotears.py`·`causal_structure.py`(numpy·scipy·networkx). MIT 라이선스 — 출처와 커밋(`b425ebc`)을 파일 머리에 적는다.
- 확인: 저장소의 합성 자료 생성기로 원본과 같은 설정을 돌려 레짐 경계·그래프가 논문 예제 수준으로 복원되는지 본다.
  (원본을 돌려 숫자를 대조하려면 옛 환경이 필요하므로 하지 않는다.)

## 2. 확인한 사실

- 저장소 5개 존재 확인(2026-11-15). `ortools`·`causal-learn`·`causalts`·`ruptures`·`hyppo` 는 Python 3.14 에서 설치된다(dry-run).
- `causalchange` 를 그대로 설치하면 numpy 가 1.26 으로 내려간다 → uv 의 `override-dependencies` 로 numpy 고정만 무시한다.
  SPACETIME 이 쓰는 선택 의존성은 `ruptures`·`hyppo`·`causal-learn` 이다(R 은 필요 없다 — `rpy2` 는 다른 알고리즘용).
- 의존성을 추가하면 `uv.lock` 이 바뀌고, `uv.lock` 은 지금 연구의 fingerprint 에 들어간다(`analysis/causal_study.py` `fingerprint`).
  → 지금 연구의 적합을 전부 다시 돌린다(약 2시간 30분). **한 번에 모두 추가해 재실행을 한 번으로 줄인다.**
- 지금 연구 코드는 방법이 `discover` 안의 분기로 박혀 있고, 그래프 하나 → 부모 → 회귀 기여로 채점한다. 레짐(주마다 다른 그래프) 개념이 없다.
- 진행 중인 재실행의 작업 프로세스는 파일을 다시 읽는다. **재실행이 끝나기 전에는 `analysis/*`·`pyproject.toml`·`uv.lock` 을 건드리지 않는다.**

## 3. 설계

### 방법은 "그래프만" 내고, 채점은 한 곳에서 한다

- 방법마다 어댑터 하나(`analysis/causal_methods/<방법>.py`): 입력 = 학습 패널, 출력 = 부모 목록 `[(변수, 시차)]`.
  레짐 방법은 레짐별 부모 목록 + 학습 주의 레짐 번호.
- 그 뒤는 공통: 부모 → 회귀 기여 → 주별 top-3 → top-3 적중률·F1·bootstrap 비교. 지금의 `contributions`·`cc.score_weeks`·`paired_ci`·`f1_scores` 를 쓴다.
- 새 실행기 `tools/run_causal_methods.py`. 결과는 `data/07_reporting/causal_methods/<방법>/<방법 지문>/` 에 지금과 같은 파일 구성으로 쓴다.
  방법마다 지문이 따로라(어댑터 파일 + 공통 채점 코드 + 입력 + 그 방법의 패키지 버전) 하나를 고쳐도 나머지 캐시는 남는다.
- `tools/log_causal_leaderboard.py` 가 이 폴더도 읽어 같은 프로젝트에 올린다(group `causal_methods.<방법>`).
- `analysis/causal_study.py` 는 건드리지 않는다. 의존성 추가로 생기는 재실행 한 번 외에는 지금 연구의 캐시가 유지된다.

### 결측

- 방금 정한 규칙을 모든 새 방법에 쓴다: 늦게 시작하는 변수의 시작 전 구간 = 0, 시작 뒤 결측은 그대로(지금 패널에는 없다). 대조군은 `matched_pcmci`.

### 레짐 방법의 채점 규칙 → **결정 필요 2**

평가 주는 학습 구간 뒤에 있다(학습 <2023-01, 채점 2023-07~). 레짐별 그래프 가운데 어느 것으로 그 주를 설명할지 정해야 한다.

- **A. 마지막 학습 레짐 고정(추천)**: 학습 구간 끝의 레짐 그래프를 평가 구간 전체에 쓴다. 평가 주의 정보를 쓰지 않는다.
- B. 주별 재배정: 평가 주마다 예측 오차가 가장 작은 레짐을 고른다. 그 주의 가격을 쓰므로 `protocol` 을 다르게 적고 순위에서 뺀다.

## 4. 단계

| 단계 | 내용 | 예상 |
| --- | --- | --- |
| 0 | (재실행 종료 뒤) 의존성 일괄 추가 → 지금 연구 재실행 1회. 새 실행기·어댑터 규약·업로드 연결. `matched_pcmci` 를 새 실행기로 재현해 숫자 일치 확인 | 반나절 + 재실행 2시간 30분 |
| 1 | CD-NOD, CD-NOTS, CDNOTS+, CEDAR (그래프 하나) | 방법당 1~2시간 + 적합 시간 |
| 2 | Regime-PCMCI, RCBNB-MB (레짐 채점 규칙 구현) | 하루 |
| 3 | SPACETIME (numpy 고정 해제), CASTOR 선형판 (옮기기) | 방법당 반나절~하루 |

- 단계마다 끝나면 리더보드에 올리고 멈춘다. 다음 단계는 결과를 보고 진행한다.

## 5. 인수 기준

- 환경은 하나다: `uv sync` 한 번으로 모든 방법이 돈다. numpy·pandas 버전은 지금과 같다(2.5·3.1).
- 단계 0: 새 실행기의 `matched_pcmci` 가 연구 실행의 값과 원자재·구간별 top-3 가 같다(차이 < 1e-12).
- 방법마다: 세 원자재 × 두 구간의 상태가 `complete`·`failed`·`timeout` 중 하나로 남고, 실패는 사유(`error.txt`)가 있다.
- 신경망 없음: 어댑터와 옮긴 코드에 `torch.nn`·`lightning` import 가 없고, Causal-TS 는 `run_cdnots`·`run_cdnots_plus`·`run_cedar` 만 부른다(테스트 하나로 확인).
- 합성 점검: 레짐 방법은 그래프가 다른 두 구간의 작은 합성 자료에서 레짐 오배정률을 한 번 찍는다(RCBNB-MB 노트북의 점검과 같은 방식).
- W&B: 방법마다 run 하나, 이름 중복 0, config 에 `method`·`protocol`·`regime_rule`·패키지 버전·옮긴 코드의 원본 커밋.
- `uv run pytest` 통과.

## 6. 위험과 대응

| 위험 | 대응 |
| --- | --- |
| 자료가 작다(15~31 변수 × 417~522주). 레짐 방법은 더 긴 자료를 전제로 한다 | 실패·불안정도 결과로 기록한다. 최소 레짐 크기는 RCBNB-MB 노트북의 규칙(52주 이상)을 공통으로 쓴다 |
| numpy 고정을 풀었더니 `causalchange` 가 조용히 다른 값을 낸다 | 저장소의 테스트(`tests/`)와 튜토리얼 예제를 지금 환경에서 돌려 통과하는지 먼저 본다. 안 되면 SPACETIME 은 `failed` 로 남긴다 |
| CASTOR 를 옮기며 동작이 달라진다(π(t) 를 로지스틱 회귀로 교체) | 저장소의 합성 예제로 레짐·그래프 복원을 확인하고, 교체한 부분을 문서에 적는다. 원본과 숫자 대조는 하지 않는다(옛 환경이 필요) |
| 방법이 많아져 선택 구간에서 고른 1위가 우연일 수 있다 | 순위는 확인 구간 + bootstrap 구간으로 읽는다. 비교한 방법 수를 리더보드에 적는다 |
| 0 으로 채운 구간이 분포 기반 방법을 왜곡한다 | `matched_pcmci` 대조군과 같이 읽고, 늦게 시작한 변수를 뺀 패널로 민감도 실행을 하나 둔다 |
| Causal-TS·tigramite 는 GPL-3 | 설치해 실행만 한다. 코드를 레포에 복사하지 않는다 |
| 의존성 추가로 지금 연구의 캐시가 무효가 된다 | 한 번에 모두 추가하고 재실행은 한 번만 한다 |

## 7. 결정 필요

1. ~~Causal-TS 3종~~ — 확정: torch 허용, 포함.
2. 레짐 채점 규칙 — 추천: A(마지막 학습 레짐)로 순위, B 는 참고용.
3. 진행 순서 — 추천: 4절의 0 → 1 → 2 → 3. 단계 0 은 진행 중인 재실행이 끝난 뒤 시작한다.

## 8. 하지 않는 것

- 방법별 하이퍼파라미터 탐색(논문 기본값 + τ_max = 4, 유의수준 0.05 를 맞춘다).
- 예측 성능 비교 — 이 리더보드는 귀속 적중률만 본다.
- 지금 연구(`causal_study`)의 선택 규칙 변경.
