# AGENTS.md — `01_riotinto` 실행 정책

## 1. 컴퓨트는 전부 Colab

| 작업 | 실행 위치 | 명령 |
|---|---|---|
| 실험 학습·walk-forward·리플레이 | **Colab VM** | `uv run python -m _colab.job run --exp E0NN --cmd "..."` |
| GPU 실험 (torch) | **Colab GPU** | `... --gpu T4 --install lightgbm` |
| Postgres 데이터 준비 (`prepare`) | Mac mini (예외) | `uv run python experiment/E021/E021.py prepare` |
| 리더보드 재집계 / sqlite 반영 | Mac mini (예외, 수 초) | 로컬 스크립트 |

- **Mac mini 에서 `E0NN.py` 를 직접 돌리지 않는다.** 2026-10-31 E021 3방향을 mini CPU로 돌려 코어 3개를 23분씩 점유한 것이 이 정책의 계기다.
- E015~E020 은 자체 `remote/runner.py`(Colab 세션)로 이미 동작한다. 신규·미이관 실험은 `_colab` 하네스를 표준으로 쓴다.

## 2. 표준 사용법

```bash
# CPU 런타임 (0.08 units/hr) — 방향 3개를 한 세션에서 순차 실행
uv run python -m _colab.job run --exp E021 --cmd "ridge" --cmd "ebh" --cmd "pooled"

# 스모크
uv run python -m _colab.job run --exp E021 --cmd "ridge --limit 40" --tag smoke --keep

# E014 스냅샷을 쓰는 실험
uv run python -m _colab.job run --exp E020 --cmd "base" --data-from E014 --gpu T4

colab sessions && colab usage     # 세션·크레딧 확인
colab stop -s <session>           # 수동 정리
```

## 3. 산출물 규칙

- 하네스는 `experiment/<EXP>/colab_runs/<tag>/{logs,results}/` 로만 내려받고 **정규 경로를 덮어쓰지 않는다**.
- 검증 후 정규 경로로 옮길 때만 `--promote`.
- `leaderboard.sqlite` 반영은 로컬 단계로 남긴다 (Colab 세션 안의 `/content/leaderboard.sqlite` 는 휘발성).
- `leaderboard.max_ape_pct` = 그 행(experiment × commodity × horizon)을 만든 TSCV 셀 중 **최악 단일 APE%**,
  상한 없음. `mape`(평균, E021·E022+는 100% 캡)와 스케일이 다르므로 **위험도 지표**로만 읽는다.
  E021은 전체 origin, E022+는 dev 구간(sealed 52 origin 제외) 셀이며 `mape`도 같은 셀집합이다.
  E001–E011 행은 예측 아티팩트가 없어 NULL이고, E012–E020은 `tools/leaderboard_max_ape.py backfill`로
  아티팩트에서 복원했다(재계산 평균==저장 `mape`·`rmse`·`n` 게이트 통과분만).

## 4. 세션 위생

- 작업 실패 시 하네스는 산출물을 회수(salvage)한 뒤 세션을 정지한다. 정지 실패 시 경고를 남기므로 `colab stop` 으로 정리한다.
- `--keep` 은 복구·디버깅용. 남긴 세션은 과금되므로 반드시 `colab stop`.
- 장시간 작업은 세션 만료에 걸릴 수 있다. `--timeout`·`--poll` 로 상한을 조절한다.

## 5. 검증 상태 (2026-10-31)

| 검증 항목 | 결과 |
|---|---|
| 단일 명령 실행 (E021 ridge, 40종목) | 5.0분, 산출물 4개 회수, 세션 자동 정지 |
| 다중 명령 1세션 (ebh→pooled) | 1.3분, 산출물 8개, 방향별 태그 충돌 없음 |
| mini vs Colab 결과 동일성 | 오차 1e-16 (float 누적 순서만 상이) |
| 소모 크레딧 | 테스트 전체 0.02 units (잔액 46.37) |

- CPU 런타임은 mini보다 느리다(40종목 5.0분 vs mini 0.7분). units은 싸지만 벽시계 시간이 길다.
- Colab VM: Python 3.13.18, numpy 2.1.3, pandas 2.2.3, pyarrow 23.0.1, 2 vCPU / 12GB
