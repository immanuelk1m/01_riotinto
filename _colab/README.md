# `_colab` — 모든 실험 실행은 Colab 런타임에서

`01_riotinto` 리서치의 **컴퓨트는 Colab VM에서 수행**한다. Mac mini는 다음 두 가지만 맡는다.

| 역할 | 위치 | 비고 |
|---|---|---|
| 오케스트레이션 (`_colab/job.py`) | Mac mini | CLI 호출만 하므로 CPU 사용 미미 |
| 데이터 준비 (`E021.py prepare` 등) | Mac mini | Postgres(`dbname=commodity`) 접근이 mini에만 있음 |
| **모델 학습 · walk-forward · 리플레이 · 리더보드 산출물 생성** | **Colab VM** | CPU 런타임 0.08 units/hr, GPU는 `--gpu` |

## 사용법

리포지토리 루트(Mac mini)에서:

```bash
# CPU 런타임, 방향 3개를 한 세션에서 순차 실행 (세션 1개만 과금)
uv run python -m _colab.job run --exp E021 --cmd "ridge" --cmd "ebh" --cmd "pooled"

# 빠른 스모크 (시리즈 40개만)
uv run python -m _colab.job run --exp E021 --cmd "ridge --limit 40" --tag smoke --keep

# GPU가 필요한 실험
uv run python -m _colab.job run --exp E017 --cmd "gold" --gpu T4 --install lightgbm

# 세션/크레딧 확인, 정리
uv run python -m _colab.job status
uv run python -m _colab.job stop --session e021-smoke
```

## 동작 방식

1. `experiment/<EXP>/`를 tar 하나로 묶는다 (`remote/`, `logs/`, `results/`, `colab_runs/` 제외) → 파일 730개를 개별 업로드하지 않는다.
2. `colab new`(4초 내외) → 업로드 → VM에서 압축 해제.
3. 각 명령을 **detached 프로세스**로 띄우고 `DONE_<tag>` / `FAILED_<tag>` 센티널을 폴링한다 (`colab exec`는 실패해도 종료코드 0을 주므로 센티널로 판정).
4. 산출물은 `experiment/<EXP>/colab_runs/<tag>/{logs,results}/` 로 내려받는다 — **정규 경로를 덮어쓰지 않는다**.
5. `finally`에서 세션을 정지한다 (`--keep`으로 유지 가능, 사고 시 복구용).

## 주의

- 검증된 산출물을 정규 경로로 옮길 때만 `--promote`를 쓴다 (덮어쓰기 발생).
- `leaderboard.sqlite` 반영은 로컬(무료) 단계다. Colab 세션 안에서는 `/content/leaderboard.sqlite`에 쓰이므로 유실되며, 필요하면 `--promote` 후 로컬에서 재집계한다.
- 장시간 작업은 세션 만료에 걸릴 수 있다. `--timeout`, `--poll`로 상한을 조절하고, 만료 시 `--keep`으로 남긴 세션에 재접속한다.
- E015~E020은 자체 `remote/` 러너(`runner.py`)가 이미 Colab GPU로 동작한다. 신규·미이관 실험은 이 하네스를 표준으로 쓴다.
- `--data-from E014` 로 다른 실험의 `data/`·`features/` 스냅샷을 번들에 넣을 수 있다(E020 등).

## 검증 기록 (2026-10-31)

| 항목 | 결과 |
|---|---|
| 단일 명령 (E021 ridge --limit 40) | 5.0분, 산출물 4개 |
| 다중 명령 1세션 (ebh → pooled) | 1.3분, 산출물 8개 |
| mini 대비 결과 오차 | 1e-16 수준 (동일) |
| 테스트 총 소모 | 0.02 units / 잔액 46.37 |

CPU 런타임(2 vCPU)은 mini보다 느리다 — 40종목 5.0분 vs mini 0.7분. 크레딧은 싸지만 벽시계 시간은 길다.
