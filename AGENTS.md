# AGENTS.md — `01_riotinto` 실행 정책

## 1. 실행

```bash
uv run python experiment/E0NN/E0NN.py prepare        # Postgres → parquet 스냅샷
uv run python experiment/E0NN/E0NN.py <direction>    # 실험 실행 → experiment/E0NN/results/
uv run python tools/leaderboard_from_cells.py E0NN --apply   # W&B 원장 반영
uv run python tools/check_all.py                     # 전체 게이트
```

- legacy 실험(E001~E021)은 코드·결과·원장 행을 전부 제거했다(git 히스토리에만 남음).

## 2. 산출물·원장 규칙

- **결과 원장은 W&B** (`sonet_krr/riotinto`). 버전 artifact `ledger:latest`(leaderboard·exogenous_round parquet; legacy 행은 제거됨)가
  유일한 원본이고, 쓸 때마다 실험 라벨 이름의 run(그룹 `E0NN`)에 행 테이블·요약이, `E0NN-results` artifact에 `results/`가 올라간다.
  sqlite 파일은 없다. 읽기는 `_exo.leaderboard.connect()`(메모리 sqlite), 쓰기는 `write()`/`commit()`.
- 인증은 mini의 `~/.netrc`(`uv run wandb login`) — 키를 레포에 넣지 않는다.
- `leaderboard.max_ape_pct` = 그 행(experiment × commodity × horizon)을 만든 TSCV 셀 중 **최악 단일 APE%**,
  상한 없음. `mape`(평균, 100% 캡)와 스케일이 다르므로 **위험도 지표**로만 읽는다.
  2025-01~2026-09 각 월 첫 주 TSCV origin 셀이며(dev/sealed 구분 없음) `mape`도 같은 셀집합이다.

## 3. 용어

- 기준선은 **naive forecast**(로그수익률 0, 즉 `P_t` 유지)라고 부른다. 개선은 "naive forecast 대비"로 쓴다.
  코드·원장의 식별자 `persistence`(모델 id, `mape_persistence` 컬럼 등)는 스키마 호환을 위해 그대로 둔다.
