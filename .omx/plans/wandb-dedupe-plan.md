# W&B 중복 run 제거 계획 (`sonet_krr/riotinto`)

현재 86 run, 이름 65개 → 정리 후 **65 run, 이름 중복 0**. 삭제 21건.

## 결정 (사용자 확정)
- 연구 간 사본: **p0_p1_v1 원본 유지**, gp_baseline_v1·lightgbm_direct_v1 사본 삭제
- 구 실험(연구 이전): **이름별 최신 1개 유지**, 연구 run 과 이름이 겹치는 구 naive·xgb_logdiff 는 삭제
- 원인 수정: 같은 실험·같은 예측이면 이후 연구가 새 run 을 만들지 않고 기존 run 을 참조

## 1. 삭제 대상 (21)

| 묶음 | 이름 | 삭제 run id | 유지 run id |
|---|---|---|---|
| 연구 사본 | naive | 657af37a4554d782529c (gp), c8851b06f13f8e0f3f22 (lgbm) | 4c52b616809db97f6518 |
| 연구 사본 | drift | a9f159c1df31291dff4c (gp), 28b9029cc05a4b7f7ba5 (lgbm) | 3369fb28a5c3cf55e3d7 |
| 연구 사본 | base_xgb | 19680b67515db10fa3c5 (gp), 0bd9677255d8e36e7d19 (lgbm) | d1049de9902a7451448b |
| 연구 사본 | xgb_logdiff | 46a5cf09ef6e79afe854 | 030e72c794e8f78f7edc |
| 연구 사본 | xgb_ret_matched | 271d17b5be3a36cebdce | 43c5dc49ba294f63edf5 |
| 연구 사본 | xgb_ret8 | d654fa2731a7d7087180 | 659062189909c0b3f4c8 |
| 연구 사본 | xgb_ret8_depth2 | 1f0a115c5368a5f75320 | 92b2cce88bf2e488f8d9 |
| 연구 사본 | xgb_ret8_depth2_lr003 | 672cf86cf570fbe01ef0 | e38cc8c7f7b7fd50165c |
| 연구 사본 | xgb_ret8_reg | 85580d62fb3b2dc64e65 | 1bd9b0fa2a8cfd3da41e |
| 연구 사본 | xgb_ret8_reg_2015 | 032920d853603601369a | 4685c2949369f36206e8 |
| 구 실험 | naive (baseline) | b630ee31f1f7, be61584446d6 | (연구 naive) |
| 구 실험 | xgb_logdiff (xgb_target) | a573d6cce90e | (연구 xgb_logdiff) |
| 구 실험 | ols_drivers | 0e6d3c611a71 | e4803a223391 |
| 구 실험 | xgb_drv | 2d3e525efd7c | 22f50ef40126 |
| 구 실험 | xgb_drv_d4 | 41c7d09bcabd | ab3480b1901d |
| 구 실험 | xgb_fnd | 93b3a26c8781 | d0c3653a5ce1 |
| 구 실험 | xgb_fnd_d4 | f456a93ce3bb | c027e8fc2408 |

안전장치: 연구 사본은 삭제 전에 **유지 run 과 summary(mape·gain_pct·시리즈별) 값이 같은지** 확인하고, 다르면 그 이름은 멈춘다.

## 2. 참조 다시 잇기 (삭제 전에)
- gp_baseline_v1·lightgbm_direct_v1 의 남는 run(base_xgb_ffill …, xgb_ret8_reg_2015_ffill, xgb_direct, base_lightgbm_direct)의
  config/summary `parent_run_id`·`root_run_id`·`component_run_ids` 를 삭제 id → 유지 id 로 바꾼다 (`run.config.update`, `run.update()`).
- 로컬 `data/07_reporting/studies/{gp_baseline_v1,lightgbm_direct_v1}/*/manifest.json` 의 사본 run `wandb.url` 을 유지 run URL 로 바꾸고
  `wandb.reused_from_study: p0_p1_v1` 를 남긴다. 산출물 파일·로컬 run_id 는 그대로 둔다(무결성 검사 대상 아님 확인 후).
- 기존 일반 실행(`conf/base`)·로컬 산출물은 삭제하지 않는다.

## 3. 원인 수정 (`src/study_runner.py`)
- 원인: run identity 에 `study_id` 가 들어가 연구마다 조상 사슬 전체가 새 id 로 다시 기록된다.
- `log_study_run` 앞에서 `dedupe_key = digest(실험 이름, spec(study·lineage.study_id 제외), data_sha, forecasts 파일 sha)` 를 계산한다.
  W&B 에 `config.dedupe_key` 가 같은 finished run 이 있으면 새로 기록하지 않고 그 run 의 id·URL 을 참조로 쓴다.
  예측이 같아야만 재사용하므로 코드가 바뀌어 결과가 달라지면 새 run 이 된다.
- 유지하는 기존 연구 run 에 `dedupe_key` 를 로컬 산출물로 계산해 config 에 채운다(backfill).
- 연구별 비교표(부모·루트 대비)는 재사용 시 W&B 에 다시 올리지 않고 로컬 `comparisons.parquet` 에만 남는다.
- 테스트 1개: 같은 이름·spec·예측이면 재사용, 예측이 다르면 새 run.

## 4. 검증
- W&B: run 65개, `name` 중복 0, 남은 run 의 `parent_run_id`·`root_run_id` 가 모두 존재하는 run 을 가리킴.
- `uv run pytest` 통과.
- 삭제 목록·매핑을 `data/07_reporting/studies/wandb_dedupe_<날짜>.json` 에 기록.

## 위험
- W&B run 삭제는 되돌릴 수 없다 → 2(참조 수정) → 삭제 목록 dry-run 출력 → 확인 후 삭제 순서.
