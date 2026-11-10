# E028 — 50% 캠페인 (Gate C 사전등록)

**등록:** 2026-11-08. 탐지기·dev 감사·봉인 해제·전방 기록 전에 작성했습니다.
**개정 4 (2026-11-09, 사후): 평가 규칙을 바꿨습니다.** dev/sealed와 전방 평가를 없애고, 2025-01~2026-09의 각 월 첫 주를
origin으로 하는 단일 TSCV(h = 1..8)로 평가합니다(PROTOCOL.md §4.2). 이 변경은 봉인 결과(pooled +0.59%, p = 0.31)를
본 **뒤**에 한 것이므로 **후향적**입니다(PROTOCOL 4.4). 새 결과는 확증이 아니라 **탐색 결과**입니다. 아래 본문의 dev·봉인·전방 서술은
등록 당시(구 규칙) 기록이며, 현행 규칙은 이 배너와 "창"·"평가" 절의 개정 4 표기가 우선합니다.
**프로토콜:** PROTOCOL.md v1.2, Gate A–F. 평가기 **cell-v4**, `PARAMS_VERSION = 3.0`.
**0단계:** [`CEILING.md`](CEILING.md). 상한은 11.5%이고, 밀·커피를 완벽 예측해도 41.6%입니다.
사용자가 상한을 확인한 뒤에도 캠페인 진행을 지시했으므로, 이 라운드의 목적은 **실제 달성치 Y를 외부 기준으로 측정하는 것**입니다.
**데이터 제약 (사용자 지시, 2026-11-08):** 비용을 쓰지 않고 `dbname=commodity`에 있는 데이터만 씁니다.
0단계에서 쓴 EIA NYMEX 파일은 DB 밖에 있으므로 이 라운드에서는 쓰지 않습니다.

## 고정 조건

- **타깃 6개(정의 그대로):** gold, copper, corn, wheat, coffee, wti. CEILING.md §0의 col_name, W-SUN last, h = 1..8.
- **1차 지표:** 시리즈 동일가중 pooled cell MAPE입니다. capped APE, §5.1, cell-v3 기준이고 naive forecast 대비 감소율로 봅니다.
  시리즈×h 표를 함께 냅니다.
- **판정 창 (개정 4):** TSCV origin 전부(월별 첫 주, 최대 21개)입니다. 봉인·전방은 없습니다.

## 가설

**H1 (1차).** 사전등록 구조 예측 `struct`는 TSCV origin에서 pooled MAPE를 naive forecast 대비 50% 이상 낮춥니다.
등록 시점의 기대는 **H1 거짓**입니다(CEILING.md). 그래서 이 라운드는 Y와 그 불확실성을 보고합니다.

**S1–S4 (2차).** 측정 가능한 시리즈별로 `struct`가 naive forecast를 이깁니다. gold, copper, corn, wti가 대상입니다.

## 후보와 기준선 (Gate F 7.2: 닫힌형 OLS, 자유 모수 없음)

각 타깃을 `r = f + b`로 분해합니다. `f`는 시장 성분으로, 예측하지 않습니다(0).
`b`는 DB에서 관측 가능한 성분이고, 이것을 자기 수준과 달력으로 추정합니다.
예측은 `r̂_h = b̂_h`이고, h마다 OLS(최소노름 lstsq)로 구합니다.
학습 표본은 origin 시점에 **이미 성숙한 라벨**(s + h ≤ t)만이고, MIN_TRAIN = 100입니다. 개정 4부터 TSCV origin마다 다시 적합합니다.

| 타깃 | 시장 성분 f | b 라벨 | 수준 x_t | DB 시리즈 |
|---|---|---|---|---|
| gold | 현물 | 캐리 = EFFR × 7h/360 (결정적, 추정 없음) | — | `Macro_NYFed_Effective_Federal_Funds_Rate_Pct` (발표 지연 때문에 타깃일 **전날까지**만 사용) |
| copper | LME 3M, 고정계약 환산 | r − Δlog3M + h·log(3M/cash)_{t+h}/13 | log(3M/cash) | `Com_LME_Cu_3M_Bid` |
| corn | CBOT nearby = cash − basis, 롤 점프 제거 | r − ΔG | log(cash/F) | `Com_Barchart_US_Corn_Basis_Index` |
| wti | Brent 현물 | r − Δlog Brent (지역 basis) | log(WTI/Brent) | `Com_EIA_Daily_Europe_Brent_Spot_Price_FOB_USD_Per_Barrel` |
| wheat | — | DB에 SRW 선물·basis 없음 | — | → `struct` = naive forecast |
| coffee | — | DB에 ICE 로부스타 없음 | — | → `struct` = naive forecast |

| 이름 | 정의 | 역할 |
|---|---|---|
| `pers` | r̂ = 0 | 기준선 (a) |
| `fut` | gold: 캐리. copper: h·log(3M/cash)/13. corn: 0 (DB에 원월물 없음 → naive forecast와 같음). wti·wheat·coffee: **DB에 없음** | 기준선 (b) 선물 내재 예측 |
| — | EIA STEO 유가·WASDE 가격·World Bank CMO·LBMA 조사는 DB에 없음. Dallas Fed DES WTI 중간값(분기)은 DB에 타깃 정의(시점·호라이즌)가 없음 | 기준선 (c): **평가 불가로 기록** |
| `struct` | b ~ 1 + x_t + 월 더미(11) | **등록 후보** |
| `lvl`, `mon` | b ~ 1 + x_t / b ~ 1 + 월 더미 | 절제 변형. **PBO 계산에만** 쓰고 선택하지 않음 |

Gate B: 데이터 축(basis·3M 곡선·Brent·캐리 성분, 거시 외생변수 없음)과 특징 축(성분 수준·달력, 추정 대상을 r에서 b로 바꿈)
두 축을 바꿨습니다. 모델 계열은 OLS 그대로입니다.

## 창 (개정 4)

| 항목 | 정의 |
|---|---|
| origin | `TSCV_START` = 2025-01 ~ `TSCV_END` = 2026-09, 각 월의 첫 W-SUN 주(`engine.tscv_origins`). 최대 21개 |
| 채택 | 6개 시리즈 모두 h = 1..8 타깃이 성숙한 origin만 pooled에 씁니다. 2026-11-09 DB에서는 LBMA 금이 2026-09-29에서 멈춰 2026-09 origin이 빠집니다(20개). DB가 갱신되면 자동으로 들어옵니다 |
| 학습 | origin마다 성숙한 라벨 전체(expanding)로 재적합 |
| 구 규칙 | dev(타깃 2025-06-29 이하)·봉인 52 origin(2025-07-06 – 2026-06-28)·전방 52 origin. 모두 폐기 |

**선언된 후향성 (PROTOCOL 4.4):** 봉인 결과를 본 뒤 규칙을 바꿨습니다. 같은 기간(2025-01 이후)에 E022–E027도 결과를 봤습니다.
`struct`의 성분 선택은 CEILING.md를 참고했습니다. 어느 것도 이 평가를 확증으로 만들지 않습니다.

## 누수 검출기 (평가보다 먼저 실행)

| ID | 시험 | 통과 조건 |
|---|---|---|
| D1 | (매주 origin; 20여 개로는 검정력이 없는 파이프라인 시험이라 격자만 넓힘) 타깃을 같은 변동성의 랜덤워크로 바꾸고, 성분은 실제 데이터 그대로 | pooled d의 26주 블록 CI 하한 ≤ 0 (개선을 찾지 못함) |
| D2 | **v2**: 실제 가격에 달력 신호를 심음(1주 R² = PLANTED_R2 = 0.10). 대상은 copper, corn, wti | 회수한 pooled 감소가 > 0 (CI 하한 > 0)이고, 신호를 아는 오라클 감소의 절반 이상 |
| D3 | **개정 4**: TSCV origin마다 그 주 이후의 일별 데이터를 전부 교란 | origin ≤ c 의 모든 예측이 비트 단위로 동일 (Gate E item 5) |
| D4 | 양성 대조: 다음 주 실제 수익률을 회귀변수로 넣은 누수 변형 | D3이 그 변형을 **검출**함 (예측이 달라짐) |

**개정 1 (2026-11-08, dev 실행 전):** D2 v1은 신호를 **랜덤워크** 위에 심었습니다.
합성 가격이 실제 basis(cents)와 무관하게 떠돌아 수준 회귀변수 x가 비정상이 됐고, 회수 −270%로 실패했습니다(오라클 +19.5%).
이는 검출기 설계 결함이므로, v2는 신호를 실제 가격 위에 심습니다. 통과 기준은 바꾸지 않았습니다.
v1 결과는 `results/detectors_v1.json`에 남겨 둡니다.

## 구 규칙 하의 기록 (2026-11-08, 개정 4 이전)

dev `struct` pooled 감소 +0.62%(CI가 0 포함, Gate A 불통과), 봉인 52 origin pooled 감소 +0.59%(DM-HLN p = 0.31)로
"차이 검출 안 됨"이었습니다. 검출기 D1–D4는 통과했습니다(`results/detectors_v1.json`은 D2 v1). 이 수치는 구 평가 집합의 것이라
새 TSCV 결과와 비교하지 않습니다.

## 평가 (TSCV origin 전체)

- **1차:** pooled MAPE 감소율, 시리즈×h 표, 26주 calendar-block 부트스트랩 CI (`engine.block_bootstrap`).
  2025-01~2026-09는 26주 블록이 4–5개뿐이라 CI는 형식상 보고만 합니다.
  추론은 아래 HAC 기반 검정이 맡습니다(래그 `ORIGIN_LAG` = 2, origin 수준 = h 평균. origin 간격이 4–5주라 8주 지평은 다음 origin까지만 겹칩니다).
- **Clark–West** (중첩, 로그수익률 MSE), **DM + HLN** (capped APE), **Giacomini–White** (도구변수 1, d_{t−2}).
- **MCS** (Hansen–Lunde–Nason, α = 0.05)와 **Hansen SPA / White RC**. `arch`, 블록 26, 1000회, 모형 {pers, fut, lvl, mon, struct}.
- **다중검정:**
  - 라운드: 명제 5개(pooled + 4 시리즈)에 DM-HLN p로 Holm.
  - **프로그램 전체:** m = 130 (E001–E027 후보 방향 126 + E028의 fut·lvl·mon·struct 4).
    pooled 명제에 Bonferroni 적용. 리더보드는 12,656행.
- **CSCV PBO** (S = 16): TSCV origin의 pooled 손실로 계산.
- **Campbell–Thompson OOS R²** (로그수익률).
- **P&L** (h = 1, origin당 1주, 비중첩):
  - 포지션 = sign(r̂_t − 이전 origin 예측의 평균)으로, 타이밍만 봅니다. origin마다 1주 보유라 Sharpe 연율화는 √12입니다(`ORIGINS_PER_YEAR`).
  - 선물 성분 f에 대한 P&L과 basis 성분 b에 대한 P&L(물리 basis 거래)을 따로 냅니다.
  - 비용 0/5/25 bp, 연율 Sharpe.
- **Gate E:** item 1–4는 `experiment/transfer_contract.evaluate_frame`으로 봅니다(대조: fut가 있으면 fut, 없으면 naive forecast). item 5는 D3입니다.

**개정 4 (2026-11-09, 봉인 결과를 본 뒤):** 위 구 규칙의 dev·봉인·전방을 폐기하고 TSCV 단일 규칙으로 바꿨습니다.

- 전방 평가(`forward`, `forward_eval`, `forward_check`, `forward_weekly.sh`, `forward/`, `data/full`)를 삭제했습니다.
  전방 로그에는 2026-11-08 발행 1건뿐이었고 평가는 시작된 적이 없습니다(개정 2·3은 이 규칙과 함께 폐기).
- `prepare`는 DB 전체를 `data/daily`에 한 번 만듭니다(절단 없음).
- 명령은 `ceiling`, `detectors`, `tscv`, `ledger`입니다. `dev`·`sealed`는 `tscv` 하나로 합쳤습니다.
- `ceiling`의 `rolling_52` 절(봉인 길이 창)을 TSCV origin 위의 상한(`tscv`)으로 바꿨습니다.
- 통계 래그는 origin 단위로 바꿨습니다(`ORIGIN_LAG` = 2).

## 판정 규칙 (셋 중 하나만)

1. TSCV pooled d의 DM-HLN 단측 p가 Holm을 통과하지 못하면 → **"차이 검출 안 됨"**.
2. 통과했지만 감소율이 50% 미만이면 → **"달성 불가: 상한 X%, 실제 달성 Y%, 위치 Z"**.
   X는 CEILING.md 값, Y는 TSCV pooled 감소율, Z는 감소가 나온 시리즈·h·성분입니다.
3. 감소율 50% 이상이면 → 아래가 **모두** 충족될 때만 **"50% 달성(외부 기준 충족)"**입니다.
   - 프로그램 전체 Bonferroni p < 0.05
   - SPA p < 0.05
   - P&L Sharpe > 0 (비용 5 bp)
   - D1–D4 통과
   봉인·전방이 없으므로 3번은 어떤 경우에도 "탐색 결과"로만 표기합니다(PROTOCOL 4.4).

어떤 시리즈의 MAPE 개선이 유의한데 그 시리즈의 두 P&L이 모두 Sharpe ≤ 0이면, 그 개선은 **인공물**로 판정합니다.

## 재현

```bash
uv run --with xlrd python experiment/E028/E028.py prepare          # mini: DB 전체 스냅샷
uv run --with arch python experiment/E028/E028.py detectors
uv run --with arch python experiment/E028/E028.py tscv
uv run --with arch python experiment/E028/E028.py ceiling
uv run python experiment/E028/E028.py ledger                       # mini: W&B 원장 반영
uv run python tools/check_all.py                                   # 셀 단위 재현 + 독립 평가기 대조 포함
```
