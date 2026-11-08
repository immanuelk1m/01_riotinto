# E028 — 50% 캠페인 (Gate C 사전등록)

**등록:** 2026-11-08. 탐지기·dev 감사·봉인 해제·전방 기록 전에 작성했습니다.
**프로토콜:** PROTOCOL.md v1.1, Gate A–F. 평가기 **cell-v3**, `PARAMS_VERSION = 2.1`.
**0단계:** [`CEILING.md`](CEILING.md). 상한은 11.5%이고, 밀·커피를 완벽 예측해도 41.6%입니다.
사용자가 상한을 확인한 뒤에도 캠페인 진행을 지시했으므로, 이 라운드의 목적은 **실제 달성치 Y를 외부 기준으로 측정하는 것**입니다.
**데이터 제약 (사용자 지시, 2026-11-08):** 비용을 쓰지 않고 `dbname=commodity`에 있는 데이터만 씁니다.
0단계에서 쓴 EIA NYMEX 파일은 DB 밖에 있으므로 이 라운드에서는 쓰지 않습니다.

## 고정 조건

- **타깃 6개(정의 그대로):** gold, copper, corn, wheat, coffee, wti. CEILING.md §0의 col_name, W-SUN last, h = 1..8.
- **1차 지표:** 시리즈 동일가중 pooled cell MAPE입니다. capped APE, §5.1, cell-v3 기준이고 persistence 대비 감소율로 봅니다.
  시리즈×h 표를 함께 냅니다.
- **판정 창:** 봉인 52 origin과 전방 52 origin뿐입니다. dev 수치는 근거가 아닙니다.

## 가설

**H1 (1차).** 사전등록 구조 예측 `struct`는 봉인 창에서 pooled MAPE를 persistence 대비 50% 이상 낮춥니다.
등록 시점의 기대는 **H1 거짓**입니다(CEILING.md). 그래서 이 라운드는 Y와 그 불확실성을 보고합니다.

**S1–S4 (2차).** 측정 가능한 시리즈별로 `struct`가 persistence를 이깁니다. gold, copper, corn, wti가 대상입니다.

## 후보와 기준선 (Gate F 7.2: 닫힌형 OLS, 자유 모수 없음)

각 타깃을 `r = f + b`로 분해합니다. `f`는 시장 성분으로, 예측하지 않습니다(0).
`b`는 DB에서 관측 가능한 성분이고, 이것을 자기 수준과 달력으로 추정합니다.
예측은 `r̂_h = b̂_h`이고, h마다 OLS(최소노름 lstsq)로 구합니다.
학습 표본은 origin 시점에 **이미 성숙한 라벨**(s + h ≤ t)만이고, MIN_TRAIN = 100, REFIT_EVERY = 4입니다.

| 타깃 | 시장 성분 f | b 라벨 | 수준 x_t | DB 시리즈 |
|---|---|---|---|---|
| gold | 현물 | 캐리 = EFFR × 7h/360 (결정적, 추정 없음) | — | `Macro_NYFed_Effective_Federal_Funds_Rate_Pct` (발표 지연 때문에 타깃일 **전날까지**만 사용) |
| copper | LME 3M, 고정계약 환산 | r − Δlog3M + h·log(3M/cash)_{t+h}/13 | log(3M/cash) | `Com_LME_Cu_3M_Bid` |
| corn | CBOT nearby = cash − basis, 롤 점프 제거 | r − ΔG | log(cash/F) | `Com_Barchart_US_Corn_Basis_Index` |
| wti | Brent 현물 | r − Δlog Brent (지역 basis) | log(WTI/Brent) | `Com_EIA_Daily_Europe_Brent_Spot_Price_FOB_USD_Per_Barrel` |
| wheat | — | DB에 SRW 선물·basis 없음 | — | → `struct` = persistence |
| coffee | — | DB에 ICE 로부스타 없음 | — | → `struct` = persistence |

| 이름 | 정의 | 역할 |
|---|---|---|
| `pers` | r̂ = 0 | 기준선 (a) |
| `fut` | gold: 캐리. copper: h·log(3M/cash)/13. corn: 0 (DB에 원월물 없음 → persistence와 같음). wti·wheat·coffee: **DB에 없음** | 기준선 (b) 선물 내재 예측 |
| — | EIA STEO 유가·WASDE 가격·World Bank CMO·LBMA 조사는 DB에 없음. Dallas Fed DES WTI 중간값(분기)은 DB에 타깃 정의(시점·호라이즌)가 없음 | 기준선 (c): **평가 불가로 기록** |
| `struct` | b ~ 1 + x_t + 월 더미(11) | **등록 후보** |
| `lvl`, `mon` | b ~ 1 + x_t / b ~ 1 + 월 더미 | 절제 변형. **PBO 계산에만** 쓰고 선택하지 않음 |

Gate B: 데이터 축(basis·3M 곡선·Brent·캐리 성분, 거시 외생변수 없음)과 특징 축(성분 수준·달력, 추정 대상을 r에서 b로 바꿈)
두 축을 바꿨습니다. 모델 계열은 OLS 그대로입니다.

## 창

| 창 | 정의 |
|---|---|
| dev | 모든 h의 타깃이 **2025-06-29 이하**인 origin. 6개 시리즈 공통 origin만 pooled에 씀 |
| 봉인 | 2025-07-06 – 2026-06-28 의 52 origin (타깃은 2026-08-23까지). **한 번만** 해제 |
| 전방 | 2026-11-08 주부터 52 origin. 마지막 origin 2027-10-31, 성숙 2027-12-26. **2028-01-10 이후 한 번** 평가. E027 전방 평가(2027-11-01)와 겹치지 않음 |

**선언된 후향성 (PROTOCOL 4.4):** 봉인 창은 E022–E027이 이미 봤습니다. 패널 집계는 출력됐고, E027은 커피·옥수수 시리즈별로 다른 모델을 해제했습니다.
따라서 봉인 창 결과는 단독 근거가 될 수 없고, 확증 창은 전방입니다.
`struct`의 성분 선택은 CEILING.md를 참고했는데, 이는 dev 데이터만으로 계산한 것입니다.

## 누수 검출기 (dev 데이터만, dev 감사보다 먼저 실행)

| ID | 시험 | 통과 조건 |
|---|---|---|
| D1 | 타깃을 같은 변동성의 랜덤워크로 바꾸고, 성분은 실제 데이터 그대로 | pooled d의 26주 블록 CI 하한 ≤ 0 (개선을 찾지 못함) |
| D2 | **v2**: 실제 가격에 달력 신호를 심음(1주 R² = PLANTED_R2 = 0.10). 대상은 copper, corn, wti | 회수한 pooled 감소가 > 0 (CI 하한 > 0)이고, 신호를 아는 오라클 감소의 절반 이상 |
| D3 | 26주 블록마다 컷오프 c 이후의 일별 데이터를 전부 교란 | origin ≤ c 의 모든 예측이 비트 단위로 동일 (Gate E item 5) |
| D4 | 양성 대조: 다음 주 실제 수익률을 회귀변수로 넣은 누수 변형 | D3이 그 변형을 **검출**함 (예측이 달라짐) |

**개정 1 (2026-11-08, dev 실행 전):** D2 v1은 신호를 **랜덤워크** 위에 심었습니다.
합성 가격이 실제 basis(cents)와 무관하게 떠돌아 수준 회귀변수 x가 비정상이 됐고, 회수 −270%로 실패했습니다(오라클 +19.5%).
이는 검출기 설계 결함이므로, v2는 신호를 실제 가격 위에 심습니다. 통과 기준은 바꾸지 않았습니다.
v1 결과는 `results/detectors_v1.json`에 남겨 둡니다.

## 봉인 해제 전 기록 (2026-11-08, dev 결과를 본 뒤, 봉인 창을 보기 전)

| 항목 | 값 |
|---|---|
| 검출기 | D1·D2(v2)·D3·D4 모두 통과 (`results/detectors.json`) |
| dev `struct` pooled 감소 | +0.62%, d CI [−0.00016, +0.00088] → Gate A **불통과** |
| dev 시리즈별 감소 | gold +1.22%, copper +0.42%, corn +1.86%, wti +0.66% (모두 CI가 0 포함) |
| 봉인 52 origin의 MDE | pooled 감소 **4.1%** (d 0.0023) |
| dev PBO | 0.555 |
| **예상 봉인 판정** | **"차이 검출 안 됨"**: MDE가 dev 효과의 약 7배 |

Gate A 불통과로 금지되는 것은 규칙·보정·적응 실험입니다(2.3). 등록된 모델 자체의 봉인 평가는 그대로 진행합니다(E027과 같은 방식).

## 평가 (봉인 창, 한 번)

- **1차:** pooled MAPE 감소율, 시리즈×h 표, 26주 calendar-block 부트스트랩 CI (`engine.block_bootstrap`).
  봉인 52 origin은 블록이 2–3개뿐이라 CI는 형식상 보고만 합니다.
  추론은 아래 HAC 기반 검정이 맡습니다(래그 H − 1 = 7, origin 수준 = h 평균).
- **Clark–West** (중첩, 로그수익률 MSE), **DM + HLN** (capped APE), **Giacomini–White** (도구변수 1, d_{t−8}).
- **MCS** (Hansen–Lunde–Nason, α = 0.05)와 **Hansen SPA / White RC**. `arch`, 블록 26, 1000회, 모형 {pers, fut, lvl, mon, struct}.
- **다중검정:**
  - 라운드: 명제 5개(pooled + 4 시리즈)에 DM-HLN p로 Holm.
  - **프로그램 전체:** m = 130 (E001–E027 후보 방향 126 + E028의 fut·lvl·mon·struct 4).
    pooled 명제에 Bonferroni 적용. 리더보드는 12,656행.
- **CSCV PBO** (S = 16): dev 창에서 pooled origin 손실로 계산.
- **Campbell–Thompson OOS R²** (로그수익률).
- **P&L** (h = 1, 주간, 비중첩):
  - 포지션 = sign(r̂_t − 과거 예측의 평균)으로, 타이밍만 봅니다.
  - 선물 성분 f에 대한 P&L과 basis 성분 b에 대한 P&L(물리 basis 거래)을 따로 냅니다.
  - 비용 0/5/25 bp, 연율 Sharpe.
- **Gate E:** item 1–4는 `experiment/transfer_contract.evaluate_frame`으로 봅니다(대조: fut가 있으면 fut, 없으면 persistence). item 5는 D3입니다.

**개정 2 (2026-11-08, 첫 전방 예측 발행 직후, 평가 전):** DB 마지막 관측이 시리즈마다 현재 주보다 늦습니다.

| 시리즈 | 마지막 관측 주 |
|---|---|
| coffee·corn | 2026-11-01 |
| copper | 2026-10-18 |
| wti | 2026-10-11 |
| gold | 2026-10-04 |
| wheat | 2026-08-23 |

따라서 전방 셀은 **목표 주의 끝이 발행 시각(`issued_utc`)보다 뒤인 것만** 셉니다.
발행 시점에 DB가 1주 넘게 늦은 시리즈는 그 주의 전방 origin에서 빠집니다.
52 origin은 6개 시리즈가 모두 최신인 발행 주만으로 셉니다.
로그는 `forward/forward_<발행일>.csv`에 쌓이며, 추가만 하고 고치지 않습니다.

**개정 3 (2026-11-08, 전방 목표가 하나도 성숙하기 전):** 이 시점에 DB는 2026-11-01 14:18 이후 갱신이 전혀 없습니다.
전방 실현값은 하나도 불러오지 않았습니다.

1. **입력 신선도.** 행의 신선도는 타깃만 보지 않고, 그 예측이 읽는 DB 입력 가운데 가장 오래된 것으로 잽니다.
   - 입력은 gold EFFR, copper LME 3M, corn basis, wti Brent입니다.
   - 발행 때 `input_stale_weeks`로 기록합니다.
   - 2026-11-08 발행분에는 이 열이 없어 셈에서 빠집니다. 개정 2로도 이미 빠지는 발행입니다(wheat 11주).
   - 그 발행의 corn struct는 08-24(10주 전) basis 위에서 나왔습니다. 이 결함을 찾은 것이 이 개정의 계기입니다.
2. **origin 가격.** 발행 때 origin 주의 로그 가격 `logp`를 기록합니다.
   실현 로그수익률은 평가 시점 DB 스냅샷의 목표 주 로그 가격에서 기록된 `logp`를 뺀 값입니다.
3. **창.**
   - 주마다 첫 발행만 씁니다.
   - 6개 시리즈 모두 입력 포함 1주 이하로 신선한 발행 주만 셉니다.
   - 처음부터 차례로 52개가 전방 창입니다.
   - 52개 발행의 목표 주가 모두 DB에 들어온 뒤 `forward_eval`로 **한 번만** 평가합니다. 그 전에 `forward_eval`은 개수만 출력하고 손실은 계산하지 않습니다.
   - 위 표의 2028-01-10은 DB가 매주 최신일 때 가장 이른 날짜입니다.
4. **평가 내용.**
   - 명제 5개에 DM-HLN과 Holm(라운드)을 적용합니다.
   - Clark–West, GW, CT R², fut 대조를 함께 냅니다.
   - pooled에는 Bonferroni를 적용합니다. m은 `results/program_family.json`의 130입니다.
   - 판정은 아래 판정 규칙을 그대로 따릅니다.
   - 전방 로그에는 fut·struct만 있으므로 SPA/MCS는 하지 않습니다.
   - 목표 주에 DB 관측이 없는 셀(결과와 무관한 결측)은 빼고 개수를 보고합니다.
5. **평가기 검증.** `forward_check`는 봉인 52 origin을 발행 로그 형태로 바꿔 넣습니다.
   2026-11-08 Colab 실행에서 52 발행, 셀 해시 = `sealed.json`, MAPE 차이 0이었습니다.
   같은 실행에서 바뀐 `forward`가 낸 48개 예측은 첫 발행과 같았습니다.
6. **주간 실행.** `forward_weekly.sh`를 쓰며, launchd가 월요일 09:00 KST에 돌립니다.
   - mini에서 `prepare full`을 돌립니다.
   - 스냅샷이 그대로면 발행하지 않습니다. 같은 주가 다시 나오므로 신선도 규칙으로 어차피 빠지기 때문입니다.
   - 바뀌었으면 Colab에서 `forward`를 돌립니다.
   - 결과는 `forward/`에 추가하고, `forward/`와 `data/full`만 커밋합니다.

## 판정 규칙 (셋 중 하나만)

1. 봉인 pooled d의 DM-HLN 단측 p가 Holm을 통과하지 못하면 → **"차이 검출 안 됨"**.
2. 통과했지만 감소율이 50% 미만이면 → **"달성 불가: 상한 X%, 실제 달성 Y%, 위치 Z"**.
   X는 CEILING.md 값, Y는 봉인 pooled 감소율, Z는 감소가 나온 시리즈·h·성분입니다.
3. 감소율 50% 이상이면 → 아래가 **모두** 충족될 때만 **"50% 달성(외부 기준 충족)"**입니다.
   - 프로그램 전체 Bonferroni p < 0.05
   - SPA p < 0.05
   - P&L Sharpe > 0 (비용 5 bp)
   - D1–D4 통과
   - 전방 창 재현
   전방 창이 성숙하지 않은 지금은 3번 판정을 낼 수 없습니다.

어떤 시리즈의 MAPE 개선이 유의한데 그 시리즈의 두 P&L이 모두 Sharpe ≤ 0이면, 그 개선은 **인공물**로 판정합니다.

## 재현

```bash
uv run --with xlrd python experiment/E028/E028.py prepare          # mini: dev 스냅샷 (cutoff 2025-06-29)
uv run python -m _colab.job run --exp E028 --cmd detectors --cmd dev --include _exo --install arch
uv run python experiment/E028/E028.py prepare full                 # mini: 봉인 해제 시점에만
uv run python -m _colab.job run --exp E028 --cmd sealed --include _exo --install arch
uv run python experiment/E028/E028.py ledger                       # mini: leaderboard.sqlite 반영
uv run python tools/check_all.py                                   # 셀 단위 재현 + 독립 평가기 대조 포함
R=experiment/E028/results                                           # results/는 번들에 안 들어감
uv run python -m _colab.job run --exp E028 --cmd forward_check --include _exo \
  --include $R/sealed.json --include $R/sealed_cells.parquet --install arch   # 전방 평가기 = 봉인 평가기
experiment/E028/forward_weekly.sh                                   # 매주 한 번 발행
uv run python -m _colab.job run --exp E028 --cmd forward_eval --include _exo \
  --include $R/program_family.json --install arch                   # 성숙 뒤 한 번
```
