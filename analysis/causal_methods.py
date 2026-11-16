"""후속 인과발견 방법의 어댑터. 방법은 "가격과 이어진 링크"만 내고, 귀속·채점은 `causal_study` 와 같은 코드가 한다.

어댑터: `fn(T, sink, tau, logp) -> dict(edges=[(변수, 시차, 표시)], extra={...})`.
- T: 학습 패널(z-score, 늦게 시작한 변수의 시작 전 = 0, 결측 없음, 0번 열 `price`). sink: 열 순서의 bool(같은 주 방향 가정 대상).
- 표시는 변수 쪽에서 본 가격과의 링크: `-->` 변수 → 가격, `<--` 가격 → 변수(시차 0 만), `o-o`·`<->` 등 그 밖은 방향 미정.
- 레짐 방법은 학습 구간 마지막 주의 레짐 그래프를 쓴다(평가 구간의 정보를 쓰지 않는다, `regime_rule = last_training_regime`).

모든 방법은 선형 검정·점수(ParCorr, Fisher-z, 선형 회귀 점수, 선형 DYNOTEARS), τ_max = 4, 유의수준 0.05 로 맞춘다. 신경망은 쓰지 않는다.
"""
import inspect
import hashlib
from pathlib import Path

import numpy as np

ALPHA = 0.05
HERE = Path(__file__).resolve().parent


def parents_from_edges(edges, sink_vars):
    """링크 → 가격의 부모 {(변수, 시차)}. 시차 ≥ 1 은 시간 순서로 변수 → 가격(가격 자기 시차는 회귀의 통제 항으로 남는다).
    시차 0 은 `-->` 이거나, sink 변수와 이어진 링크(표시와 무관 — PCMCI 의 같은 주 방향 가정을 모든 방법에 사후로 똑같이 준다)."""
    out = set()
    for v, lag, mark in edges:
        if v == "price" and lag == 0:
            continue
        if lag > 0 or mark == "-->" or v in sink_vars:
            out.add((v, int(lag)))
    return out


def _tigramite_edges(g, cols, price=0):
    return [(cols[i], t, str(g[i, price, t])) for i in range(len(cols)) for t in range(g.shape[2]) if (i, t) != (price, 0) and g[i, price, t] != ""]


def matched_pcmci(T, sink, tau, logp):
    """대조군: `causal_study` 의 matched_pcmci 와 같은 호출(PCMCI+ · ParCorr · 같은 주 방향 가정)."""
    from analysis.causal_study import Spec, discover
    g = discover(T, sink, Spec(matched=True, tau=tau))
    return dict(edges=_tigramite_edges(g, list(T.columns)))


def cdnod(T, sink, tau, logp):
    """CD-NOD (Huang et al., JMLR 2020; causal-learn). 시차 펼침 [X_t | X_{t-1} | … | X_{t-τ}] + 시간 인덱스 C.
    배경 지식: 현재 → 과거 금지, 가격(t) → sink 변수(t) 금지(같은 주 방향 가정)."""
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.search.ConstraintBased.CDNOD import cdnod as run
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
    V, cols = T.values, list(T.columns)
    n, d = V.shape
    E = np.column_stack([V[tau - k:n - k] for k in range(tau + 1)])  # 열 k·d + i = 변수 i 의 시차 k
    node = lambda j: GraphNode(f"X{j + 1}")  # noqa: E731 — causal-learn 의 기본 노드 이름
    bk = BackgroundKnowledge()
    for i in range(d):
        for j in range(d, E.shape[1]):
            bk.add_forbidden_by_node(node(i), node(j))
        if sink[i]:
            bk.add_forbidden_by_node(node(0), node(i))
    g = run(E, np.arange(len(E), dtype=float).reshape(-1, 1), ALPHA, "fisherz", background_knowledge=bk, show_progress=False).G.graph
    edges = []
    for j in range(1, E.shape[1]):
        if g[j, 0] == 0 and g[0, j] == 0:
            continue
        v, lag = cols[j % d], j // d
        mark = "-->" if (g[j, 0] == -1 and g[0, j] == 1) else "<--" if (g[j, 0] == 1 and g[0, j] == -1) else "o-o"
        edges.append((v, lag, mark))
    return dict(edges=edges)


def _causalts(result, cols):
    m, names = result.to_marks(), list(result.var_names)
    p = names.index("price")
    return [(names[i], t, str(m[i, p, t])) for i in range(len(names)) for t in range(m.shape[2])
            if names[i] in cols and (i, t) != (p, 0) and m[i, p, t] != ""]


def cdnots(T, sink, tau, logp):
    """CD-NOTS (Sadeghi et al. 2025; Causal-TS). 시간 인덱스 노드로 비정상성을 다루는 PC 계열. ParCorr."""
    from causalts import run_cdnots
    from causalts.ci_tests import create_ci_test
    df = T.reset_index(drop=True)
    return dict(edges=_causalts(run_cdnots(df, create_ci_test("parcorr", df.values), num_lags=tau, alpha=ALPHA), list(T.columns)))


def cdnots_plus(T, sink, tau, logp):
    """CDNOTS+ (Causal-TS): CD-NOTS 에 PCMCI+ 식 2단계 뼈대(MCI). ParCorr."""
    from causalts import run_cdnots_plus
    from causalts.ci_tests import create_ci_test
    df = T.reset_index(drop=True)
    return dict(edges=_causalts(run_cdnots_plus(df, create_ci_test("parcorr", df.values), num_lags=tau, alpha=ALPHA), list(T.columns)))


def cedar(T, sink, tau, logp):
    """CEDAR (Fesanghary et al. 2026; Causal-TS). 최소 시차 선택 + 쌍별 검정. ParCorr, 같은 주 링크 포함."""
    from causalts import run_cedar
    from causalts.ci_tests import create_ci_test
    df = T.reset_index(drop=True)
    r = run_cedar(df, create_ci_test("parcorr", df.values), tau, alpha_cond1=ALPHA, alpha_cond2=ALPHA, include_lag0=True)
    return dict(edges=_causalts(r, list(T.columns)))


def regime_pcmci(T, sink, tau, logp):
    """Regime-PCMCI (Saggioro et al., Chaos 2020; tigramite RPCMCI). 레짐 2개, 전환 ≤ 20% 전환점 수, 시차 링크만(τ_min = 1)."""
    from analysis.rcbnb import turning_points
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.rpcmci import RPCMCI
    n_switch = turning_points(logp)
    out = RPCMCI(pp.DataFrame(T.values), ParCorr(significance="analytic"), seed=0, verbosity=-1).run_rpcmci(
        num_regimes=2, max_transitions=n_switch, tau_max=tau, pc_alpha=ALPHA, alpha_level=ALPHA, n_jobs=-1)
    labels = np.argmax(out["regimes"], axis=0)
    g = out["causal_results"][int(labels[-1])]["graph"]
    return dict(edges=_tigramite_edges(g, list(T.columns)),
                extra=dict(labels=labels.tolist(), max_transitions=n_switch, last_regime=int(labels[-1])))


def rcbnb_mb(T, sink, tau, logp):
    """RCBNB-MB (`RCBNB-MB.ipynb` 의 방식 C: 레짐 2개, 전환 ≤ 20% 전환점 수). 최소 레짐 크기 max((2τ+1)·d, 52) 를
    두 레짐이 채우지 못하면 노트북 규칙(r_max = T // N_l)대로 레짐 1개(CBNB)가 된다."""
    from analysis import rcbnb
    d = T.shape[1]
    n_min = max((2 * tau + 1) * d, rcbnb.SEASON)
    r = 2 if len(T) >= 2 * n_min else 1
    n_switch = rcbnb.turning_points(logp)
    best = rcbnb.fit(T.values, r, n_min, n_switch, tau, ALPHA, sink)
    k = int(best["labels"][-1])
    A = best["graphs"][k]
    cols = list(T.columns)
    return dict(edges=[(cols[i], int(t), "-->") for i, t in zip(*np.nonzero(A[:, 0])) if (i, t) != (0, 0)],
                extra=dict(labels=best["labels"].tolist(), regimes=r, min_regime=n_min, max_transitions=n_switch,
                           last_regime=k, converged=best["converged"], inits_ok=best["n_ok"]))


def spacetime(T, sink, tau, logp):
    """SPACETIME (Mameche et al., AAAI 2025; causalchange). 변화점(PELT) 탐지 + 시간 인과 그래프. 선형 점수, 최소 구간 52주.
    그래프는 구간 사이에 공유되고 메커니즘만 바뀐다 — 레짐 규칙이 필요 없다."""
    from causalchange import SpaceTime
    from causalchange.scoring import statistical_tests as st
    from scipy.stats import ks_2samp
    mmd = st.SCMEqualityTestKCI._mmd_or_ks_pvalue
    def mmd_or_ks(self, ya, yb):  # MMD(hyppo)는 분산 0 입력을 거부한다(시작 전 = 0 인 변수). 패키지에 주석으로 남은 KS 대체 검정을 쓴다
        try:
            return mmd(self, ya, yb)
        except ValueError:
            return float(ks_2samp(ya, yb).pvalue)
    st.SCMEqualityTestKCI._mmd_or_ks_pvalue = mmd_or_ks
    m = SpaceTime(data_mode="time-contexts", score_type="lin", context_col="context", tau_max=tau, changepoint_mode="detect",
                  changepoint_scope="global", changepoint_method="pelt", d_min=52, max_iter=3, seed=0)
    m.fit(T.reset_index(drop=True).assign(context="all"))
    edges = [(u[0], int(u[1]), "-->") for u, v in m.graph_.edges if v == ("price", 0)]
    edges += [(v[0], 0, "<--") for u, v in m.graph_.edges if u == ("price", 0) and v[1] == 0]
    return dict(edges=edges, extra=dict(changepoints=[int(c) for c in m.changepoints_]))


def castor_linear(T, sink, tau, logp, w_threshold=0.4):
    """CASTOR 선형판 (Rahmani & Frossard, AISTATS 2025; `analysis/castor_linear.py`). 초기 창 104주, 최소 레짐 52주.
    가중치 문턱 기본값 0.4 = 저장소 튜토리얼 값."""
    from analysis import castor_linear as cl
    out = cl.castor(T.values, p=tau, window=104, zeta=52, w_threshold=w_threshold)
    k = int(out["labels"][-1])
    W, A = out["graphs"][k]
    cols, d = list(T.columns), T.shape[1]
    edges = [(cols[i], 0, "-->") for i in np.flatnonzero(W[:, 0]) if i != 0]
    edges += [(cols[i], 0, "<--") for i in np.flatnonzero(W[0]) if i != 0]
    edges += [(cols[i % d], i // d + 1, "-->") for i in np.flatnonzero(A[:, 0])]
    return dict(edges=edges, extra=dict(labels=out["labels"].tolist(), regimes=len(out["graphs"]), last_regime=k,
                                        iterations=out["iterations"], w_threshold=w_threshold))


def castor_linear_t01(T, sink, tau, logp):
    """민감도: CASTOR 가중치 문턱 0.4 → 0.1. 튜토리얼 값은 주간 수익률(표준화)의 작은 효과를 모두 지운다."""
    return castor_linear(T, sink, tau, logp, w_threshold=0.1)


def castor_linear_t005(T, sink, tau, logp):
    """민감도: CASTOR 가중치 문턱 0.05 (L1 벌점 λ = 0.05 와 같은 크기)."""
    return castor_linear(T, sink, tau, logp, w_threshold=0.05)


# 이름 → (어댑터, 이 방법의 결과를 바꾸는 파일·패키지). 순서 = 실행 순서.
METHODS = {
    "matched_pcmci": (matched_pcmci, ["analysis/causal_study.py"], ["tigramite"]),
    "cdnod": (cdnod, [], ["causal-learn"]),
    "cdnots": (cdnots, [], ["causalts"]),
    "cdnots_plus": (cdnots_plus, [], ["causalts"]),
    "cedar": (cedar, [], ["causalts"]),
    "regime_pcmci": (regime_pcmci, ["analysis/rcbnb.py"], ["tigramite", "ortools"]),
    "rcbnb_mb": (rcbnb_mb, ["analysis/rcbnb.py"], ["tigramite", "lingam"]),
    "spacetime": (spacetime, [], ["causalchange", "ruptures"]),
    "castor_linear": (castor_linear, ["analysis/castor_linear.py"], []),
    # 민감도 실행: 네 번째 항목 = 결과를 바꾸는 다른 어댑터 함수(캐시 열쇠에 넣는다)
    "castor_linear_t01": (castor_linear_t01, ["analysis/castor_linear.py"], [], [castor_linear]),
    "castor_linear_t005": (castor_linear_t005, ["analysis/castor_linear.py"], [], [castor_linear]),
}
REGIME = {"regime_pcmci", "rcbnb_mb", "castor_linear", "castor_linear_t01", "castor_linear_t005"}
SENSITIVITY = {"castor_linear_t01": ("castor_linear", dict(w_threshold=0.1)), "castor_linear_t005": ("castor_linear", dict(w_threshold=0.05))}


def method_spec(name, root):
    """이 방법의 캐시 열쇠: 어댑터 소스 + 딸린 파일 + 패키지 버전 + 공통 설정."""
    import importlib.metadata
    fn, files, packages, *helpers = METHODS[name]
    h = hashlib.sha256(inspect.getsource(fn).encode() + inspect.getsource(parents_from_edges).encode())
    for g in (helpers[0] if helpers else []):
        h.update(inspect.getsource(g).encode())
    for f in files:
        h.update((Path(root) / f).read_bytes())
    base, changes = SENSITIVITY.get(name, (None, {}))
    return dict(method=name, source_sha=h.hexdigest()[:16], alpha=ALPHA, matched=True,
                regime_rule="last_training_regime" if name in REGIME else None,
                packages={p: importlib.metadata.version(p) for p in packages},
                **(dict(sensitivity_of=base, **changes) if base else {}))
