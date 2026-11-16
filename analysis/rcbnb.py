"""RCBNB-MB (Zan et al., arXiv 2609.05150) — `RCBNB-MB.ipynb` 4절 구현을 모듈로 옮긴 것. 알고리즘은 노트북과 같다.

레짐별 CBNB 그래프(PCMCI+ 뼈대·시차 방향 + LiNGAM 동시점 방향) ↔ 각 주를 Markov blanket 회귀 손실이 가장 작은 레짐에 배정(MILP)을 번갈아 반복한다.
노트북과 다른 점: 캐시는 실행기(`tools/run_causal_methods.py`)가 맡고, 결측은 받지 않는다(입력 패널은 이미 채워져 있다).
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.signal import find_peaks
from scipy.sparse.csgraph import connected_components

SEASON = 52          # 최소 레짐 크기 하한: 작황·계절 한 바퀴
BULL_BEAR = 0.20     # 강세·약세 전환 기준: 20% 등락
N_INIT, N_ITER = 20, 20
MILP_SECONDS = 10


def turning_points(logp):
    """로그 가격에서 20% 이상 오르내린 고점·저점 수(최소 1) — 전환 횟수 상한 N_c."""
    prom = np.log(1 + BULL_BEAR)
    return max(len(find_peaks(logp, prominence=prom)[0]) + len(find_peaks(-logp, prominence=prom)[0]), 1)


def lagged(X, tau):
    T, d = X.shape
    Z = np.full((T, d, 2 * tau + 1), np.nan)
    for o in range(-tau, tau + 1):
        lo, hi = max(0, -o), min(T, T - o)
        Z[lo:hi, :, tau + o] = X[lo + o:hi + o]
    return Z


def ols_predict(P, y, fit, ok):
    D = np.column_stack([np.ones(len(P)), P])
    return D[ok] @ np.linalg.lstsq(D[fit], y[fit], rcond=None)[0]


def cbnb(X, Z, sel, tau, alpha, sink=None):
    import lingam
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI
    T, d = X.shape
    frame = pp.DataFrame(X, mask=np.repeat(~sel[:, None], d, 1))
    out = PCMCI(frame, ParCorr(significance="analytic", mask_type="y"), verbosity=0).run_pcmciplus(tau_min=0, tau_max=tau, pc_alpha=alpha)
    A, S = out["graph"] == "-->", out["graph"][:, :, 0] != ""
    A[:, :, 0] = False  # 동시점 방향은 아래 noise-based 단계가 정한다
    n_comp, comp = connected_components(sparse.csr_matrix(S), directed=False)
    for c in range(n_comp):
        m = np.flatnonzero(comp == c)
        if len(m) < 2:
            continue
        par = sorted({(int(i), int(t)) for j in m for i, t in zip(*np.nonzero(A[:, j]))})
        P = np.column_stack([Z[:, i, tau - t] for i, t in par]) if par else np.empty((T, 0))
        ok = sel & ~np.isnan(P).any(1)
        E = np.column_stack([X[ok, j] - ols_predict(P, X[:, j], ok, ok) for j in m])
        rank = np.argsort(lingam.DirectLiNGAM().fit(E).causal_order_)
        for a in range(len(m)):
            for b in range(len(m)):
                if S[m[a], m[b]] and rank[a] < rank[b]:
                    A[m[a], m[b], 0] = True
    if sink is not None:  # 같은 주 방향 가정: sink 변수와 가격(0번)의 같은 주 링크는 변수 → 가격
        fix = sink & S[:, 0]
        A[fix, 0, 0], A[0, fix, 0] = True, False
    return A


def markov_blankets(A):
    out = []
    for j in range(A.shape[0]):
        mb = {(int(i), -int(t)) for i, t in zip(*np.nonzero(A[:, j]))}
        for c, t in zip(*np.nonzero(A[j])):
            mb.add((int(c), int(t)))
            mb |= {(int(m), int(t - t2)) for m, t2 in zip(*np.nonzero(A[:, c]))}
        out.append(sorted(mb - {(j, 0)}))
    return out


def sq_errors(X, Z, mbs, train, tau):
    r, (T, d) = len(mbs), X.shape
    E = np.full((r, T, d), np.nan)
    for k in range(r):
        for j in range(d):
            P = np.column_stack([Z[:, i, tau + o] for i, o in mbs[k][j]]) if mbs[k][j] else np.empty((T, 0))
            ok = ~np.isnan(P).any(1)
            fit = ok & train[k]
            if fit.sum() > P.shape[1] + 1:
                E[k, ok, j] = (X[ok, j] - ols_predict(P, X[:, j], fit, ok)) ** 2
    E[:, np.isnan(E).any(0)] = 0
    return E


def assign(cost, n_min, n_switch):
    r, T = cost.shape
    n, m = r * T, r * (T - 1)
    D = sparse.kron(sparse.eye(r), sparse.diags([-1.0, 1.0], [0, 1], shape=(T - 1, T)))
    rows = sparse.bmat([
        [sparse.kron(np.ones((1, r)), sparse.eye(T)), None],
        [D, -sparse.eye(m)], [-D, -sparse.eye(m)],
        [None, sparse.kron(sparse.eye(r), np.ones((1, T - 1)))],
        [sparse.kron(sparse.eye(r), np.ones((1, T))), None],
    ], format="csr")
    lo = np.r_[np.ones(T), np.full(2 * m, -np.inf), np.zeros(r), np.full(r, n_min)]
    hi = np.r_[np.ones(T), np.zeros(2 * m), np.full(r, n_switch), np.full(r, T)]
    res = milp(np.r_[cost.ravel(), np.zeros(m)], constraints=LinearConstraint(rows, lo, hi),
               integrality=np.r_[np.ones(n), np.zeros(m)], bounds=Bounds(0, 1), options={"time_limit": MILP_SECONDS})
    if res.x is None:
        raise ValueError("No feasible regime assignment")
    return res.x[:n].reshape(r, T).argmax(0)


def run_init(X, r, n_min, n_switch, seed, tau, alpha, sink=None):
    T, d = X.shape
    rng, Z = np.random.default_rng(seed), lagged(X, tau)
    slack = T - r * n_min
    sizes = np.diff(np.r_[0, np.sort(rng.integers(0, slack + 1, r - 1)), slack]) + n_min
    labels = np.repeat(rng.permutation(r), sizes)
    try:
        for it in range(N_ITER):
            train = [labels == k for k in range(r)]
            graphs = [cbnb(X, Z, sel, tau, alpha, sink) for sel in train]
            cost = sq_errors(X, Z, [markov_blankets(A) for A in graphs], train, tau).sum(-1)
            new = assign(cost, n_min, n_switch) if r > 1 else labels
            done, labels = bool((new == labels).all()), new
            if done:
                break
    except (ValueError, np.linalg.LinAlgError):  # 표본 부족·특이 행렬·해 없음: 이 초기화는 버린다
        return None
    order = pd.unique(labels)
    return dict(labels=np.argsort(order)[labels], graphs=[graphs[k] for k in order],
                loss=float(cost[labels, np.arange(T)].sum()), converged=done, iters=it + 1)


def fit(X, r, n_min, n_switch, tau, alpha, sink=None):
    runs = Parallel(n_jobs=-1)(delayed(run_init)(X, r, n_min, n_switch, s, tau, alpha, sink) for s in range(N_INIT if r > 1 else 1))
    runs = [x for x in runs if x is not None]
    if not runs:
        raise ValueError("Every RCBNB initialisation failed")
    best = min(runs, key=lambda x: x["loss"])
    return best | dict(n_ok=len(runs))
