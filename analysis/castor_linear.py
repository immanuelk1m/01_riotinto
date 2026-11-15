"""CASTOR (선형판) — Rahmani & Frossard, AISTATS 2025. 레짐 수·구간·레짐별 시차 DAG 를 EM 으로 함께 추정한다.

원본: https://github.com/arahmani1/CASTOR (MIT, 커밋 b425ebc) 의 `CASTOR.run_linear`, `dynotears.py`(causalnex DYNOTEARS).
지금 환경(numpy 2, torch 없음)으로 옮기며 바꾼 것:
- 레짐 비율 π(t) = softmax(시간의 1차식) 을 torch `nn.Linear` + Adam 대신 같은 모형인 다항 로지스틱 회귀(sklearn)로 맞춘다.
- 원본은 마지막 반복에서 작은 레짐을 지운 뒤 그래프를 다시 맞추지 않아 레짐 번호와 그래프가 어긋난다. 최종 레짐으로 한 번 더 맞춘다.
- DYNOTEARS 는 W(같은 주)·A(시차) 행렬만 돌려준다(StructureModel 변환은 뺐다).
"""
import warnings

import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.stats import multivariate_normal
from sklearn.linear_model import LogisticRegression


def lag_matrix(X, p):
    """X (T, d) → (X_t, [X_{t-1} | … | X_{t-p}]) 의 행 t = p..T-1."""
    return X[p:], np.hstack([X[p - k:len(X) - k] for k in range(1, p + 1)])


def dynotears(X, Xlags, lambda_w=0.05, lambda_a=0.05, max_iter=100, h_tol=1e-8, w_threshold=0.0):
    """DYNOTEARS (Pamfil et al. 2020). 반환: W (d, d) 같은 주 i→j, A (p·d, d) 시차 (k-1)·d+i → j."""
    n, d = X.shape
    p = Xlags.shape[1] // d
    bnds = 2 * [(0, 0) if i == j else (0, None) for i in range(d) for j in range(d)] + 2 * p * d * d * [(0, None)]

    def unpack(v):
        w = v.reshape([2 * (p + 1) * d, d])
        a = w[2 * d:].reshape(2 * p, d * d)
        return w[:d] - w[d:2 * d], (a[::2] - a[1::2]).reshape(d * p, d)

    def h(v):
        W, _ = unpack(v)
        return np.trace(slin.expm(W * W)) - d

    def func(v):
        W, A = unpack(v)
        hv = h(v)
        return (0.5 / n * np.square(np.linalg.norm(X @ (np.eye(d) - W) - Xlags @ A, "fro"))
                + 0.5 * rho * hv * hv + alpha * hv + lambda_w * v[:2 * d * d].sum() + lambda_a * v[2 * d * d:].sum())

    def grad(v):
        W, A = unpack(v)
        E = slin.expm(W * W)
        R = X @ (np.eye(d) - W) - Xlags @ A
        gw = -1.0 / n * (X.T @ R) + (rho * (np.trace(E) - d) + alpha) * E.T * W * 2
        ga = (-1.0 / n * (Xlags.T @ R)).reshape(p, d * d)
        return np.r_[np.append(gw, -gw, axis=0).ravel() + lambda_w, np.hstack((ga, -ga)).ravel() + lambda_a]

    est = np.zeros(2 * (p + 1) * d * d)
    rho, alpha, h_value, h_new = 1.0, 0.0, np.inf, np.inf
    for it in range(max_iter):
        while rho < 1e20 and (h_new > 0.25 * h_value or h_new == np.inf):
            new = sopt.minimize(func, est, method="L-BFGS-B", jac=grad, bounds=bnds).x
            h_new = h(new)
            if h_new > 0.25 * h_value:
                rho *= 10
        est, h_value = new, h_new
        alpha += rho * h_value
        if h_value <= h_tol:
            break
        if it == max_iter - 1:
            warnings.warn("DYNOTEARS did not converge")
    W, A = unpack(est)
    W[np.abs(W) < w_threshold] = 0
    A[np.abs(A) < w_threshold] = 0
    return W, A


def castor(X, p, window, zeta, max_it=5, w_threshold=0.4, lambda_w=0.05, lambda_a=0.05):
    """X (T, d) 표준화된 패널 → dict(labels (T-p,), graphs [(W, A)], iterations). 행 t 는 원래 행 t + p.

    초기 레짐 = 길이 `window` 의 연속 구간들. 반복: (M) 레짐별 DYNOTEARS(그 레짐 행만), (E) 잔차의 표준 정규 우도 × π(t) 로 행을 다시 배정,
    π(t) 재적합, 행이 `zeta` 개보다 적은 레짐은 지운다.
    """
    Y, L = lag_matrix(np.asarray(X, float), p)
    m, d = Y.shape
    r = max(m // window, 1)
    labels = np.minimum(np.arange(m) // window, r - 1)
    prior = np.eye(r)[labels]
    t = np.linspace(0, 1, m).reshape(-1, 1)
    fit = lambda rows: dynotears(Y[rows], L[rows], lambda_w, lambda_a, w_threshold=w_threshold)  # noqa: E731
    for it in range(max_it):
        graphs = [fit(labels == k) for k in range(r)]
        lik = np.column_stack([multivariate_normal.pdf(Y - Y @ W - L @ A, mean=np.zeros(d), cov=np.eye(d)) for W, A in graphs])
        new = np.argmax(prior * lik, axis=1)
        keep = [k for k in range(r) if (new == k).sum() >= zeta]
        if not keep:  # 어떤 레짐도 zeta 행을 넘지 못하면 하나로 합친다
            keep, new = [0], np.zeros(m, int)
        remap = {k: i for i, k in enumerate(keep)}
        if len(keep) < r:  # 지운 레짐의 행은 남은 레짐 중 우도가 가장 큰 쪽으로
            new = np.array([k if k in remap else keep[int(np.argmax((prior * lik)[i, keep]))] for i, k in enumerate(new)])
        new = np.array([remap[k] for k in new])
        done = it > 0 and len(keep) == r and (new == labels).all()  # 첫 반복의 π 는 초기 구간 그대로라 배정이 바뀔 수 없다
        labels, r = new, len(keep)
        prior = (LogisticRegression(max_iter=1000).fit(t, labels).predict_proba(t) if r > 1 else np.ones((m, 1)))
        if done:
            break
    return dict(labels=labels, graphs=[fit(labels == k) for k in range(r)], iterations=it + 1)


def demo():
    """두 구간의 그래프가 다른 선형 VAR 에서 레짐과 시차 링크를 복원하는지 본다."""
    rng = np.random.default_rng(0)
    T, d = 600, 4
    X = np.zeros((T, d))
    for t in range(1, T):
        e = rng.normal(scale=0.5, size=d)
        X[t] = e + (np.array([0.0, 0.9 * X[t - 1, 0], 0.0, 0.0]) if t < T // 2 else np.array([0.0, 0.0, 0.0, 0.9 * X[t - 1, 2]]))
        X[t, [0, 2]] += 0.3 * X[t - 1, [0, 2]]
    X = (X - X.mean(0)) / X.std(0)
    out = castor(X, p=1, window=150, zeta=100)
    truth = (np.arange(T - 1) + 1 >= T // 2).astype(int)
    err = min((out["labels"] != truth).mean(), (out["labels"] != 1 - truth).mean())
    return err, out


if __name__ == "__main__":
    err, out = demo()
    print("regimes", len(out["graphs"]), "mis-assignment", round(err, 3))
    for k, (W, A) in enumerate(out["graphs"]):
        print(k, "lag-1 links", [(i, j) for i, j in zip(*np.nonzero(A[:4]))])
    assert len(out["graphs"]) == 2 and err < 0.1
