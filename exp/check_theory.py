"""Numerical check of Lemma 1, Proposition 2 and the argmin claim of Eq. (2) on
random instances, with rank-deficient X and damping from 1e-8 to 10 times the
mean diagonal of C.  Needs numpy only:  python exp/check_theory.py"""
import numpy as np

rng = np.random.default_rng(0)


def trunc(A, r):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return (U[:, :r] * s[:r]) @ Vt[:r]


worst = dict(lemma=0.0, argmin=0.0, prop2a=0.0)
viol_damp = viol_gamma = 0
comp_ratio = []
for trial in range(300):
    d, p = (int(x) for x in rng.integers(20, 60, size=2))
    r = int(rng.integers(1, min(d, p)))
    k = int(rng.integers(1, d + 1))                            # rank of X, often < d
    n = int(rng.integers(k, 3 * d))
    X = rng.standard_normal((n, k)) @ (rng.standard_normal((k, d)) * rng.lognormal(0, 1.5, d))
    W = rng.standard_normal((d, p)) * rng.lognormal(0, 1, p)
    C = X.T @ X
    tau = 10 ** rng.uniform(-8, 1) * np.trace(C) / d
    B = np.linalg.cholesky(C + tau * np.eye(d)).T              # B^T B = C + tau I
    Wh = np.linalg.solve(B, trunc(B @ W, r))
    Q = np.linalg.svd(B @ W, full_matrices=False)[2][:r].T
    worst["lemma"] = max(worst["lemma"], np.abs(Wh - W @ Q @ Q.T).max() / np.abs(W).max())
    # Prop. 2(b): 0 <= s_i(XW)^2 - s_i(X W_tau)^2 <= tau ||W||_2^2
    s0 = np.linalg.svd(X @ W, compute_uv=False)[:r]
    gap = s0 ** 2 - np.linalg.svd(X @ Wh, compute_uv=False)[:r] ** 2
    tol = 1e-8 * s0[0] ** 2
    viol_damp += int((gap < -tol).any() or (gap > tau * np.linalg.norm(W, 2) ** 2 + tol).any())
    # Lemma 1: no target-W estimator raises ||W||_2 (one-shot, sequential, diagonal, plain)
    Xh = X + 0.3 * X.std() * rng.standard_normal(X.shape)
    Bs = np.linalg.cholesky(Xh.T @ Xh + tau * np.eye(d)).T
    S = np.sqrt(np.diag(C)) + 1e-8
    cands = [Wh, np.linalg.solve(Bs, trunc(Bs @ W, r)), trunc(S[:, None] * W, r) / S[:, None], trunc(W, r)]
    viol_gamma += int(any(np.linalg.norm(c, 2) > np.linalg.norm(W, 2) * (1 + 1e-9) for c in cands))
    # compensating calibration, argmin ||X W - Xh Z||, is not of that form
    Bc = np.linalg.cholesky(Xh.T @ Xh + tau * np.eye(d)).T
    Zc = np.linalg.solve(Bc, trunc(np.linalg.solve(Bc.T, Xh.T @ (X @ W)), r))
    comp_ratio.append(np.linalg.norm(Zc, 2) / np.linalg.norm(W, 2))
    # Eq. (2) and Prop. 2(a) with the compact factor B = Sigma V^T, any rank of X
    U, sx, Vt = np.linalg.svd(X, full_matrices=False)
    m = int((sx > 1e-10 * sx[0]).sum())
    U, sx, V = U[:, :m], sx[:m], Vt[:m].T
    W2 = (V / sx) @ trunc(sx[:, None] * V.T @ W, r)
    q = min(r, m)
    worst["prop2a"] = max(worst["prop2a"], np.abs(np.linalg.svd(X @ W2, compute_uv=False)[:q] - s0[:q]).max() / s0[0])
    worst["argmin"] = max(worst["argmin"], abs(np.linalg.norm(X @ W - X @ W2)
                                               - np.linalg.norm(X @ W - trunc(X @ W, r))) / np.linalg.norm(X @ W))

print(f"Lemma 1, W_hat = W Q_r Q_r^T, max rel. error : {worst['lemma']:.1e}")
print(f"Lemma 1, trials with ||W_hat|| > ||W||       : {viol_gamma} / 300")
print(f"  compensating ||Z||/||W||: median {np.median(comp_ratio):.2f}, max {max(comp_ratio):.2f}")
print(f"Prop. 2(a), any rank of X, max rel. error     : {worst['prop2a']:.1e}")
print(f"Prop. 2(b), trials violating Eq. (6)          : {viol_damp} / 300")
print(f"Eq. (2) argmin claim, max rel. error          : {worst['argmin']:.1e}")
