"""Core utilities for data-driven low-rank compression experiments.

Conventions
-----------
A layer maps X (n x d) -> X W with W (d x p).
Gram matrix  C = X^T X  (unnormalised, d x d, PSD).
All compressors return a rank-r matrix W_hat of the same shape as W.
"""
import math
import torch


# --------------------------------------------------------------------------
# rank-r compressors
# --------------------------------------------------------------------------
def _truncate(A, r):
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    return (U[:, :r] * S[:r]) @ Vh[:r]


def lr_agnostic(W, C, r, **kw):
    """Data-agnostic truncated SVD of W (ignores C)."""
    return _truncate(W.double(), r).to(W.dtype)


def lr_diag(W, C, r, alpha=1.0, ridge=1e-8, **kw):
    """Diagonal (activation-magnitude) weighting, ASVD-style.

    s_j = C_jj^{alpha/2}; truncate diag(s) W and undo the scaling.
    alpha = 1 reproduces column-norm weighting, alpha = 0.5 the ASVD default.
    """
    Wd = W.double()
    s = torch.diagonal(C.double()).clamp_min(0).sqrt() ** alpha
    s = s + ridge * s.max()
    return (_truncate(s.unsqueeze(1) * Wd, r) / s.unsqueeze(1)).to(W.dtype)


def whitening_factor(C, ridge=1e-6):
    """Upper-triangular B with B^T B = C + tau I, tau ridge-scaled until PD.

    Cholesky rather than an eigendecomposition: for the widest layers (e.g. the
    8192-dimensional down-projection of Llama-3.2-1B) this is the difference
    between seconds and minutes per layer.
    """
    Cd = C.double()
    Cd = 0.5 * (Cd + Cd.T)
    tau = ridge * torch.diagonal(Cd).mean().clamp_min(1e-30)
    eye = torch.eye(Cd.shape[0], dtype=Cd.dtype, device=Cd.device)
    for _ in range(10):
        B, info = torch.linalg.cholesky_ex(Cd + tau * eye, upper=True)
        if int(info) == 0:
            return B
        tau = tau * 10
    raise RuntimeError("Cholesky failed: calibration Gram is not positive definite")


def lr_whitened(W, C, r, ridge=1e-6, **kw):
    """Covariance-weighted (whitened) truncated SVD.

    W_hat = B^{-1} [B W]_r with B^T B = C + tau I, i.e. the minimiser of
    ||X W - X Z||_F over rank(Z) <= r.
    """
    B = whitening_factor(C, ridge)
    T = _truncate(B @ W.double(), r)
    return torch.linalg.solve_triangular(B, T, upper=True).to(W.dtype)


def lr_normal_eq(R, C, r, ridge=1e-6):
    """Rank-r argmin_Z ||Y - X Z||_F from the normal-equation statistics
    C = X^T X and R = X^T Y.  Explicitly  Z = Q L^{-1/2} [ L^{-1/2} Q^T R ]_r
    with C = Q L Q^T.

    All three calibration schemes are this estimator with different (C, R):
      one-shot     C = X^T X,          R = C W          (X = original input)
      sequential   C = Xh^T Xh,        R = C W          (Xh = compressed input)
      compensated  C = Xh^T Xh,        R = Xh^T (X W)   (target = clean output)
    """
    B = whitening_factor(C, ridge)
    Z = torch.linalg.solve_triangular(B.T, R.double(), upper=False)   # B^{-T} R
    T = _truncate(Z, r)
    return torch.linalg.solve_triangular(B, T, upper=True).to(R.dtype)


COMPRESSORS = {"agnostic": lr_agnostic, "diag": lr_diag, "whitened": lr_whitened}


# --------------------------------------------------------------------------
# diagnostics used to validate Theorem 2
# --------------------------------------------------------------------------
def gamma_of(W_hat):
    """gamma_l = ||W_hat||_2."""
    return torch.linalg.matrix_norm(W_hat.double(), 2).item()


def kappa_of(C_delta, W_hat, tol=1e-10):
    """Subspace-filtering coefficient kappa = ||Delta P||_F / ||Delta||_F.

    P projects onto col(W_hat) (the r-dimensional input subspace the compressed
    layer actually reads).  Uses only the Gram matrix C_delta = Delta^T Delta,
    since ||Delta P||_F^2 = tr(P C_delta P) = tr(C_delta P).
    """
    Cd = C_delta.double()
    denom = torch.diagonal(Cd).sum()
    if denom <= tol:
        return float("nan")
    U, S, _ = torch.linalg.svd(W_hat.double(), full_matrices=False)
    k = int((S > tol * S.max()).sum())
    A = U[:, :k]                                   # orthonormal basis of col(W_hat)
    num = (A * (Cd @ A)).sum()
    return math.sqrt(max(num.item(), 0.0) / denom.item())


def cond_gram(C, tol=0.0):
    """Condition number of the Gram matrix restricted to its numerical range."""
    lam = torch.linalg.eigvalsh(0.5 * (C.double() + C.double().T)).clamp_min(0)
    hi = lam.max()
    lo = lam[lam > tol * hi].min() if tol > 0 else lam.min().clamp_min(1e-30)
    return (hi / lo).item()


# --------------------------------------------------------------------------
# layer <-> matrix views
# --------------------------------------------------------------------------
def unfold_input(x, conv):
    """Conv2d input -> (n x d) patch matrix matching W = weight.reshape(out,-1).T."""
    return torch.nn.functional.unfold(
        x, conv.kernel_size, dilation=conv.dilation,
        padding=conv.padding, stride=conv.stride
    ).transpose(1, 2).reshape(-1, conv.weight[0].numel())


def layer_matrix(mod):
    """W (d x p) view of a Conv2d / Linear weight."""
    return mod.weight.detach().reshape(mod.weight.shape[0], -1).T


def set_layer_matrix(mod, W):
    with torch.no_grad():
        mod.weight.copy_(W.T.reshape(mod.weight.shape).to(mod.weight.dtype))


def rank_for_budget(d, p, rho):
    """Rank giving a fraction rho of the layer's d*p parameters."""
    r = int(round(rho * d * p / (d + p)))
    return max(1, min(r, min(d, p)))


class GramAccumulator:
    """Streams  C = sum_batches X^T X  for one layer, in float64."""

    def __init__(self, mod, device):
        self.mod = mod
        self.d = mod.weight[0].numel()
        self.C = torch.zeros(self.d, self.d, dtype=torch.float64, device=device)
        self.n = 0

    def add(self, x):
        X = unfold_input(x, self.mod) if x.dim() == 4 else x.reshape(-1, self.d)
        X = X.double()
        self.C += X.T @ X
        self.n += X.shape[0]
