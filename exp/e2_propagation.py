"""E2 -- layer-wise error-propagation diagnostics on the VGG-16 convolutional stack.

VGG-16's feature extractor is a plain 13-layer chain (conv -> ReLU -> optional
max-pool), the architecture Theorem 1 is stated for.  Every interleaved op is
1-Lipschitz, and the biases cancel in Z - Z_hat because compression keeps them.

Conv layer l is compressed as the matrix W (C_in k^2 x C_out) acting on the im2col
patch matrix U(A) (one row per output pixel), using the patch Gram matrix of the
calibration images.  The propagation quantities are measured on feature maps:
    delta_l   compression residual at the clean operating point
    gamma_l   ||T_hat_l||_op, the conv operator on feature maps (power iteration)
    kappa_l   ||T_hat_l(Delta)||_F / (gamma_l ||Delta||_F)
    kbar_l    ||U(Delta) Pi_l||_F / ||U(Delta)||_F, in patch space
    c_l       ||W_hat||_2 ||U(Delta)||_F / (gamma_l ||Delta||_F), so kappa_l <= c_l kbar_l
    theta_l   ||A_l - A_hat_l||_F / ||Z_l - Z_hat_l||_F   (ReLU and max-pool)
    kyfan_l   top-r eigenvalue fraction of U(Delta)^T U(Delta): the largest value
              kbar_l^2 can take over all rank-r subspaces (calibration only)
    err_rel   the propagated relative error eps_l
The network is compressed once from the calibration images; the same quantities
are then measured on the calibration images and on disjoint held-out sets.
Outputs results/e2.json
"""
import argparse, copy, json, os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lrc import COMPRESSORS, GramAccumulator, rank_for_budget, set_layer_matrix, layer_matrix
from data import calib_loader, heldout_loader

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "e2.json")

# full fp32 convolutions: on Ampere cuDNN would otherwise use TF32
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def conv_opnorm(conv, shape, iters=500, device="cuda", seed=0):
    """Spectral norm of the convolution as an operator on feature maps.

    Power iteration on T^T T.  Returns the estimate after `iters` steps and the
    estimate after 60 steps (the earlier setting), to document convergence.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    v = torch.randn(1, *shape, device=device, generator=g)
    v /= v.norm()
    s, s60 = torch.tensor(0.0), None
    for i in range(iters):
        u = F.conv2d(v, conv.weight, None, conv.stride, conv.padding, conv.dilation, conv.groups)
        v = F.conv_transpose2d(u, conv.weight, None, conv.stride, conv.padding,
                               0, conv.groups, conv.dilation)
        v = v[..., :shape[1], :shape[2]]
        s = v.norm()
        v = v / s
        if i == 59:
            s60 = float(s.sqrt())
    return float(s.sqrt()), s60


def conv_layers(model):
    return [m for m in model.features if isinstance(m, nn.Conv2d)]


@torch.no_grad()
def input_shapes(model, device, size=224):
    """(C, H, W) of the feature map entering each conv layer."""
    a = torch.zeros(1, 3, size, size, device=device)
    shapes = []
    for m in model.features:
        if isinstance(m, nn.Conv2d):
            shapes.append(tuple(a.shape[1:]))
        a = m(a)
    return shapes


@torch.no_grad()
def collect_grams(model, loader, device, max_batches):
    convs = conv_layers(model)
    accs = {i: GramAccumulator(c, device) for i, c in enumerate(convs)}
    for b, (x, _) in enumerate(loader):
        if b >= max_batches:
            break
        a = x.to(device)
        for m in model.features:
            if isinstance(m, nn.Conv2d):
                accs[convs.index(m)].add(a)
            a = m(a)
    return accs


@torch.no_grad()
def diagnostics(ref, cmp_, loader, device, max_batches, kyfan=False):
    convs_r, convs_c = conv_layers(ref), conv_layers(cmp_)
    L = len(convs_r)
    # index l = 0..L-1 refers to conv layer l+1;  "in2"/"D2" carry L+1 entries so
    # that entry L holds the norms of the stack output A_L (input of the head).
    acc = {k: torch.zeros(L, dtype=torch.float64) for k in
           ("delta2", "F2", "e2", "ref2", "uD2", "uPi2", "sup", "sup_clean", "numel")}
    acc["in2"] = torch.zeros(L + 1, dtype=torch.float64)
    acc["D2"] = torch.zeros(L + 1, dtype=torch.float64)
    shapes = [None] * L

    # Pi_l = A A^T with A the left singular vectors of W_hat (its column space)
    bases = []
    for mc in convs_c:
        Wh = layer_matrix(mc).double()
        U = torch.linalg.svd(Wh, full_matrices=False)[0]
        r = int(torch.linalg.matrix_rank(Wh, rtol=1e-8))
        bases.append(U[:, :max(r, 1)])
    G = [torch.zeros(A.shape[0], A.shape[0], dtype=torch.float64, device=device)
         for A in bases] if kyfan else None

    for b, (x, _) in enumerate(loader):
        if b >= max_batches:
            break
        a = ah = x.to(device)
        li = 0
        for mr, mc in zip(ref.features, cmp_.features):
            if isinstance(mr, nn.Conv2d):
                if shapes[li] is None:
                    shapes[li] = tuple(a.shape[1:])
                z_clean = mr(a)                      # C_l(A_{l-1})
                z_cmp_clean = mc(a)                  # C_hat_l(A_{l-1})
                z_cmp = mc(ah)                       # C_hat_l(A_hat_{l-1})
                d = a - ah                           # propagated input error
                cd = mc(d) - (mc.bias.view(1, -1, 1, 1) if mc.bias is not None else 0)

                acc["delta2"][li] += (z_clean - z_cmp_clean).double().pow(2).sum().cpu()
                acc["F2"][li] += cd.double().pow(2).sum().cpu()
                acc["D2"][li] += d.double().pow(2).sum().cpu()
                acc["e2"][li] += (z_clean - z_cmp).double().pow(2).sum().cpu()
                acc["ref2"][li] += z_clean.double().pow(2).sum().cpu()
                acc["in2"][li] += a.double().pow(2).sum().cpu()

                A_ = bases[li]
                ud = F.unfold(d, mc.kernel_size, mc.dilation, mc.padding, mc.stride)
                ud = ud.transpose(1, 2).reshape(-1, A_.shape[0]).double()
                acc["uD2"][li] += ud.pow(2).sum().cpu()
                acc["uPi2"][li] += (ud @ A_).pow(2).sum().cpu()
                if kyfan:
                    G[li] += ud.T @ ud
                # ReLU support: union of clean and compressed (the gates the error can
                # pass), and clean only (available before the compressed net is run)
                acc["sup"][li] += ((z_clean > 0) | (z_cmp > 0)).double().sum().cpu()
                acc["sup_clean"][li] += (z_clean > 0).double().sum().cpu()
                acc["numel"][li] += z_clean.numel()

                a, ah, li = z_clean, z_cmp, li + 1
            else:
                a, ah = mr(a), mr(ah)
        acc["in2"][L] += a.double().pow(2).sum().cpu()
        acc["D2"][L] += (a - ah).double().pow(2).sum().cpu()

    ky = None
    if kyfan:
        ky = []
        for Gl, A_ in zip(G, bases):
            ev = torch.linalg.eigvalsh(Gl).flip(0).clamp_min(0)
            tot = float(ev.sum())
            ky.append(float(ev[:A_.shape[1]].sum()) / tot if tot > 0 else None)
    return acc, shapes, ky


def make_rows(acc, gam, wh_norm, ranks, dims, ky=None):
    """Per-layer quantities and the three recursions, from accumulated statistics.

    Relative-error recursion on the post-activation chain A_0 -> ... -> A_L:
        eps_l <= theta_l delta_l/u_{l+1} + (theta_l gamma_l kappa_l u_l/u_{l+1}) eps_{l-1}
    with u_l = ||A_{l-1}||_F (K = 1: ReLU and max-pool are 1-Lipschitz).
    """
    L = len(ranks)
    u = [float(acc["in2"][l].sqrt()) for l in range(L + 1)]
    rows = []
    e_nai = e_fil = e_prj = e_cpr = 0.0
    for l in range(L):
        D = float(acc["D2"][l].sqrt())
        uD = float(acc["uD2"][l].sqrt())
        delta = float(acc["delta2"][l].sqrt())
        kap = float(acc["F2"][l].sqrt()) / (gam[l] * D) if D > 0 else 0.0
        kbar = min(float((acc["uPi2"][l] / acc["uD2"][l]).sqrt()), 1.0) \
            if acc["uD2"][l] > 0 else 0.0
        c = wh_norm[l] * uD / (gam[l] * D) if D > 0 else None
        # activation gating: theta_l = ||A_l - A_hat_l||_F / ||Z_l - Z_hat_l||_F
        th = float((acc["D2"][l + 1] / acc["e2"][l]).sqrt()) if acc["e2"][l] > 0 else 1.0
        g = u[l] / u[l + 1]                       # inverse signal gain 1/g_l
        a_nai = gam[l] * g                        # Lipschitz amplification
        a_fil = th * gam[l] * kap * g             # Theorem 1
        a_prj = th * gam[l] * kbar * g            # kappa replaced by the patch kbar (no c_l)
        a_cpr = th * gam[l] * (c or 0.0) * kbar * g   # kappa replaced by c_l kbar (a bound)
        dt = th * delta / u[l + 1]
        e_nai = delta / u[l + 1] + a_nai * e_nai
        e_fil, e_prj, e_cpr = dt + a_fil * e_fil, dt + a_prj * e_prj, dt + a_cpr * e_cpr
        rows.append(dict(
            layer=l + 1, rank=ranks[l], gamma=gam[l], kappa=kap, kappa_bar=kbar,
            c_conv=c, w_hat_norm=wh_norm[l], theta=th,
            relu_support=float(acc["sup"][l] / acc["numel"][l]),
            relu_support_clean=float(acc["sup_clean"][l] / acc["numel"][l]),
            rd_ratio=ranks[l] / dims[l], gain=u[l + 1] / u[l],
            kyfan=None if ky is None else ky[l],
            alpha_naive=a_nai, alpha_filtered=a_fil, alpha_proj=a_prj, alpha_cproj=a_cpr,
            delta_rel=dt, delta_plain=delta / u[l + 1],
            err_rel=float(acc["D2"][l + 1].sqrt()) / u[l + 1],
            preact_err_rel=float(acc["e2"][l].sqrt()) / float(acc["ref2"][l].sqrt()),
            bound_naive=e_nai, bound_filtered=e_fil, bound_proj=e_prj, bound_cproj=e_cpr,
        ))
    return rows


def run(ref, grams, rho, methods, nb, n_held, device, calib_offset=0):
    out = {}
    for meth in methods:
        cmp_ = copy.deepcopy(ref)
        ranks, dims, params_full, params_lr = [], [], 0, 0
        for i, c in enumerate(conv_layers(cmp_)):
            W = layer_matrix(c)
            d, p = W.shape
            r = rank_for_budget(d, p, rho)
            set_layer_matrix(c, COMPRESSORS[meth](W, grams[i].C, r))
            ranks.append(r)
            dims.append(d)
            params_full += d * p
            params_lr += r * (d + p)
        wh_norm = [torch.linalg.matrix_norm(layer_matrix(c).double(), 2).item()
                   for c in conv_layers(cmp_)]
        cal = calib_loader(n_per_class=1, batch_size=8, offset=calib_offset)
        acc, shapes, ky = diagnostics(ref, cmp_, cal, device, nb, kyfan=True)
        gam, gam60 = zip(*[conv_opnorm(c, shapes[i], device=device)
                           for i, c in enumerate(conv_layers(cmp_))])
        rows = make_rows(acc, gam, wh_norm, ranks, dims, ky)
        for row, g60 in zip(rows, gam60):
            row["gamma_60it"] = g60
        held = []
        for k in range(n_held):
            hl = heldout_loader(nb * 8, offset=k * nb * 8, batch_size=8)
            acc_k, _, _ = diagnostics(ref, cmp_, hl, device, nb)
            held.append(make_rows(acc_k, gam, wh_norm, ranks, dims))
        out[meth] = dict(rows=rows, heldout=held, param_ratio=params_lr / params_full)
        L = len(rows)
        held_eps = ", ".join("%.4f" % h[-1]["err_rel"] for h in held)
        print(f"  [{meth}] eps_L={rows[-1]['err_rel']:.4f} (held-out {held_eps}) | bounds: "
              f"filtered={rows[-1]['bound_filtered']:.3f} c-proj={rows[-1]['bound_cproj']:.3f} "
              f"naive={rows[-1]['bound_naive']:.3e} | mean alpha (l>=2): "
              f"fil={sum(r['alpha_filtered'] for r in rows[1:])/(L-1):.3f} "
              f"nai={sum(r['alpha_naive'] for r in rows[1:])/(L-1):.3f} | "
              f"max gamma change 60->500 it: "
              f"{max(abs(r['gamma'] / r['gamma_60it'] - 1) for r in rows):.2e}", flush=True)
        del cmp_
        torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vgg16")
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.5, 0.3, 0.15])
    ap.add_argument("--batches", type=int, default=16, help="calibration batches of 8 images")
    ap.add_argument("--heldout-sets", type=int, default=4,
                    help="disjoint held-out sets, each as large as the calibration set")
    ap.add_argument("--calib-offset", type=int, default=0,
                    help="skip this many calibration images (a disjoint calibration set)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out:
        OUT = a.out if os.path.isabs(a.out) else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), a.out)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[E2] {a.model} on {dev}, {a.batches} calib batches of 8, "
          f"{a.heldout_sets} held-out sets of {a.batches * 8}")
    ref = getattr(tvm, a.model)(weights="IMAGENET1K_V1").to(dev).eval()
    print("  accumulating Gram matrices ...", flush=True)
    grams = collect_grams(ref, calib_loader(n_per_class=1, batch_size=8, offset=a.calib_offset),
                          dev, a.batches)
    shapes0 = input_shapes(ref, dev)
    res = {"model": a.model, "n_calib": a.batches * 8, "n_heldout": a.batches * 8,
           "calib_offset": a.calib_offset,
           "gamma_orig": [conv_opnorm(c, shapes0[i], device=dev)[0]
                          for i, c in enumerate(conv_layers(ref))],
           "w_norm": [torch.linalg.matrix_norm(layer_matrix(c).double(), 2).item()
                      for c in conv_layers(ref)],
           "by_rho": {}}
    for rho in a.rhos:
        print(f"rho = {rho}", flush=True)
        res["by_rho"][str(rho)] = run(ref, grams, rho, list(COMPRESSORS), a.batches,
                                      a.heldout_sets, dev, a.calib_offset)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", OUT)
