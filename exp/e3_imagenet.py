"""E3 -- end-to-end ImageNet top-1 after post-training low-rank compression.

Compares data-agnostic SVD, diagonal (ASVD-style) weighting and the
covariance-weighted (whitened) estimator, each under two calibration schemes:

  one-shot    every layer is calibrated on the *original* network's activations
  sequential  layer l is calibrated on the activations of the network whose
              layers 1..l-1 are already compressed (the operating point the
              propagation theorem identifies as the relevant one)

No fine-tuning and no BatchNorm re-estimation.  Outputs results/e3.json
"""
import argparse, copy, json, os, sys, time
import torch
import torch.nn as nn
import torchvision.models as tvm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lrc import (COMPRESSORS, GramAccumulator, lr_normal_eq, rank_for_budget,
                 layer_matrix, set_layer_matrix, unfold_input)
from data import calib_loader, eval_loader, top1

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "e3.json")


def targets(model):
    """(name, module) for every compressible Conv2d / Linear, in execution order."""
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, nn.Linear) or (isinstance(m, nn.Conv2d) and m.groups == 1)]


def plan(model, rho):
    """Per-layer rank, skipping layers where factorisation would not save."""
    out = {}
    for n, m in targets(model):
        d, p = m.weight[0].numel(), m.weight.shape[0]
        r = rank_for_budget(d, p, rho)
        out[n] = r if r * (d + p) < d * p else None
    return out


def param_ratio(model, ranks):
    tot = sum(q.numel() for q in model.parameters())
    saved = 0
    for n, m in targets(model):
        r = ranks.get(n)
        if r is None:
            continue
        d, p = m.weight[0].numel(), m.weight.shape[0]
        saved += d * p - r * (d + p)
    return (tot - saved) / tot


def _mat_in(mod, x):
    """(n x d) input view: im2col patches for Conv2d, flattened tokens for Linear."""
    return unfold_input(x, mod) if isinstance(mod, nn.Conv2d) else \
        x.reshape(-1, mod.in_features)


def _mat_out(mod, y):
    """(n x p) output view, bias removed, rows aligned with _mat_in."""
    if isinstance(mod, nn.Conv2d):
        y = y.flatten(2).transpose(1, 2).reshape(-1, y.shape[1])
    else:
        y = y.reshape(-1, y.shape[-1])
    return y - mod.bias if mod.bias is not None else y


@torch.no_grad()
def calib_stats(src, tgt, name, mod_s, mod_t, loader, device, nb, compensate):
    """Accumulate C = X_s^T X_s and, if `compensate`, R = X_s^T Y_t.

    `src` supplies the input the compressed layer will actually see; `tgt`
    supplies the clean target output.  Both are run on the same batches.
    """
    d = mod_t.weight[0].numel()
    C = torch.zeros(d, d, dtype=torch.float64, device=device)
    R = torch.zeros(d, mod_t.weight.shape[0], dtype=torch.float64, device=device) \
        if compensate else None
    store = {}
    hs = [mod_s.register_forward_pre_hook(lambda m, i: store.__setitem__("x", i[0]))]
    if compensate:
        hs.append(mod_t.register_forward_hook(lambda m, i, o: store.__setitem__("y", o)))
    for b, (x, _) in enumerate(loader):
        if b >= nb:
            break
        x = x.to(device)
        if compensate:
            tgt(x)
            Y = _mat_out(mod_t, store["y"]).double()
        src(x)
        X = _mat_in(mod_s, store["x"]).double()
        C += X.T @ X
        if compensate:
            R += X.T @ Y
    for h in hs:
        h.remove()
    return C, R


def compress(ref, cmp_, ranks, meth, loader, device, nb, scheme, ridge=1e-6):
    """scheme in {one-shot, sequential, compensated}; meth in COMPRESSORS."""
    pairs = [(n, dict(targets(ref))[n], m) for n, m in targets(cmp_) if ranks.get(n)]
    gains = []
    for n, m_ref, m in pairs:
        src, mod_s = (ref, m_ref) if scheme == "one-shot" else (cmp_, m)
        C, R = calib_stats(src, ref, n, mod_s, m_ref, loader, device, nb,
                           compensate=(scheme == "compensated"))
        W = layer_matrix(m)
        Wh = lr_normal_eq(R, C, ranks[n], ridge) if scheme == "compensated" \
            else COMPRESSORS[meth](W, C, ranks[n])
        gains.append((torch.linalg.matrix_norm(Wh.double(), 2)
                      / torch.linalg.matrix_norm(W.double(), 2)).item())
        set_layer_matrix(m, Wh)
    return cmp_, gains


@torch.no_grad()
def compress_oneshot_fast(ref, cmp_, ranks, meth, loader, device, nb):
    """One-shot needs only the original activations -> a single forward pass."""
    mods = [(n, m) for n, m in targets(ref) if ranks.get(n)]
    accs, hooks = {}, []
    for n, m in mods:
        accs[n] = GramAccumulator(m, device)
        hooks.append(m.register_forward_pre_hook(lambda mod, inp, n=n: accs[n].add(inp[0])))
    for b, (x, _) in enumerate(loader):
        if b >= nb:
            break
        ref(x.to(device))
    for h in hooks:
        h.remove()
    gains = []
    for n, m in targets(cmp_):
        if ranks.get(n):
            W = layer_matrix(m)
            Wh = COMPRESSORS[meth](W, accs[n].C, ranks[n])
            gains.append((torch.linalg.matrix_norm(Wh.double(), 2)
                          / torch.linalg.matrix_norm(W.double(), 2)).item())
            set_layer_matrix(m, Wh)
    return cmp_, gains


def main(a):
    global OUT
    if a.out:
        OUT = a.out if os.path.isabs(a.out) else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), a.out)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cal = calib_loader(n_per_class=1, batch_size=a.calib_bs)
    ev = eval_loader(n_per_class=a.eval_per_class, batch_size=a.eval_bs)
    nb = a.calib // a.calib_bs
    res = {"n_calib": a.calib, "n_eval": a.eval_per_class * 1000, "models": {}}

    for mname in a.models:
        ref = getattr(tvm, mname)(weights="IMAGENET1K_V1").to(dev).eval()
        base = top1(ref, ev, dev)
        print(f"\n=== {mname}: uncompressed top-1 = {base:.2f}", flush=True)
        entry = {"baseline_top1": base, "runs": []}
        for rho in a.rhos:
            ranks = plan(ref, rho)
            pr = param_ratio(ref, ranks)
            for scheme in a.schemes:
                meths = ["whitened"] if scheme == "compensated" else a.methods
                for meth in meths:
                    t0 = time.time()
                    cmp_ = copy.deepcopy(ref)
                    m, gains = (compress_oneshot_fast(ref, cmp_, ranks, meth, cal, dev, nb)
                                if scheme == "one-shot" else
                                compress(ref, cmp_, ranks, meth, cal, dev, nb,
                                         scheme, a.ridge))
                    acc = top1(m, ev, dev)
                    gmean = sum(gains) / len(gains)
                    del m
                    torch.cuda.empty_cache()
                    entry["runs"].append(dict(rho=rho, param_ratio=pr, scheme=scheme,
                                              method=meth, top1=acc,
                                              gain_mean=gmean, gain_max=max(gains)))
                    print(f"  rho={rho} params={pr:.3f} {scheme:<11} {meth:<9} "
                          f"top-1={acc:6.2f}  gamma/gamma0: mean={gmean:.2f} "
                          f"max={max(gains):.2f}  ({time.time()-t0:.0f}s)", flush=True)
        res["models"][mname] = entry
        del ref
        torch.cuda.empty_cache()
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet18", "resnet34", "resnet50"])
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.6, 0.4])
    ap.add_argument("--methods", nargs="+", default=["agnostic", "diag", "whitened"])
    ap.add_argument("--schemes", nargs="+",
                    default=["one-shot", "sequential", "compensated"])
    ap.add_argument("--calib", type=int, default=256)
    ap.add_argument("--calib-bs", type=int, default=16)
    ap.add_argument("--eval-per-class", type=int, default=10)
    ap.add_argument("--eval-bs", type=int, default=100)
    ap.add_argument("--ridge", type=float, default=1e-6)
    ap.add_argument("--out", default=None,
                    help="override the results path (use for side sweeps so they "
                         "cannot clobber the main run)")
    main(ap.parse_args())
