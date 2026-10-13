"""E5 -- alignment of the propagated error with the retained subspace, measured on
the linear layers of transformer language models.  These are plain matrix layers,
so Proposition 1 applies without the im2col constant of the convolutional case.

Every linear map of every block is compressed one-shot with the covariance-weighted
estimator at budget rho, exactly as in E4.  The original and the compressed model
are then run side by side on held-out text (the WikiText-2 test split).  For each
compressed map W_hat (d x p, rank r) we accumulate the Gram matrix of
Delta = X - X_hat, the difference between the inputs the map receives in the two
models, and record
    kbar2   ||Delta Pi||_F^2 / ||Delta||_F^2, Pi the projector onto col(W_hat)
    rd      r / d, the expected value of kbar2 for a Haar-random rank-r Pi
    kyfan   top-r eigenvalue fraction of Delta^T Delta, the largest kbar2 any
            rank-r subspace can capture
    kappa   ||Delta W_hat||_F / (||W_hat||_2 ||Delta||_F)
Maps whose input is unchanged (q/k/v of the first block) have Delta = 0 and are
skipped.  Outputs results/e5.json
"""
import argparse, copy, gc, json, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lrc import layer_matrix
from e4_llm import (load, wikitext, blocks, block_linears, capture_block_inputs,
                    _advance, compress_model)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "e5.json")


@torch.no_grad()
def measure(ref, cmp_, chunks, device):
    cache_r = capture_block_inputs(ref, chunks, device)
    cache_c = capture_block_inputs(cmp_, chunks, device)
    rows = []
    for b, (bref, bcmp) in enumerate(zip(blocks(ref), blocks(cmp_))):
        lin_r, lin_c = dict(block_linears(bref)), dict(block_linears(bcmp))
        cur_r, cur_c, hooks = {}, {}, []
        for n in lin_c:
            hooks.append(lin_r[n].register_forward_pre_hook(
                lambda m, i, n=n: cur_r.__setitem__(n, i[0])))
            hooks.append(lin_c[n].register_forward_pre_hook(
                lambda m, i, n=n: cur_c.__setitem__(n, i[0])))
        G = {n: torch.zeros(m.in_features, m.in_features, dtype=torch.float64,
                            device=device) for n, m in lin_c.items()}
        X2 = dict.fromkeys(lin_c, 0.0)
        for i in range(len(cache_r)):
            a, k = cache_r[i]
            bref(*a, **k)
            a, k = cache_c[i]
            bcmp(*a, **k)
            for n, m in lin_c.items():
                Xr = cur_r[n].reshape(-1, m.in_features).double()
                D = Xr - cur_c[n].reshape(-1, m.in_features).double()
                G[n] += D.T @ D
                X2[n] += float(Xr.pow(2).sum())
        for h in hooks:
            h.remove()

        for n, m in lin_c.items():
            Wh = layer_matrix(m).double()                      # d x p
            if torch.equal(Wh, layer_matrix(lin_r[n]).double()):
                continue                                       # left uncompressed
            tot = float(torch.trace(G[n]))
            if tot <= 0:
                continue                                       # Delta = 0
            U, S, _ = torch.linalg.svd(Wh, full_matrices=False)
            r = int((S > 1e-8 * S[0]).sum())
            A = U[:, :r]
            ev = torch.linalg.eigvalsh(G[n]).flip(0).clamp_min(0)
            rows.append(dict(
                block=b, name=n, d=m.in_features, p=m.out_features, rank=r,
                rd_ratio=r / m.in_features,
                kbar2=float((A * (G[n] @ A)).sum()) / tot,
                kappa=(float(torch.trace(Wh.T @ G[n] @ Wh)) / tot) ** 0.5 / float(S[0]),
                kyfan=float(ev[:r].sum()) / float(ev.sum()),
                delta_rel=(tot / X2[n]) ** 0.5))
        del G, cur_r, cur_c
        gc.collect()
        torch.cuda.empty_cache()
        _advance(bref, cache_r)
        _advance(bcmp, cache_c)
    return rows


def main(a):
    global OUT
    if a.out:
        OUT = a.out if os.path.isabs(a.out) else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), a.out)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {"seqlen": a.seqlen, "n_calib_seq": a.calib_seq, "n_eval_seq": a.eval_seq,
           "models": {}}
    for mid in a.models:
        print(f"\n=== {mid}", flush=True)
        ref, tok = load(mid, dev)
        train_ids = tok(wikitext("train"), return_tensors="pt").input_ids
        test_ids = tok(wikitext("test"), return_tensors="pt").input_ids
        g = torch.Generator().manual_seed(0)                   # same calibration as E4
        starts = torch.randint(0, train_ids.numel() - a.seqlen - 1, (a.calib_seq,), generator=g)
        calib = [train_ids[:, s:s + a.seqlen] for s in starts.tolist()]
        held = [test_ids[:, i * a.seqlen:(i + 1) * a.seqlen] for i in range(a.eval_seq)]
        res["models"][mid] = {}
        for rho in a.rhos:
            t0 = time.time()
            cmp_ = copy.deepcopy(ref)
            compress_model(ref, cmp_, rho, "whitened", "one-shot", calib, dev, a.ridge)
            rows = measure(ref, cmp_, held, dev)
            res["models"][mid][str(rho)] = rows
            mu = sorted(r["kbar2"] / r["rd_ratio"] for r in rows)
            print(f"  rho={rho}: {len(rows)} maps, median mu={mu[len(mu) // 2]:.2f} "
                  f"[{mu[0]:.2f}, {mu[-1]:.2f}]  ({time.time() - t0:.0f}s)", flush=True)
            del cmp_
            gc.collect()
            torch.cuda.empty_cache()
            os.makedirs(os.path.dirname(OUT), exist_ok=True)
            json.dump(res, open(OUT, "w"), indent=1)
        del ref
        gc.collect()
        torch.cuda.empty_cache()
    print("wrote", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-0.5B",
                                                    "meta-llama/Llama-3.2-1B-Instruct"])
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.9, 0.8, 0.6])
    ap.add_argument("--calib-seq", type=int, default=32)
    ap.add_argument("--eval-seq", type=int, default=16)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--ridge", type=float, default=1e-6)
    ap.add_argument("--out", default=None)
    main(ap.parse_args())
