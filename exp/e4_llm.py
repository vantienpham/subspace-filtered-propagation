"""E4 -- WikiText-2 perplexity after post-training low-rank compression of an LLM.

Transformer blocks are pure matrix layers, so the propagation analysis applies
verbatim.  Calibration is done block-by-block: for `one-shot` the Gram matrices
come from the *original* model's activations, for `sequential` from the model
whose earlier blocks are already compressed.  Outputs results/e4.json
"""
import argparse, copy, gc, glob, json, os, sys, time
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lrc import COMPRESSORS, lr_normal_eq, rank_for_budget, layer_matrix, set_layer_matrix

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "e4.json")
HUB = os.environ.get("HF_HUB_CACHE") or os.path.join(
    os.environ.get("HF_HOME", "~/.cache/huggingface"), "hub")
WIKI = os.path.join(os.path.expanduser(HUB),
                    "datasets--Salesforce--wikitext/snapshots/*/wikitext-2-raw-v1")


def wikitext(split):
    import pandas as pd
    f = sorted(glob.glob(os.path.join(WIKI, f"{split}-*.parquet")))
    assert f, f"no parquet for split {split}"
    return "\n\n".join(pd.read_parquet(f[0])["text"].tolist())


def load(model_id, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    try:
        m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    return m.to(device).eval(), tok


def blocks(model):
    for attr in ("model.layers", "model.decoder.layers", "transformer.h", "gpt_neox.layers"):
        o = model
        try:
            for a in attr.split("."):
                o = getattr(o, a)
            return list(o)
        except AttributeError:
            continue
    raise RuntimeError("could not locate transformer blocks")


def block_linears(block):
    return [(n, m) for n, m in block.named_modules() if isinstance(m, nn.Linear)]


@torch.no_grad()
def ppl(model, ids, device, seqlen=2048):
    n = ids.numel() // seqlen
    nll, ntok = 0.0, 0
    for i in range(n):
        x = ids[:, i * seqlen:(i + 1) * seqlen].to(device)
        out = model(x, labels=x)
        nll += out.loss.float().item() * (seqlen - 1)
        ntok += seqlen - 1
    return float(torch.exp(torch.tensor(nll / ntok)))


class _Stop(Exception):
    pass


@torch.no_grad()
def capture_block_inputs(model, chunks, device):
    """Run the model once per chunk and capture the (args, kwargs) entering block 0.

    Blocks can then be executed one at a time on the cached stream, so calibrating
    all L blocks costs one model pass rather than L.

    The KV cache the model creates is dropped from the kwargs.  Each block is run
    more than once on the same stream (calibration, then advance); with the cache
    kept, every call appends to it and the causal attention of later calls reads
    the keys and values of the first call, i.e. of the weights before compression.
    """
    cache = []

    def hook(mod, args, kwargs):
        kwargs = {k: v for k, v in kwargs.items()
                  if k not in ("past_key_values", "past_key_value")}
        kwargs["use_cache"] = False
        cache.append((args, kwargs))
        raise _Stop
    h = blocks(model)[0].register_forward_pre_hook(hook, with_kwargs=True)
    for c in chunks:
        try:
            model(c.to(device))
        except _Stop:
            pass
    h.remove()
    return cache


def _advance(block, cache):
    """Run `block` over the cached stream and replace the hidden states in place."""
    for i, (args, kwargs) in enumerate(cache):
        out = block(*args, **kwargs)
        h = out[0] if isinstance(out, tuple) else out
        cache[i] = ((h,) + tuple(args[1:]), kwargs)


@torch.no_grad()
def compress_model(ref, cmp_, rho, meth, scheme, chunks, device, ridge=1e-6):
    """Block-by-block compression; returns (param_ratio, mean/max spectral gain)."""
    comp = scheme == "compensated"
    one = scheme == "one-shot"
    cache_r = capture_block_inputs(ref, chunks, device) if (one or comp) else None
    cache_c = capture_block_inputs(cmp_, chunks, device) if not one else None

    full = lr = 0
    gains = []
    for bref, bcmp in zip(blocks(ref), blocks(cmp_)):
        src_b, src_cache = (bref, cache_r) if one else (bcmp, cache_c)
        lin_s, lin_t = block_linears(src_b), block_linears(bref)
        C = {n: torch.zeros(m.in_features, m.in_features, dtype=torch.float64,
                            device=device) for n, m in lin_s}
        R = {n: torch.zeros(m.in_features, m.out_features, dtype=torch.float64,
                            device=device) for n, m in lin_t} if comp else {}
        cur, hooks = {}, []
        for n, m in lin_s:
            hooks.append(m.register_forward_pre_hook(
                lambda mod, inp, n=n: cur.__setitem__(n, inp[0])))
        if comp:
            hooks.append(None)
            hooks[-1:] = [m.register_forward_hook(
                lambda mod, inp, out, n=n: cur.__setitem__("y_" + n, out))
                for n, m in lin_t]

        for i in range(len(src_cache)):
            if comp:
                a, k = cache_r[i]
                bref(*a, **k)
                ys = {n: cur["y_" + n].reshape(-1, cur["y_" + n].shape[-1]).double()
                      - (m.bias.double() if m.bias is not None else 0)
                      for n, m in lin_t}
            a, k = src_cache[i]
            src_b(*a, **k)
            for n, m in lin_s:
                X = cur[n].reshape(-1, m.in_features).double()
                C[n] += X.T @ X
                if comp:
                    R[n] += X.T @ ys[n]
        for h in hooks:
            h.remove()

        for n, m in block_linears(bcmp):
            W = layer_matrix(m)
            d, p = W.shape
            r = rank_for_budget(d, p, rho)
            full += d * p
            if r * (d + p) >= d * p:
                lr += d * p
                continue
            lr += r * (d + p)
            Wh = lr_normal_eq(R[n], C[n], r, ridge) if comp else COMPRESSORS[meth](W, C[n], r)
            gains.append((torch.linalg.matrix_norm(Wh.double(), 2)
                          / torch.linalg.matrix_norm(W.double(), 2)).item())
            set_layer_matrix(m, Wh)
        del C, R, cur
        gc.collect()
        torch.cuda.empty_cache()

        if cache_r is not None:
            _advance(bref, cache_r)
        if cache_c is not None:
            _advance(bcmp, cache_c)
    return lr / full, sum(gains) / len(gains), max(gains)


def main(a):
    global OUT
    if a.out:
        OUT = a.out if os.path.isabs(a.out) else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), a.out)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {"seqlen": a.seqlen, "n_calib_seq": a.calib_seq, "models": {}}
    for mid in a.models:
        print(f"\n=== {mid}", flush=True)
        ref, tok = load(mid, dev)
        test_ids = tok(wikitext("test"), return_tensors="pt").input_ids
        train_ids = tok(wikitext("train"), return_tensors="pt").input_ids
        g = torch.Generator().manual_seed(0)
        starts = torch.randint(0, train_ids.numel() - a.seqlen - 1, (a.calib_seq,), generator=g)
        chunks = [train_ids[:, s:s + a.seqlen] for s in starts.tolist()]

        base = ppl(ref, test_ids, dev, a.seqlen)
        print(f"  dense ppl = {base:.3f}", flush=True)
        entry = {"baseline_ppl": base, "runs": []}
        for rho in a.rhos:
            for scheme in a.schemes:
                meths = ["whitened"] if scheme == "compensated" else a.methods
                for meth in meths:
                    t0 = time.time()
                    cmp_ = copy.deepcopy(ref)
                    pr, gm, gx = compress_model(ref, cmp_, rho, meth, scheme,
                                                chunks, dev, a.ridge)
                    p = ppl(cmp_, test_ids, dev, a.seqlen)
                    del cmp_
                    gc.collect()
                    torch.cuda.empty_cache()
                    entry["runs"].append(dict(rho=rho, param_ratio=pr, scheme=scheme,
                                              method=meth, ppl=p,
                                              gain_mean=gm, gain_max=gx))
                    print(f"  rho={rho} lin-params={pr:.3f} {scheme:<11} {meth:<9} "
                          f"ppl={p:10.3f}  gamma/gamma0: mean={gm:.2f} max={gx:.2f}"
                          f"  ({time.time()-t0:.0f}s)", flush=True)
        res["models"][mid] = entry
        del ref
        gc.collect()
        torch.cuda.empty_cache()
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-0.5B",
                                                    "meta-llama/Llama-3.2-1B-Instruct"])
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.8, 0.6])
    ap.add_argument("--methods", nargs="+", default=["agnostic", "diag", "whitened"])
    ap.add_argument("--schemes", nargs="+",
                    default=["one-shot", "sequential", "compensated"])
    ap.add_argument("--ridge", type=float, default=1e-6)
    ap.add_argument("--out", default=None,
                    help="override the results path (use for side sweeps so they "
                         "cannot clobber the main run)")
    ap.add_argument("--calib-seq", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=2048)
    main(ap.parse_args())
