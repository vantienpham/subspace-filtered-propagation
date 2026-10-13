"""Re-check the quantitative claims of the manuscript against results/*.json.

Run from the directory that contains numbers.tex and exp/:
    python exp/verify_claims.py
Every check prints PASS or FAIL with the numbers involved.
"""
import json, os, re, statistics as st

ROOT = os.getcwd()
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
n = dict(re.findall(r'\\newcommand\{\\n(\w+)\}\{(.*)\}', open(os.path.join(ROOT, "numbers.tex")).read()))
J = lambda name: json.load(open(os.path.join(RES, name)))
e2, e2b, e3, e4, e5 = J("e2.json"), J("e2_calib2.json"), J("e3.json"), J("e4.json"), J("e5.json")
ok = []


def chk(name, cond, detail):
    ok.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'}  {name}: {detail}")


def mu(r):
    return r["kappa_bar"] ** 2 / r["rd_ratio"]


rhos = sorted(e2["by_rho"], key=float, reverse=True)
W = {rho: e2["by_rho"][rho]["whitened"] for rho in rhos}
r = W["0.5"]["rows"]

# ---- headline numbers (abstract, introduction, Sec. IV-A)
chk("eps_L", abs(r[-1]["err_rel"] - float(n["EpsL"])) < 5e-3, f"{r[-1]['err_rel']:.4f} vs {n['EpsL']}")
chk("filtered bound", abs(r[-1]["bound_filtered"] - float(n["BoundFilt"])) < 5e-3,
    f"{r[-1]['bound_filtered']:.3f} vs {n['BoundFilt']}")
chk("tightness factor", abs(r[-1]["bound_filtered"] / r[-1]["err_rel"] - float(n["TightFactor"])) < 0.05,
    n["TightFactor"])
chk("Lipschitz bound exceeds the error by > 1e11", r[-1]["bound_naive"] / r[-1]["err_rel"] > 1e11,
    f"{r[-1]['bound_naive']:.2e}")
chk("alpha > 1 count (l >= 2)", sum(x["alpha_filtered"] > 1 for x in r[1:]) == int(n["NOver"]),
    f"{n['NOver']} of {n['NDef']}")

# ---- Sec. III: the Lipschitz version of alpha_l < 1 never holds
naive_min = min(x["alpha_naive"] for rho in rhos for m in e2["by_rho"][rho].values() for x in m["rows"])
chk("gamma_l / g_l >= NaiveMin > 1 at every layer", naive_min >= float(n["NaiveMin"]) - 0.05 and naive_min > 1,
    f"min {naive_min:.2f}")
chk("signal does not vanish", abs(__import__('math').prod(x["gain"] for x in r) - float(n["GainProd"])) < 5e-3,
    f"prod g_l = {n['GainProd']}")

# ---- Lemma 1: target-W estimators never raise the operator norm
chk("Lemma 1, VGG im2col: ||W_hat||_2 <= ||W||_2",
    all(x["w_hat_norm"] <= wn * (1 + 1e-6) for rho in rhos for m in e2["by_rho"][rho].values()
        for x, wn in zip(m["rows"], e2["w_norm"])), "all layers, budgets, compressors (float32 tolerance)")
chk("Lemma 1, VGG feature maps: gamma_l <= ||T_l||",
    all(x["gamma"] <= g0 * (1 + 1e-3) for rho in rhos for m in e2["by_rho"][rho].values()
        for x, g0 in zip(m["rows"], e2["gamma_orig"])), "power-iteration tolerance 1e-3")
allr = [x for d in (e3, e4) for v in d["models"].values() for x in v["runs"]]
chk("Lemma 1, ResNets and LLMs: gamma_bar <= 1 unless compensating",
    all(x["gain_max"] <= 1 + 1e-4 for x in allr if x["scheme"] != "compensated"),
    f"max {max(x['gain_max'] for x in allr if x['scheme'] != 'compensated'):.6f}")

# ---- Sec. IV-A, convolutions: kappa_l <= c_l kbar_l everywhere
viol = [(rho, m, x["layer"]) for rho in rhos for m, v in e2["by_rho"][rho].items()
        for rows in [v["rows"]] + v["heldout"] for x in rows[1:]
        if x["kappa"] > x["c_conv"] * x["kappa_bar"] * (1 + 1e-9)]
chk("kappa_l <= c_l kbar_l (calibration and held-out sets)", not viol, f"violations: {viol[:5]}")
cs = [x["c_conv"] for rho in rhos for m in e2["by_rho"][rho].values() for x in m["rows"][1:]]
chk("c_l range", f"{min(cs):.2f}" == n["CLo"] and f"{max(cs):.2f}" == n["CHi"], f"[{n['CLo']}, {n['CHi']}]")

# ---- held-out transfer and a second calibration set
ratios = [R["rows"][-1]["bound_filtered"] / h[-1]["err_rel"]
          for rho in rhos for R in e2["by_rho"][rho].values() for h in R["heldout"]]
chk("calibration bound exceeds every held-out error", min(ratios) > 1,
    f"factor {min(ratios):.2f} to {max(ratios):.2f}")
devs = [abs(h[i]["alpha_filtered"] / W[rho]["rows"][i]["alpha_filtered"] - 1)
        for rho in rhos for h in W[rho]["heldout"] for i in range(1, len(r))]
chk("alpha_l stable on held-out sets", st.median(devs) < 0.02, f"median change {100 * st.median(devs):.2f}%")
dv = [abs(xb["alpha_filtered"] / xa["alpha_filtered"] - 1) for rho in rhos
      for xa, xb in zip(W[rho]["rows"][1:], e2b["by_rho"][rho]["whitened"]["rows"][1:])]
chk("alpha_l stable across calibration sets", st.median(dv) < 0.02, f"median change {100 * st.median(dv):.2f}%")

# ---- general position
meds = [st.median(mu(x) for x in W[rho]["rows"][1:]) for rho in rhos]
chk("median mu grows with compression", meds[0] < meds[1] < meds[2], " < ".join(f"{m:.1f}" for m in meds))
chk("contribution (iii): 'MuLoAll to MuHiAll times'", f"{min(meds):.0f}" == n["MuLoAll"] and f"{max(meds):.0f}" == n["MuHiAll"],
    f"{n['MuLoAll']} to {n['MuHiAll']}")
kb = [st.mean(x["kappa_bar"] for x in W[rho]["rows"][1:]) for rho in rhos]
rd = [st.mean(x["rd_ratio"] ** .5 for x in W[rho]["rows"][1:]) for rho in rhos]
chk("kbar flat while sqrt(r/N) falls", abs(kb[-1] / kb[0] - 1) < 0.1 and rd[-1] < 0.6 * rd[0],
    f"kbar {kb[0]:.2f}->{kb[-1]:.2f}, sqrt(r/N) {rd[0]:.2f}->{rd[-1]:.2f}")
am = [st.mean(x["alpha_filtered"] for x in W[rho]["rows"][1:]) for rho in rhos]
chk("mean alpha does not decrease with compression", min(am[1:]) >= am[0] - 5e-3, ", ".join(f"{a:.3f}" for a in am))
llm_meds = [st.median(x["kbar2"] / x["rd_ratio"] for x in rows) for m in e5["models"].values() for rows in m.values()]
chk("transformer linear layers much closer to general position", max(llm_meds) < min(meds) / 2,
    f"LLM medians {min(llm_meds):.2f}-{max(llm_meds):.2f} vs VGG >= {min(meds):.1f}")

# ---- Sec. IV-B, ResNets
for tag, m in [("RA", "resnet18"), ("RB", "resnet34"), ("RC", "resnet50")]:
    R = e3["models"][m]["runs"]
    rr = sorted({x["rho"] for x in R}, reverse=True)
    pick = lambda rho, s: next(x for x in R if x["rho"] == rho and x["scheme"] == s and x["method"] == "whitened")["top1"]
    win = [rho for rho in rr if pick(rho, "compensated") > pick(rho, "one-shot")]
    chk(f"{m} crossover", f"{max(win):g}" == n["Xover" + tag], f"largest winning rho {max(win):g}")
    chk(f"{m} sequential never wins", all(pick(rho, "sequential") <= pick(rho, "one-shot") for rho in rr), "")
chk("damping changes compensated ResNet-18 by < 1 point", float(n["RidgeSpread"]) < 1, f"{n['RidgeSpread']} points")

# ---- Sec. IV-C, language models
for tag, m in zip(["LA", "LB"], e4["models"]):
    R = e4["models"][m]["runs"]
    rr = sorted({x["rho"] for x in R})
    g = lambda rho, s: next(x for x in R if x["rho"] == rho and x["scheme"] == s and x["method"] == "whitened")["ppl"]
    short = m.split("/")[-1]
    chk(f"{short}: sequential beats one-shot at every budget", all(g(rho, "sequential") < g(rho, "one-shot") for rho in rr),
        ", ".join(f"{g(rho, 'one-shot'):.1f}->{g(rho, 'sequential'):.1f}" for rho in rr))
    chk(f"{short}: compensation never wins", all(g(rho, "compensated") > g(rho, "one-shot") for rho in rr), "")
chk("LLM inflation exceeds the deepest ResNet's", min(float(n["GamMaxLA"]), float(n["GamMaxLB"])) > float(n["GamMaxRC"]),
    f"{n['GamMaxLA']}, {n['GamMaxLB']} vs {n['GamMaxRC']}")

print(f"\n{sum(ok)}/{len(ok)} checks passed")
