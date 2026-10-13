"""Build the manuscript figure, the LaTeX tables and the in-text numbers from
results/*.json.

Writes fig_propagation.pdf, tab_e2.tex, tab_e3.tex, tab_e4.tex and numbers.tex
to the directory above exp/ (override with OUT_DIR; read results from RESULTS_DIR).
No number in the manuscript is typed by hand: prose numbers are the macros of
numbers.tex.
"""
import json, math, os, statistics as st, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("RESULTS_DIR", os.path.join(HERE, "results"))
DOC = os.environ.get("OUT_DIR", os.path.abspath(os.path.join(HERE, "..")))

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42,       # embed TrueType, not Type 3
    "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 2.8,
    "legend.frameon": False, "figure.dpi": 300,
})

PRETTY = {"resnet18": "ResNet-18", "resnet34": "ResNet-34", "resnet50": "ResNet-50",
          "vgg16": "VGG-16", "Qwen2.5-0.5B": "Qwen2.5-0.5B",
          "Llama-3.2-1B-Instruct": "Llama-3.2-1B-Instruct"}


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def write(name, lines):
    p = os.path.join(DOC, name)
    open(p, "w").write("\n".join(lines) + "\n")
    print("wrote", p)


def _fmt_sci(x):
    s = f"{x:.1e}".split("e")
    return f"${s[0]}\\!\\cdot\\!10^{{{int(s[1])}}}$"


# ------------------------------------------------------- E2 derived quantities
def mu(r):
    """Alignment mu_l = kbar_l^2 / (r_l/N_{l-1}); 1 in general position."""
    return r["kappa_bar"] ** 2 / r["rd_ratio"]


def align_index(r):
    """Position of kbar^2 between general position (0) and the Ky Fan maximum (1)."""
    return (r["kappa_bar"] ** 2 - r["rd_ratio"]) / (r["kyfan"] - r["rd_ratio"])


def unroll(alphas, deltas):
    e = 0.0
    for a, dt in zip(alphas, deltas):
        e = dt + a * e
    return e


def apriori_bound(rows):
    """Bound with kappa_l -> sqrt(r_l/N_{l-1}) and theta_l -> sqrt(s_l), s_l the ReLU
    support of the uncompressed network; gamma_l, g_l and delta_l need only the
    compressed weights and clean activations.  Returns (alphas, final bound)."""
    al = [0.0] + [r["relu_support_clean"] ** .5 * r["gamma"] * r["rd_ratio"] ** .5 / r["gain"]
                  for r in rows[1:]]
    dt = [r["relu_support_clean"] ** .5 * r["delta_plain"] for r in rows]
    return al, unroll(al, dt)


def mean_l2(rows, key):
    return st.mean(r[key] for r in rows[1:])


# ---------------------------------------------------------------- figure
def figure(rho="0.5"):
    d = load("e2.json")
    R = d["by_rho"][rho]["whitened"]
    rows, held = R["rows"], R["heldout"]
    L = [r["layer"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.85))

    a = ax[0]
    r2, L2 = rows[1:], L[1:]                 # Delta = 0 at layer 1
    a.plot(L2, [r["c_conv"] * r["kappa_bar"] for r in r2], "s-", color="#c44e52",
           label=r"$c_l\bar\kappa_l$")
    a.plot(L2, [r["kappa_bar"] for r in r2], "s:", color="#c44e52", mfc="white",
           label=r"$\bar\kappa_l$")
    a.plot(L2, [r["kappa"] for r in r2], "o-", color="#4c72b0", label=r"$\kappa_l$")
    a.plot(L2, [r["rd_ratio"] ** .5 for r in r2], "--", color="#8172b2",
           label=r"$\sqrt{r_l/N_{l-1}}$")
    a.plot(L, [r["theta"] for r in rows], "^-", color="#55a868", label=r"$\theta_l$")
    a.axhline(1.0, color="k", lw=0.5, ls="--")
    a.set_ylim(0, 1.5)
    a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    a.set_xlabel("layer $l$")
    a.set_ylabel("transmission")
    a.set_title("(a) transmission coefficients")
    a.legend(loc="upper center", ncol=3, handlelength=1.5, borderpad=0.05,
             labelspacing=0.1, handletextpad=0.3, columnspacing=0.7, fontsize=5.8)

    b = ax[1]
    b.semilogy(L, [r["bound_naive"] for r in rows], "s--", color="#c44e52",
               label="Lipschitz, Eq. (1)")
    b.semilogy(L, [r["bound_filtered"] for r in rows], "o-", color="#4c72b0",
               label="filtered, Eq. (4)")
    b.semilogy(L, [r["err_rel"] for r in rows], "^-", color="#55a868", label="measured")
    b.semilogy(L, [st.mean(h[i]["err_rel"] for h in held) for i in range(len(L))],
               "v:", color="#2d6b3c", mfc="white", label="held-out")
    b.axhline(1.0, color="k", lw=0.5, ls="--")
    b.set_xlabel("layer $l$")
    b.set_ylabel("relative error $\\varepsilon_l$")
    b.set_title("(b) bound vs. measurement")
    b.legend(loc="upper left", handlelength=1.6, borderpad=0.1,
             labelspacing=0.15, handletextpad=0.4)

    for x in ax:
        x.grid(True, ls=":", lw=0.4, alpha=0.6)
        x.tick_params(length=2, width=0.5)
    fig.tight_layout(pad=0.25, w_pad=1.0)
    out = os.path.join(DOC, "fig_propagation.pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    print("wrote", out)


# ---------------------------------------------------------------- tables
def tab_e2():
    d = load("e2.json")
    rhos = sorted(d["by_rho"], key=float, reverse=True)
    L = [r"\begin{table}[!t]", r"\centering", r"\footnotesize",
         r"\setlength{\tabcolsep}{2.6pt}",
         r"\caption{VGG-16 ($L=13$). Median $\mu_l$, mean $\alpha_l$ ($l\ge2$) and bounds "
         r"on $\varepsilon_L$ from the 128 calibration images; $\varepsilon_L$ measured on "
         r"them and on four held-out sets of 128 images (largest of the four).}",
         r"\label{tab:e2}",
         r"\begin{tabular}{cl cc cc cc}", r"\toprule",
         r"& & & & \multicolumn{2}{c}{bound on $\varepsilon_L$} & "
         r"\multicolumn{2}{c}{measured $\varepsilon_L$} \\",
         r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
         r"$\rho$ & compr. & $\mu_l$ & $\alpha_l$ & Eq.~\eqref{eq:naive} & "
         r"Eq.~\eqref{eq:unrolled} & calib. & held-out \\", r"\midrule"]
    for k, rho in enumerate(rhos):
        for j, meth in enumerate(["agnostic", "whitened"]):
            R = d["by_rho"][rho][meth]
            rr = R["rows"]
            head = f"\\multirow{{2}}{{*}}{{{float(rho):.2f}}}" if j == 0 else ""
            short = {"agnostic": "agn.", "whitened": "cov.-wt."}[meth]
            L.append(f"{head} & {short} & {st.median(mu(r) for r in rr[1:]):.1f} & "
                     f"{mean_l2(rr, 'alpha_filtered'):.2f} & "
                     f"{_fmt_sci(rr[-1]['bound_naive'])} & {rr[-1]['bound_filtered']:.2f} & "
                     f"{rr[-1]['err_rel']:.2f} & "
                     f"{max(h[-1]['err_rel'] for h in R['heldout']):.2f} \\\\")
        if k < len(rhos) - 1:
            L.append(r"\addlinespace[1pt]")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_e2.tex", L)


def _perf_table(d, key, fmt, caption, label, base_key, base_fmt, unit_note):
    models = list(d["models"])
    rhos = sorted({r["rho"] for r in d["models"][models[0]]["runs"]}, reverse=True)

    def get(m, rho, sch, meth):
        for r in d["models"][m]["runs"]:
            if r["rho"] == rho and r["scheme"] == sch and r["method"] == meth:
                return r
        return None

    L = [r"\begin{table}[!t]", r"\centering", r"\scriptsize",
         r"\setlength{\tabcolsep}{3pt}",
         r"\caption{" + caption + " " + unit_note + "}",
         r"\label{" + label + "}",
         r"\begin{tabular}{lc ccc cc c}", r"\toprule",
         r"& & \multicolumn{3}{c}{one-shot calibration} & "
         r"\multicolumn{2}{c}{covariance-weighted} & \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
         r"model & $\rho$ & agn. & diag. & cov.-wt. & seq. & comp. & "
         r"$\bar\gamma$ \\", r"\midrule"]
    for k, m in enumerate(models):
        short = m.split("/")[-1]
        short = PRETTY.get(short, short)
        base = base_fmt.format(d["models"][m][base_key])
        for i2, rho in enumerate(rhos):
            head = (f"\\multirow{{{len(rhos)}}}{{*}}{{\\shortstack[l]{{{short}\\\\({base})}}}}"
                    if i2 == 0 else "")
            cells = []
            for sch, meth in [("one-shot", "agnostic"), ("one-shot", "diag"),
                              ("one-shot", "whitened"), ("sequential", "whitened"),
                              ("compensated", "whitened")]:
                r = get(m, rho, sch, meth)
                cells.append("--" if r is None else fmt(r[key]))
            rc = get(m, rho, "compensated", "whitened")
            g = "--" if rc is None else f"{rc['gain_mean']:.2f}"
            L.append(f"{head} & {rho:g} & " + " & ".join(cells) + f" & {g} \\\\")
        if k < len(models) - 1:
            L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return L


def _acc(x):
    return f"{x:.1f}"


def _ppl(x):
    return f"{x:.1f}" if x < 1e4 else _fmt_sci(x)


def tab_e3():
    d = load("e3.json")
    write("tab_e3.tex", _perf_table(
        d, "top1", _acc,
        r"ImageNet top-1 (\%) without fine-tuning; dense top-1 in parentheses.",
        "tab:e3", "baseline_top1", "{:.1f}",
        r"$\bar\gamma$: mean of $\|\widehat W^{(l)}\|_2/\|W^{(l)}\|_2$ under "
        r"compensating calibration ($\le1$ otherwise, Lemma~\ref{lem:proj})."))


def tab_e4():
    d = load("e4.json")
    write("tab_e4.tex", _perf_table(
        d, "ppl", _ppl,
        r"WikiText-2 perplexity, all linear layers compressed; dense in parentheses.",
        "tab:e4", "baseline_ppl", "{:.1f}",
        r"$\bar\gamma$ as in Table~\ref{tab:e3}."))


# ------------------------------------------------------- in-text numbers
def numbers(rho_fig="0.5"):
    """Emit \\newcommand macros so no number in the prose is typed by hand."""
    out = {}
    d2 = load("e2.json")
    rhos = sorted(d2["by_rho"], key=float, reverse=True)           # 0.5, 0.3, 0.15
    W = {rho: d2["by_rho"][rho]["whitened"] for rho in rhos}
    allr = W[rho_fig]["rows"]
    rows = allr[1:]                                                # layer 1 has Delta = 0
    n = len(allr)
    for name, k in [("Kappa", "kappa"), ("Kbar", "kappa_bar"), ("Theta", "theta")]:
        out[name + "Lo"] = f"{min(r[k] for r in rows):.2f}"
        out[name + "Hi"] = f"{max(r[k] for r in rows):.2f}"
    cs = [r["c_conv"] for rho in rhos for m in d2["by_rho"][rho].values() for r in m["rows"][1:]]
    out["CLo"], out["CHi"] = f"{min(cs):.2f}", f"{max(cs):.2f}"
    out["AlphaNaive"] = f"{mean_l2(allr, 'alpha_naive'):.1f}"
    out["AlphaFilt"] = f"{mean_l2(allr, 'alpha_filtered'):.2f}"
    out["BoundNaive"] = _fmt_sci(allr[-1]["bound_naive"]).strip("$")
    out["BoundFilt"] = f"{allr[-1]['bound_filtered']:.2f}"
    out["EpsL"] = f"{allr[-1]['err_rel']:.2f}"
    out["TightFactor"] = f"{allr[-1]['bound_filtered'] / allr[-1]['err_rel']:.1f}"
    out["NLayers"] = str(n)
    out["NDef"] = str(n - 1)
    over = [x["layer"] for x in rows if x["alpha_filtered"] > 1.0]
    out["NOver"] = str(len(over))
    out["NUnder"] = str(len(rows) - len(over))
    out["AlphaMax"] = f"{max(x['alpha_filtered'] for x in rows):.2f}"
    out["RhoFig"] = f"{float(rho_fig):g}"
    out["RhoMild"], out["RhoHard"] = f"{float(rhos[0]):g}", f"{float(rhos[-1]):g}"
    out["NaiveMin"] = f"{min(r['alpha_naive'] for rho in rhos for m in d2['by_rho'][rho].values() for r in m['rows']):.1f}"
    out["GainProd"] = f"{math.prod(r['gain'] for r in allr):.2f}"
    out["NCalib"] = str(d2["n_calib"])
    out["GamIterDev"] = f"{100 * max(abs(r['gamma'] / r['gamma_60it'] - 1) for rho in rhos for m in d2['by_rho'][rho].values() for r in m['rows']):.1f}"

    # alignment (Remark 2): per-budget medians and trends, covariance-weighted
    tags = dict(zip(rhos, ["A", "B", "C"]))
    meds = []
    for rho in rhos:
        rr = W[rho]["rows"][1:]
        t = tags[rho]
        m_ = st.median(mu(r) for r in rr)
        meds.append(m_)
        out["MuMed" + t] = f"{m_:.1f}"
        out["AIdx" + t] = f"{st.median(align_index(r) for r in rr):.2f}"
        out["KbarMean" + t] = f"{st.mean(r['kappa_bar'] for r in rr):.2f}"
        out["KappaMean" + t] = f"{st.mean(r['kappa'] for r in rr):.2f}"
        out["RdMean" + t] = f"{st.mean(r['rd_ratio'] ** .5 for r in rr):.2f}"
        out["ThetaMean" + t] = f"{st.mean(r['theta'] for r in rr):.2f}"
        out["AlphaMean" + t] = f"{st.mean(r['alpha_filtered'] for r in rr):.2f}"
        out["Rho" + t] = f"{float(rho):g}"
    out["MuLoAll"], out["MuHiAll"] = f"{min(meds):.0f}", f"{max(meds):.0f}"
    allmu = [mu(r) for rho in rhos for r in W[rho]["rows"][1:]]
    out["MuMin"], out["MuMax"] = f"{min(allmu):.1f}", f"{max(allmu):.1f}"

    # held-out transfer: coefficients and bound from calibration vs held-out sets
    dev = {"kappa": [], "theta": [], "alpha_filtered": []}
    ratio, dmean = [], []
    for rho in rhos:
        for meth in ("agnostic", "diag", "whitened"):
            R = d2["by_rho"][rho][meth]
            for h in R["heldout"]:
                ratio.append(R["rows"][-1]["bound_filtered"] / h[-1]["err_rel"])
                if meth != "whitened":
                    continue
                for rc, rh in zip(R["rows"][1:], h[1:]):
                    for k in dev:
                        dev[k].append(abs(rh[k] / rc[k] - 1))
                dmean.append(abs(mean_l2(h, "alpha_filtered") / mean_l2(R["rows"], "alpha_filtered") - 1))
    out["DevKappa"] = f"{100 * st.median(dev['kappa']):.1f}"
    out["DevTheta"] = f"{100 * st.median(dev['theta']):.1f}"
    out["DevAlpha"] = f"{100 * st.median(dev['alpha_filtered']):.1f}"
    out["DevAlphaMax"] = f"{100 * max(dev['alpha_filtered']):.1f}"
    out["DevAlphaMean"] = f"{100 * max(dmean):.1f}"
    out["TransferMin"] = f"{min(ratio):.1f}"
    out["TransferMax"] = f"{max(ratio):.1f}"
    H = W[rho_fig]["heldout"]
    out["EpsHeldLo"] = f"{min(h[-1]['err_rel'] for h in H):.2f}"
    out["EpsHeldHi"] = f"{max(h[-1]['err_rel'] for h in H):.2f}"

    # second, disjoint calibration set
    d2b = load("e2_calib2.json")
    if d2b:
        dv, eps2 = [], []
        for rho in rhos:
            ra, rb = W[rho]["rows"], d2b["by_rho"][rho]["whitened"]["rows"]
            for xa, xb in zip(ra[1:], rb[1:]):
                dv.append(abs(xb["alpha_filtered"] / xa["alpha_filtered"] - 1))
            eps2.append(abs(st.mean(h[-1]["err_rel"] for h in d2b["by_rho"][rho]["whitened"]["heldout"])
                            / st.mean(h[-1]["err_rel"] for h in W[rho]["heldout"]) - 1))
        out["CalTwoDevAlpha"] = f"{100 * st.median(dv):.1f}"
        out["CalTwoDevAlphaMax"] = f"{100 * max(dv):.1f}"
        out["CalTwoDevEps"] = f"{100 * max(eps2):.1f}"

    # a-priori surrogate, judged against the held-out error
    for rho in (rhos[0], rhos[-1]):
        t = tags[rho]
        R = W[rho]
        al, b = apriori_bound(R["rows"])
        out["Apri" + t] = f"{b:.2f}"
        out["ApriUnder" + t] = str(sum(a_ < r["alpha_filtered"] for a_, r in zip(al[1:], R["rows"][1:])))
        out["EpsHeldMean" + t] = f"{st.mean(h[-1]['err_rel'] for h in R['heldout']):.2f}"

    # ImageNet: crossover budget vs. the gamma inflation compensation induces
    d3 = load("e3.json")
    R = d3["models"]["resnet18"]["runs"]
    rr = sorted({r["rho"] for r in R})
    for tag, rho in [("Hi", max(rr)), ("Lo", min(rr))]:
        pick = lambda sch: next(r for r in R if r["rho"] == rho and r["scheme"] == sch
                                and r["method"] == "whitened")
        out["One" + tag] = f"{pick('one-shot')['top1']:.1f}"
        out["Cmp" + tag] = f"{pick('compensated')['top1']:.1f}"
        out["Gam" + tag] = f"{pick('compensated')['gain_mean']:.2f}"
        out["Rho" + tag] = f"{rho:g}"
    for m, tag in zip(["resnet18", "resnet34", "resnet50"], ["RA", "RB", "RC"]):
        R = d3["models"][m]["runs"]
        rr = sorted({r["rho"] for r in R}, reverse=True)
        win = None
        for rho in rr:
            o = next(x for x in R if x["rho"] == rho and x["scheme"] == "one-shot"
                     and x["method"] == "whitened")
            c = next(x for x in R if x["rho"] == rho and x["scheme"] == "compensated")
            if c["top1"] > o["top1"] and win is None:
                win = rho
        lo = next(x for x in R if x["rho"] == min(rr) and x["scheme"] == "compensated")
        out["GamMax" + tag] = f"{lo['gain_max']:.1f}"
        out["Xover" + tag] = f"{win:g}" if win is not None else "--"
        out["Dense" + tag] = f"{d3['models'][m]['baseline_top1']:.1f}"
    # damping sweep of compensating calibration: ResNet-18 at the tightest budget
    R = d3["models"]["resnet18"]["runs"]
    rlo = min(r["rho"] for r in R)
    accs = [next(x for x in R if x["rho"] == rlo and x["scheme"] == "compensated")["top1"]]
    for ridge in ("1e-4", "1e-2"):
        dr = load(f"e3_ridge_{ridge}.json")
        if dr:
            accs.append(dr["models"]["resnet18"]["runs"][0]["top1"])
    out["RidgeSpread"] = f"{max(accs) - min(accs):.1f}"

    # language models
    d4 = load("e4.json")
    for m, tag in zip(list(d4["models"])[:2], ["LA", "LB"]):
        R = d4["models"][m]["runs"]
        rr = sorted({r["rho"] for r in R})
        lo, hi = min(rr), max(rr)
        pick = lambda rho, s: next(x for x in R if x["rho"] == rho
                                   and x["scheme"] == s and x["method"] == "whitened")
        for tg, rho in [("Hi", hi), ("Lo", lo)]:
            out[f"Ppl{tag}One{tg}"] = f"{pick(rho, 'one-shot')['ppl']:.1f}"
            out[f"Ppl{tag}Seq{tg}"] = f"{pick(rho, 'sequential')['ppl']:.1f}"
            out[f"Ppl{tag}Cmp{tg}"] = f"{pick(rho, 'compensated')['ppl']:.0f}"
        out[f"Rho{tag}Hi"], out[f"Rho{tag}Lo"] = f"{hi:g}", f"{lo:g}"
        out[f"Dense{tag}"] = f"{d4['models'][m]['baseline_ppl']:.1f}"
        out[f"GamMax{tag}"] = f"{max(x['gain_max'] for x in R if x['scheme'] == 'compensated'):.0f}"
        out[f"SeqGain{tag}"] = f"{pick(lo, 'one-shot')['ppl'] / pick(lo, 'sequential')['ppl']:.1f}"
    # covariance-weighted vs. the better baseline, over models and budgets
    ratios = []
    for m in d4["models"]:
        R = d4["models"][m]["runs"]
        for rho in {r["rho"] for r in R}:
            one = {x["method"]: x["ppl"] for x in R if x["rho"] == rho and x["scheme"] == "one-shot"}
            ratios.append(min(one["agnostic"], one["diag"]) / one["whitened"])
    out["PplBaseRatioMin"] = f"{min(ratios):.0f}"
    # damping sweep of compensating calibration: first model at its mildest budget
    m0 = list(d4["models"])[0]
    R = d4["models"][m0]["runs"]
    hi = max(r["rho"] for r in R)
    sweep = [next(x for x in R if x["rho"] == hi and x["scheme"] == "compensated")]
    for ridge in ("1e-4", "1e-2"):
        dr = load(f"e4_ridge_{ridge}.json")
        if dr:
            sweep.append(dr["models"][m0]["runs"][0])
    for t, x in zip("ABC", sweep):
        out["PplCmpRidge" + t] = f"{x['ppl']:.0f}"
        out["GamCmpRidge" + t] = f"{x['gain_mean']:.2f}"
    out["RhoRidgeLLM"] = f"{hi:g}"

    # alignment on transformer linear layers
    d5 = load("e5.json")
    if d5:
        for m, tag in zip(list(d5["models"])[:2], ["LA", "LB"]):
            meds = []
            for rho, rows5 in sorted(d5["models"][m].items(), key=lambda kv: -float(kv[0])):
                m5 = st.median(r["kbar2"] / r["rd_ratio"] for r in rows5)
                meds.append(m5)
            out[f"Mu{tag}Lo"], out[f"Mu{tag}Hi"] = f"{min(meds):.1f}", f"{max(meds):.1f}"
            rows5 = [r for v in d5["models"][m].values() for r in v]
            out[f"Kap{tag}Max"] = f"{max(r['kappa'] for r in rows5):.2f}"
            out[f"Kap{tag}Med"] = f"{st.median(r['kappa'] for r in rows5):.2f}"
        allmeds = [st.median(r["kbar2"] / r["rd_ratio"] for r in rows5)
                   for m in d5["models"].values() for rows5 in m.values()]
        out["MuLLMLo"], out["MuLLMHi"] = f"{min(allmeds):.1f}", f"{max(allmeds):.1f}"

    write("numbers.tex", [f"\\newcommand{{\\n{k}}}{{{v}}}" for k, v in out.items()])


if __name__ == "__main__":
    rho = sys.argv[1] if len(sys.argv) > 1 else "0.5"
    figure(rho)
    tab_e2()
    tab_e3()
    tab_e4()
    numbers(rho)
