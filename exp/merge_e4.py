"""Merge the E4 runs (results/e4_part_*.json, one file per model or per model and
budget) into results/e4.json, with the models in the order of Table IV and the
runs of each model in decreasing budget."""
import glob, json, os

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ORDER = ["Qwen/Qwen2.5-0.5B", "meta-llama/Llama-3.2-1B-Instruct"]

out, models = None, {}
for p in sorted(glob.glob(os.path.join(RES, "e4_part_*.json"))):
    d = json.load(open(p))
    if out is None:
        out = {k: v for k, v in d.items() if k != "models"}
    for m, e in d["models"].items():
        tgt = models.setdefault(m, {"baseline_ppl": e["baseline_ppl"], "runs": []})
        tgt["runs"] += e["runs"]
order = {s: i for i, s in enumerate(["one-shot", "sequential", "compensated"])}
for e in models.values():
    e["runs"].sort(key=lambda r: (-r["rho"], order[r["scheme"]]))
out["models"] = {m: models[m] for m in ORDER if m in models}
p = os.path.join(RES, "e4.json")
json.dump(out, open(p, "w"), indent=1)
print("wrote", p, {m: len(e["runs"]) for m, e in out["models"].items()})
