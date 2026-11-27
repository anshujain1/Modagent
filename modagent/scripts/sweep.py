"""Offline escalation-threshold sweep: no API calls.

Two finished runs on the same items (a weak-only run and a strong-only run, same dataset/tools/limit) contain every
number needed to replay the escalation policy at any threshold t:
    escalate if the weak answer was unparseable or its confidence < t, then take the strong model's answer
    (falling back to the weak answer if the strong one is unparseable, as the live agent does)
    cost    = weak cost (+ strong cost when escalated)
    latency = weak latency (+ strong latency when escalated)

Usage:
  python scripts/run_eval.py --dataset spam --provider anthropic --weak W --no-escalate --prices prices.json --run spam_weak_only
  python scripts/run_eval.py --dataset spam --provider anthropic --weak S --no-escalate --prices prices.json --run spam_strong_only
  python scripts/sweep.py --weak results/spam_weak_only.json --strong results/spam_strong_only.json --name spam --plot

Writes results/<name>_sweep.md / .json (and .png with --plot if matplotlib is installed).
Also prints a calibration table: does the weak model's self-reported confidence predict correctness? If not,
confidence-based escalation is not buying what it looks like it buys.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from modagent.evaluate import metrics  # noqa: E402

DEFAULT_THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]   # 1.01 = escalate everything
CAL_EDGES = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0001]


def load(path):
    d = json.load(open(path))
    return d["config"], {r["id"]: r for r in d["items"]}


def binary(pred):
    return "violating" if pred == "violating" else "ok"


def _add(a, b):
    return None if a is None or b is None else a + b


def simulate(weak, strong, ids, t):
    gold, pred, cost, lat, esc = [], [], [], [], 0
    for i in ids:
        w, s = weak[i], strong[i]
        c, lt, label = w["cost_usd"], w["latency_s"], w["pred"]
        if w["parse_failed"] or w["confidence"] < t:
            esc += 1
            c, lt = _add(c, s["cost_usd"]), lt + s["latency_s"]
            if not (s["parse_failed"] and not w["parse_failed"]):   # live agent keeps weak's answer if strong failed
                label = s["pred"]
        gold.append(weak[i]["gold"])
        pred.append(binary(label))
        cost.append(c)
        lat.append(lt)
    m = metrics(gold, pred)
    total = None if any(c is None for c in cost) else sum(cost)
    return {"threshold": t, "f1": m["f1"], "precision": m["precision"], "recall": m["recall"],
            "escalation_rate": round(esc / len(ids), 4),
            "cost_per_1k": None if total is None else round(total / len(ids) * 1000, 4),
            "mean_latency_s": round(float(np.mean(lat)), 3)}


def pure(rows, ids):
    gold = [rows[i]["gold"] for i in ids]
    m = metrics(gold, [binary(rows[i]["pred"]) for i in ids])
    costs = [rows[i]["cost_usd"] for i in ids]
    total = None if any(c is None for c in costs) else sum(costs)
    return {"f1": m["f1"], "precision": m["precision"], "recall": m["recall"],
            "cost_per_1k": None if total is None else round(total / len(ids) * 1000, 4),
            "mean_latency_s": round(float(np.mean([rows[i]["latency_s"] for i in ids])), 3)}


def calibration(weak, ids):
    ok_rows = [weak[i] for i in ids if not weak[i]["parse_failed"]]
    out = []
    for lo, hi in zip(CAL_EDGES, CAL_EDGES[1:]):
        b = [r for r in ok_rows if lo <= r["confidence"] < hi]
        if b:
            out.append({"bin": f"[{lo:.2f}, {min(hi, 1.0):.2f}{']' if hi > 1 else ')'}", "n": len(b),
                        "mean_conf": round(float(np.mean([r["confidence"] for r in b])), 3),
                        "accuracy": round(float(np.mean([binary(r["pred"]) == r["gold"] for r in b])), 3)})
    return out, len(ids) - len(ok_rows)


def fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weak", required=True, help="results/<run>.json of a weak-model, --no-escalate run")
    ap.add_argument("--strong", required=True, help="results/<run>.json of a strong-model, --no-escalate run")
    ap.add_argument("--name", help="output basename (default: dataset name)")
    ap.add_argument("--thresholds", default=",".join(map(str, DEFAULT_THRESHOLDS)))
    ap.add_argument("--tolerance", type=float, default=0.01, help="F1 drop vs strong-only you are willing to accept")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    a = ap.parse_args()

    (wc, weak), (sc, strong) = load(a.weak), load(a.strong)
    warnings = []
    for key in ("dataset", "tools"):
        if wc.get(key) != sc.get(key):
            warnings.append(f"config mismatch on '{key}': weak={wc.get(key)} strong={sc.get(key)}")
    if wc.get("strong_model") or sc.get("strong_model"):
        warnings.append("one of the runs used escalation; pass --no-escalate runs so each file is one model's own answers")
    ids = [i for i in weak if i in strong]
    if not ids:
        sys.exit("no item ids in common between the two runs")
    if len(ids) < min(len(weak), len(strong)):
        warnings.append(f"only {len(ids)} items are in both runs ({len(weak)} weak, {len(strong)} strong)")

    ts = [float(x) for x in a.thresholds.split(",")]
    rows = [simulate(weak, strong, ids, t) for t in ts]
    w_only, s_only = pure(weak, ids), pure(strong, ids)
    cal, unparsed = calibration(weak, ids)

    # cheapest threshold whose F1 is within tolerance of strong-only (fall back to lowest escalation rate)
    ok = [r for r in rows if r["f1"] >= s_only["f1"] - a.tolerance]
    key = (lambda r: (r["cost_per_1k"], r["escalation_rate"])) if all(r["cost_per_1k"] is not None for r in rows) \
        else (lambda r: (r["escalation_rate"], 0))
    best = min(ok, key=key) if ok else None

    name = a.name or wc.get("dataset", "run")
    L = [f"# Threshold sweep: {name}", "",
         f"weak = `{wc.get('weak_model')}`, strong = `{sc.get('weak_model')}`, n = {len(ids)} items, tools = {wc.get('tools')}", ""]
    L += [f"> WARNING: {w}" for w in warnings] + ([""] if warnings else [])
    L += ["| setting | F1 | precision | recall | escalated | $ / 1k items | mean latency (s) |", "|---|---|---|---|---|---|---|",
          f"| weak only (file) | {fmt(w_only['f1'])} | {fmt(w_only['precision'])} | {fmt(w_only['recall'])} | 0% | "
          f"{fmt(w_only['cost_per_1k'], 4)} | {fmt(w_only['mean_latency_s'])} |"]
    for r in rows:
        label = "escalate everything" if r["threshold"] > 1 else f"escalate if conf < {r['threshold']:.2f}"
        L.append(f"| {label} | {fmt(r['f1'])} | {fmt(r['precision'])} | {fmt(r['recall'])} | "
                 f"{r['escalation_rate']:.1%} | {fmt(r['cost_per_1k'], 4)} | {fmt(r['mean_latency_s'])} |")
    L += [f"| strong only (file) | {fmt(s_only['f1'])} | {fmt(s_only['precision'])} | {fmt(s_only['recall'])} | 100% | "
          f"{fmt(s_only['cost_per_1k'], 4)} | {fmt(s_only['mean_latency_s'])} |", ""]
    if best:
        saved = ""
        if best["cost_per_1k"] is not None and s_only["cost_per_1k"]:
            saved = f", {1 - best['cost_per_1k'] / s_only['cost_per_1k']:.0%} cheaper than strong-only"
        L.append(f"**Cheapest threshold within {a.tolerance:.2f} F1 of strong-only:** conf < {best['threshold']:.2f} "
                 f"(F1 {best['f1']:.3f}, escalates {best['escalation_rate']:.1%}{saved}). "
                 "Differences below the bootstrap CI on a 400-item set are noise.")
    else:
        L.append(f"**No threshold gets within {a.tolerance:.2f} F1 of strong-only.**")
    L += ["", "## Calibration of the weak model's confidence", "",
          "| confidence bin | n | mean confidence | accuracy |", "|---|---|---|---|"]
    L += [f"| {c['bin']} | {c['n']} | {c['mean_conf']:.3f} | {c['accuracy']:.3f} |" for c in cal]
    L += ["", f"(excluded {unparsed} items whose weak output was unparseable)",
          "If accuracy does not rise with confidence, the confidence signal is not informative and escalation on it "
          "is no better than escalating a random subset."]

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}_sweep.md").write_text("\n".join(L) + "\n")
    json.dump({"weak_only": w_only, "strong_only": s_only, "sweep": rows, "best": best, "calibration": cal,
               "warnings": warnings}, open(out / f"{name}_sweep.json", "w"), indent=1)
    print("\n".join(L))

    if a.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping --plot", file=sys.stderr)
            return
        use_cost = all(r["cost_per_1k"] is not None for r in rows)
        xs = [r["cost_per_1k"] if use_cost else r["escalation_rate"] for r in rows]
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        ax.plot(xs, [r["f1"] for r in rows], marker="o", color="#2a6fdb")
        for x, r in zip(xs, rows):
            if r["threshold"] <= 1:
                ax.annotate(f"{r['threshold']:.2f}", (x, r["f1"]), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.set_xlabel("cost per 1,000 items (USD)" if use_cost else "escalation rate")
        ax.set_ylabel("F1")
        ax.set_title(f"{name}: escalation threshold trade-off (labels = confidence threshold)", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / f"{name}_sweep.png")


if __name__ == "__main__":
    main()
