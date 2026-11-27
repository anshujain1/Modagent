"""Metrics and report writing."""
import numpy as np


def confusion(y_true, y_pred, pos="violating"):
    tp = sum(t == pos and p == pos for t, p in zip(y_true, y_pred))
    fp = sum(t != pos and p == pos for t, p in zip(y_true, y_pred))
    fn = sum(t == pos and p != pos for t, p in zip(y_true, y_pred))
    tn = sum(t != pos and p != pos for t, p in zip(y_true, y_pred))
    return tp, fp, fn, tn


def metrics(y_true, y_pred) -> dict:
    tp, fp, fn, tn = confusion(y_true, y_pred)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"n": len(y_true), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "accuracy": round((tp + tn) / max(1, len(y_true)), 4)}


def bootstrap_f1_ci(y_true, y_pred, iters=1000, seed=0):
    """95% bootstrap CI for F1, so small differences between runs are not over-read."""
    rng = np.random.default_rng(seed)
    yt, yp = np.array(y_true), np.array(y_pred)
    vals = []
    for _ in range(iters):
        idx = rng.integers(0, len(yt), len(yt))
        vals.append(metrics(yt[idx].tolist(), yp[idx].tolist())["f1"])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def latency_stats(lat):
    a = np.array(lat)
    return {"mean": round(float(a.mean()), 3), "p50": round(float(np.percentile(a, 50)), 3),
            "p95": round(float(np.percentile(a, 95)), 3)}


def cost_usd(in_tok: dict, out_tok: dict, prices: dict):
    """prices: {model: [usd_per_1M_input, usd_per_1M_output]}. Returns None if any model has no price."""
    total = 0.0
    for m in set(in_tok) | set(out_tok):
        if m not in prices:
            return None
        pi, po = prices[m]
        total += in_tok.get(m, 0) / 1e6 * pi + out_tok.get(m, 0) / 1e6 * po
    return total


def write_report(path, title, cfg, m, ci, lat, items, per_cat, cost_total, esc_rate, tool_rate, parse_fail):
    n = len(items)
    review = sum(bool(r.get("needs_review")) for r in items)
    L = [f"# {title}", "", "## Config", "```", *[f"{k}: {v}" for k, v in cfg.items()], "```", "",
         "## Quality", f"- precision **{m['precision']:.3f}**, recall **{m['recall']:.3f}**, F1 **{m['f1']:.3f}** "
         f"(95% bootstrap CI {ci[0]:.3f}-{ci[1]:.3f}), accuracy {m['accuracy']:.3f}, n={m['n']}",
         f"- confusion: TP {m['tp']}  FP {m['fp']}  FN {m['fn']}  TN {m['tn']}"]
    for c, v in per_cat.items():
        L.append(f"- gold category `{c}`: {v['correct']}/{v['n']} flagged correctly")
    L += ["", "## Latency and cost",
          f"- per-item latency (s): mean {lat['mean']}, p50 {lat['p50']}, p95 {lat['p95']}",
          f"- escalation rate: {esc_rate:.1%}; tool-use rate: {tool_rate:.1%}; weak-model unparseable (before escalation): {parse_fail}",
          f"- needs human review (no usable answer from any model): {review} ({review / n:.1%}); "
          "metrics above count these as not flagged"]
    if cost_total is None:
        L.append("- cost: not computed (no --prices file supplied)")
    else:
        L.append(f"- cost: ${cost_total:.4f} total, **${cost_total / n * 1000:.3f} per 1,000 items**")
    for kind, flt in (("False positives (flagged, gold ok)", lambda r: r["pred"] == "violating" and r["gold"] == "ok"),
                      ("False negatives (missed violations)", lambda r: r["pred"] != "violating" and r["gold"] == "violating")):
        rows = [r for r in items if flt(r)][:8]
        L += ["", f"## {kind}"]
        for r in rows:
            L.append(f"- [{r['confidence']:.2f}] {r['text'][:140]!r} -> {r['reason']}")
        if not rows:
            L.append("- none")
    open(path, "w").write("\n".join(L) + "\n")
