"""Run the agent over an eval set and write results/<run>.json + .md.

Examples (run on your machine; needs an API key):
  export ANTHROPIC_API_KEY=...
  python scripts/run_eval.py --dataset spam --provider anthropic --weak <small-model> --strong <big-model> \
      --prices prices.json --run spam_escalate
  python scripts/run_eval.py --dataset spam --provider anthropic --weak <small-model> --no-escalate --run spam_weak_only
  python scripts/run_eval.py --dataset spam --provider anthropic --weak <small-model> --no-tools --no-escalate --run spam_notools
prices.json: {"<model name>": [usd_per_1M_input_tokens, usd_per_1M_output_tokens]}
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from modagent.agent import ModerationAgent  # noqa: E402
from modagent.evaluate import bootstrap_f1_ci, cost_usd, latency_stats, metrics, write_report  # noqa: E402
from modagent.providers import make_provider  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["spam", "abuse"], required=True)
    ap.add_argument("--provider", choices=["openai", "anthropic", "mock"], required=True)
    ap.add_argument("--weak", required=True)
    ap.add_argument("--strong")
    ap.add_argument("--no-escalate", action="store_true")
    ap.add_argument("--no-tools", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--prices")
    ap.add_argument("--run", required=True)
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    a = ap.parse_args()

    items = [json.loads(l) for l in open(ROOT / f"data/{a.dataset}_eval.jsonl")][: a.limit]
    prices = json.load(open(a.prices)) if a.prices else {}
    strong = None if a.no_escalate else a.strong
    agent = ModerationAgent(make_provider(a.provider), a.dataset, a.weak, strong, a.threshold, not a.no_tools)

    def work(r):
        d = agent.moderate(r["text"])
        return {"id": r["id"], "text": r["text"], "gold": r["gold"], "gold_category": r["gold_category"],
                "pred": d.label, "category": d.category, "confidence": d.confidence, "reason": d.reason,
                "escalated": d.escalated, "parse_failed": d.parse_failed, "tools_used": d.tools_used,
                "latency_s": d.latency_s, "in_tok": d.input_tokens, "out_tok": d.output_tokens,
                "cost_usd": cost_usd(d.input_tokens, d.output_tokens, prices) if prices else None}

    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(work, items))

    gold, pred = [r["gold"] for r in rows], [r["pred"] for r in rows]
    m = metrics(gold, pred)
    ci = bootstrap_f1_ci(gold, pred)
    cats = {}
    for r in rows:
        if r["gold"] == "violating":
            c = cats.setdefault(r["gold_category"], {"n": 0, "correct": 0})
            c["n"] += 1
            c["correct"] += r["pred"] == "violating"
    costs = [r["cost_usd"] for r in rows]
    total = None if (not prices or any(c is None for c in costs)) else sum(costs)
    n = len(rows)
    cfg = {"dataset": a.dataset, "provider": a.provider, "weak_model": a.weak, "strong_model": strong,
           "threshold": a.threshold, "tools": not a.no_tools, "n": n, "workers": a.workers}
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"config": cfg, "metrics": m, "f1_ci95": ci, "items": rows}, open(out / f"{a.run}.json", "w"), indent=1)
    write_report(out / f"{a.run}.md", f"Run: {a.run}", cfg, m, ci, latency_stats([r["latency_s"] for r in rows]), rows,
                 cats, total, sum(r["escalated"] for r in rows) / n,
                 sum(bool(r["tools_used"]) for r in rows) / n, sum(r["parse_failed"] for r in rows))
    print(json.dumps({"run": a.run, **m, "f1_ci95": ci}))


if __name__ == "__main__":
    main()
