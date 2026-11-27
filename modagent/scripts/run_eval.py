"""Run the agent over an eval set and write results/<run>.json + .md.

Fault tolerant: every finished item is appended to results/<run>.items.jsonl as soon as it completes, a failing item
is recorded as an error instead of killing the run, and --resume re-runs only what is missing or failed.
Items that still errored are left out of the metrics and the script exits with code 3 so it cannot pass silently.

Examples (run on your machine; needs an API key):
  export ANTHROPIC_API_KEY=...
  python scripts/run_eval.py --dataset spam --provider anthropic --weak <small-model> --strong <big-model> \
      --prices prices.json --run spam_escalate
  python scripts/run_eval.py --dataset spam --provider anthropic --weak <small-model> --no-escalate --run spam_weak_only
  python scripts/run_eval.py --dataset spam --provider anthropic --weak <small-model> --no-tools --no-escalate --run spam_notools
  python scripts/run_eval.py ... --run spam_escalate --resume        # after a crash / rate-limit failure
prices.json: {"<model name>": [usd_per_1M_input_tokens, usd_per_1M_output_tokens]}
"""
import argparse
import json
import sys
import threading
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
    ap.add_argument("--data", help="eval jsonl to use instead of data/<dataset>_eval.jsonl (policy still comes from --dataset)")
    ap.add_argument("--provider", choices=["openai", "anthropic", "mock"], required=True)
    ap.add_argument("--weak", required=True)
    ap.add_argument("--strong")
    ap.add_argument("--no-escalate", action="store_true")
    ap.add_argument("--no-tools", action="store_true")
    ap.add_argument("--fail-mode", choices=["review", "open", "closed"], default="review",
                    help="label when no model returns a usable answer (default: review queue)")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--prices")
    ap.add_argument("--run", required=True)
    ap.add_argument("--resume", action="store_true", help="reuse successful items from results/<run>.items.jsonl")
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    a = ap.parse_args()

    data_path = Path(a.data) if a.data else ROOT / f"data/{a.dataset}_eval.jsonl"
    items = [json.loads(l) for l in open(data_path)][: a.limit]
    prices = json.load(open(a.prices)) if a.prices else {}
    strong = None if a.no_escalate else a.strong
    agent = ModerationAgent(make_provider(a.provider), a.dataset, a.weak, strong, a.threshold, not a.no_tools,
                            fail_mode=a.fail_mode)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / f"{a.run}.items.jsonl"
    done = {}
    if a.resume and ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if "error" not in r:          # failed items are retried
                    done[r["id"]] = r
    ckpt.write_text("".join(json.dumps(r) + "\n" for r in done.values()))
    if done:
        print(f"resuming: {len(done)} items already done", file=sys.stderr)
    lock = threading.Lock()

    def work(r):
        base = {"id": r["id"], "text": r["text"], "gold": r["gold"], "gold_category": r["gold_category"]}
        try:
            d = agent.moderate(r["text"])
            row = {**base, "pred": d.label, "category": d.category, "confidence": d.confidence, "reason": d.reason,
                   "escalated": d.escalated, "parse_failed": d.parse_failed, "needs_review": d.needs_review,
                   "tools_used": d.tools_used,
                   "latency_s": d.latency_s, "in_tok": d.input_tokens, "out_tok": d.output_tokens,
                   "cost_usd": cost_usd(d.input_tokens, d.output_tokens, prices) if prices else None}
        except Exception as e:                # one bad item must not lose the rest of the run
            row = {**base, "error": f"{type(e).__name__}: {e}"[:300]}
        with lock:
            with ckpt.open("a") as f:
                f.write(json.dumps(row) + "\n")
        return row

    todo = [r for r in items if r["id"] not in done]
    with ThreadPoolExecutor(a.workers) as ex:
        fresh = {r["id"]: r for r in ex.map(work, todo)}
    by_id = {**done, **fresh}
    all_rows = [by_id[r["id"]] for r in items]
    errors = [{"id": r["id"], "error": r["error"]} for r in all_rows if "error" in r]
    rows = [r for r in all_rows if "error" not in r]
    if not rows:
        sys.exit(f"every item failed; first error: {errors[0]['error'] if errors else 'no items'}")

    gold = [r["gold"] for r in rows]
    pred = ["violating" if r["pred"] == "violating" else "ok" for r in rows]   # "review" = not flagged until a human looks
    m = metrics(gold, pred)
    m["needs_review"] = sum(bool(r.get("needs_review")) for r in rows)
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
           "threshold": a.threshold, "tools": not a.no_tools, "n": n, "workers": a.workers,
           "requested": len(items), "errors": len(errors), "fail_mode": a.fail_mode}
    json.dump({"config": cfg, "metrics": m, "f1_ci95": ci, "errors": errors, "items": rows},
              open(out / f"{a.run}.json", "w"), indent=1)
    write_report(out / f"{a.run}.md", f"Run: {a.run}", cfg, m, ci, latency_stats([r["latency_s"] for r in rows]), rows,
                 cats, total, sum(r["escalated"] for r in rows) / n,
                 sum(bool(r["tools_used"]) for r in rows) / n, sum(r["parse_failed"] for r in rows))
    print(json.dumps({"run": a.run, **m, "f1_ci95": ci, "errors": len(errors)}))
    if errors:
        print(f"WARNING: {len(errors)}/{len(items)} items failed and are excluded from the metrics "
              f"(first: {errors[0]['error']}). Re-run with --resume to retry them.", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
