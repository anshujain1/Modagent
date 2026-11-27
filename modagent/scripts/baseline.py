"""Classical baseline: TF-IDF (word + char n-grams) + logistic regression, trained on the non-eval rows.
Gives the LLM agent something honest to be compared against (accuracy, latency, ~zero cost)."""
import json
import sys
import time
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modagent.evaluate import bootstrap_f1_ci, metrics  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def load(p):
    return [json.loads(l) for l in open(p)]


if __name__ == "__main__":
    out = {}
    for name in ("spam", "abuse"):
        tr, ev = load(ROOT / f"data/{name}_train.jsonl"), load(ROOT / f"data/{name}_eval.jsonl")
        pipe = make_pipeline(
            make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                       TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)),
            LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))
        pipe.fit([r["text"] for r in tr], [r["gold"] for r in tr])
        t0 = time.perf_counter()
        pred = [pipe.predict([r["text"]])[0] for r in ev]     # one-at-a-time, like serving
        per_item_ms = (time.perf_counter() - t0) / len(ev) * 1000
        gold = [r["gold"] for r in ev]
        m = metrics(gold, pred)
        m["f1_ci95"] = bootstrap_f1_ci(gold, pred)
        m["per_item_ms"] = round(per_item_ms, 2)
        m["train_n"] = len(tr)
        out[name] = m
        print(name, m)
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(out, open(ROOT / "results/baseline.json", "w"), indent=2)
