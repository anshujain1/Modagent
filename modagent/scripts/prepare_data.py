"""Build fixed, seeded eval sets from the two public datasets.

spam : UCI SMS Spam Collection (data/sms.tsv)           gold violating = spam
abuse: Davidson et al. 2017 tweets (data/davidson.csv)  gold violating = hate (0) or offensive (1); ok = neither (2)
Writes data/<name>_eval.jsonl (the held-out eval set) and data/<name>_train.jsonl (everything else, for the baseline).
Eval sets are class-enriched (see --pos-frac) because natural prevalence differs a lot; precision depends on it.
"""
import argparse
import csv
import html
import json
import random
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def load_spam():
    rows = []
    for i, line in enumerate(open(DATA / "sms.tsv", encoding="utf-8", errors="replace")):
        lab, _, text = line.rstrip("\n").partition("\t")
        if text:
            rows.append({"id": f"sms{i}", "text": text, "gold": "violating" if lab == "spam" else "ok",
                         "gold_category": "spam" if lab == "spam" else "none"})
    return rows


def load_abuse():
    rows = []
    with open(DATA / "davidson.csv", encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            c = int(r["class"])
            text = html.unescape(r["tweet"]).strip()
            rows.append({"id": f"dav{r['']}", "text": text, "gold": "ok" if c == 2 else "violating",
                         "gold_category": {0: "hate", 1: "offensive", 2: "none"}[c]})
    return rows


def dedupe(rows):
    """Drop repeated texts (ignoring case, whitespace and the 'RT @user:' prefix) so no eval text is also in train."""
    seen, out = set(), []
    for r in rows:
        k = re.sub(r"\s+", " ", re.sub(r"^rt @\w+:\s*", "", r["text"].lower())).strip()
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def split(rows, n, pos_frac, rng):
    pos = [r for r in rows if r["gold"] == "violating"]
    neg = [r for r in rows if r["gold"] == "ok"]
    rng.shuffle(pos), rng.shuffle(neg)
    k = round(n * pos_frac)
    ev = pos[:k] + neg[:n - k]
    rng.shuffle(ev)
    ids = {r["id"] for r in ev}
    return ev, [r for r in rows if r["id"] not in ids]


def write(name, rows):
    with open(DATA / name, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--spam-pos-frac", type=float, default=0.3)
    ap.add_argument("--abuse-pos-frac", type=float, default=0.5)
    a = ap.parse_args()
    for name, loader, pf in (("spam", load_spam, a.spam_pos_frac), ("abuse", load_abuse, a.abuse_pos_frac)):
        ev, tr = split(dedupe(loader()), a.n, pf, random.Random(a.seed))
        write(f"{name}_eval.jsonl", ev)
        write(f"{name}_train.jsonl", tr)
        print(f"{name}: eval {len(ev)} ({sum(r['gold'] == 'violating' for r in ev)} violating), train {len(tr)}")
