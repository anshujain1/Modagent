# ModAgent: LLM content-moderation agent with an offline evaluation harness

A small tool-using LLM agent that decides whether a piece of user content violates a policy, plus a harness that
measures **precision / recall / F1, latency, token cost, escalation rate and failure cases** against labelled data and
against a classical baseline. Built for evaluating *whether an LLM moderation system is worth its cost*, not just
whether it works.

## How the agent works
1. A **cheap model** reads the text (treated as untrusted data, so injected instructions are ignored) and may call tools:
   - `lookup_policy`: returns the written policy (`policies/*.md`)
   - `check_signals`: deterministic features (URLs, phone numbers, short codes, promo/urgency terms, profanity)
2. It returns strict JSON: `label`, `category`, `confidence`, `reason`.
3. If the output is **unparseable or confidence < threshold**, a **stronger model** re-judges the item from scratch.
4. Every call's tokens and latency are accumulated per item, so cost per 1,000 items falls out of the run.
Unparseable output everywhere fails open to `ok` and is counted (`unparseable outputs` in the report).

Providers: Anthropic, any OpenAI-compatible endpoint (Groq, OpenAI, vLLM...), and an offline mock for tests.

## Data (public, downloaded by you; fixed seed)
| set | source | violating = |
|---|---|---|
| `spam` | UCI SMS Spam Collection | spam |
| `abuse` | Davidson et al. 2017 tweets | hate speech or offensive language (crowd majority label) |

`scripts/prepare_data.py` de-duplicates texts first (the SMS set has many exact repeats; without this, ~15% of eval
messages also sat in the training data and inflated the baseline), then draws a 400-item eval set per task (spam 30%
violating, abuse 50% violating). **Precision depends on prevalence**; real traffic has far fewer violations than
these sets, so quote precision together with the mix. Davidson labels are known to be noisy at the hate/offensive
boundary, so expect a ceiling below 100%.

## Run
```bash
pip install -r requirements.txt
python scripts/prepare_data.py          # builds data/*_eval.jsonl and *_train.jsonl
python scripts/baseline.py              # TF-IDF + logistic regression, no API needed
python -m pytest -q                     # 13 tests, offline (mock provider)

export ANTHROPIC_API_KEY=...            # or MODAGENT_API_KEY / MODAGENT_BASE_URL for OpenAI-compatible
echo '{"<weak-model>": [IN_PER_1M, OUT_PER_1M], "<strong-model>": [IN_PER_1M, OUT_PER_1M]}' > prices.json   # current list prices in USD

# the four runs that make the comparison (repeat with --dataset abuse)
python scripts/run_eval.py --dataset spam --provider anthropic --weak <weak-model> --strong <strong-model> --prices prices.json --run spam_escalate
python scripts/run_eval.py --dataset spam --provider anthropic --weak <weak-model> --no-escalate           --prices prices.json --run spam_weak_only
python scripts/run_eval.py --dataset spam --provider anthropic --weak <strong-model> --no-escalate         --prices prices.json --run spam_strong_only
python scripts/run_eval.py --dataset spam --provider anthropic --weak <weak-model> --no-escalate --no-tools --prices prices.json --run spam_weak_notools
```
Each run writes `results/<run>.md` (metrics, bootstrap CI on F1, latency p50/p95, cost per 1,000 items, escalation and
tool-use rates, false positives/negatives with the model's reasons) and `results/<run>.json` (per item).

## Results
**Classical baseline (measured, 400 held-out items, no leakage):**

| set | precision | recall | F1 (95% CI) | latency/item |
|---|---|---|---|---|
| spam | 0.983 | 0.933 | 0.957 (0.926-0.983) | ~1.5 ms |
| abuse | 0.923 | 0.955 | 0.939 (0.913-0.961) | ~1.5 ms |

**LLM agent runs: fill in after you run the commands above.** (The mock provider is only for testing the loop; never
quote its numbers.) The baseline is strong, so the honest question this project answers is *what the LLM buys you*
(zero-shot coverage of new abuse patterns, explanations, no training data) and *what it costs* (latency and dollars),
and whether escalation recovers most of the strong model's quality at a fraction of its cost.

## Limitations
- Static public datasets, not live traffic; no online experiments or monitoring.
- Binary labels; the policy text was written for this project, not taken from a real platform.
- Single seed and 400 items per task: differences smaller than the CI are noise.
