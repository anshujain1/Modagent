"""LLM provider layer. Real providers use httpx; MockProvider is deterministic and offline (for tests/CI only)."""
import json
import os
import re
import time
from dataclasses import dataclass

import httpx

from .rules import signals


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float


class Provider:
    def complete(self, system: str, messages: list, model: str, max_tokens: int = 400) -> LLMResponse:
        raise NotImplementedError


def _post_with_retry(client, url, headers, payload, retries=6):
    r = None
    for attempt in range(retries):
        r = client.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code in (429, 500, 502, 503, 529):
            time.sleep(min(2 ** attempt, 20))
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


class OpenAICompatProvider(Provider):
    """Any OpenAI-compatible chat endpoint (Groq, OpenAI, Together, vLLM, ...).
    Env: MODAGENT_API_KEY, MODAGENT_BASE_URL (default https://api.groq.com/openai/v1)."""

    def __init__(self):
        self.key = os.environ["MODAGENT_API_KEY"]
        self.base = os.environ.get("MODAGENT_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
        self.client = httpx.Client()

    def complete(self, system, messages, model, max_tokens=400):
        t0 = time.perf_counter()
        data = _post_with_retry(
            self.client, f"{self.base}/chat/completions",
            {"Authorization": f"Bearer {self.key}"},
            {"model": model, "temperature": 0, "max_tokens": max_tokens,
             "messages": [{"role": "system", "content": system}] + messages},
        )
        u = data.get("usage", {})
        return LLMResponse(data["choices"][0]["message"]["content"], u.get("prompt_tokens", 0),
                           u.get("completion_tokens", 0), time.perf_counter() - t0)


class AnthropicProvider(Provider):
    """Env: ANTHROPIC_API_KEY."""

    def __init__(self):
        self.key = os.environ["ANTHROPIC_API_KEY"]
        self.client = httpx.Client()

    def complete(self, system, messages, model, max_tokens=400):
        t0 = time.perf_counter()
        data = _post_with_retry(
            self.client, "https://api.anthropic.com/v1/messages",
            {"x-api-key": self.key, "anthropic-version": "2023-06-01"},
            {"model": model, "temperature": 0, "max_tokens": max_tokens, "system": system, "messages": messages},
        )
        u = data.get("usage", {})
        return LLMResponse(data["content"][0]["text"], u.get("input_tokens", 0),
                           u.get("output_tokens", 0), time.perf_counter() - t0)


class MockProvider(Provider):
    """Deterministic stand-in so the agent loop and harness can be tested offline.
    'mock-weak' is unsure on borderline items (exercises escalation); 'mock-strong' is decisive.
    Its metrics mean nothing: never report them."""

    def complete(self, system, messages, model, max_tokens=400):
        t0 = time.perf_counter()
        text = messages[0]["content"].split("TEXT:\n", 1)[1]
        tool_turns = [m for m in messages if m["role"] == "user" and m["content"].startswith("TOOL_RESULT")]
        tools_on = "check_signals" in system
        if tools_on and not tool_turns:
            out = {"action": "tool", "tool": "check_signals", "args": {}}
        else:
            s = signals(text)
            abuse = "ABUSE" in system or "abuse" in system.lower().split("domain:")[-1][:20]
            score = (s["urls"] + s["phone_numbers"] + s["shortcodes"] + len(s["promo_terms"]) + len(s["urgency_terms"])
                     + 2 * len(s["profanity"]))
            violating = score >= 2
            conf = 0.95 if (model.endswith("strong") or score == 0 or score >= 3) else 0.5
            if "FORCE_BAD_JSON" in text and not model.endswith("strong"):
                return LLMResponse("not json at all", 50, 5, time.perf_counter() - t0)
            out = {"action": "final", "label": "violating" if violating else "ok",
                   "category": ("offensive" if abuse else "spam") if violating else "none",
                   "confidence": conf, "reason": f"mock score {score}"}
        raw = json.dumps(out)
        return LLMResponse(raw, (len(system) + sum(len(m["content"]) for m in messages)) // 4, len(raw) // 4,
                           time.perf_counter() - t0 + 0.001)


def make_provider(name: str) -> Provider:
    return {"openai": OpenAICompatProvider, "anthropic": AnthropicProvider, "mock": MockProvider}[name]()
