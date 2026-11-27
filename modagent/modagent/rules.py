"""Cheap deterministic signals. Exposed to the agent as the `check_signals` tool."""
import re

URL_RE = re.compile(r"(https?://|www\.)\S+|\b\w+\.(com|co\.uk|net|org|info)\b", re.I)
PHONE_RE = re.compile(r"\b\d{9,12}\b")
SHORTCODE_RE = re.compile(r"\b\d{5}\b")
PROMO = ["free", "win", "won", "prize", "claim", "cash", "offer", "reward", "bonus", "winner", "guaranteed", "£", "$"]
URGENCY = ["urgent", "now", "today", "limited", "last chance", "expires", "hurry", "immediately", "call"]
PROFANITY = ["fuck", "shit", "bitch", "ass", "damn", "dick", "crap", "piss", "bastard"]
SECOND_PERSON = re.compile(r"\b(you|your|youre|you're|ur|u)\b", re.I)


def signals(text: str) -> dict:
    low = text.lower()
    words = re.findall(r"[a-z']+", low)
    letters = [c for c in text if c.isalpha()]
    caps = sum(c.isupper() for c in letters) / len(letters) if letters else 0.0
    return {
        "urls": len(URL_RE.findall(text)),
        "phone_numbers": len(PHONE_RE.findall(text)),
        "shortcodes": len(SHORTCODE_RE.findall(text)),
        "promo_terms": sorted({p for p in PROMO if p in low}),
        "urgency_terms": sorted({u for u in URGENCY if re.search(rf"\b{re.escape(u)}\b", low)}),
        "caps_ratio": round(caps, 2),
        "profanity": sorted({p for p in PROFANITY if any(p in w for w in words)}),
        "second_person": len(SECOND_PERSON.findall(text)),
        "length_chars": len(text),
    }
