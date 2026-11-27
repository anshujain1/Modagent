"""Cheap deterministic signals. Exposed to the agent as the `check_signals` tool.

Term matching is word-boundary based. Plain substring matching (`"ass" in word`) flagged "class", "assembly" and
"Dickens" as profanity and "window" as a promo term, which fed false evidence to the model.
"""
import re

URL_RE = re.compile(r"(https?://|www\.)\S+|\b\w+\.(com|co\.uk|net|org|info)\b", re.I)
PHONE_RE = re.compile(r"\b\d{9,12}\b")
SHORTCODE_RE = re.compile(r"\b\d{5}\b")

# Promo words may carry a plural / verb suffix ("prizes", "claimed", "winning").
PROMO_WORDS = ["free", "win", "won", "prize", "claim", "cash", "offer", "reward", "bonus", "winner", "guaranteed"]
PROMO_SYMBOLS = ["£", "$"]            # not word characters, so \b does not apply
_PROMO_SUFFIX = r"(?:s|es|ed|ing|ning)?"
_PROMO_RES = {p: re.compile(rf"\b{re.escape(p)}{_PROMO_SUFFIX}\b") for p in PROMO_WORDS}

URGENCY = ["urgent", "now", "today", "limited", "last chance", "expires", "hurry", "immediately", "call"]

# canonical term -> regex of the surface forms we accept (whole words only)
PROFANITY_FORMS = {
    "fuck": r"(?:mother)?fuck(?:s|ed|er|ers|ing|in)?",
    "shit": r"(?:bull|dip)?shit(?:s|ty|ting|head|heads)?",
    "bitch": r"bitch(?:es|y|ing)?",
    "ass": r"(?:bad|dumb|jack|smart)?ass(?:es|hole|holes)?",
    "damn": r"damn(?:ed|it)?|dammit",
    "dick": r"dick(?:s|head|heads)?",
    "crap": r"crap(?:s|py)?",
    "piss": r"piss(?:ed|es|ing|y)?",
    "bastard": r"bastards?",
}
PROFANITY = sorted(PROFANITY_FORMS)
_PROFANITY_RES = {k: re.compile(rf"\b(?:{v})\b") for k, v in PROFANITY_FORMS.items()}

SECOND_PERSON = re.compile(r"\b(you|your|youre|you're|ur|u)\b", re.I)


def signals(text: str) -> dict:
    low = text.lower()
    letters = [c for c in text if c.isalpha()]
    caps = sum(c.isupper() for c in letters) / len(letters) if letters else 0.0
    promo = {p for p, rx in _PROMO_RES.items() if rx.search(low)} | {s for s in PROMO_SYMBOLS if s in low}
    return {
        "urls": len(URL_RE.findall(text)),
        "phone_numbers": len(PHONE_RE.findall(text)),
        "shortcodes": len(SHORTCODE_RE.findall(text)),
        "promo_terms": sorted(promo),
        "urgency_terms": sorted({u for u in URGENCY if re.search(rf"\b{re.escape(u)}\b", low)}),
        "caps_ratio": round(caps, 2),
        "profanity": sorted(k for k, rx in _PROFANITY_RES.items() if rx.search(low)),
        "second_person": len(SECOND_PERSON.findall(text)),
        "length_chars": len(text),
    }
