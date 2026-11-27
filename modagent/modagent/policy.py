"""Policy lookup. Exposed to the agent as the `lookup_policy` tool."""
from pathlib import Path

POLICY_DIR = Path(__file__).resolve().parent.parent / "policies"


def get_policy(domain: str) -> str:
    path = POLICY_DIR / f"{domain}.md"
    if not path.exists():
        return f"No policy found for domain '{domain}'. Available: " + ", ".join(sorted(p.stem for p in POLICY_DIR.glob("*.md")))
    return path.read_text().strip()
