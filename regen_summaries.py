"""
Regenerate executive summaries for both events with clean ASCII output.
Reads credentials from .streamlit/secrets.toml, saves back to fusion_results.json.
"""
import json, re, sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── load secrets ──────────────────────────────────────────────────────────────
try:
    import tomllib
except ImportError:
    import tomli as tomllib  # pip install tomli for Python < 3.11

secrets_path = os.path.join(os.path.dirname(__file__), ".streamlit", "secrets.toml")
with open(secrets_path, "rb") as f:
    secrets = tomllib.load(f)

import boto3

session = boto3.Session(
    aws_access_key_id     = secrets["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key = secrets["AWS_SECRET_ACCESS_KEY"],
    aws_session_token     = secrets.get("AWS_SESSION_TOKEN"),
    region_name           = secrets.get("AWS_DEFAULT_REGION", "us-east-1"),
)
bedrock = session.client("bedrock-runtime", region_name="us-east-1")

MODEL = secrets.get("CLAUDE_MODEL_ID", "us.anthropic.claude-sonnet-4-6")

# ── helpers ───────────────────────────────────────────────────────────────────
def sanitize(text: str) -> str:
    """Replace any remaining non-ASCII characters with safe equivalents."""
    replacements = {
        "—": "--",   # em dash
        "–": "-",    # en dash
        "‘": "'",    # left single quote
        "’": "'",    # right single quote
        "“": '"',    # left double quote
        "”": '"',    # right double quote
        "•": "-",    # bullet
        " ": " ",    # non-breaking space
        "�": "",     # replacement character (already corrupted)
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    # strip any remaining non-ASCII
    return text.encode("ascii", errors="ignore").decode("ascii")

def call_claude(prompt: str) -> str:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = bedrock.invoke_model(
        modelId=MODEL,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    return json.loads(resp["body"].read())["content"][0]["text"]

def generate_summary(event_name: str, event_date: str, counts: dict, top_unreported: list) -> str:
    prompt = f"""Write a FEMA Preliminary Damage Assessment executive summary.

IMPORTANT FORMATTING RULES:
- Use ONLY plain ASCII characters. No em dashes, no curly quotes, no special symbols.
- Use a regular hyphen (-) instead of a dash.
- Use straight quotes only.
- Do not use bullet symbols; use a plain hyphen (-) for list items.

Event: {event_name} -- {event_date}
AI-Assessed Damage Inventory:
- Validated damage (multi-source confirmed): {counts['validated']} structures
- Unreported damage (AI-detected, not in field reports): {counts['unreported']} structures
- Conflicts (source disagreement): {counts['conflicts']} structures

Top unreported structures (missed by field teams):
{json.dumps(top_unreported, indent=2)}

Write 3-4 formal paragraphs covering:
1. Incident and scope
2. Validated damage summary
3. Unreported damage as a key intelligence finding
4. Recommended immediate follow-up actions

Use formal emergency management language. Plain ASCII only."""
    return sanitize(call_claude(prompt))

# ── main ──────────────────────────────────────────────────────────────────────
with open("fusion_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

events = {
    "hurricane_milton":   ("Hurricane Milton",   "October 9, 2024"),
    "palisades_wildfire": ("Palisades Wildfire",  "January 7, 2025"),
}

for event_id, (event_name, event_date) in events.items():
    print(f"\nGenerating summary for {event_name}...")
    result = data[event_id]
    counts = {k: len(v) for k, v in result.items() if isinstance(v, list)}
    top_unreported = result.get("unreported", [])[:3]

    summary = generate_summary(event_name, event_date, counts, top_unreported)
    data[event_id]["executive_summary"] = summary

    print(f"  OK -- {len(summary)} chars, ASCII clean: {summary.isascii()}")
    print(summary[:300])

with open("fusion_results.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=True)

print("\nSaved fusion_results.json with clean summaries.")
