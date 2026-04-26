"""
Generate realistic mock field reports and parcel CSVs using Claude on Bedrock.
Intentional discrepancies are baked in for the fusion engine to surface.
"""
import json, csv, io, re
import config
from utils import aws_session

SEVERITY_LEVELS = ["none", "minor", "moderate", "severe", "destroyed"]


def generate_field_reports(event_id: str, num_reports: int = 3) -> list[dict]:
    """
    Ask Claude to generate realistic preliminary damage assessment reports.
    Each report is a dict with keys: source, date, entries[{address, severity, notes}]
    """
    event = config.EVENTS[event_id]
    prompt = f"""Generate {num_reports} realistic preliminary damage assessment field reports
for {event['name']} ({event['date']}) in {event['state']}.

Each report is from a different source team (volunteer fire dept, county building inspector, Red Cross).
Include 5-8 property entries per report with realistic street addresses near {event['center']}.
IMPORTANT: introduce these intentional discrepancies across reports:
- 2 properties reported as "moderate" that are actually "severe" or "destroyed"
- 1 property reported that doesn't actually exist (wrong address)
- 2 badly damaged properties MISSING from all reports (unreported damage)

Return ONLY valid JSON array:
[
  {{
    "source": "<team name>",
    "date": "{event['date']}",
    "entries": [
      {{
        "address": "<street address>",
        "severity": "<none|minor|moderate|severe|destroyed>",
        "notes": "<field observation>",
        "lat": <decimal>,
        "lon": <decimal>
      }}
    ]
  }}
]"""

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }
    response = aws_session.bedrock_runtime().invoke_model(
        modelId=config.CLAUDE_MODEL_ID,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    text = json.loads(response["body"].read())["content"][0]["text"]
    try:
        reports = json.loads(text)
    except Exception:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        reports = json.loads(match.group()) if match else []

    print(f"[MockData] Generated {len(reports)} field reports for {event['name']}")
    return reports


def generate_parcel_csv(event_id: str, num_rows: int = 80) -> str:
    """
    Ask Claude to generate a realistic parcel CSV for the event area.
    Returns CSV string.
    """
    event = config.EVENTS[event_id]
    prompt = f"""Generate a realistic county parcel dataset for {event['name']} area in {event['state']}.
{num_rows} rows representing residential/commercial structures near {event['center']}.
Coordinates should be within bbox: {event['bbox']}

Return ONLY a CSV with these exact columns:
parcel_id,address,lat,lon,structure_type,year_built,assessed_value,owner_name,stories,sq_ft

Use realistic values. Mix residential, commercial, and multi-family structures."""

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": prompt}],
    }
    response = aws_session.bedrock_runtime().invoke_model(
        modelId=config.CLAUDE_MODEL_ID,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    text = json.loads(response["body"].read())["content"][0]["text"]

    # strip any markdown code fences
    text = re.sub(r"```[a-z]*\n?", "", text).strip()
    print(f"[MockData] Generated parcel CSV for {event['name']} ({text.count(chr(10))} rows)")
    return text


def save_to_s3(event_id: str, reports: list[dict], parcel_csv: str) -> dict:
    """Upload generated mock data to S3, return keys."""
    s3 = aws_session.s3()
    keys = {}

    reports_key = f"{config.S3_REPORTS_PREFIX}/{event_id}/field_reports.json"
    s3.put_object(
        Bucket=config.S3_BUCKET,
        Key=reports_key,
        Body=json.dumps(reports, indent=2).encode(),
        ContentType="application/json",
    )
    keys["reports"] = reports_key

    parcels_key = f"{config.S3_PARCELS_PREFIX}/{event_id}/parcels.csv"
    s3.put_object(
        Bucket=config.S3_BUCKET,
        Key=parcels_key,
        Body=parcel_csv.encode(),
        ContentType="text/csv",
    )
    keys["parcels"] = parcels_key

    print(f"[MockData] Saved to S3: {keys}")
    return keys
