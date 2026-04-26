"""
Fusion engine — combines Pegasus video detections, Maxar satellite change detection,
and field reports into validated/unreported/conflict damage clusters.
"""
import json, re
import pandas as pd
from rapidfuzz import fuzz
import config
from utils import aws_session

SEVERITY_RANK = {"none": 0, "minor": 1, "moderate": 2, "severe": 3, "destroyed": 4}


def run_fusion(
    pegasus_results: list[dict],
    geo_results: list[dict],
    satellite_results: list[dict],
    field_reports: list[dict],
    parcel_df: pd.DataFrame,
) -> dict:
    """
    Merge all sources into a unified damage table.

    Returns:
        {
          "validated":  [DamageRecord, ...],
          "unreported": [DamageRecord, ...],
          "conflicts":  [DamageRecord, ...]
        }
    """
    # Build video-detected damage list with geo coords
    video_detections = _build_video_detections(pegasus_results, geo_results)

    # Build field report lookup by address
    report_lookup = _build_report_lookup(field_reports)

    # Flatten satellite results by location
    sat_lookup = {(r["lat"], r["lon"]): r for r in satellite_results if r.get("lat")}

    validated, unreported, conflicts = [], [], []

    for det in video_detections:
        lat, lon = det["lat"], det["lon"]
        sev_video = det.get("damage_severity", "none")

        # Satellite signal
        sat = _nearest_satellite(sat_lookup, lat, lon, radius_deg=0.005)
        sev_sat = sat.get("damage_class", "unknown") if sat else "unknown"

        # Field report match
        report_match = _match_field_report(det, report_lookup, parcel_df)
        sev_report = report_match.get("severity", "unknown") if report_match else "unknown"

        # Parcel match
        parcel = _nearest_parcel(parcel_df, lat, lon, radius_deg=0.002)

        # Composite confidence score
        score = _score(sev_video, sev_sat, report_match, parcel)

        record = {
            "lat": lat,
            "lon": lon,
            "address": parcel.get("address", det.get("location_name", det.get("location_description", "Unknown"))) if parcel else det.get("location_name", det.get("location_description", "Unknown")),
            "parcel_id": parcel.get("parcel_id") if parcel else None,
            "assessed_value": parcel.get("assessed_value") if parcel else None,
            "severity_video": sev_video,
            "severity_satellite": sev_sat,
            "severity_report": sev_report,
            "confidence": score,
            "video_description": det.get("description", ""),
            "satellite_notes": sat.get("notes", "") if sat else "",
            "report_source": report_match.get("source", "") if report_match else "",
            "damage_indicators": det.get("damage_indicators", []),
            "geo_reasoning": det.get("reasoning", ""),
        }

        if not report_match and score >= 0.4:
            record["category"] = "unreported"
            unreported.append(record)
        elif _is_conflict(sev_video, sev_report):
            record["category"] = "conflict"
            conflicts.append(record)
        elif score >= config.VALIDATED_THRESHOLD:
            record["category"] = "validated"
            validated.append(record)
        else:
            record["category"] = "validated"
            validated.append(record)

    # Sort each list by confidence descending
    for lst in [validated, unreported, conflicts]:
        lst.sort(key=lambda x: x["confidence"], reverse=True)

    print(f"[Fusion] validated={len(validated)} unreported={len(unreported)} conflicts={len(conflicts)}")
    return {"validated": validated, "unreported": unreported, "conflicts": conflicts}


def generate_executive_summary(fusion_result: dict, event_id: str) -> str:
    """Ask Claude to write a FEMA-tone one-page executive summary."""
    event = config.EVENTS[event_id]
    counts = {k: len(v) for k, v in fusion_result.items()}
    top_unreported = fusion_result["unreported"][:3]

    prompt = f"""Write a one-page executive summary in FEMA Preliminary Damage Assessment tone.

Event: {event['name']} — {event['date']}
AI-Assessed Damage Inventory:
- Validated damage (multi-source confirmed): {counts['validated']} structures
- Unreported damage (detected but not in field reports): {counts['unreported']} structures
- Conflicts (source disagreement): {counts['conflicts']} structures

Top unreported structures (missed by field teams):
{json.dumps(top_unreported[:3], indent=2)}

The summary should:
1. State the incident and scope
2. Summarize validated damage
3. Highlight the unreported damage as a key intelligence finding
4. Note confidence methodology
5. Recommend immediate follow-up actions

Write in formal emergency management language. 3-4 paragraphs."""

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }
    response = aws_session.bedrock_runtime().invoke_model(
        modelId=config.CLAUDE_MODEL_ID,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    return json.loads(response["body"].read())["content"][0]["text"]


# --- helpers ---

def _build_video_detections(pegasus_results, geo_results):
    detections = []
    for i, peg in enumerate(pegasus_results):
        if peg.get("damage_severity", "none") == "none":
            continue
        geo = geo_results[i] if i < len(geo_results) else {}
        detections.append({**peg, **geo})
    return detections


def _build_report_lookup(field_reports):
    lookup = []
    for report in field_reports:
        for entry in report.get("entries", []):
            lookup.append({**entry, "source": report.get("source", "")})
    return lookup


def _match_field_report(detection, report_lookup, parcel_df, threshold=80):
    desc = (detection.get("location_name", "") + " " +
            detection.get("location_description", "") + " " +
            detection.get("description", ""))
    best, best_score = None, 0
    for entry in report_lookup:
        score = fuzz.token_set_ratio(desc.lower(), entry.get("address", "").lower())
        if score > best_score:
            best_score, best = score, entry
    return best if best_score >= threshold else None


def _nearest_satellite(sat_lookup, lat, lon, radius_deg=0.005):
    best, best_dist = None, float("inf")
    for (slat, slon), rec in sat_lookup.items():
        dist = ((slat - lat) ** 2 + (slon - lon) ** 2) ** 0.5
        if dist < best_dist and dist < radius_deg:
            best_dist, best = dist, rec
    return best


def _nearest_parcel(parcel_df, lat, lon, radius_deg=0.002):
    if parcel_df is None or parcel_df.empty:
        return None
    df = parcel_df.copy()
    df["dist"] = ((df["lat"] - lat) ** 2 + (df["lon"] - lon) ** 2) ** 0.5
    nearest = df.nsmallest(1, "dist").iloc[0]
    if nearest["dist"] > radius_deg:
        return None
    return nearest.to_dict()


def _score(sev_video, sev_sat, report_match, parcel):
    w = config.FUSION_WEIGHTS
    score = 0.0
    if sev_video not in ("none", "unknown", ""):
        score += w["video_damage"]
    if sev_sat not in ("none", "unknown", ""):
        score += w["satellite_change"]
    if report_match:
        score += w["field_report"]
    if parcel:
        score += w["parcel_exists"]
    return round(min(score, 1.0), 3)


def _is_conflict(sev_video, sev_report):
    rv = SEVERITY_RANK.get(sev_video, -1)
    rr = SEVERITY_RANK.get(sev_report, -1)
    if rv < 0 or rr < 0:
        return False
    return abs(rv - rr) >= config.CONFLICT_SEVERITY_GAP
