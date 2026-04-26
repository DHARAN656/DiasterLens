"""
Maxar Open Data satellite imagery fetcher.
Uses the public STAC catalog to find pre/post event tiles
for a given bounding box and incident date.
"""
import requests, json, io, base64
from datetime import datetime, timedelta
from PIL import Image
import config
from utils import aws_session

STAC_BASE = "https://maxar-opendata.s3.amazonaws.com/events"

# Known Maxar Open Data collection slugs
COLLECTION_SLUGS = {
    "hurricane_milton":    "Hurricane-Milton-Oct-2024",
    "palisades_wildfire":  "Palisades-and-Eaton-fires-Jan-2025",
}


def fetch_tile_pair(event_id: str, lat: float, lon: float) -> dict:
    """
    Fetch pre-event and post-event Maxar tiles for a point.
    Returns {pre_url, post_url, pre_date, post_date, pre_b64, post_b64}
    """
    slug = COLLECTION_SLUGS.get(event_id)
    if not slug:
        return _fallback_tiles(event_id, lat, lon)

    event_date = config.EVENTS[event_id]["date"]

    pre_url  = _find_tile_url(slug, lat, lon, before=event_date)
    post_url = _find_tile_url(slug, lat, lon, after=event_date)

    result = {
        "event_id": event_id,
        "lat": lat,
        "lon": lon,
        "pre_url": pre_url,
        "post_url": post_url,
        "pre_b64": _url_to_b64(pre_url) if pre_url else None,
        "post_b64": _url_to_b64(post_url) if post_url else None,
    }
    print(f"[Maxar] {event_id} ({lat:.4f},{lon:.4f}) pre={'OK' if pre_url else 'MISS'} post={'OK' if post_url else 'MISS'}")
    return result


def _find_tile_url(slug: str, lat: float, lon: float, before: str = None, after: str = None) -> str | None:
    """
    Query the Maxar STAC catalog for a tile covering (lat, lon).
    before/after are ISO date strings used to filter pre/post event.
    """
    catalog_url = f"{STAC_BASE}/{slug}/collection.json"
    try:
        catalog = requests.get(catalog_url, timeout=10).json()
    except Exception as e:
        print(f"[Maxar] Catalog fetch failed: {e}")
        return None

    # Try links to child catalogs/items
    links = [l for l in catalog.get("links", []) if l.get("rel") in ("item", "items", "child")]
    if not links:
        return None

    # Walk first few items looking for one that covers our point
    for link in links[:20]:
        href = link.get("href", "")
        if not href.startswith("http"):
            href = f"{STAC_BASE}/{slug}/{href}"
        try:
            item = requests.get(href, timeout=10).json()
        except Exception:
            continue

        item_date = item.get("properties", {}).get("datetime", "")[:10]
        bbox = item.get("bbox", [])
        if len(bbox) == 4:
            w, s, e, n = bbox
            if not (s <= lat <= n and w <= lon <= e):
                continue

        if before and item_date >= before:
            continue
        if after and item_date < after:
            continue

        assets = item.get("assets", {})
        for key in ("visual", "overview", "thumbnail"):
            if key in assets:
                return assets[key].get("href")

    return None


def _url_to_b64(url: str) -> str | None:
    try:
        resp = requests.get(url, timeout=15)
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img.thumbnail((512, 512))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"[Maxar] Image fetch failed {url}: {e}")
        return None


def compare_tiles_with_claude(pre_b64: str, post_b64: str, event_id: str, location_desc: str = "") -> dict:
    """
    Send pre/post tile pair to Claude Vision for change detection.
    Returns {damage_class, bbox_pct, confidence, notes}
    """
    if not pre_b64 or not post_b64:
        return {"damage_class": "unknown", "confidence": 0.0, "notes": "missing tile"}

    prompt = f"""Compare these two satellite images of the same location.
Image 1 (BEFORE): taken before the {config.EVENTS[event_id]['name']} on {config.EVENTS[event_id]['date']}
Image 2 (AFTER):  taken after the disaster
Location: {location_desc or 'disaster zone'}

Return ONLY valid JSON:
{{
  "damage_class": "<none|minor|moderate|severe|destroyed>",
  "change_detected": <true|false>,
  "damage_area_pct": <0-100>,
  "confidence": <0.0-1.0>,
  "indicators": ["<building_footprint_loss|vegetation_loss|debris|flooding|burn_scar|structural_collapse>"],
  "notes": "<one sentence describing the change>"
}}"""

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": pre_b64}},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": post_b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    }

    response = aws_session.bedrock_runtime().invoke_model(
        modelId=config.CLAUDE_MODEL_ID,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    text = json.loads(response["body"].read())["content"][0]["text"]
    try:
        return json.loads(text)
    except Exception:
        import re
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(match.group()) if match else {"damage_class": "unknown", "confidence": 0.0}


def _fallback_tiles(event_id: str, lat: float, lon: float) -> dict:
    """Return empty tile dict when Maxar catalog doesn't have the event."""
    return {"event_id": event_id, "lat": lat, "lon": lon,
            "pre_url": None, "post_url": None, "pre_b64": None, "post_b64": None}
