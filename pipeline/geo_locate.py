"""
Claude Vision geo-location inference — prompt-engineered for disaster footage.

Strategy:
- Extract 3 frames per video (early, mid, late) for triangulation
- Chain-of-thought prompt forces Claude to reason visually before committing to coords
- Event-specific anchor hints narrow the search space
- Pegasus description feeds as corroborating context
- Returns confidence-ranked candidates, picks best
"""
import json, base64, re, io
import config
from utils import aws_session
from PIL import Image

# Event-specific visual anchors Claude should look for
EVENT_ANCHORS = {
    "hurricane_milton": {
        "terrain": "flat coastal Florida, Gulf of Mexico side. Barrier islands (Siesta Key, Longboat Key, Anna Maria Island, Fort Myers Beach, Sanibel). Inland: Charlotte Harbor, Peace River delta.",
        "visual_cues": [
            "Gulf-facing beaches with white sand and clear shallow water",
            "Flat terrain, no hills or elevation changes",
            "Florida ranch-style and concrete block homes common",
            "Palm trees (sabal/cabbage palms are native), mangroves near water",
            "Street grid: east-west causeways connecting barrier islands to mainland",
            "Fort Myers Beach: Estero Boulevard runs the length of the island",
            "US-41 (Tamiami Trail) is the main inland corridor",
            "Mobile home parks concentrated in inland areas",
            "I-75 runs north-south inland",
            "Storm surge damage: floodwater line on structures, boats displaced inland",
        ],
        "known_landmarks": "Fort Myers Beach, Sanibel Causeway, Estero Boulevard, Cape Coral bridges, Charlotte Harbor, Punta Gorda, Englewood Beach, Venice Beach, Siesta Key, Sarasota Bay",
        "anti_cues": "No mountains, no cliffs, no canyon roads, no dry vegetation — if you see any of these it is NOT this event area",
    },
    "palisades_wildfire": {
        "terrain": "Santa Monica Mountains coastal edge. Pacific Palisades is between the mountains and Pacific Ocean. Pacific Coast Highway (PCH) runs along the coast. Topanga Canyon, Malibu Canyon nearby.",
        "visual_cues": [
            "Pacific Coast Highway (PCH / CA-1) — 4-lane divided highway right at beach edge",
            "Santa Monica Mountains steep hillsides — chaparral (now burnt), dramatic slopes",
            "Upscale California homes: stucco, tile roofs, hillside terracing, canyon lots",
            "Pacific Ocean visible to the south/southwest — deep blue, no barrier islands",
            "Malibu Pier visible in some shots (wooden pier extending into ocean)",
            "Topanga State Beach — wide sandy beach, PCH runs directly behind it",
            "Palisades neighborhood: grid streets on terraced hillside above PCH",
            "Carbon Beach / Malibu Colony — beachfront homes right on sand",
            "Getty Villa vicinity: Mediterranean-style architecture on bluff above PCH",
            "Fire damage pattern: white/grey ash, blackened hillsides, exposed foundations",
        ],
        "known_landmarks": "Pacific Coast Highway, Malibu Pier, Topanga Canyon Boulevard, Sunset Boulevard at PCH, Getty Villa, Carbon Beach, Las Virgenes Road, Kanan Dume Road, Zuma Beach, Point Dume",
        "anti_cues": "No flat terrain, no Gulf of Mexico, no mangroves, no causeways — if you see these it is NOT this event area",
    },
}

# Chain-of-thought geo-location prompt template
GEO_PROMPT_TEMPLATE = """You are a geospatial intelligence analyst specializing in disaster response.
Your task is to pinpoint the GPS location shown in this {frame_label} from disaster footage.

## Event Context
- **Disaster:** {event_name}
- **Date:** {event_date}
- **General area:** {terrain_description}
- **AI damage assessment of this video:** {pegasus_desc}

## Known visual signatures for this area
{visual_cues}

## Landmarks to watch for
{known_landmarks}

## IMPORTANT: What rules out this area
{anti_cues}

---

## Your task — reason step by step:

**Step 1 — Terrain & geography:**
Describe the terrain type, elevation, water bodies, vegetation. Does it match the event area?

**Step 2 — Built environment:**
Describe building styles, street layout, density. What does this suggest about the specific neighborhood?

**Step 3 — Distinctive features:**
List ANY identifiable features: street signs, landmarks, unique structures, coastline shape, road configurations.

**Step 4 — Cross-reference:**
Given steps 1-3 and the event context, which specific sub-location within the event area is most likely?

**Step 5 — Coordinate estimate:**
Based on your reasoning, provide your best GPS estimate.

---

Return ONLY valid JSON (no markdown, no extra text):
{{
  "reasoning": {{
    "terrain": "<what terrain/geography you observe>",
    "built_environment": "<building styles, street layout>",
    "distinctive_features": ["<list of identifiable features>"],
    "location_inference": "<which specific sub-area within the event zone>"
  }},
  "candidates": [
    {{
      "lat": <decimal degrees>,
      "lon": <decimal degrees>,
      "location_name": "<human-readable name e.g. Fort Myers Beach near Estero Blvd>",
      "confidence": <0.0-1.0>,
      "supporting_evidence": "<key visual cue that anchors this estimate>"
    }}
  ],
  "best_estimate": {{
    "lat": <decimal degrees>,
    "lon": <decimal degrees>,
    "confidence": <0.0-1.0>,
    "location_name": "<best guess location name>"
  }}
}}

If the frame shows no distinctive features (e.g. pure sky or interior), set confidence < 0.15 and use the event center coordinates."""


def geolocate_video(
    local_video_path: str,
    event_id: str,
    pegasus_result: dict,
    num_frames: int = 3,
) -> dict:
    """
    Extract num_frames from the video, run geo-location on each,
    triangulate, and return the best coordinate estimate.
    """
    event = config.EVENTS[event_id]
    anchors = EVENT_ANCHORS[event_id]

    # Extract frames at different timestamps for triangulation
    duration_s = _get_video_duration(local_video_path)
    timestamps = _pick_timestamps(duration_s, num_frames)

    print(f"[GeoLocate] {event['name']} | {local_video_path}")
    print(f"[GeoLocate] Extracting {num_frames} frames at {[f'{t:.0f}s' for t in timestamps]}")

    frame_results = []
    for i, ts in enumerate(timestamps):
        frame_bytes = extract_frame(local_video_path, ts)
        if frame_bytes is None:
            continue
        label = f"frame at {ts:.0f}s (position: {'early' if i==0 else 'middle' if i==1 else 'late'} in video)"
        result = _infer_single_frame(frame_bytes, event_id, pegasus_result, label, anchors, event)
        result["timestamp_s"] = ts
        frame_results.append(result)
        conf = result.get("best_estimate", {}).get("confidence", 0)
        loc = result.get("best_estimate", {}).get("location_name", "unknown")
        print(f"[GeoLocate]   Frame {i+1} @ {ts:.0f}s: {loc} (conf={conf:.2f})")

    if not frame_results:
        return _fallback(event_id, event)

    # Pick the highest-confidence frame result
    best = max(frame_results, key=lambda r: r.get("best_estimate", {}).get("confidence", 0))
    best_est = best.get("best_estimate", {})

    # Clamp to event bounding box
    bbox = event["bbox"]
    lat = max(bbox["south"], min(bbox["north"], best_est.get("lat", event["center"]["lat"])))
    lon = max(bbox["west"],  min(bbox["east"],  best_est.get("lon", event["center"]["lon"])))

    return {
        "event_id": event_id,
        "lat": lat,
        "lon": lon,
        "confidence": best_est.get("confidence", 0.1),
        "location_name": best_est.get("location_name", "unknown"),
        "reasoning": best.get("reasoning", {}),
        "all_frame_results": frame_results,
        "source_video": local_video_path,
    }


def _infer_single_frame(
    frame_bytes: bytes,
    event_id: str,
    pegasus_result: dict,
    frame_label: str,
    anchors: dict,
    event: dict,
) -> dict:
    b64 = _resize_and_encode(frame_bytes)
    pegasus_desc = (
        f"Severity={pegasus_result.get('damage_severity')}, "
        f"Indicators={pegasus_result.get('damage_indicators', [])}, "
        f"\"{pegasus_result.get('description', '')}\""
    )
    visual_cues_str = "\n".join(f"  - {c}" for c in anchors["visual_cues"])

    prompt = GEO_PROMPT_TEMPLATE.format(
        frame_label=frame_label,
        event_name=event["name"],
        event_date=event["date"],
        terrain_description=anchors["terrain"],
        pegasus_desc=pegasus_desc,
        visual_cues=visual_cues_str,
        known_landmarks=anchors["known_landmarks"],
        anti_cues=anchors["anti_cues"],
    )

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1500,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                },
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
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return {"best_estimate": {"lat": event["center"]["lat"], "lon": event["center"]["lon"], "confidence": 0.1, "location_name": "fallback"}}


def _pick_timestamps(duration_s: float, n: int) -> list[float]:
    """Pick n evenly-spaced timestamps, avoiding first/last 5% of video."""
    start = duration_s * 0.05
    end = duration_s * 0.95
    if n == 1:
        return [duration_s * 0.5]
    step = (end - start) / (n - 1)
    return [start + i * step for i in range(n)]


def _get_video_duration(video_path: str) -> float:
    try:
        import imageio
        reader = imageio.get_reader(video_path)
        meta = reader.get_meta_data()
        reader.close()
        fps = meta.get("fps", 30)
        nframes = meta.get("nframes", 0)
        duration = meta.get("duration", 0)
        if duration:
            return float(duration)
        if nframes and fps:
            return nframes / fps
    except Exception:
        pass
    return 60.0  # safe fallback


def extract_frame(video_path: str, time_sec: float = 30.0) -> bytes | None:
    """Extract a single JPEG frame from a local video file at time_sec."""
    try:
        import imageio
        reader = imageio.get_reader(video_path)
        fps = reader.get_meta_data().get("fps", 30)
        frame_idx = int(fps * time_sec)
        try:
            frame = reader.get_data(frame_idx)
        except Exception:
            frame = reader.get_data(0)
        reader.close()
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception as e:
        print(f"[GeoLocate] Frame extraction failed at {time_sec}s: {e}")
        return None


def _resize_and_encode(frame_bytes: bytes, max_dim: int = 1120) -> str:
    img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
    img.thumbnail((max_dim, max_dim))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _fallback(event_id: str, event: dict) -> dict:
    return {
        "event_id": event_id,
        "lat": event["center"]["lat"],
        "lon": event["center"]["lon"],
        "confidence": 0.05,
        "location_name": f"{event['name']} area (fallback)",
        "reasoning": {},
        "all_frame_results": [],
    }
