import json, re, sys, requests, math, io, base64, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from PIL import Image
import config
from utils import aws_session

WAYBACK_RELEASE = 10   # earliest available — shows intact pre-disaster neighborhoods


def latlon_to_tile(lat, lon, zoom):
    lat_r = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n)
    return x, y


def stitch(lat, lon, zoom, source, radius=1):
    x0, y0 = latlon_to_tile(lat, lon, zoom)
    size = (2 * radius + 1) * 256
    canvas = Image.new("RGB", (size, size))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            tx, ty = x0 + dx, y0 + dy
            if source == "current":
                url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{ty}/{tx}"
            else:
                url = (f"https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery"
                       f"/WMTS/1.0.0/default028mm/MapServer/tile/{WAYBACK_RELEASE}/{zoom}/{ty}/{tx}")
            try:
                r = requests.get(url, timeout=10, headers={"User-Agent": "DisasterLens/1.0"})
                if r.status_code == 200:
                    canvas.paste(Image.open(io.BytesIO(r.content)), ((dx + radius) * 256, (dy + radius) * 256))
            except Exception:
                pass
            time.sleep(0.03)
    return canvas


def to_b64(img, max_dim=768):
    img.thumbnail((max_dim, max_dim))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


PROMPT = (
    "You are a satellite imagery analyst for FEMA disaster assessment.\n"
    "Image 1 = PRE-event: intact neighborhood before disaster\n"
    "Image 2 = POST-event: same location after disaster\n\n"
    "Analyze visible changes: building footprints, vegetation, burn scars, flooding, debris.\n\n"
    "Return ONLY valid JSON:\n"
    '{"damage_class": "<none|minor|moderate|severe|destroyed>", '
    '"change_detected": <true/false>, '
    '"damage_area_pct": <0-100>, '
    '"confidence": <0.0-1.0>, '
    '"indicators": ["<building_footprint_loss|vegetation_loss|debris|flooding|burn_scar|structural_collapse>"], '
    '"pre_description": "<one sentence describing pre-event state>", '
    '"post_description": "<one sentence describing post-event state>", '
    '"notes": "<key observable difference>"}'
)


def compare(pre_b64, post_b64, event_id, label):
    event = config.EVENTS[event_id]
    context = f"Event: {event['name']} ({event['date']}). Location: {label}.\n\n"
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 800,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": pre_b64}},
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": post_b64}},
            {"type": "text", "text": context + PROMPT},
        ]}],
    }
    resp = aws_session.bedrock_runtime().invoke_model(
        modelId=config.CLAUDE_MODEL_ID,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    text = json.loads(resp["body"].read())["content"][0]["text"]
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(m.group()) if m else {"damage_class": "unknown", "confidence": 0.0, "notes": text[:200]}


locations = [
    ("hurricane_milton",   "Fort Myers Beach",   27.30,  -82.10),
    ("palisades_wildfire", "Sunset/Temescal",    34.043, -118.527),
    ("palisades_wildfire", "Malibu Colony PCH",  34.034, -118.685),
    ("palisades_wildfire", "Las Tunas Beach",    34.043, -118.612),
]

results = {}
for event_id, label, lat, lon in locations:
    print(f"Processing: {label}")
    pre_img  = stitch(lat, lon, zoom=16, source="wayback")
    post_img = stitch(lat, lon, zoom=16, source="current")
    pre_b64  = to_b64(pre_img)
    post_b64 = to_b64(post_img)

    result = compare(pre_b64, post_b64, event_id, label)
    key = f"{event_id}_{label.replace(' ', '_').replace('/', '_')}"
    results[key] = {
        "event_id": event_id, "label": label, "lat": lat, "lon": lon,
        "pre_b64": pre_b64, "post_b64": post_b64,
        "comparison": result,
    }
    print(f"  damage_class={result.get('damage_class')}  conf={result.get('confidence')}  area={result.get('damage_area_pct')}%")
    print(f"  pre : {result.get('pre_description', '')[:100]}")
    print(f"  post: {result.get('post_description', '')[:100]}")
    print(f"  notes: {str(result.get('notes', ''))[:120]}")
    print()

with open("satellite_results.json", "w") as f:
    json.dump(results, f, indent=2)

aws_session.s3().put_object(
    Bucket=config.S3_BUCKET,
    Key="hackathon/outputs/satellite_results.json",
    Body=json.dumps(results, indent=2).encode(),
    ContentType="application/json",
)
print("Saved satellite_results.json + S3")
