"""
Pegasus 1.2 video-to-text pipeline via Bedrock sync invoke.
Input:  S3 URI of a short clip (or full video — Pegasus handles chunking)
Output: structured damage description dict
"""
import json, re, base64
import config
from utils import aws_session

DAMAGE_PROMPT = ('Analyze this disaster footage. Return ONLY valid JSON: '
    '{"structures_visible": <int>, "damage_severity": "<none|minor|moderate|severe|destroyed>", '
    '"damage_indicators": ["<roof_loss|wall_collapse|flooding|fire_damage|debris_field|foundation_failure|burnt_structure|ash_field>"], '
    '"infrastructure": {"roads": "<passable|blocked|destroyed|unknown>", "utilities": "<intact|damaged|destroyed|unknown>"}, '
    '"vegetation": "<intact|scorched|destroyed|unknown>", '
    '"confidence": <0.0-1.0>, "description": "<one sentence summary of damage visible>"}')


def _parse_response(raw: dict) -> dict:
    # Pegasus returns {"message": "<json string>", "stopReason": "stop"}
    text = raw.get("message") or raw.get("content", [{}])[0].get("text", "{}")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(match.group()) if match else {"raw": text, "confidence": 0.0}


def describe_video(s3_key: str) -> dict:
    """Run Pegasus on a full video S3 key. Returns structured damage dict."""
    s3_uri = f"s3://{config.S3_BUCKET}/{s3_key}"
    account_id = aws_session.get_session().client("sts").get_caller_identity()["Account"]
    print(f"[Pegasus] Describing: {s3_uri}")

    body = {
        "inputPrompt": DAMAGE_PROMPT,
        "mediaSource": {
            "s3Location": {"uri": s3_uri, "bucketOwner": account_id}
        },
    }
    response = aws_session.bedrock_runtime().invoke_model(
        modelId=config.PEGASUS_MODEL_ID,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    result = _parse_response(json.loads(response["body"].read()))
    result["source_key"] = s3_key
    print(f"[Pegasus] Severity={result.get('damage_severity')}  Confidence={result.get('confidence')}")
    return result
