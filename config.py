from dotenv import load_dotenv
import os

load_dotenv()


def _get(key, default=""):
    """Read from Streamlit secrets first (Cloud), fall back to env vars (local)."""
    try:
        import streamlit as st
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)


# AWS
AWS_REGION        = _get("AWS_DEFAULT_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = _get("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY    = _get("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = _get("AWS_SESSION_TOKEN")

# S3
S3_BUCKET            = _get("S3_BUCKET", "twelvelabs-bedrock-workshop-workshopbucket-f7a0zzoflpzw")
S3_REPORTS_PREFIX    = "hackathon/reports"
S3_PARCELS_PREFIX    = "hackathon/parcels"
S3_EMBEDDINGS_PREFIX = "hackathon/embeddings"
S3_OUTPUTS_PREFIX    = "hackathon/outputs"

# Bedrock model IDs
MARENGO_MODEL_ID = _get("MARENGO_MODEL_ID", "twelvelabs.marengo-embed-3-0-v1:0")
PEGASUS_MODEL_ID = _get("PEGASUS_MODEL_ID", "twelvelabs.pegasus-1-2-v1:0")
CLAUDE_MODEL_ID  = _get("CLAUDE_MODEL_ID",  "us.anthropic.claude-sonnet-4-6")

# Disaster events
EVENTS = {
    "hurricane_milton": {
        "name": "Hurricane Milton",
        "date": "2024-10-09",
        "s3_videos": [
            "hurricane_milton/zJyDF8_NHcs.mp4",
        ],
        "bbox": {"west": -82.70, "south": 27.30, "east": -82.10, "north": 27.80},
        "center": {"lat": 27.55, "lon": -82.45},
        "state": "FL",
        "color": "#e74c3c",
    },
    "palisades_wildfire": {
        "name": "Palisades Wildfire",
        "date": "2025-01-07",
        "s3_videos": [
            "palisades/palisades_fox11_destroyed.mp4",
            "palisades/palisades_burn_zone_devastation.mp4",
            "palisades/palisades_drone_devastation.mp4",
            "palisades/palisades_malibu_desolation.mp4",
            "palisades/palisades_malibu_drone.mp4",
        ],
        "bbox": {"west": -118.85, "south": 33.98, "east": -118.45, "north": 34.12},
        "center": {"lat": 34.05, "lon": -118.65},
        "state": "CA",
        "color": "#e67e22",
    },
}

# Maxar Open Data STAC
MAXAR_STAC_BASE = "https://maxar-opendata.s3.amazonaws.com/events"
MAXAR_EVENTS = {
    "hurricane_milton":  "Hurricane-Milton-Oct-2024",
    "palisades_wildfire": "Palisades-and-Eaton-fires-Jan-2025",
}

# Fusion scoring weights
FUSION_WEIGHTS = {
    "video_damage":     0.40,
    "satellite_change": 0.30,
    "field_report":     0.20,
    "parcel_exists":    0.10,
}
VALIDATED_THRESHOLD   = 0.70
CONFLICT_SEVERITY_GAP = 2
