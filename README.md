# DisasterLens

Multimodal damage assessment for disaster response, powered by Amazon Bedrock with TwelveLabs Marengo 3.0, TwelveLabs Pegasus 1.2, and Anthropic Claude Sonnet.

Built for the Geospatial Video Intelligence Hackathon, Track 03 (Multimodal Geospatial Workloads), April 2026.

> Live application: deployed on Streamlit Community Cloud. The deployed URL is added to the repository description once provisioning completes.

## The Problem in One Paragraph

When a major disaster strikes the United States, the gating activity that decides how quickly federal money reaches survivors is the Preliminary Damage Assessment, often shortened to PDA. State, local, tribal, and territorial emergency managers must inventory damaged structures, classify each one by severity, and produce a defensible report that triggers the Stafford Act process for a Presidential disaster declaration. Today this work is done largely by hand. Field teams drive through neighborhoods, drone pilots upload footage to laptops, and analysts manually correlate three or four data sources to produce a spreadsheet. Every hour spent on this assessment is an hour that survivors are not receiving Individual Assistance or Public Assistance funds.

## Why This Matters Right Now

Two recent disasters demonstrate the timeline problem.

**Hurricane Milton** made landfall near Siesta Key, Florida on October 9, 2024 as a Category 3 storm. The Presidential major disaster declaration came two days later on October 11, 2024 (FEMA DR-4834-FL). However, the structural damage inventory needed for full Individual Assistance distribution continued for several weeks. Initial damage estimates exceeded 34 billion United States dollars, with roughly 35 confirmed fatalities. (Source: FEMA disaster declaration page DR-4834-FL; NOAA National Centers for Environmental Information event report.)

**The Palisades Fire** ignited on January 7, 2025 in the Pacific Palisades neighborhood of Los Angeles. The federal major disaster declaration arrived on January 8, 2025 (FEMA DR-4856-CA), but the damage assessment process to confirm the destruction of more than 6,800 structures took multiple weeks of joint county, state, and federal field surveys. (Source: FEMA disaster declaration page DR-4856-CA; CalFire incident report.)

In both cases, the speed of the initial declaration was acceptable. The pain point sits one step later: producing the granular damage inventory that survivors and case workers actually need. That work consistently takes days to weeks.

## Current Gaps

Three gaps repeat across every major disaster.

1. **Field teams cannot cover the whole impact zone safely.** In wildfires, active embers and unstable structures keep inspectors at a distance. In floods, washed out roads cut off entire neighborhoods.
2. **Multiple data sources never get cross checked in one place.** Drone footage, satellite imagery, and field reports are reviewed by different teams on different timelines. Discrepancies between sources are noticed days later, if at all.
3. **Unreported damage is the silent failure mode.** If a property is missed by every field team, it does not appear in the federal package, and the homeowner may not receive timely assistance.

## What DisasterLens Does

DisasterLens ingests four data sources and produces one validated damage inventory.

| Source | What it provides |
|---|---|
| Drone or aerial video (MP4 in S3) | Visual evidence of structural damage |
| Satellite tile pairs (Esri Wayback historical and Esri World Imagery current) | Pre versus post change detection |
| Field reports (JSON from ground teams) | Ground truth from human inspectors |
| County parcel records (CSV) | Property identity and assessed value |

The system fuses these into three categories.

| Category | Meaning |
|---|---|
| Validated | Confirmed by two or more sources with confidence at or above 0.70 |
| Unreported | Detected by AI in video or satellite imagery, missing from every field report |
| Conflict | Two sources disagree on severity by two or more levels |

The unreported and conflict categories are the key intelligence outputs. They tell emergency managers exactly which addresses need a follow up visit before the federal package gets submitted.

## Foundation Models

All models run inside Amazon Bedrock in the us-east-1 region. No personal API keys are used. Inference stays inside the AWS account boundary.

| Model | Bedrock model identifier | Purpose |
|---|---|---|
| TwelveLabs Marengo 3.0 | twelvelabs.marengo-embed-3-0-v1:0 | Generates 512 dimensional embeddings for every roughly 6 second visual clip in a video. Used for semantic search, similarity, and downstream retrieval. |
| TwelveLabs Pegasus 1.2 | twelvelabs.pegasus-1-2-v1:0 | Watches the full video and returns structured damage JSON: severity level, damage indicators, infrastructure status, and a one sentence description. |
| Anthropic Claude Sonnet 4.6 | us.anthropic.claude-sonnet-4-6 | Performs three jobs: GPS location inference from video frames using chain of thought visual reasoning, pre versus post satellite tile comparison, and FEMA style executive summary generation. |

Why this combination works: Marengo provides retrieval, Pegasus provides description, and Claude provides judgement. Each model handles the task it was built for, instead of forcing one large language model to do everything.

## Architecture

```
DATA SOURCES
  Drone Video (S3 MP4)   Satellite Tiles   Field Reports   Parcels
        |                      |                |             |
   [Marengo 3.0 Async]         |          [Field reports]    |
   [Pegasus 1.2 Sync]   [Claude Vision]   (JSON ingest)   [Parcel CSV]
   [Claude Vision]      [pre/post diff]
   geo-locate frames    damage_class
        |                      |                |             |
        +----------------------+----------------+-------------+
                               |
                    FUSION ENGINE (pipeline/fusion.py)
                    - Spatial join, RapidFuzz address match
                    - Confidence scoring 0.0 to 1.0
                    - Weights: Video 0.40 + Satellite 0.30
                              + Report 0.20 + Parcel 0.10
                               |
            +------------------+------------------+
       VALIDATED          UNREPORTED          CONFLICTS
                               |
                    Claude Sonnet FEMA brief
                    GeoJSON and CSV export
                    Streamlit interactive map
```

## Tech Stack

* Python 3.10 or newer
* Streamlit for the interactive application
* Amazon Bedrock for all model inference
* boto3 for AWS access
* Folium and streamlit folium for the damage map
* Pandas for data manipulation
* RapidFuzz for fuzzy address matching in fusion
* Pillow for image handling
* Esri ArcGIS World Imagery and Esri Wayback for satellite tiles

## Local Setup

```bash
git clone https://github.com/DHARAN656/DiasterLens.git
cd DiasterLens

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Fill in your AWS credentials in .streamlit/secrets.toml

streamlit run app/streamlit_app.py
```

Open the local URL printed in the terminal (typically http://localhost:8501).

## Deployment to Streamlit Community Cloud

1. Visit https://share.streamlit.io and sign in with GitHub.
2. Create a new app pointing at this repository, branch main, main file path `app/streamlit_app.py`.
3. Open Advanced settings and paste your AWS credentials and S3 bucket name into the Secrets text box using the format shown in `.streamlit/secrets.toml.example`.
4. Click Deploy.

The pre computed result files (pegasus_results.json, geo_results.json, satellite_results.json, fusion_results.json, frame previews) are checked into the repository. The application reads them directly, so the demo continues to render maps, fusion tables, and frame grids even if AWS credentials expire. Only video playback, the S3 storage browser, and FEMA summary generation require live AWS access.

## Repository Structure

```
DiasterLens/
  app/
    streamlit_app.py          Main Streamlit application
  pipeline/
    marengo.py                Marengo 3.0 embedding pipeline
    pegasus.py                Pegasus 1.2 video to text pipeline
    geo_locate.py             Claude Vision geo location inference
    fusion.py                 Fusion engine and FEMA summary generator
    mock_data.py              Field report and parcel CSV generators
    maxar.py                  Maxar Open Data STAC client (alternative satellite source)
  utils/
    aws_session.py            Boto3 session helper
  static/
    frames/                   Pre extracted video frame previews (base64 JPEG)
  config.py                   Event registry and Bedrock model identifiers
  run_satellite_compare.py    One off script that ran satellite comparisons
  run_mock_and_fusion.py      One off script that generated reports and ran fusion
  pegasus_results.json        Cached Pegasus output per video
  geo_results.json            Cached geo location output per video
  satellite_results.json      Cached satellite comparison output
  fusion_results.json         Final fusion output for both events
  field_reports_*.json        Mock field reports for both events
  parcels_*.csv               Mock parcel datasets for both events
  HACKATHON_REPORT.md         Detailed project report (separate file)
```

## Detailed Project Report

For the deeper write up with use case scenarios, validation metrics, and the full mission impact analysis, see `HACKATHON_REPORT.md` in this repository.

## License

This project was built for educational and hackathon demonstration purposes. The TwelveLabs and Anthropic model usage is governed by their respective Amazon Bedrock terms of service.

## Acknowledgements

* TwelveLabs for the Marengo and Pegasus video understanding models
* Anthropic for Claude Sonnet
* Amazon Web Services for the Bedrock platform and the workshop environment
* FEMA for publishing the 2025 Preliminary Damage Assessment Guide that informed the system requirements
