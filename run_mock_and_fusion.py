"""Step 5+6: Generate mock field reports/parcels and run fusion engine."""
import json, sys, io
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pipeline.mock_data import generate_field_reports, generate_parcel_csv, save_to_s3
from pipeline.fusion import run_fusion, generate_executive_summary

# Load upstream results
with open("pegasus_results.json") as f:
    pegasus_results = json.load(f)
with open("geo_results.json") as f:
    geo_results = json.load(f)
with open("satellite_results.json") as f:
    satellite_results = json.load(f)

all_fusion = {}

for event_id in ["hurricane_milton", "palisades_wildfire"]:
    print(f"\n{'='*60}")
    print(f"EVENT: {event_id}")
    print('='*60)

    # Step 5: Generate mock data
    reports = generate_field_reports(event_id, num_reports=3)
    parcel_csv = generate_parcel_csv(event_id, num_rows=60)
    save_to_s3(event_id, reports, parcel_csv)

    # Save locally
    with open(f"field_reports_{event_id}.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)
    with open(f"parcels_{event_id}.csv", "w", encoding="utf-8") as f:
        f.write(parcel_csv)

    # Parse parcel CSV → DataFrame
    try:
        parcel_df = pd.read_csv(io.StringIO(parcel_csv))
    except Exception as e:
        print(f"[Warning] parcel CSV parse failed: {e}")
        parcel_df = pd.DataFrame()

    # Filter upstream results for this event (as lists)
    peg_ev = pegasus_results.get(event_id, [])
    geo_ev = geo_results.get(event_id, [])
    # satellite_results values as list, with lat/lon injected from their stored keys
    sat_ev = []
    for k, v in satellite_results.items():
        if v.get("event_id") == event_id:
            sat_ev.append({**v.get("comparison", {}), "lat": v["lat"], "lon": v["lon"], "label": v["label"]})

    # Step 6: Run fusion
    fusion_result = run_fusion(peg_ev, geo_ev, sat_ev, reports, parcel_df)
    print(f"\n[Fusion] Validated incidents: {len(fusion_result.get('validated', []))}")
    print(f"[Fusion] Unreported damage:    {len(fusion_result.get('unreported', []))}")
    print(f"[Fusion] Conflicts detected:   {len(fusion_result.get('conflicts', []))}")

    # Executive summary
    summary = generate_executive_summary(fusion_result, event_id)
    fusion_result["executive_summary"] = summary
    print(f"\n[Summary excerpt] {summary[:400]}...")

    all_fusion[event_id] = fusion_result

# Save combined fusion output
with open("fusion_results.json", "w", encoding="utf-8") as f:
    json.dump(all_fusion, f, indent=2)
print("\nSaved fusion_results.json")
