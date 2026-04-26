"""
Marengo 3.0 embedding pipeline via Bedrock async invoke.
Input:  S3 URI of an MP4
Output: list of {segment_start, segment_end, embedding, s3_uri}
"""
import json, time, uuid
import config
from utils import aws_session

POLL_INTERVAL = 10  # seconds between status checks


def embed_video(s3_key: str, event_id: str) -> list[dict]:
    """
    Submit a video for Marengo embedding and poll until complete.
    Returns segment-level embedding records.
    """
    client = aws_session.bedrock_runtime()
    s3_uri = f"s3://{config.S3_BUCKET}/{s3_key}"
    output_prefix = f"s3://{config.S3_BUCKET}/{config.S3_EMBEDDINGS_PREFIX}/{event_id}/"

    print(f"[Marengo] Submitting: {s3_uri}")
    import boto3
    account_id = aws_session.get_session().client("sts").get_caller_identity()["Account"]

    response = client.start_async_invoke(
        modelId=config.MARENGO_MODEL_ID,
        modelInput={
            "inputType": "video",
            "video": {
                "mediaSource": {
                    "s3Location": {"uri": s3_uri, "bucketOwner": account_id}
                }
            },
        },
        outputDataConfig={"s3OutputDataConfig": {"s3Uri": output_prefix}},
    )
    invocation_arn = response["invocationArn"]
    print(f"[Marengo] Job ARN: {invocation_arn}")

    return _poll_and_parse(invocation_arn, output_prefix, s3_key)


def _poll_and_parse(invocation_arn: str, output_prefix: str, source_key: str) -> list[dict]:
    client = aws_session.bedrock_runtime()

    while True:
        status_resp = client.get_async_invoke(invocationArn=invocation_arn)
        status = status_resp["status"]
        print(f"[Marengo] Status: {status}")

        if status == "Completed":
            break
        if status in ("Failed", "Expired"):
            raise RuntimeError(f"Marengo job {status}: {status_resp.get('failureMessage')}")

        time.sleep(POLL_INTERVAL)

    return _load_embeddings(output_prefix, source_key)


def _load_embeddings(output_prefix: str, source_key: str) -> list[dict]:
    s3_client = aws_session.s3()
    prefix = output_prefix.replace(f"s3://{config.S3_BUCKET}/", "")

    resp = s3_client.list_objects_v2(Bucket=config.S3_BUCKET, Prefix=prefix)
    output_files = [o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith("output.json")]

    segments = []
    for key in output_files:
        obj = s3_client.get_object(Bucket=config.S3_BUCKET, Key=key)
        raw = json.loads(obj["Body"].read())
        # Real output format: {"data": [{embedding, embeddingOption, embeddingScope, startSec, endSec}]}
        for seg in raw.get("data", []):
            if seg.get("embeddingScope") != "clip":
                continue
            if seg.get("embeddingOption") != "visual":
                continue
            segments.append({
                "source_key": source_key,
                "segment_start": seg.get("startSec", 0),
                "segment_end": seg.get("endSec", 6),
                "embedding": seg.get("embedding", []),
                "embedding_key": key,
            })

    print(f"[Marengo] Loaded {len(segments)} visual clip segments from {source_key}")
    return segments
