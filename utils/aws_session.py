import boto3
import config

_session = None

def get_session() -> boto3.Session:
    global _session
    if _session is None:
        _session = boto3.Session(
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_KEY,
            aws_session_token=config.AWS_SESSION_TOKEN,
            region_name=config.AWS_REGION,
        )
    return _session

def s3():
    return get_session().client("s3")

def bedrock_runtime():
    return get_session().client("bedrock-runtime", region_name=config.AWS_REGION)

def bedrock():
    return get_session().client("bedrock", region_name=config.AWS_REGION)
