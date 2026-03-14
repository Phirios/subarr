from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Critical - env only
    redis_url: str = "redis://127.0.0.1:6379/0"
    gemini_api_key: str
    pyannote_auth_token: str
    tmdb_api_key: str | None = None
    log_level: str = "info"
    post_id_merge_threshold: float = 0.75
    device: str = "auto"  # auto, cuda, mps, cpu

    # MinIO/S3 storage
    s3_endpoint: str = "http://minio-service.phirios.svc.cluster.local:9000"
    s3_access_key: str = "phirios"
    s3_secret_key: str = "changeme"
    s3_bucket: str = "subarr"

    model_config = {"env_prefix": "SUBARR_", "env_file": ".env"}


settings = Settings()
