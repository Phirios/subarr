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

    model_config = {"env_prefix": "SUBARR_", "env_file": ".env"}


settings = Settings()
