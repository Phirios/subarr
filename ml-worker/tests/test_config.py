import os
import pytest


def test_config_loads_from_env(monkeypatch):
    monkeypatch.setenv("SUBARR_GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("SUBARR_PYANNOTE_AUTH_TOKEN", "hf-test")
    monkeypatch.setenv("SUBARR_REDIS_URL", "redis://localhost:6379/1")

    # Re-import to pick up env
    from config import Settings
    s = Settings()
    assert s.gemini_api_key == "test-key"
    assert s.pyannote_auth_token == "hf-test"
    assert s.redis_url == "redis://localhost:6379/1"


def test_config_defaults(monkeypatch):
    monkeypatch.setenv("SUBARR_GEMINI_API_KEY", "key")
    monkeypatch.setenv("SUBARR_PYANNOTE_AUTH_TOKEN", "token")
    monkeypatch.delenv("SUBARR_REDIS_URL", raising=False)
    monkeypatch.delenv("SUBARR_TMDB_API_KEY", raising=False)

    from config import Settings
    s = Settings()
    assert s.redis_url == "redis://127.0.0.1:6379/0"
    assert s.tmdb_api_key is None


def test_config_missing_required_key(monkeypatch):
    monkeypatch.delenv("SUBARR_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("SUBARR_PYANNOTE_AUTH_TOKEN", raising=False)

    from config import Settings
    with pytest.raises(Exception):
        Settings()
