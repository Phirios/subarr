import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_settings(monkeypatch):
    monkeypatch.setenv("SUBARR_GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("SUBARR_PYANNOTE_AUTH_TOKEN", "hf-test")
    from config import Settings
    return Settings()


@pytest.fixture
def sample_srt():
    return """1
00:00:01,000 --> 00:00:04,000
Hello, how are you?

2
00:00:05,000 --> 00:00:08,000
I'm fine, thanks!
"""


@pytest.fixture
def sample_task(sample_srt):
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.wav")
        subtitle_path = os.path.join(tmpdir, "subtitle.srt")

        # Create dummy files
        with open(audio_path, "wb") as f:
            f.write(b"\x00" * 100)
        with open(subtitle_path, "w") as f:
            f.write(sample_srt)

        yield {
            "job_id": "test-job-1",
            "audio_path": audio_path,
            "subtitle_path": subtitle_path,
            "target_language": "tr",
            "metadata": {"title": "Test Show", "tmdb_id": 12345},
        }


def test_pipeline_updates_status(mock_settings, sample_task):
    """Pipeline should update job status at each stage."""
    mock_redis = MagicMock()
    mock_redis.get.return_value = json.dumps({
        "id": "test-job-1",
        "status": "queued",
        "target_language": "tr",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "error": None,
        "result_path": None,
        "metadata": None,
    })

    with patch("pipeline.Diarizer") as MockDiarizer, \
         patch("pipeline.EmotionDetector") as MockEmotion, \
         patch("pipeline.Translator") as MockTranslator:

        MockDiarizer.return_value.process.return_value = [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}
        ]
        MockEmotion.return_value.process.return_value = [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00", "emotion": "neutral"}
        ]
        MockTranslator.return_value.translate.return_value = [
            {"start_ms": 1000, "end_ms": 4000, "text": "Merhaba", "speaker": "SPEAKER_00", "emotion": "neutral", "color": "FFFFFF"}
        ]

        from pipeline import Pipeline
        pipe = Pipeline(mock_settings)
        result = pipe.process(sample_task, mock_redis)

        assert "output_path" in result
        # Should have called redis.set for status updates (diarizing, detecting_emotion, translating, formatting)
        assert mock_redis.set.call_count >= 4


def test_pipeline_writes_output(mock_settings, sample_task):
    """Pipeline should write result JSON to output path."""
    mock_redis = MagicMock()
    mock_redis.get.return_value = json.dumps({
        "id": "test-job-1",
        "status": "queued",
        "target_language": "tr",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "error": None,
        "result_path": None,
        "metadata": None,
    })

    translated_data = [
        {"start_ms": 1000, "end_ms": 4000, "text": "Merhaba, nasılsın?", "speaker": "SPEAKER_00", "emotion": "neutral", "color": "FFFFFF"},
    ]

    with patch("pipeline.Diarizer") as MockDiarizer, \
         patch("pipeline.EmotionDetector") as MockEmotion, \
         patch("pipeline.Translator") as MockTranslator:

        MockDiarizer.return_value.process.return_value = []
        MockEmotion.return_value.process.return_value = []
        MockTranslator.return_value.translate.return_value = translated_data

        from pipeline import Pipeline
        pipe = Pipeline(mock_settings)
        result = pipe.process(sample_task, mock_redis)

        with open(result["output_path"]) as f:
            output = json.load(f)
        assert len(output) == 1
        assert output[0]["text"] == "Merhaba, nasılsın?"
