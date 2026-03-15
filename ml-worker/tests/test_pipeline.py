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


def _mock_redis():
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
    return mock_redis


def test_pipeline_updates_status(mock_settings, sample_task):
    """Pipeline should update job status at each stage."""
    mock_redis = _mock_redis()

    with patch("pipeline.Diarizer") as MockDiarizer, \
         patch("pipeline.EmotionDetector") as MockEmotion, \
         patch("pipeline.Translator") as MockTranslator, \
         patch("pipeline.TMDBClient") as MockTMDB, \
         patch("pipeline.CharacterIdentifier") as MockCharId:

        MockDiarizer.return_value.process.return_value = {
            "segments": [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}],
            "embeddings": None,
        }
        # EmotionDetector.process now receives mapped_subtitles and returns them with emotion
        MockEmotion.return_value.process.side_effect = lambda audio, subs: [
            {**s, "emotion": "neutral"} for s in subs
        ]
        MockTranslator.return_value.translate.return_value = [
            {"start_ms": 1000, "end_ms": 4000, "text": "Merhaba", "speaker": "SPEAKER_00", "emotion": "neutral", "color": "FFFFFF"}
        ]
        MockTMDB.return_value.get_characters.return_value = []
        MockCharId.return_value.identify.return_value = {}

        from pipeline import Pipeline
        pipe = Pipeline(mock_settings)
        result = pipe.process(sample_task, mock_redis)

        assert "output_path" in result
        # Status updates: diarizing, mapping_speakers, detecting_emotion, translating, formatting
        assert mock_redis.set.call_count >= 5


def test_pipeline_writes_output(mock_settings, sample_task):
    """Pipeline should write result JSON to output path."""
    mock_redis = _mock_redis()

    translated_data = [
        {"start_ms": 1000, "end_ms": 4000, "text": "Merhaba, nasılsın?", "speaker": "SPEAKER_00", "emotion": "neutral", "color": "FFFFFF"},
    ]

    with patch("pipeline.Diarizer") as MockDiarizer, \
         patch("pipeline.EmotionDetector") as MockEmotion, \
         patch("pipeline.Translator") as MockTranslator, \
         patch("pipeline.TMDBClient") as MockTMDB, \
         patch("pipeline.CharacterIdentifier") as MockCharId:

        MockDiarizer.return_value.process.return_value = {
            "segments": [],
            "embeddings": None,
        }
        MockEmotion.return_value.process.side_effect = lambda audio, subs: [
            {**s, "emotion": "neutral"} for s in subs
        ]
        MockTranslator.return_value.translate.return_value = translated_data
        MockTMDB.return_value.get_characters.return_value = []
        MockCharId.return_value.identify.return_value = {}

        from pipeline import Pipeline
        pipe = Pipeline(mock_settings)
        result = pipe.process(sample_task, mock_redis)

        with open(result["output_path"]) as f:
            output = json.load(f)
        assert len(output) == 1
        assert output[0]["text"] == "Merhaba, nasılsın?"


def test_pipeline_with_character_identification(monkeypatch, sample_task):
    """Pipeline should use character names when TMDB cast is available."""
    monkeypatch.setenv("SUBARR_GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("SUBARR_PYANNOTE_AUTH_TOKEN", "hf-test")
    monkeypatch.setenv("SUBARR_TMDB_API_KEY", "tmdb-test-key")
    from config import Settings
    settings_with_tmdb = Settings()

    mock_redis = _mock_redis()

    with patch("pipeline.Diarizer") as MockDiarizer, \
         patch("pipeline.EmotionDetector") as MockEmotion, \
         patch("pipeline.Translator") as MockTranslator, \
         patch("pipeline.TMDBClient") as MockTMDB, \
         patch("pipeline.CharacterIdentifier") as MockCharId:

        MockDiarizer.return_value.process.return_value = {
            "segments": [{"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"}],
            "embeddings": None,
        }
        MockEmotion.return_value.process.side_effect = lambda audio, subs: [
            {**s, "emotion": "neutral"} for s in subs
        ]
        MockTranslator.return_value.translate.return_value = [
            {"start_ms": 1000, "end_ms": 4000, "text": "Merhaba", "speaker": "SPEAKER_00", "emotion": "neutral", "color": "FFFFFF"}
        ]
        MockTMDB.return_value.get_characters.return_value = [
            {"actor": "Actor A", "character": "Shizuka"}
        ]
        MockCharId.return_value.identify.return_value = {"SPEAKER_00": "Shizuka"}

        from pipeline import Pipeline
        pipe = Pipeline(settings_with_tmdb)
        pipe.process(sample_task, mock_redis)

        # Translator should have been called with character_map
        translate_call = MockTranslator.return_value.translate.call_args
        assert translate_call.kwargs.get("character_map") == {"SPEAKER_00": "Shizuka"}


def test_pipeline_graceful_no_tmdb_key(monkeypatch, sample_task):
    """Pipeline should work without TMDB API key — skip character ID steps."""
    monkeypatch.setenv("SUBARR_GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("SUBARR_PYANNOTE_AUTH_TOKEN", "hf-test")
    monkeypatch.delenv("SUBARR_TMDB_API_KEY", raising=False)
    from config import Settings
    settings = Settings()

    mock_redis = _mock_redis()

    with patch("pipeline.Diarizer") as MockDiarizer, \
         patch("pipeline.EmotionDetector") as MockEmotion, \
         patch("pipeline.Translator") as MockTranslator, \
         patch("pipeline.CharacterIdentifier") as MockCharId:

        MockDiarizer.return_value.process.return_value = {
            "segments": [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}],
            "embeddings": None,
        }
        MockEmotion.return_value.process.side_effect = lambda audio, subs: [
            {**s, "emotion": "neutral"} for s in subs
        ]
        MockTranslator.return_value.translate.return_value = [
            {"start_ms": 1000, "end_ms": 4000, "text": "Merhaba", "speaker": "SPEAKER_00", "emotion": "neutral", "color": "FFFFFF"}
        ]

        from pipeline import Pipeline
        pipe = Pipeline(settings)
        result = pipe.process(sample_task, mock_redis)

        assert "output_path" in result
        # CharacterIdentifier.identify should NOT have been called (no cast)
        MockCharId.return_value.identify.assert_not_called()
        # Translator called with character_map=None
        translate_call = MockTranslator.return_value.translate.call_args
        assert translate_call.kwargs.get("character_map") is None
