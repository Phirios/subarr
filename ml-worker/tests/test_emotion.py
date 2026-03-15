import sys
import types
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# Provide a fake soundfile module so emotion.py can import it at runtime
_fake_sf = types.ModuleType("soundfile")
_fake_sf.read = MagicMock()
sys.modules.setdefault("soundfile", _fake_sf)


def _patch_sf_read(fake_audio, sample_rate):
    """Helper to patch soundfile.read with fake audio data."""
    return patch.dict(sys.modules, {"soundfile": _fake_sf}), \
           patch.object(_fake_sf, "read", return_value=(fake_audio, sample_rate))


def test_emotion_detector_returns_enriched_subtitles():
    from emotion import EmotionDetector

    detector = EmotionDetector()
    detector._model = MagicMock(return_value=[{"label": "happy", "score": 0.9}])

    mapped = [
        {"start_ms": 0, "end_ms": 2000, "text": "Hello", "speaker": "SPEAKER_00"},
        {"start_ms": 3000, "end_ms": 5000, "text": "Hi there", "speaker": "SPEAKER_01"},
    ]

    fake_audio = np.zeros(16000 * 6, dtype=np.float32)
    _fake_sf.read = MagicMock(return_value=(fake_audio, 16000))

    result = detector.process("dummy.wav", mapped)

    assert len(result) == 2
    assert all("emotion" in r for r in result)
    assert result[0]["emotion"] == "happy"
    assert result[0]["text"] == "Hello"
    assert result[0]["speaker"] == "SPEAKER_00"


def test_emotion_detector_empty_subtitles():
    from emotion import EmotionDetector

    detector = EmotionDetector()
    detector._model = MagicMock()

    fake_audio = np.zeros(16000, dtype=np.float32)
    _fake_sf.read = MagicMock(return_value=(fake_audio, 16000))

    result = detector.process("dummy.wav", [])
    assert result == []


def test_emotion_detector_short_audio_gets_neutral():
    from emotion import EmotionDetector

    detector = EmotionDetector()
    detector._model = MagicMock(return_value=[{"label": "happy", "score": 0.9}])

    mapped = [{"start_ms": 0, "end_ms": 200, "text": "Hey", "speaker": "SPEAKER_00"}]

    fake_audio = np.zeros(16000, dtype=np.float32)
    _fake_sf.read = MagicMock(return_value=(fake_audio, 16000))

    result = detector.process("dummy.wav", mapped)

    assert result[0]["emotion"] == "neutral"
    detector._model.assert_not_called()


def test_emotion_detector_preserves_subtitle_data():
    from emotion import EmotionDetector

    detector = EmotionDetector()
    detector._model = MagicMock(return_value=[{"label": "sad", "score": 0.8}])

    mapped = [{"start_ms": 1000, "end_ms": 3500, "text": "Goodbye", "speaker": "SPEAKER_00"}]

    fake_audio = np.zeros(16000 * 5, dtype=np.float32)
    _fake_sf.read = MagicMock(return_value=(fake_audio, 16000))

    result = detector.process("dummy.wav", mapped)

    assert result[0]["start_ms"] == 1000
    assert result[0]["end_ms"] == 3500
    assert result[0]["text"] == "Goodbye"
    assert result[0]["speaker"] == "SPEAKER_00"
    assert result[0]["emotion"] == "sad"


def test_emotion_detector_handles_stereo_audio():
    from emotion import EmotionDetector

    detector = EmotionDetector()
    detector._model = MagicMock(return_value=[{"label": "angry", "score": 0.7}])

    mapped = [{"start_ms": 0, "end_ms": 2000, "text": "Stop!", "speaker": "SPEAKER_00"}]

    fake_audio = np.zeros((16000 * 3, 2), dtype=np.float32)
    _fake_sf.read = MagicMock(return_value=(fake_audio, 16000))

    result = detector.process("dummy.wav", mapped)
    assert result[0]["emotion"] == "angry"


def test_emotion_detector_handles_model_error():
    from emotion import EmotionDetector

    detector = EmotionDetector()
    detector._model = MagicMock(side_effect=RuntimeError("Model error"))

    mapped = [{"start_ms": 0, "end_ms": 2000, "text": "Test", "speaker": "SPEAKER_00"}]

    fake_audio = np.zeros(16000 * 3, dtype=np.float32)
    _fake_sf.read = MagicMock(return_value=(fake_audio, 16000))

    result = detector.process("dummy.wav", mapped)
    assert result[0]["emotion"] == "neutral"
