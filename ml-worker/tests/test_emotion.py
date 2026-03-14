import pytest
from unittest.mock import patch, MagicMock


def test_emotion_detector_returns_enriched_segments():
    from emotion import EmotionDetector

    detector = EmotionDetector()
    detector._model = MagicMock()  # Skip real model loading

    segments = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
        {"start": 3.0, "end": 5.0, "speaker": "SPEAKER_01"},
    ]

    result = detector.process("dummy_path.wav", segments)
    assert len(result) == 2
    assert all("emotion" in r for r in result)
    assert all(r["speaker"] == s["speaker"] for r, s in zip(result, segments))


def test_emotion_detector_empty_segments():
    from emotion import EmotionDetector

    detector = EmotionDetector()
    detector._model = MagicMock()

    result = detector.process("dummy.wav", [])
    assert result == []


def test_emotion_detector_preserves_segment_data():
    from emotion import EmotionDetector

    detector = EmotionDetector()
    detector._model = MagicMock()

    segments = [{"start": 1.5, "end": 3.5, "speaker": "SPEAKER_00"}]

    result = detector.process("dummy.wav", segments)
    assert result[0]["start"] == 1.5
    assert result[0]["end"] == 3.5
    assert result[0]["speaker"] == "SPEAKER_00"
