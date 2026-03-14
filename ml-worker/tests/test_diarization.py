import pytest
from unittest.mock import patch, MagicMock


def test_diarizer_lazy_loads_model():
    from diarization import Diarizer

    d = Diarizer("fake-token")
    assert d._pipeline is None


def test_diarizer_process_returns_segments():
    from diarization import Diarizer

    d = Diarizer("fake-token")

    # Mock the pyannote pipeline
    mock_turn1 = MagicMock()
    mock_turn1.start = 0.0
    mock_turn1.end = 2.5
    mock_turn2 = MagicMock()
    mock_turn2.start = 3.0
    mock_turn2.end = 5.0

    mock_diarization = MagicMock()
    mock_diarization.itertracks.return_value = [
        (mock_turn1, None, "SPEAKER_00"),
        (mock_turn2, None, "SPEAKER_01"),
    ]

    mock_pipeline = MagicMock(return_value=mock_diarization)
    d._pipeline = mock_pipeline

    segments = d.process("test.wav")
    assert len(segments) == 2
    assert segments[0]["speaker"] == "SPEAKER_00"
    assert segments[0]["start"] == 0.0
    assert segments[0]["end"] == 2.5
    assert segments[1]["speaker"] == "SPEAKER_01"


def test_diarizer_process_empty_audio():
    from diarization import Diarizer

    d = Diarizer("fake-token")

    mock_diarization = MagicMock()
    mock_diarization.itertracks.return_value = []

    mock_pipeline = MagicMock(return_value=mock_diarization)
    d._pipeline = mock_pipeline

    segments = d.process("empty.wav")
    assert segments == []
