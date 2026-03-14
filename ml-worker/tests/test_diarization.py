import pytest
from unittest.mock import patch, MagicMock


def test_diarizer_lazy_loads_model():
    from diarization import Diarizer

    d = Diarizer("fake-token")
    assert d._pipeline is None


def test_diarizer_process_returns_dict_with_segments():
    from diarization import Diarizer

    d = Diarizer("fake-token")

    mock_turn1 = MagicMock()
    mock_turn1.start = 0.0
    mock_turn1.end = 2.5
    mock_turn2 = MagicMock()
    mock_turn2.start = 3.0
    mock_turn2.end = 5.0

    mock_result = MagicMock()
    mock_result.speaker_diarization.itertracks.return_value = [
        (mock_turn1, None, "SPEAKER_00"),
        (mock_turn2, None, "SPEAKER_01"),
    ]
    mock_result.speaker_embeddings = None

    mock_pipeline = MagicMock(return_value=mock_result)
    d._pipeline = mock_pipeline

    result = d.process("test.wav")
    assert isinstance(result, dict)
    assert "segments" in result
    assert "embeddings" in result
    segments = result["segments"]
    assert len(segments) == 2
    assert segments[0]["speaker"] == "SPEAKER_00"
    assert segments[0]["start"] == 0.0
    assert segments[0]["end"] == 2.5
    assert segments[1]["speaker"] == "SPEAKER_01"


def test_diarizer_process_empty_audio():
    from diarization import Diarizer

    d = Diarizer("fake-token")

    mock_result = MagicMock()
    mock_result.speaker_diarization.itertracks.return_value = []
    mock_result.speaker_embeddings = None

    mock_pipeline = MagicMock(return_value=mock_result)
    d._pipeline = mock_pipeline

    result = d.process("empty.wav")
    assert result["segments"] == []
    assert result["embeddings"] is None
