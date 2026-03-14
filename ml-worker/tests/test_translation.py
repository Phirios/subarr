import json
from unittest.mock import MagicMock, patch

import pytest


def test_translator_builds_context():
    from translation import Translator

    t = Translator.__new__(Translator)
    t.client = None

    segments = [
        {"start": 0, "end": 2, "speaker": "SPEAKER_00"},
        {"start": 3, "end": 5, "speaker": "SPEAKER_01"},
    ]
    emotions = [
        {"start": 0, "end": 2, "speaker": "SPEAKER_00", "emotion": "happy"},
        {"start": 3, "end": 5, "speaker": "SPEAKER_01", "emotion": "sad"},
    ]
    metadata = {"title": "Breaking Bad", "season": 1, "episode": 1, "tmdb_id": 1396}

    ctx = t._build_context(segments, emotions, metadata)
    assert "Breaking Bad" in ctx
    assert "Season 1" in ctx
    assert "2 speakers" in ctx
    assert "happy" in ctx


def test_translator_builds_context_no_metadata():
    from translation import Translator

    t = Translator.__new__(Translator)
    t.client = None

    ctx = t._build_context(
        [{"speaker": "SPEAKER_00"}],
        [{"emotion": "neutral"}],
        None,
    )
    assert "1 speakers" in ctx
    assert "neutral" in ctx


def test_translator_builds_context_with_character_map():
    from translation import Translator

    t = Translator.__new__(Translator)
    t.client = None

    segments = [{"start": 0, "end": 2, "speaker": "SPEAKER_00"}]
    emotions = [{"emotion": "neutral"}]
    character_map = {"SPEAKER_00": "Shizuka"}

    ctx = t._build_context(segments, emotions, None, character_map)
    assert "SPEAKER_00 = Shizuka" in ctx


def test_translator_parses_json_response():
    from translation import Translator

    t = Translator.__new__(Translator)

    mock_response = MagicMock()
    mock_response.text = json.dumps([
        {"start_ms": 1000, "end_ms": 4000, "text": "Merhaba", "speaker": None, "emotion": None, "color": "FFFFFF"}
    ])

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    t.client = mock_client

    result = t.translate("1\n00:00:01,000 --> 00:00:04,000\nHello", [], [], "tr", None)
    assert len(result) == 1
    assert result[0]["text"] == "Merhaba"


def test_translator_parses_json_with_markdown_wrapper():
    from translation import Translator

    t = Translator.__new__(Translator)

    mock_response = MagicMock()
    mock_response.text = '```json\n[{"start_ms": 0, "end_ms": 1000, "text": "Test", "speaker": null, "emotion": null, "color": null}]\n```'

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    t.client = mock_client

    result = t.translate("1\n00:00:00,000 --> 00:00:01,000\nTest", [], [], "tr", None)
    assert len(result) == 1
    assert result[0]["text"] == "Test"


def test_translator_uses_character_names_in_prompt():
    from translation import Translator

    t = Translator.__new__(Translator)

    mock_response = MagicMock()
    mock_response.text = json.dumps([
        {"start_ms": 1000, "end_ms": 4000, "text": "Merhaba", "speaker": "SPEAKER_00", "emotion": None, "color": "FF4444"}
    ])
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    t.client = mock_client

    mapped = [{"start_ms": 1000, "end_ms": 4000, "text": "Hello", "speaker": "SPEAKER_00"}]
    character_map = {"SPEAKER_00": "Shizuka"}

    t.translate("", [], [], "tr", None, mapped_subtitles=mapped, character_map=character_map)

    # Check the prompt sent to Gemini contains character name
    call_args = mock_client.models.generate_content.call_args
    prompt = call_args.kwargs.get("contents") or call_args[1].get("contents") or call_args[0][0]
    assert "Shizuka" in str(prompt)
