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
    metadata = {"title": "Breaking Bad", "season": 1, "episode": 1, "tmdb_id": 1396}
    mapped_subtitles = [
        {"start_ms": 0, "end_ms": 2000, "text": "Hello", "speaker": "SPEAKER_00", "emotion": "happy"},
        {"start_ms": 3000, "end_ms": 5000, "text": "Hi", "speaker": "SPEAKER_01", "emotion": "sad"},
    ]

    ctx = t._build_context(segments, metadata, mapped_subtitles=mapped_subtitles)
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
        None,
    )
    assert "1 speakers" in ctx


def test_translator_builds_context_with_character_map():
    from translation import Translator

    t = Translator.__new__(Translator)
    t.client = None

    segments = [{"start": 0, "end": 2, "speaker": "SPEAKER_00"}]
    character_map = {"SPEAKER_00": "Shizuka"}

    ctx = t._build_context(segments, None, character_map)
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

    result = t.translate("1\n00:00:01,000 --> 00:00:04,000\nHello", [], "tr", None)
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

    result = t.translate("1\n00:00:00,000 --> 00:00:01,000\nTest", [], "tr", None)
    assert len(result) == 1
    assert result[0]["text"] == "Test"


def test_translator_uses_character_names_in_prompt():
    from translation import Translator

    t = Translator.__new__(Translator)

    mock_response = MagicMock()
    mock_response.text = json.dumps([
        {"start_ms": 1000, "end_ms": 4000, "text": "Merhaba", "speaker": "SPEAKER_00", "emotion": "happy", "color": "FF4444"}
    ])
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    t.client = mock_client

    mapped = [{"start_ms": 1000, "end_ms": 4000, "text": "Hello", "speaker": "SPEAKER_00", "emotion": "happy"}]
    character_map = {"SPEAKER_00": "Shizuka"}

    t.translate("", [], "tr", None, mapped_subtitles=mapped, character_map=character_map)

    # Check the prompt sent to Gemini contains character name
    call_args = mock_client.models.generate_content.call_args
    prompt = call_args.kwargs.get("contents") or call_args[1].get("contents") or call_args[0][0]
    assert "Shizuka" in str(prompt)


def test_translator_post_process_replaces_speaker_with_character():
    from translation import Translator

    t = Translator.__new__(Translator)

    results = [
        {"start_ms": 1000, "end_ms": 4000, "text": "Merhaba", "speaker": "SPEAKER_00", "emotion": "happy"},
        {"start_ms": 5000, "end_ms": 8000, "text": "Selam", "speaker": "SPEAKER_01", "emotion": "neutral"},
    ]
    character_map = {"SPEAKER_00": "Shizuka", "SPEAKER_01": "Takopi"}

    processed = t._post_process(results, character_map)
    assert processed[0]["speaker"] == "Shizuka"
    assert processed[1]["speaker"] == "Takopi"
    assert processed[0]["color"] is not None
    assert processed[0]["color"] != processed[1]["color"]


def test_translator_post_process_normalizes_typos():
    from translation import Translator

    t = Translator.__new__(Translator)

    results = [
        {"start_ms": 1000, "end_ms": 4000, "text": "Test", "speaker": "SPEAPER_05", "emotion": "neutral"},
    ]

    processed = t._post_process(results, None)
    assert processed[0]["speaker"] == "SPEAKER_05"


def test_translator_includes_emotion_in_prompt():
    from translation import Translator

    t = Translator.__new__(Translator)

    mock_response = MagicMock()
    mock_response.text = json.dumps([
        {"start_ms": 1000, "end_ms": 4000, "text": "Merhaba", "speaker": "SPEAKER_00", "emotion": "sad"}
    ])
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    t.client = mock_client

    mapped = [{"start_ms": 1000, "end_ms": 4000, "text": "Hello", "speaker": "SPEAKER_00", "emotion": "sad"}]

    t.translate("", [], "tr", None, mapped_subtitles=mapped)

    call_args = mock_client.models.generate_content.call_args
    prompt = call_args.kwargs.get("contents") or call_args[1].get("contents") or call_args[0][0]
    assert "sad" in str(prompt)
