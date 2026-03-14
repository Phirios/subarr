import json
from unittest.mock import MagicMock

from character_id import CharacterIdentifier


class TestCharacterIdentifier:
    def _make_identifier(self, response_text: str) -> CharacterIdentifier:
        ci = CharacterIdentifier.__new__(CharacterIdentifier)
        mock_response = MagicMock()
        mock_response.text = response_text
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        ci.client = mock_client
        return ci

    def test_identify_returns_mapping(self):
        response = json.dumps({"SPEAKER_00": "Shizuka", "SPEAKER_01": "Takopi"})
        ci = self._make_identifier(response)

        subs = [
            {"start_ms": 0, "end_ms": 1000, "text": "Hello", "speaker": "SPEAKER_00"},
            {"start_ms": 2000, "end_ms": 3000, "text": "Hi", "speaker": "SPEAKER_01"},
        ]
        cast = [
            {"actor": "Actor A", "character": "Shizuka"},
            {"actor": "Actor B", "character": "Takopi"},
        ]

        result = ci.identify(subs, cast)
        assert result == {"SPEAKER_00": "Shizuka", "SPEAKER_01": "Takopi"}

    def test_empty_cast_returns_empty(self):
        ci = self._make_identifier("{}")
        subs = [{"start_ms": 0, "end_ms": 1000, "text": "Hello", "speaker": "SPEAKER_00"}]
        assert ci.identify(subs, []) == {}

    def test_no_speakers_returns_empty(self):
        ci = self._make_identifier("{}")
        subs = [{"start_ms": 0, "end_ms": 1000, "text": "Hello", "speaker": None}]
        cast = [{"actor": "A", "character": "B"}]
        assert ci.identify(subs, cast) == {}

    def test_llm_error_returns_empty(self):
        ci = CharacterIdentifier.__new__(CharacterIdentifier)
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API error")
        ci.client = mock_client

        subs = [{"start_ms": 0, "end_ms": 1000, "text": "Hello", "speaker": "SPEAKER_00"}]
        cast = [{"actor": "A", "character": "B"}]
        assert ci.identify(subs, cast) == {}

    def test_filters_invalid_keys(self):
        response = json.dumps({"SPEAKER_00": "Shizuka", "FAKE_SPEAKER": "Nobody"})
        ci = self._make_identifier(response)

        subs = [{"start_ms": 0, "end_ms": 1000, "text": "Hello", "speaker": "SPEAKER_00"}]
        cast = [{"actor": "A", "character": "Shizuka"}]

        result = ci.identify(subs, cast)
        assert "SPEAKER_00" in result
        assert "FAKE_SPEAKER" not in result
