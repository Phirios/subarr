import pytest
from speaker_mapping import parse_srt_entries, compute_overlap, map_speakers_to_subtitles


class TestParseSrtEntries:
    def test_basic_parsing(self):
        srt = "1\n00:00:01,000 --> 00:00:03,000\nHello\n\n2\n00:00:04,000 --> 00:00:06,000\nWorld"
        entries = parse_srt_entries(srt)
        assert len(entries) == 2
        assert entries[0]["start_ms"] == 1000
        assert entries[0]["end_ms"] == 3000
        assert entries[0]["text"] == "Hello"
        assert entries[1]["start_ms"] == 4000

    def test_multiline_text(self):
        srt = "1\n00:00:01,000 --> 00:00:03,000\nLine one\nLine two"
        entries = parse_srt_entries(srt)
        assert entries[0]["text"] == "Line one Line two"

    def test_empty_input(self):
        assert parse_srt_entries("") == []


class TestComputeOverlap:
    def test_full_overlap(self):
        assert compute_overlap(1.0, 3.0, 1000, 3000) == pytest.approx(1.0)

    def test_partial_overlap(self):
        assert compute_overlap(1.0, 2.0, 1000, 3000) == pytest.approx(0.5)

    def test_no_overlap(self):
        assert compute_overlap(5.0, 6.0, 1000, 3000) == 0.0

    def test_segment_inside_subtitle(self):
        assert compute_overlap(1.5, 2.5, 1000, 3000) == pytest.approx(0.5)


class TestMapSpeakersToSubtitles:
    def test_basic_mapping(self):
        segments = [
            {"start": 1.0, "end": 3.0, "speaker": "SPEAKER_00"},
            {"start": 4.0, "end": 6.0, "speaker": "SPEAKER_01"},
        ]
        srt = "1\n00:00:01,000 --> 00:00:03,000\nHello\n\n2\n00:00:04,000 --> 00:00:06,000\nWorld"
        mapped = map_speakers_to_subtitles(segments, srt)
        assert mapped[0]["speaker"] == "SPEAKER_00"
        assert mapped[1]["speaker"] == "SPEAKER_01"

    def test_returns_list(self):
        segments = [{"start": 1.0, "end": 3.0, "speaker": "SPEAKER_00"}]
        srt = "1\n00:00:01,000 --> 00:00:03,000\nHello"
        result = map_speakers_to_subtitles(segments, srt)
        assert isinstance(result, list)

    def test_no_segments(self):
        srt = "1\n00:00:01,000 --> 00:00:03,000\nHello"
        mapped = map_speakers_to_subtitles([], srt)
        assert mapped[0]["speaker"] is None

    def test_unassigned_subtitle(self):
        segments = [{"start": 100.0, "end": 102.0, "speaker": "SPEAKER_00"}]
        srt = "1\n00:00:01,000 --> 00:00:03,000\nHello"
        mapped = map_speakers_to_subtitles(segments, srt)
        assert mapped[0]["speaker"] is None

    def test_orphan_speakers_not_merged(self):
        """Orphan speakers should NOT be merged — that's post_id_merge's job now."""
        segments = [
            {"start": 1.0, "end": 3.0, "speaker": "SPEAKER_00"},
            {"start": 3.5, "end": 4.0, "speaker": "SPEAKER_02"},  # orphan
            {"start": 5.0, "end": 7.0, "speaker": "SPEAKER_00"},
        ]
        srt = "1\n00:00:01,000 --> 00:00:03,000\nHello\n\n2\n00:00:05,000 --> 00:00:07,000\nWorld"
        mapped = map_speakers_to_subtitles(segments, srt)
        # SPEAKER_02 segments unchanged (no merge in speaker_mapping)
        assert segments[1]["speaker"] == "SPEAKER_02"
