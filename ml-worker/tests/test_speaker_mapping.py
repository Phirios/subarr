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
        # Segment fully covers subtitle
        assert compute_overlap(1.0, 3.0, 1000, 3000) == pytest.approx(1.0)

    def test_partial_overlap(self):
        # Segment covers first half
        assert compute_overlap(1.0, 2.0, 1000, 3000) == pytest.approx(0.5)

    def test_no_overlap(self):
        assert compute_overlap(5.0, 6.0, 1000, 3000) == 0.0

    def test_segment_inside_subtitle(self):
        # Segment is inside subtitle
        assert compute_overlap(1.5, 2.5, 1000, 3000) == pytest.approx(0.5)


class TestMapSpeakersToSubtitles:
    def test_basic_mapping(self):
        segments = [
            {"start": 1.0, "end": 3.0, "speaker": "SPEAKER_00"},
            {"start": 4.0, "end": 6.0, "speaker": "SPEAKER_01"},
        ]
        srt = "1\n00:00:01,000 --> 00:00:03,000\nHello\n\n2\n00:00:04,000 --> 00:00:06,000\nWorld"
        mapped, _ = map_speakers_to_subtitles(segments, srt)
        assert mapped[0]["speaker"] == "SPEAKER_00"
        assert mapped[1]["speaker"] == "SPEAKER_01"

    def test_orphan_merge(self):
        # SPEAKER_02 only appears between SPEAKER_00 segments, should merge
        segments = [
            {"start": 1.0, "end": 3.0, "speaker": "SPEAKER_00"},
            {"start": 3.5, "end": 4.0, "speaker": "SPEAKER_02"},  # orphan - no subtitle overlap
            {"start": 5.0, "end": 7.0, "speaker": "SPEAKER_00"},
        ]
        srt = "1\n00:00:01,000 --> 00:00:03,000\nHello\n\n2\n00:00:05,000 --> 00:00:07,000\nWorld"
        mapped, merged_segments = map_speakers_to_subtitles(segments, srt)
        # SPEAKER_02 should have been merged into SPEAKER_00
        assert all(s["speaker"] == "SPEAKER_00" for s in merged_segments)

    def test_no_segments(self):
        srt = "1\n00:00:01,000 --> 00:00:03,000\nHello"
        mapped, segments = map_speakers_to_subtitles([], srt)
        assert mapped[0]["speaker"] is None

    def test_unassigned_subtitle(self):
        # Subtitle far from any segment
        segments = [{"start": 100.0, "end": 102.0, "speaker": "SPEAKER_00"}]
        srt = "1\n00:00:01,000 --> 00:00:03,000\nHello"
        mapped, _ = map_speakers_to_subtitles(segments, srt)
        assert mapped[0]["speaker"] is None
