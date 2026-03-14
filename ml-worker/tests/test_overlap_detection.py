from overlap_detection import detect_overlaps


class TestDetectOverlaps:
    def test_no_overlap(self):
        segments = [
            {"start": 1.0, "end": 3.0, "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 7.0, "speaker": "SPEAKER_01"},
        ]
        subs = [
            {"start_ms": 1000, "end_ms": 3000, "text": "Hello", "speaker": "SPEAKER_00"},
            {"start_ms": 5000, "end_ms": 7000, "text": "World", "speaker": "SPEAKER_01"},
        ]
        assert detect_overlaps(segments, subs) == []

    def test_overlap_detected(self):
        segments = [
            {"start": 1.0, "end": 4.0, "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 5.0, "speaker": "SPEAKER_01"},
        ]
        subs = [
            {"start_ms": 1000, "end_ms": 5000, "text": "Hello", "speaker": "SPEAKER_00"},
        ]
        result = detect_overlaps(segments, subs)
        assert len(result) == 1
        assert result[0]["assigned_speaker"] == "SPEAKER_00"
        assert "SPEAKER_01" in result[0]["other_speakers"]

    def test_min_overlap_duration_filter(self):
        segments = [
            {"start": 1.0, "end": 3.0, "speaker": "SPEAKER_00"},
            {"start": 2.8, "end": 3.1, "speaker": "SPEAKER_01"},  # only 0.2s overlap with sub
        ]
        subs = [
            {"start_ms": 1000, "end_ms": 3000, "text": "Hello", "speaker": "SPEAKER_00"},
        ]
        # Default min_overlap_duration=0.5, so SPEAKER_01's 0.2s shouldn't flag
        assert detect_overlaps(segments, subs) == []

    def test_empty_inputs(self):
        assert detect_overlaps([], []) == []
        assert detect_overlaps([], [{"start_ms": 0, "end_ms": 1000, "text": "Hi", "speaker": "S"}]) == []
        assert detect_overlaps([{"start": 0, "end": 1, "speaker": "S"}], []) == []
