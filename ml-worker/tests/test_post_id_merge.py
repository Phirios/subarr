import numpy as np

from post_id_merge import post_id_merge, _cosine_similarity


class TestCosineSimlarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert _cosine_similarity(v, v) == 1.0

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 1.0])
        assert _cosine_similarity(a, b) == 0.0


class TestPostIdMerge:
    def test_merges_similar_speaker(self):
        character_map = {"SPEAKER_00": "Shizuka"}
        embeddings = {
            "vectors": np.array([
                [1.0, 0.0, 0.0],  # SPEAKER_00
                [0.95, 0.05, 0.0],  # SPEAKER_01 - very similar to 00
            ]),
            "speakers": ["SPEAKER_00", "SPEAKER_01"],
        }
        segments = [
            {"start": 0, "end": 1, "speaker": "SPEAKER_00"},
            {"start": 2, "end": 3, "speaker": "SPEAKER_01"},
        ]
        subs = [
            {"start_ms": 0, "end_ms": 1000, "text": "Hi", "speaker": "SPEAKER_00"},
            {"start_ms": 2000, "end_ms": 3000, "text": "Hey", "speaker": "SPEAKER_01"},
        ]

        result = post_id_merge(character_map, embeddings, segments, subs, threshold=0.75)
        assert result["SPEAKER_01"] == "Shizuka"
        # Segments and subs should be updated in-place
        assert segments[1]["speaker"] == "SPEAKER_00"
        assert subs[1]["speaker"] == "SPEAKER_00"

    def test_no_merge_below_threshold(self):
        character_map = {"SPEAKER_00": "Shizuka"}
        embeddings = {
            "vectors": np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],  # orthogonal - very different
            ]),
            "speakers": ["SPEAKER_00", "SPEAKER_01"],
        }
        segments = [{"start": 0, "end": 1, "speaker": "SPEAKER_01"}]
        subs = [{"start_ms": 0, "end_ms": 1000, "text": "Hi", "speaker": "SPEAKER_01"}]

        result = post_id_merge(character_map, embeddings, segments, subs, threshold=0.75)
        assert "SPEAKER_01" not in result
        assert segments[0]["speaker"] == "SPEAKER_01"  # unchanged

    def test_no_embeddings_noop(self):
        character_map = {"SPEAKER_00": "Shizuka"}
        result = post_id_merge(character_map, None, [], [])
        assert result == character_map

    def test_empty_character_map_noop(self):
        embeddings = {"vectors": np.array([[1.0]]), "speakers": ["SPEAKER_00"]}
        result = post_id_merge({}, embeddings, [], [])
        assert result == {}
