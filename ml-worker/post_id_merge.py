"""
Post-identification merge: merge unidentified speakers into identified ones
using embedding cosine similarity.
"""

import logging

import numpy as np

logger = logging.getLogger("subarr-worker")


def post_id_merge(
    character_map: dict[str, str],
    embeddings: dict | None,
    segments: list[dict],
    mapped_subtitles: list[dict],
    threshold: float = 0.75,
) -> dict[str, str]:
    """
    Merge unidentified speakers into identified ones based on embedding similarity.

    Args:
        character_map: current speaker→character mapping (e.g. {"SPEAKER_00": "Shizuka"})
        embeddings: {"vectors": np.array (num_speakers, dim), "speakers": ["SPEAKER_00", ...]}
        segments: diarization segments
        mapped_subtitles: subtitle entries with speaker assigned
        threshold: cosine similarity threshold for merging

    Returns:
        Updated character_map with merged speakers added.
    """
    if not embeddings or not character_map:
        return character_map

    vectors = embeddings.get("vectors")
    speaker_list = embeddings.get("speakers", [])

    if vectors is None or len(speaker_list) == 0:
        return character_map

    identified = set(character_map.keys())
    all_speakers = set(speaker_list)
    unidentified = all_speakers - identified

    if not unidentified:
        return character_map

    # Build speaker index → vector mapping
    speaker_to_idx = {sp: i for i, sp in enumerate(speaker_list)}

    updated_map = dict(character_map)
    merge_map = {}

    for unknown_sp in sorted(unidentified):
        if unknown_sp not in speaker_to_idx:
            continue

        unknown_vec = vectors[speaker_to_idx[unknown_sp]]
        best_match = None
        best_sim = -1.0

        for known_sp in sorted(identified):
            if known_sp not in speaker_to_idx:
                continue

            known_vec = vectors[speaker_to_idx[known_sp]]
            sim = _cosine_similarity(unknown_vec, known_vec)

            if sim > best_sim:
                best_sim = sim
                best_match = known_sp

        if best_match and best_sim >= threshold:
            char_name = character_map[best_match]
            updated_map[unknown_sp] = char_name
            merge_map[unknown_sp] = best_match
            logger.info(
                f"Post-ID merge: {unknown_sp} → {best_match} ({char_name}) "
                f"similarity={best_sim:.3f}"
            )

    # Apply merge to segments and subtitles
    if merge_map:
        for seg in segments:
            if seg["speaker"] in merge_map:
                seg["speaker"] = merge_map[seg["speaker"]]
        for sub in mapped_subtitles:
            if sub.get("speaker") in merge_map:
                sub["speaker"] = merge_map[sub["speaker"]]

    return updated_map


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
