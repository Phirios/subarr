"""
Overlap detection: flags subtitle lines where multiple speakers are active.
"""

import logging

logger = logging.getLogger("subarr-worker")


def detect_overlaps(
    segments: list[dict],
    mapped_subtitles: list[dict],
    min_overlap_duration: float = 0.5,
) -> list[dict]:
    """
    Detect subtitle lines where more than one speaker is active.

    Args:
        segments: diarization segments [{"start", "end", "speaker"}, ...]
        mapped_subtitles: subtitle entries with speaker assigned
        min_overlap_duration: minimum overlap in seconds to count a speaker

    Returns:
        List of overlap flags:
        [{"subtitle_index", "assigned_speaker", "other_speakers", "note"}, ...]
    """
    if not segments or not mapped_subtitles:
        return []

    overlaps = []
    for i, sub in enumerate(mapped_subtitles):
        sub_start = sub["start_ms"] / 1000.0
        sub_end = sub["end_ms"] / 1000.0
        assigned = sub.get("speaker")

        active_speakers = set()
        for seg in segments:
            overlap_start = max(seg["start"], sub_start)
            overlap_end = min(seg["end"], sub_end)
            if overlap_end - overlap_start >= min_overlap_duration:
                active_speakers.add(seg["speaker"])

        other_speakers = active_speakers - {assigned}
        if other_speakers:
            overlaps.append({
                "subtitle_index": i,
                "assigned_speaker": assigned,
                "other_speakers": sorted(other_speakers),
                "note": f"Multiple speakers active: {assigned} + {', '.join(sorted(other_speakers))}",
            })

    if overlaps:
        logger.info(f"Detected {len(overlaps)} subtitle(s) with speaker overlap")

    return overlaps
