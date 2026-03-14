"""
Speaker-to-subtitle mapping and speaker merging.
Maps diarization segments to subtitle lines by time overlap,
then merges orphan/fragmented speakers into existing ones.
"""

import logging
import re

logger = logging.getLogger("subarr-worker")


def parse_srt_entries(srt_content: str) -> list[dict]:
    """Parse SRT content into list of {start_ms, end_ms, text} entries."""
    entries = []
    blocks = srt_content.strip().split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        timestamp_line = lines[1] if lines[0].strip().isdigit() else lines[0]
        match = re.match(
            r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})",
            timestamp_line,
        )
        if not match:
            continue
        g = [int(x) for x in match.groups()]
        start_ms = g[0] * 3600000 + g[1] * 60000 + g[2] * 1000 + g[3]
        end_ms = g[4] * 3600000 + g[5] * 60000 + g[6] * 1000 + g[7]
        text_lines = lines[2:] if lines[0].strip().isdigit() else lines[1:]
        text = " ".join(t.strip() for t in text_lines if t.strip())
        entries.append({"start_ms": start_ms, "end_ms": end_ms, "text": text})
    return entries


def compute_overlap(seg_start: float, seg_end: float, sub_start_ms: int, sub_end_ms: int) -> float:
    """Compute overlap ratio between a diarization segment and a subtitle entry."""
    sub_start = sub_start_ms / 1000.0
    sub_end = sub_end_ms / 1000.0
    overlap_start = max(seg_start, sub_start)
    overlap_end = min(seg_end, sub_end)
    if overlap_end <= overlap_start:
        return 0.0
    overlap_duration = overlap_end - overlap_start
    sub_duration = max(sub_end - sub_start, 0.001)
    return overlap_duration / sub_duration


def map_speakers_to_subtitles(
    segments: list[dict],
    srt_content: str,
    overlap_threshold: float = 0.3,
) -> tuple[list[dict], list[dict]]:
    """
    Map diarization speaker segments to subtitle entries.

    Returns:
        - mapped_subtitles: subtitle entries with speaker assigned
        - segments: updated segments (after merging)
    """
    srt_entries = parse_srt_entries(srt_content)

    if not segments or not srt_entries:
        return [
            {**entry, "speaker": None} for entry in srt_entries
        ], segments

    # Step 1: Assign best-matching speaker to each subtitle line
    subtitle_speakers = []
    for entry in srt_entries:
        best_speaker = None
        best_overlap = 0.0
        for seg in segments:
            overlap = compute_overlap(
                seg["start"], seg["end"],
                entry["start_ms"], entry["end_ms"],
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = seg["speaker"]

        subtitle_speakers.append(
            best_speaker if best_overlap >= overlap_threshold else None
        )

    # Step 2: Find speakers that mapped to subtitles vs orphans
    mapped_speakers = set(s for s in subtitle_speakers if s is not None)
    all_speakers = set(seg["speaker"] for seg in segments)
    orphan_speakers = all_speakers - mapped_speakers

    if orphan_speakers:
        logger.info(
            f"Found {len(orphan_speakers)} orphan speaker(s): {orphan_speakers}, "
            f"attempting merge into {mapped_speakers}"
        )

    # Step 3: Merge orphan speakers into nearest mapped speaker
    merge_map = {}
    for orphan in orphan_speakers:
        orphan_segs = [s for s in segments if s["speaker"] == orphan]
        best_target = None
        best_distance = float("inf")

        for mapped_sp in mapped_speakers:
            mapped_segs = [s for s in segments if s["speaker"] == mapped_sp]
            # Find minimum time distance between orphan and mapped speaker segments
            for o_seg in orphan_segs:
                o_mid = (o_seg["start"] + o_seg["end"]) / 2
                for m_seg in mapped_segs:
                    m_mid = (m_seg["start"] + m_seg["end"]) / 2
                    dist = abs(o_mid - m_mid)
                    if dist < best_distance:
                        best_distance = dist
                        best_target = mapped_sp

        if best_target is not None:
            merge_map[orphan] = best_target
            logger.info(f"Merging {orphan} -> {best_target} (distance: {best_distance:.1f}s)")

    # Step 4: Apply merge to segments
    if merge_map:
        for seg in segments:
            if seg["speaker"] in merge_map:
                seg["speaker"] = merge_map[seg["speaker"]]

        # Re-assign subtitles with merged speakers
        for i, entry in enumerate(srt_entries):
            best_speaker = None
            best_overlap = 0.0
            for seg in segments:
                overlap = compute_overlap(
                    seg["start"], seg["end"],
                    entry["start_ms"], entry["end_ms"],
                )
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = seg["speaker"]
            subtitle_speakers[i] = (
                best_speaker if best_overlap >= overlap_threshold else None
            )

    # Build result
    mapped_subtitles = []
    for entry, speaker in zip(srt_entries, subtitle_speakers):
        mapped_subtitles.append({**entry, "speaker": speaker})

    final_speakers = set(s for s in subtitle_speakers if s is not None)
    unassigned = sum(1 for s in subtitle_speakers if s is None)
    logger.info(
        f"Speaker mapping: {len(final_speakers)} speakers, "
        f"{len(srt_entries) - unassigned}/{len(srt_entries)} lines assigned, "
        f"{unassigned} unassigned"
    )

    return mapped_subtitles, segments
