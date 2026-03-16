"""
Speaker-to-subtitle mapping.
Maps diarization segments to subtitle lines by time overlap.
No merging — merge decisions are handled by post_id_merge.
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
) -> list[dict]:
    """
    Map diarization speaker segments to subtitle entries.
    Filter + assign only — no orphan merging.

    Returns:
        mapped_subtitles: subtitle entries with speaker assigned
    """
    srt_entries = parse_srt_entries(srt_content)

    if not segments or not srt_entries:
        return [{**entry, "speaker": None} for entry in srt_entries]

    mapped_subtitles = []
    for entry in srt_entries:
        sub_start = entry["start_ms"] / 1000.0

        # Collect all segments that overlap this subtitle
        candidates = []
        for seg in segments:
            overlap = compute_overlap(
                seg["start"], seg["end"],
                entry["start_ms"], entry["end_ms"],
            )
            if overlap >= overlap_threshold:
                candidates.append((seg, overlap))

        if not candidates:
            mapped_subtitles.append({**entry, "speaker": None})
            continue

        if len(candidates) == 1:
            best_speaker = candidates[0][0]["speaker"]
        else:
            # Multiple segments overlap — if top candidates are close in overlap,
            # prefer the one starting closest to the subtitle's start time.
            # This prevents a trailing segment from the previous speaker from winning.
            candidates.sort(key=lambda x: x[1], reverse=True)
            top_overlap = candidates[0][1]
            close = [(seg, ov) for seg, ov in candidates if ov >= top_overlap - 0.15]

            if len(close) > 1:
                best_seg, _ = min(close, key=lambda x: abs(x[0]["start"] - sub_start))
                best_speaker = best_seg["speaker"]
            else:
                best_speaker = candidates[0][0]["speaker"]

        mapped_subtitles.append({**entry, "speaker": best_speaker})

    assigned = sum(1 for s in mapped_subtitles if s["speaker"] is not None)
    speakers = set(s["speaker"] for s in mapped_subtitles if s["speaker"] is not None)
    logger.info(
        f"Speaker mapping: {len(speakers)} speakers, "
        f"{assigned}/{len(srt_entries)} lines assigned, "
        f"{len(srt_entries) - assigned} unassigned"
    )

    return mapped_subtitles
