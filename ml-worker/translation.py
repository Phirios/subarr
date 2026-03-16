"""
Context-aware subtitle translation using Gemini API.
Takes into account speaker identity, emotion, and series context.
"""

import json
import logging
import time
from google import genai
from google.genai import types

logger = logging.getLogger("subarr-worker")


class Translator:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def translate(
        self,
        subtitle_content: str,
        segments: list[dict],
        target_language: str = "tr",
        metadata: dict | None = None,
        mapped_subtitles: list[dict] | None = None,
        character_map: dict[str, str] | None = None,
        tmdb_context: dict | None = None,
    ) -> list[dict]:
        """
        Translates subtitle entries with context awareness.
        Uses mapped_subtitles (with speaker info and emotion) if available,
        otherwise falls back to raw SRT content.
        """
        context_block = self._build_context(segments, metadata, character_map, mapped_subtitles, tmdb_context)

        if mapped_subtitles:
            batches = self._split_entries(mapped_subtitles, max_entries=100)
        else:
            srt_batches = self._split_srt(subtitle_content, max_entries=100)
            batches = [{"srt": batch} for batch in srt_batches]

        logger.info(f"Translating {len(batches)} batch(es)")
        all_results = []
        for i, batch in enumerate(batches):
            if i > 0:
                logger.info("Rate limit pause (15s)...")
                time.sleep(15)
            logger.info(f"Translating batch {i+1}/{len(batches)}")
            if isinstance(batch, dict) and "srt" in batch:
                result = self._translate_batch_srt(batch["srt"], context_block, target_language)
            else:
                result = self._translate_batch_mapped(batch, context_block, target_language, character_map)
            all_results.extend(result)

        # Post-process: replace speaker IDs with character names and assign consistent colors
        all_results = self._post_process(all_results, character_map)

        return all_results

    def _translate_batch_mapped(self, entries: list[dict], context_block: str, target_language: str, character_map: dict[str, str] | None = None) -> list[dict]:
        """Translate a batch of mapped subtitle entries with speaker info."""
        # Build character legend: numeric ID -> name
        char_legend = ""
        if character_map:
            legend_lines = []
            for speaker, name in sorted(character_map.items()):
                num = speaker.replace("SPEAKER_", "")
                legend_lines.append(f"  {num}: {name}")
            char_legend = "Characters:\n" + "\n".join(legend_lines) + "\n\n"

        # Format entries: [timestamp] (character/id, emotion) text
        lines = []
        for e in entries:
            # Prefer per-line character name over speaker ID
            char = e.get("character")
            if not char:
                speaker = e.get("speaker") or "Unknown"
                char = speaker.replace("SPEAKER_", "") if speaker.startswith("SPEAKER_") else speaker
            emotion = e.get("emotion", "neutral")
            lines.append(f"[{self._ms_to_timestamp(e['start_ms'])}] ({char}, {emotion}) {e['text']}")
        source_block = "\n".join(lines)

        prompt = f"""You are a professional subtitle translator. Translate the following subtitles to {target_language}.

{context_block}

{char_legend}Lines:
{source_block}

Rules:
- Translate each line to natural, conversational {target_language}
- Adapt tone based on who is speaking (see character legend) and their emotion
- Keep translations concise to fit subtitle timing
- Return ONLY a JSON array of translated strings, one per input line, in the same order
- The array must have exactly {len(entries)} elements

Example output: ["Translated line 1", "Translated line 2", ...]"""

        translations = self._call_gemini(prompt)

        # Map translations back to full entries
        results = []
        for i, e in enumerate(entries):
            text = translations[i] if i < len(translations) else e["text"]
            entry = {
                "start_ms": e["start_ms"],
                "end_ms": e["end_ms"],
                "text": text,
                "speaker": e.get("character") or e.get("speaker"),
                "emotion": e.get("emotion", "neutral"),
            }
            results.append(entry)
        return results

    def _translate_batch_srt(self, srt_batch: str, context_block: str, target_language: str) -> list[dict]:
        """Fallback: translate raw SRT batch without speaker info."""
        # Parse SRT blocks to count entries
        blocks = [b.strip() for b in srt_batch.strip().split("\n\n") if b.strip()]

        prompt = f"""You are a professional subtitle translator. Translate the following subtitles to {target_language}.

{context_block}

Rules:
- Translate each subtitle block to natural, conversational {target_language}
- Keep translations concise to fit subtitle timing
- Return ONLY a JSON array of translated strings, one per subtitle block, in the same order
- The array must have exactly {len(blocks)} elements

Source subtitles (SRT format):
{srt_batch}

Example output: ["Translated line 1", "Translated line 2", ...]"""

        translations = self._call_gemini(prompt)

        # Parse SRT blocks and map translations back
        import re
        results = []
        for i, block in enumerate(blocks):
            block_lines = block.strip().split("\n")
            if len(block_lines) < 2:
                continue
            time_line = block_lines[1] if len(block_lines) > 1 else ""
            time_match = re.match(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})", time_line)
            if time_match:
                g = time_match.groups()
                start_ms = int(g[0])*3600000 + int(g[1])*60000 + int(g[2])*1000 + int(g[3])
                end_ms = int(g[4])*3600000 + int(g[5])*60000 + int(g[6])*1000 + int(g[7])
            else:
                start_ms, end_ms = 0, 0
            text = translations[i] if i < len(translations) else "\n".join(block_lines[2:])
            results.append({
                "start_ms": start_ms,
                "end_ms": end_ms,
                "text": text,
                "speaker": None,
                "emotion": None,
            })
        return results

    def _call_gemini(self, prompt: str, max_retries: int = 3) -> list[dict]:
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        http_options=types.HttpOptions(timeout=300_000),
                    ),
                )
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    wait = 30 * (attempt + 1)
                    logger.warning(f"Rate limited, waiting {wait}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait)
                    continue
                raise

            response_text = response.text
            if not response_text:
                logger.warning(f"Empty response from Gemini (attempt {attempt + 1}/{max_retries})")
                time.sleep(5)
                continue
            try:
                return json.loads(response_text)
            except (json.JSONDecodeError, TypeError):
                # Try extracting JSON array
                start = response_text.find("[")
                end = response_text.rfind("]") + 1
                if start != -1 and end > start:
                    text = response_text[start:end]
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        pass
                    # Fix invalid \escapes from LLM (e.g. \n inside already-quoted strings)
                    import re
                    text = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', text)
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        pass
                raise ValueError("Failed to parse translation response as JSON")

        raise RuntimeError(f"Gemini API failed after {max_retries} retries")

    @staticmethod
    def _split_srt(content: str, max_entries: int = 50) -> list[str]:
        """Split SRT content into batches of max_entries subtitle blocks."""
        blocks = content.strip().split("\n\n")
        batches = []
        for i in range(0, len(blocks), max_entries):
            batches.append("\n\n".join(blocks[i:i + max_entries]))
        return batches

    @staticmethod
    def _split_entries(entries: list[dict], max_entries: int = 50) -> list[list[dict]]:
        """Split mapped subtitle entries into batches."""
        batches = []
        for i in range(0, len(entries), max_entries):
            batches.append(entries[i:i + max_entries])
        return batches

    @staticmethod
    def _ms_to_timestamp(ms: int) -> str:
        h = ms // 3600000
        m = (ms % 3600000) // 60000
        s = (ms % 60000) // 1000
        ms_r = ms % 1000
        return f"{h:02d}:{m:02d}:{s:02d},{ms_r:03d}"

    # Consistent color palette for speakers
    SPEAKER_COLORS = [
        "FF4444",  # red
        "4488FF",  # blue
        "44DD44",  # green
        "FFAA00",  # orange
        "AA44FF",  # purple
        "FF44AA",  # pink
        "44FFDD",  # cyan
        "FFDD44",  # yellow
        "8888FF",  # light blue
        "FF8844",  # coral
    ]

    def _post_process(self, results: list[dict], character_map: dict[str, str] | None) -> list[dict]:
        """Assign consistent colors per unique speaker/character."""
        unique_speakers = sorted(set(
            e.get("speaker") for e in results if e.get("speaker")
        ))
        color_map = {}
        for i, speaker in enumerate(unique_speakers):
            color_map[speaker] = self.SPEAKER_COLORS[i % len(self.SPEAKER_COLORS)]

        for entry in results:
            speaker = entry.get("speaker")
            if speaker:
                entry["color"] = color_map.get(speaker, "FFFFFF")

        return results

    def _build_context(self, segments, metadata, character_map=None, mapped_subtitles=None, tmdb_context=None) -> str:
        lines = ["Context for translation:"]

        if metadata:
            if metadata.get("title"):
                lines.append(f"- Title: {metadata['title']}")
            if metadata.get("season") and metadata.get("episode"):
                lines.append(f"- Season {metadata['season']}, Episode {metadata['episode']}")

        # Show/episode context from TMDB
        if tmdb_context:
            if tmdb_context.get("show_genres"):
                lines.append(f"- Genres: {', '.join(tmdb_context['show_genres'])}")
            if tmdb_context.get("show_overview"):
                lines.append(f"- Show synopsis: {tmdb_context['show_overview']}")
            if tmdb_context.get("episode_name"):
                lines.append(f"- Episode: {tmdb_context['episode_name']}")
            if tmdb_context.get("episode_overview"):
                lines.append(f"- Episode synopsis: {tmdb_context['episode_overview']}")

        # Summarize speakers
        speakers = set(s.get("speaker", "unknown") for s in segments)
        lines.append(f"- {len(speakers)} speakers detected: {', '.join(sorted(speakers))}")

        # Character identifications
        if character_map:
            lines.append("- Character identifications:")
            for speaker, character in sorted(character_map.items()):
                lines.append(f"  {speaker} = {character}")

        # Summarize emotions from mapped subtitles
        if mapped_subtitles:
            emotion_counts = {}
            for e in mapped_subtitles:
                em = e.get("emotion", "neutral")
                emotion_counts[em] = emotion_counts.get(em, 0) + 1
            lines.append(f"- Emotion distribution: {emotion_counts}")

        return "\n".join(lines)
