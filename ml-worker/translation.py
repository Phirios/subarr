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
        emotions: list[dict],
        target_language: str = "tr",
        metadata: dict | None = None,
        mapped_subtitles: list[dict] | None = None,
        character_map: dict[str, str] | None = None,
    ) -> list[dict]:
        """
        Translates subtitle entries with context awareness.
        Uses mapped_subtitles (with speaker info) if available,
        otherwise falls back to raw SRT content.
        """
        context_block = self._build_context(segments, emotions, metadata, character_map)

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

        return all_results

    def _translate_batch_mapped(self, entries: list[dict], context_block: str, target_language: str, character_map: dict[str, str] | None = None) -> list[dict]:
        """Translate a batch of mapped subtitle entries with speaker info."""
        # Format entries for the prompt, using character names if available
        lines = []
        for e in entries:
            speaker = e.get("speaker") or "Unknown"
            if character_map and speaker in character_map:
                speaker = f"{character_map[speaker]} ({speaker})"
            lines.append(f"[{self._ms_to_timestamp(e['start_ms'])} --> {self._ms_to_timestamp(e['end_ms'])}] ({speaker}) {e['text']}")
        source_block = "\n".join(lines)

        prompt = f"""You are a professional subtitle translator. Translate the following subtitles to {target_language}.

{context_block}

Rules:
- Preserve the timing and structure exactly
- Adapt tone based on who is speaking and their emotion
- Use natural, conversational {target_language}
- Keep translations concise to fit subtitle timing
- Keep the speaker identifier exactly as provided

Source subtitles (with speaker assignments):
{source_block}

Return a JSON array where each element has:
- "start_ms": start time in milliseconds (preserve exact values)
- "end_ms": end time in milliseconds (preserve exact values)
- "text": translated text
- "speaker": speaker identifier exactly as provided (e.g. "SPEAKER_00", or null)
- "emotion": detected emotion (or null)
- "color": hex color for the speaker (assign a unique consistent color per speaker, e.g. "FF4444", "44FF44", "4444FF", "FFAA00", "AA44FF")
"""

        return self._call_gemini(prompt)

    def _translate_batch_srt(self, srt_batch: str, context_block: str, target_language: str) -> list[dict]:
        """Fallback: translate raw SRT batch without speaker info."""
        prompt = f"""You are a professional subtitle translator. Translate the following subtitles to {target_language}.

{context_block}

Rules:
- Preserve the timing and structure
- Use natural, conversational {target_language}
- Keep translations concise to fit subtitle timing

Source subtitles (SRT format):
{srt_batch}

Return a JSON array where each element has:
- "start_ms": start time in milliseconds
- "end_ms": end time in milliseconds
- "text": translated text
- "speaker": null
- "emotion": null
- "color": null
"""

        return self._call_gemini(prompt)

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
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                start = response_text.find("[")
                end = response_text.rfind("]") + 1
                if start != -1 and end > start:
                    return json.loads(response_text[start:end])
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

    def _build_context(self, segments, emotions, metadata, character_map=None) -> str:
        lines = ["Context for translation:"]

        if metadata:
            if metadata.get("title"):
                lines.append(f"- Title: {metadata['title']}")
            if metadata.get("season") and metadata.get("episode"):
                lines.append(f"- Season {metadata['season']}, Episode {metadata['episode']}")
            if metadata.get("tmdb_id"):
                lines.append(f"- TMDB ID: {metadata['tmdb_id']}")

        # Summarize speakers
        speakers = set(s.get("speaker", "unknown") for s in segments)
        lines.append(f"- {len(speakers)} speakers detected: {', '.join(sorted(speakers))}")

        # Character identifications
        if character_map:
            lines.append("- Character identifications:")
            for speaker, character in sorted(character_map.items()):
                lines.append(f"  {speaker} = {character}")

        # Summarize emotions
        emotion_counts = {}
        for e in emotions:
            em = e.get("emotion", "neutral")
            emotion_counts[em] = emotion_counts.get(em, 0) + 1
        lines.append(f"- Emotion distribution: {emotion_counts}")

        return "\n".join(lines)
