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
    ) -> list[dict]:
        """
        Translates subtitle entries with context awareness.
        Splits large SRT files into batches to avoid output token limits.
        """
        context_block = self._build_context(segments, emotions, metadata)
        batches = self._split_srt(subtitle_content, max_entries=50)

        logger.info(f"Translating {len(batches)} batch(es)")
        all_results = []
        for i, batch in enumerate(batches):
            if i > 0:
                logger.info("Rate limit pause (15s)...")
                time.sleep(15)
            logger.info(f"Translating batch {i+1}/{len(batches)}")
            result = self._translate_batch(batch, context_block, target_language)
            all_results.extend(result)

        return all_results

    def _translate_batch(self, srt_batch: str, context_block: str, target_language: str) -> list[dict]:
        prompt = f"""You are a professional subtitle translator. Translate the following subtitles to {target_language}.

{context_block}

Rules:
- Preserve the timing and structure
- Adapt tone based on the speaker's emotion and character personality
- Use natural, conversational {target_language}
- Keep translations concise to fit subtitle timing

Source subtitles (SRT format):
{srt_batch}

Return a JSON array where each element has:
- "start_ms": start time in milliseconds
- "end_ms": end time in milliseconds
- "text": translated text
- "speaker": speaker identifier (from diarization, or null)
- "emotion": detected emotion (or null)
- "color": hex color for the speaker (assign consistent colors, e.g. "FF4444" for angry, "44AAFF" for calm)
"""

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                http_options=types.HttpOptions(timeout=300_000),
            ),
        )

        response_text = response.text

        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            start = response_text.find("[")
            end = response_text.rfind("]") + 1
            if start != -1 and end > start:
                return json.loads(response_text[start:end])
            raise ValueError("Failed to parse translation response as JSON")

    @staticmethod
    def _split_srt(content: str, max_entries: int = 50) -> list[str]:
        """Split SRT content into batches of max_entries subtitle blocks."""
        blocks = content.strip().split("\n\n")
        batches = []
        for i in range(0, len(blocks), max_entries):
            batches.append("\n\n".join(blocks[i:i + max_entries]))
        return batches

    def _build_context(self, segments, emotions, metadata) -> str:
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
        lines.append(f"- {len(speakers)} speakers detected: {', '.join(speakers)}")

        # Summarize emotions
        emotion_counts = {}
        for e in emotions:
            em = e.get("emotion", "neutral")
            emotion_counts[em] = emotion_counts.get(em, 0) + 1
        lines.append(f"- Emotion distribution: {emotion_counts}")

        return "\n".join(lines)
