"""
Context-aware subtitle translation using Gemini API.
Takes into account speaker identity, emotion, and series context.
"""

import json
import logging
from google import genai

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
        Returns list of translated entries ready for ASS formatting:
        [{"start_ms": 1000, "end_ms": 4000, "text": "...", "speaker": "...", "emotion": "...", "color": "FF0000"}, ...]
        """
        context_block = self._build_context(segments, emotions, metadata)

        prompt = f"""You are a professional subtitle translator. Translate the following subtitles to {target_language}.

{context_block}

Rules:
- Preserve the timing and structure
- Adapt tone based on the speaker's emotion and character personality
- Use natural, conversational {target_language}
- Keep translations concise to fit subtitle timing
- Return ONLY valid JSON array

Source subtitles (SRT format):
{subtitle_content}

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
        )

        response_text = response.text

        # Extract JSON from response
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to find JSON array in response
            start = response_text.find("[")
            end = response_text.rfind("]") + 1
            if start != -1 and end > start:
                return json.loads(response_text[start:end])
            raise ValueError("Failed to parse translation response as JSON")

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
