"""
LLM-based character identification.
Uses Gemini to match SPEAKER_XX labels to TMDB character names.
"""

import json
import logging

from google import genai
from google.genai import types

logger = logging.getLogger("subarr-worker")


class CharacterIdentifier:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def identify(
        self,
        mapped_subtitles: list[dict],
        cast: list[dict],
        overlap_flags: list[dict] | None = None,
        metadata: dict | None = None,
    ) -> dict[str, str]:
        """
        Use LLM to match speaker labels to character names from TMDB cast.

        Args:
            mapped_subtitles: subtitle entries with speaker assigned
            cast: TMDB cast list [{"actor": "...", "character": "..."}, ...]
            overlap_flags: overlap detection results (optional context)
            metadata: series/movie metadata (optional context)

        Returns:
            Mapping of speaker label to character name.
            e.g. {"SPEAKER_00": "Shizuka", "SPEAKER_01": "Takopi"}
            Returns {} on error or empty cast.
        """
        if not cast:
            logger.info("No cast data, skipping character identification")
            return {}

        speakers = sorted(set(
            s.get("speaker") for s in mapped_subtitles if s.get("speaker")
        ))
        if not speakers:
            return {}

        # Build speaker dialogue summary
        speaker_lines = {sp: [] for sp in speakers}
        for sub in mapped_subtitles:
            sp = sub.get("speaker")
            if sp and sp in speaker_lines:
                speaker_lines[sp].append(sub["text"])

        prompt = self._build_prompt(speakers, speaker_lines, cast, overlap_flags, metadata)

        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        http_options=types.HttpOptions(timeout=60_000),
                    ),
                )
                result = json.loads(response.text)
                if not isinstance(result, dict):
                    logger.warning(f"Character ID returned non-dict: {type(result)}")
                    return {}

                character_map = {k: v for k, v in result.items() if k in speakers and isinstance(v, str)}
                logger.info(f"Character identification: {character_map}")
                return character_map

            except Exception as e:
                error_str = str(e)
                if ("429" in error_str or "RESOURCE_EXHAUSTED" in error_str) and attempt < 2:
                    wait = 30 * (attempt + 1)
                    logger.warning(f"Character ID rate limited, waiting {wait}s (attempt {attempt + 1}/3)")
                    import time
                    time.sleep(wait)
                    continue
                logger.error(f"Character identification failed: {e}")
                return {}

    def _build_prompt(
        self,
        speakers: list[str],
        speaker_lines: dict[str, list[str]],
        cast: list[dict],
        overlap_flags: list[dict] | None,
        metadata: dict | None,
    ) -> str:
        parts = ["You are an expert at identifying anime/TV characters from their dialogue."]

        if metadata:
            title = metadata.get("title", "Unknown")
            parts.append(f"\nShow/Movie: {title}")
            if metadata.get("season") and metadata.get("episode"):
                parts.append(f"Season {metadata['season']}, Episode {metadata['episode']}")

        parts.append("\nKnown cast (from TMDB):")
        for c in cast:
            parts.append(f"- {c['character']} (played by {c['actor']})")

        parts.append("\nSpeakers and their dialogue:")
        for sp in speakers:
            lines = speaker_lines[sp][:15]  # Limit to avoid token overflow
            parts.append(f"\n{sp}:")
            for line in lines:
                parts.append(f'  "{line}"')

        if overlap_flags:
            parts.append("\nNote: Some lines have overlapping speakers:")
            for flag in overlap_flags[:5]:
                parts.append(f"  - Line {flag['subtitle_index']}: {flag['note']}")

        parts.append("""
Based on the dialogue content and character personalities, match each speaker to a character.
Return a JSON object mapping speaker labels to character names.
Example: {"SPEAKER_00": "Shizuka", "SPEAKER_01": "Takopi"}
Only include speakers you can confidently identify. Omit uncertain ones.""")

        return "\n".join(parts)
