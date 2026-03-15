"""
LLM-based character identification.
Uses Gemini to assign character names per subtitle line using TMDB cast + speaker hints.
"""

import json
import logging
import time

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
    ) -> list[str | None]:
        """
        Use LLM to assign a character name to each subtitle line.

        Speaker IDs from diarization are used as hints, but the LLM can
        override them based on dialogue content (e.g. speech patterns,
        character names mentioned, tone).

        Returns:
            List of character names, one per mapped_subtitle entry.
            None for lines that couldn't be identified.
        """
        if not cast:
            logger.info("No cast data, skipping character identification")
            return [None] * len(mapped_subtitles)

        # Process in batches to stay within token limits
        batch_size = 100
        all_characters = []
        batches = [mapped_subtitles[i:i + batch_size] for i in range(0, len(mapped_subtitles), batch_size)]

        for batch_idx, batch in enumerate(batches):
            if batch_idx > 0:
                time.sleep(5)
            characters = self._identify_batch(batch, cast, overlap_flags, metadata, batch_idx, len(batches))
            all_characters.extend(characters)

        # Log summary
        from collections import Counter
        counts = Counter(c for c in all_characters if c)
        logger.info(f"Character identification: {len([c for c in all_characters if c])}/{len(all_characters)} lines identified, {dict(counts)}")

        return all_characters

    def _identify_batch(
        self,
        entries: list[dict],
        cast: list[dict],
        overlap_flags: list[dict] | None,
        metadata: dict | None,
        batch_idx: int,
        total_batches: int,
    ) -> list[str | None]:
        prompt = self._build_prompt(entries, cast, overlap_flags, metadata)

        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        http_options=types.HttpOptions(timeout=120_000),
                    ),
                )
                if not response.text:
                    logger.warning(f"Empty response for character ID batch {batch_idx + 1}/{total_batches}")
                    time.sleep(5)
                    continue

                result = json.loads(response.text)
                if not isinstance(result, list):
                    logger.warning(f"Character ID returned non-list: {type(result)}")
                    return [None] * len(entries)

                # Pad or trim to match entry count
                characters = []
                for i in range(len(entries)):
                    if i < len(result) and result[i] and str(result[i]).lower() != "null":
                        characters.append(str(result[i]))
                    else:
                        characters.append(None)
                return characters

            except Exception as e:
                error_str = str(e)
                if ("429" in error_str or "RESOURCE_EXHAUSTED" in error_str) and attempt < 2:
                    wait = 30 * (attempt + 1)
                    logger.warning(f"Character ID rate limited, waiting {wait}s (attempt {attempt + 1}/3)")
                    time.sleep(wait)
                    continue
                logger.error(f"Character identification failed: {e}")
                return [None] * len(entries)

        return [None] * len(entries)

    def _build_prompt(
        self,
        entries: list[dict],
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

        parts.append("\nSubtitle lines to identify (format: [speaker_hint] text):")
        for i, e in enumerate(entries):
            speaker = e.get("speaker", "?")
            num = speaker.replace("SPEAKER_", "") if speaker and speaker.startswith("SPEAKER_") else "?"
            parts.append(f"{i}. [{num}] {e['text']}")

        parts.append(f"""
IMPORTANT: The speaker numbers are hints from audio diarization, but they can be WRONG.
One speaker number might contain dialogue from multiple characters.
Use the actual dialogue content (speech patterns, names mentioned, context) to determine
who is really speaking each line.

Return a JSON array with exactly {len(entries)} elements.
Each element should be a character name from the cast list, or null if uncertain.
Example: ["Takopi", "Shizuka Kuze", null, "Takopi", ...]""")

        return "\n".join(parts)
