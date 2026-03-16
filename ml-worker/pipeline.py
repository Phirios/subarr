"""
Main ML pipeline orchestrator.
7-step pipeline:
1. Diarization + TMDB Fetch (parallel)
2. Subtitle Mapping (filter + assign, no merge)
3. Overlap Detection
4. Emotion Detection
5. Character ID (LLM) — skip if no TMDB cast
6. Post-ID Merge — merge unidentified speakers via embedding similarity
7. Translation (LLM)
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from character_id import CharacterIdentifier
from config import Settings
from diarization import Diarizer
from emotion import EmotionDetector
from overlap_detection import detect_overlaps
from post_id_merge import post_id_merge
from speaker_mapping import map_speakers_to_subtitles
from tmdb import TMDBClient
from translation import Translator

logger = logging.getLogger("subarr-worker")


class Pipeline:
    def __init__(self, settings: Settings):
        self.diarizer = Diarizer(settings.pyannote_auth_token, device=settings.device, clustering_threshold=settings.diarization_threshold)
        self.emotion_detector = EmotionDetector(device=settings.device)
        self.translator = Translator(settings.gemini_api_key)
        self.tmdb_client = TMDBClient(settings.tmdb_api_key) if settings.tmdb_api_key else None
        self.character_identifier = CharacterIdentifier(settings.gemini_api_key)
        self.post_id_merge_threshold = settings.post_id_merge_threshold

    def process(self, task: dict, redis_client) -> dict:
        job_id = task["job_id"]
        audio_path = task["audio_path"]
        subtitle_path = task["subtitle_path"]
        target_language = task.get("target_language", "tr")
        metadata = task.get("metadata")

        with open(subtitle_path, "r", encoding="utf-8") as f:
            subtitle_content = f.read()

        # Step 1: Diarization + TMDB fetch (parallel)
        self._update_status(redis_client, job_id, "diarizing")
        diarization_result, cast, tmdb_context = self._step1_parallel(audio_path, metadata)

        segments = diarization_result["segments"]
        embeddings = diarization_result.get("embeddings")

        # Step 2: Subtitle mapping (filter + assign, no merge)
        self._update_status(redis_client, job_id, "mapping_speakers")
        mapped_subtitles = map_speakers_to_subtitles(segments, subtitle_content)

        # Step 3: Overlap detection
        overlap_flags = detect_overlaps(segments, mapped_subtitles)

        # Step 4: Emotion detection (on mapped subtitles — only dialogue audio)
        self._update_status(redis_client, job_id, "detecting_emotion")
        mapped_subtitles = self.emotion_detector.process(audio_path, mapped_subtitles)

        # Step 5: Character ID (LLM) — per-line character assignment
        character_names = [None] * len(mapped_subtitles)
        if cast:
            self._update_status(redis_client, job_id, "identifying_characters")
            character_names = self.character_identifier.identify(
                mapped_subtitles, cast, overlap_flags, metadata, tmdb_context
            )

        # Apply character names to mapped_subtitles
        for i, name in enumerate(character_names):
            if name:
                mapped_subtitles[i]["character"] = name

        # Step 6: Post-ID merge — for unidentified lines, use embedding similarity
        # Build character_map from per-line assignments for translation context
        character_map = {}
        for sub in mapped_subtitles:
            speaker = sub.get("speaker")
            char = sub.get("character")
            if speaker and char and speaker not in character_map:
                character_map[speaker] = char

        # Step 7: Translation (LLM)
        self._update_status(redis_client, job_id, "translating")
        translated = self.translator.translate(
            subtitle_content=subtitle_content,
            segments=segments,
            target_language=target_language,
            metadata=metadata,
            mapped_subtitles=mapped_subtitles,
            character_map=character_map if character_map else None,
            tmdb_context=tmdb_context,
        )

        # Write output
        self._update_status(redis_client, job_id, "formatting")
        output_path = subtitle_path.replace("subtitle.srt", "result.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(translated, f, ensure_ascii=False)

        return {"output_path": output_path}

    def _step1_parallel(self, audio_path: str, metadata: dict | None) -> tuple[dict, list[dict], dict]:
        """Run diarization, TMDB cast fetch, and TMDB context fetch in parallel."""
        def fetch_cast():
            if not self.tmdb_client or not metadata:
                return []
            try:
                return self.tmdb_client.get_characters(metadata)
            except Exception as e:
                logger.error(f"TMDB cast fetch failed: {e}")
                return []

        def fetch_context():
            if not self.tmdb_client or not metadata:
                return {}
            try:
                return self.tmdb_client.get_context(metadata)
            except Exception as e:
                logger.error(f"TMDB context fetch failed: {e}")
                return {}

        with ThreadPoolExecutor(max_workers=3) as executor:
            diarization_future = executor.submit(self.diarizer.process, audio_path)
            cast_future = executor.submit(fetch_cast)
            context_future = executor.submit(fetch_context)

            diarization_result = diarization_future.result()
            cast = cast_future.result()
            tmdb_context = context_future.result()

        return diarization_result, cast, tmdb_context

    def _update_status(self, redis_client, job_id: str, status: str):
        job = json.loads(redis_client.get(f"job:{job_id}"))
        job["status"] = status
        job["updated_at"] = datetime.now(timezone.utc).isoformat()
        redis_client.set(f"job:{job_id}", json.dumps(job))
