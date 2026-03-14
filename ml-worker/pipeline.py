"""
Main ML pipeline orchestrator.
Diarization → Emotion Detection → Translation
"""

import json
import logging
from datetime import datetime, timezone

from diarization import Diarizer
from emotion import EmotionDetector
from translation import Translator

logger = logging.getLogger("subarr-worker")


class Pipeline:
    def __init__(self, settings):
        self.diarizer = Diarizer(settings.pyannote_auth_token)
        self.emotion_detector = EmotionDetector()
        self.translator = Translator(settings.gemini_api_key)

    def process(self, task: dict, redis_client) -> dict:
        job_id = task["job_id"]
        audio_path = task["audio_path"]
        subtitle_path = task["subtitle_path"]
        target_language = task.get("target_language", "tr")
        metadata = task.get("metadata")

        # Read source subtitle
        with open(subtitle_path, "r", encoding="utf-8") as f:
            subtitle_content = f.read()

        # Step 1: Speaker diarization
        self._update_status(redis_client, job_id, "diarizing")
        segments = self.diarizer.process(audio_path)

        # Step 2: Emotion detection
        self._update_status(redis_client, job_id, "detecting_emotion")
        emotions = self.emotion_detector.process(audio_path, segments)

        # Step 3: Translation
        self._update_status(redis_client, job_id, "translating")
        translated = self.translator.translate(
            subtitle_content=subtitle_content,
            segments=segments,
            emotions=emotions,
            target_language=target_language,
            metadata=metadata,
        )

        # Step 4: Write output
        self._update_status(redis_client, job_id, "formatting")
        output_path = subtitle_path.replace("subtitle.srt", "result.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(translated, f, ensure_ascii=False)

        return {"output_path": output_path}

    def _update_status(self, redis_client, job_id: str, status: str):
        job = json.loads(redis_client.get(f"job:{job_id}"))
        job["status"] = status
        job["updated_at"] = datetime.now(timezone.utc).isoformat()
        redis_client.set(f"job:{job_id}", json.dumps(job))
