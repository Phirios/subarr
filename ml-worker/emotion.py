"""
Emotion detection from audio segments.
Uses a speech emotion recognition model to detect emotional tone.
"""

import logging

logger = logging.getLogger("subarr-worker")


class EmotionDetector:
    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is None:
            from transformers import pipeline
            logger.info("Loading emotion detection model...")
            self._model = pipeline(
                "audio-classification",
                model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
            )
            logger.info("Emotion model loaded")

    def process(self, audio_path: str, segments: list[dict]) -> list[dict]:
        """
        For each speaker segment, detect the dominant emotion.
        Returns list matching segments with emotion labels added:
        [{"start": 0.5, "end": 2.3, "speaker": "SPEAKER_00", "emotion": "happy"}, ...]
        """
        self._load_model()

        # TODO: Slice audio by segment timestamps and classify each slice
        # For now, return segments with neutral emotion as placeholder
        enriched = []
        for seg in segments:
            enriched.append({
                **seg,
                "emotion": "neutral",
            })

        logger.info(f"Emotion detection done for {len(enriched)} segments")
        return enriched
