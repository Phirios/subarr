"""
Emotion detection from audio segments.
Uses a speech emotion recognition model to detect emotional tone
for each mapped subtitle line.
"""

import logging

logger = logging.getLogger("subarr-worker")


class EmotionDetector:
    def __init__(self, device: str = "cpu"):
        self._model = None
        self._device = device

    def _load_model(self):
        if self._model is None:
            from transformers import pipeline
            logger.info("Loading emotion detection model...")
            device_arg = 0 if self._device.startswith("cuda") else -1
            self._model = pipeline(
                "audio-classification",
                model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                device=device_arg,
            )
            logger.info("Emotion model loaded")

    def process(self, audio_path: str, mapped_subtitles: list[dict]) -> list[dict]:
        """
        For each mapped subtitle, detect emotion from its audio slice.
        Returns mapped_subtitles with 'emotion' field added.
        Only processes segments with enough audio (>= 0.3s).
        """
        self._load_model()

        import numpy as np
        import soundfile as sf

        audio, sample_rate = sf.read(audio_path, dtype="float32")
        # Convert stereo to mono if needed
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        enriched = []
        detected = 0
        for sub in mapped_subtitles:
            start_sample = int(sub["start_ms"] / 1000 * sample_rate)
            end_sample = int(sub["end_ms"] / 1000 * sample_rate)
            audio_slice = audio[start_sample:end_sample]

            if len(audio_slice) >= int(sample_rate * 0.3):
                try:
                    result = self._model(
                        {"raw": audio_slice.astype(np.float32), "sampling_rate": sample_rate},  # noqa: F821
                        top_k=1,
                    )
                    emotion = result[0]["label"] if result else "neutral"
                    detected += 1
                except Exception as e:
                    logger.warning(f"Emotion detection failed for subtitle at {sub['start_ms']}ms: {e}")
                    emotion = "neutral"
            else:
                emotion = "neutral"

            enriched.append({**sub, "emotion": emotion})

        logger.info(f"Emotion detection done: {detected}/{len(mapped_subtitles)} subtitles classified")
        return enriched
