"""
Speaker diarization using pyannote.audio
Identifies who speaks when in the audio.
"""

import logging

logger = logging.getLogger("subarr-worker")


class Diarizer:
    def __init__(self, auth_token: str):
        self.auth_token = auth_token
        self._pipeline = None

    def _load_model(self):
        if self._pipeline is None:
            from pyannote.audio import Pipeline
            logger.info("Loading pyannote diarization model...")
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.auth_token,
            )
            logger.info("Diarization model loaded")

    def process(self, audio_path: str) -> list[dict]:
        """
        Returns list of segments:
        [{"start": 0.5, "end": 2.3, "speaker": "SPEAKER_00"}, ...]
        """
        self._load_model()
        diarization = self._pipeline(audio_path)

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
            })

        logger.info(f"Found {len(segments)} speaker segments")
        return segments
