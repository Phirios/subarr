"""
Speaker diarization using pyannote.audio
Identifies who speaks when in the audio.
"""

import logging

import torch

logger = logging.getLogger("subarr-worker")


def _resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Diarizer:
    def __init__(self, auth_token: str, device: str = "auto"):
        self.auth_token = auth_token
        self.device = _resolve_device(device)
        self._pipeline = None

    def _load_model(self):
        if self._pipeline is None:
            from pyannote.audio import Pipeline
            logger.info(f"Loading pyannote diarization model on {self.device}...")
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=self.auth_token,
            )
            self._pipeline.to(self.device)
            logger.info(f"Diarization model loaded on {self.device}")

    def process(self, audio_path: str) -> dict:
        """
        Returns:
        {
            "segments": [{"start": 0.5, "end": 2.3, "speaker": "SPEAKER_00"}, ...],
            "embeddings": {
                "vectors": numpy array (num_speakers, 256),
                "speakers": ["SPEAKER_00", "SPEAKER_01", ...]
            }
        }
        """
        self._load_model()
        # pyannote 4.x returns Annotation directly, 3.x returns wrapper
        result = self._pipeline(audio_path)

        # Handle both pyannote 3.x (wrapper) and 4.x (Annotation directly)
        if hasattr(result, "speaker_diarization"):
            diarization = result.speaker_diarization
        else:
            diarization = result

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
            })

        # Extract speaker embeddings for similarity analysis
        speakers = sorted(set(s["speaker"] for s in segments))
        embeddings = None
        if hasattr(result, "speaker_embeddings") and result.speaker_embeddings is not None:
            embeddings = {
                "vectors": result.speaker_embeddings,
                "speakers": speakers,
            }

        logger.info(f"Found {len(segments)} segments, {len(speakers)} speakers")
        return {"segments": segments, "embeddings": embeddings}
