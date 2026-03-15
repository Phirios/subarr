"""
Subarr ML Worker
Listens on Redis queue "subarr:tasks", processes ML pipeline, pushes results.
Downloads job files from MinIO, uploads results back.
Reconnects automatically if Redis is unreachable.
"""

# Patch torchaudio for compatibility with 2.10+ (removed APIs that pyannote expects)
import torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
if not hasattr(torchaudio, "AudioMetaData"):
    torchaudio.AudioMetaData = type("AudioMetaData", (), {})

# Patch torch.load for PyTorch 2.6+ (weights_only=True by default breaks pyannote)
import torch
_orig_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_load

import json
import logging
import os
import shutil
import tempfile
import time
import redis
from datetime import datetime, timezone

from config import settings
from pipeline import Pipeline
from storage import StorageClient

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger("subarr-worker")

RETRY_INTERVAL = 10  # seconds between reconnect attempts


def connect_redis(url: str) -> redis.Redis:
    while True:
        try:
            r = redis.from_url(url)
            r.ping()
            return r
        except redis.ConnectionError as e:
            logger.warning(f"Redis unreachable ({e}), retrying in {RETRY_INTERVAL}s...")
            time.sleep(RETRY_INTERVAL)


def main():
    r = connect_redis(settings.redis_url)
    pipe = Pipeline(settings)
    storage = StorageClient(
        endpoint=settings.s3_endpoint,
        access_key=settings.s3_access_key,
        secret_key=settings.s3_secret_key,
        bucket=settings.s3_bucket,
    )

    logger.info(f"Subarr ML Worker started (device={settings.device}), waiting for tasks...")

    while True:
        try:
            result = r.brpop("subarr:tasks", timeout=30)
            if result is None:
                continue

            _, task_json = result
            task = json.loads(task_json)
            job_id = task["job_id"]

            logger.info(f"Processing job {job_id}")
            work_dir = tempfile.mkdtemp(prefix=f"subarr-{job_id}-")

            try:
                # Download files from MinIO
                audio_path, subtitle_path = storage.download_job_files(job_id, work_dir)

                # Override task paths with local temp paths
                task["audio_path"] = audio_path
                task["subtitle_path"] = subtitle_path

                output = pipe.process(task, r)

                # Upload result to MinIO
                result_key = f"{job_id}/result.json"
                storage.upload(output["output_path"], result_key)

                job = json.loads(r.get(f"job:{job_id}"))
                job["status"] = "completed"
                job["result_path"] = result_key
                job["updated_at"] = datetime.now(timezone.utc).isoformat()
                r.set(f"job:{job_id}", json.dumps(job))

                logger.info(f"Job {job_id} completed")

            except Exception as e:
                import traceback
                logger.error(f"Job {job_id} failed: {e}\n{traceback.format_exc()}")
                try:
                    job = json.loads(r.get(f"job:{job_id}"))
                    job["status"] = "failed"
                    job["error"] = str(e)
                    job["updated_at"] = datetime.now(timezone.utc).isoformat()
                    r.set(f"job:{job_id}", json.dumps(job))
                except Exception:
                    logger.error(f"Could not update failed status for job {job_id}")
            finally:
                shutil.rmtree(work_dir, ignore_errors=True)

        except redis.ConnectionError:
            logger.warning("Redis connection lost, reconnecting...")
            r = connect_redis(settings.redis_url)
            logger.info("Redis reconnected, resuming...")


if __name__ == "__main__":
    main()
