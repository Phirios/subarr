"""
Subarr ML Worker
Listens on Redis queue "subarr:tasks", processes ML pipeline, pushes results.
Reconnects automatically if Redis is unreachable.
"""

import json
import logging
import time
import redis
from datetime import datetime, timezone

from config import settings
from pipeline import Pipeline

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

    logger.info(f"Subarr ML Worker started (device={settings.device}), waiting for tasks...")

    while True:
        try:
            result = r.brpop("subarr:tasks", timeout=30)
            if result is None:
                continue  # timeout, loop back to check connection

            _, task_json = result
            task = json.loads(task_json)
            job_id = task["job_id"]

            logger.info(f"Processing job {job_id}")

            try:
                output = pipe.process(task, r)

                job = json.loads(r.get(f"job:{job_id}"))
                job["status"] = "completed"
                job["result_path"] = output["output_path"]
                job["updated_at"] = datetime.now(timezone.utc).isoformat()
                r.set(f"job:{job_id}", json.dumps(job))

                logger.info(f"Job {job_id} completed")

            except Exception as e:
                logger.error(f"Job {job_id} failed: {e}")
                try:
                    job = json.loads(r.get(f"job:{job_id}"))
                    job["status"] = "failed"
                    job["error"] = str(e)
                    job["updated_at"] = datetime.now(timezone.utc).isoformat()
                    r.set(f"job:{job_id}", json.dumps(job))
                except Exception:
                    logger.error(f"Could not update failed status for job {job_id}")

        except redis.ConnectionError:
            logger.warning("Redis connection lost, reconnecting...")
            r = connect_redis(settings.redis_url)
            logger.info("Redis reconnected, resuming...")


if __name__ == "__main__":
    main()
